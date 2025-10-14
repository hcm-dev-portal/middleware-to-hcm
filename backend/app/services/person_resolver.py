# backend/app/services/person_resolver.py
from __future__ import annotations

import os
import re
import json
import logging
from typing import Dict, List, Optional, Any, Iterable, Tuple

from app.services.db_service import SQLServerDatabaseService, DatabaseQueryError

logger = logging.getLogger(__name__)


def _coalesce_str(*vals) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _format_name(row: Dict[str, Optional[str]]) -> str:
    # Prefer TRUENAME; then assembled English; then ENGNAME; then EMPLOYEEID; then PERSONID.
    full_en = " ".join(
        p for p in [
            (row.get("FIRSTNAME") or "").strip(),
            (row.get("MIDDLENAME") or "").strip(),
            (row.get("LASTNAME") or "").strip(),
        ] if p
    )
    return _coalesce_str(
        row.get("TRUENAME"),
        full_en if full_en else None,
        row.get("ENGNAME"),
        row.get("EMPLOYEEID"),
        row.get("PERSONID"),
    ) or (row.get("PERSONID") or "")


class PersonResolver:
    """
    Resolve identifiers to readable texts:
      • PERSONID / EMPLOYEEID → name, employee_id, email, dept info
      • Department UNITID / BRANCHID → department_name / department_code
      • Leave types: ID / CLASSCODE / CLASSNAME → CLASSNAME

    Department table is configurable via ORG_TABLE env (default dbo.ORGStdStruct).

    COMPATIBILITY GOALS (for PersonEnrichmentService):
      - resolve_many(person_ids=[...], employee_ids=[...]) returns a dict keyed by the SAME
        identifiers that were provided (both pids and eids), each mapping to a normalized info dict:
        {
          person_id, name, employee_id, email, cardnum,
          department_id, department_name, department_code
        }
      - department_id is always a trimmed string if present (NVARCHAR(100) semantics).
    """

    _BATCH_SIZE = 1000  # safely below 2100 param limit

    def __init__(
        self,
        db_service: SQLServerDatabaseService,
        storage_dir: Optional[str] = None,
        cache_cap: int = 5000,
    ):
        self.db = db_service
        self.cache_cap = cache_cap
        # NOTE: cache keyed by PERSONID (authoritative id). EMPLOYEEID results are also returned
        # to the caller, but we store by PERSONID internally for dedupe/size efficiency.
        self.cache: Dict[str, Dict[str, Optional[str]]] = {}
        self.storage_dir = storage_dir or os.getenv("STORAGE_DIR", "./storage")
        self._local_index: Dict[str, Dict[str, Optional[str]]] = {}

        self._table_exists_cache: Dict[str, bool] = {}

        # External tables
        self._org_table = os.getenv("ORG_TABLE", "dbo.ORGStdStruct")
        self._leave_table = os.getenv("LEAVE_CLASS_TABLE", "dbo.ATDATTENDANCECLASS")

        # Presence detection
        self._have_psnaccount = self._table_exists("dbo", "PSNACCOUNT")
        self._have_org = self._fq_table_exists(self._org_table)
        self._have_leave = self._fq_table_exists(self._leave_table)

        if not self._have_psnaccount:
            logger.warning("dbo.PSNACCOUNT not found. PersonResolver will serve only from cache/local index.")
        if not self._have_org:
            logger.warning("%s not found. Department fields may be None.", self._org_table)
        if not self._have_leave:
            logger.warning("%s not found. Leave type resolution will fall back to raw values.", self._leave_table)

        self._org_cache: Dict[str, Dict[str, Optional[str]]] = {}  # UNITID -> {department_name, department_code}
        self._leave_cache: Dict[str, str] = {}  # key (id/classcode/classname norm) -> CLASSNAME
        self._load_local_index()

    # ---------- public: PERSON ----------

    def resolve(self, person_id: str) -> Dict[str, Optional[str]]:
        pid = (person_id or "").strip()
        empty = {
            "person_id": person_id, "name": None, "employee_id": None, "email": None, "cardnum": None,
            "department_id": None, "department_name": None, "department_code": None
        }
        if not pid:
            return empty

        hit = self.cache.get(pid)
        if hit:
            return hit

        li = self._local_index.get(pid)
        if li:
            info = self._normalize_index_record(pid, li)
            self._cache_put(pid, info)
            return info

        results = self.resolve_many([pid])
        return results.get(pid, {**empty, "name": pid})

    def resolve_many(
        self,
        person_ids: List[str],
        *,
        employee_ids: List[str] | None = None
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Return a mapping keyed by the SAME identifiers as provided:
          - every PERSONID in person_ids will be a key
          - every EMPLOYEEID in employee_ids will be a key

        This is critical for PersonEnrichmentService, which looks up the returned dict by
        whatever identifier column it found in the result set.
        """
        clean_pid = [str(p).strip() for p in (person_ids or []) if p and str(p).strip()]
        clean_eid = [str(e).strip() for e in (employee_ids or []) if e and str(e).strip()]

        out: Dict[str, Dict[str, Optional[str]]] = {}

        # Cache/local hits for PID
        remaining_pid: List[str] = []
        for pid in clean_pid:
            if pid in self.cache:
                out[pid] = self.cache[pid]
            elif pid in self._local_index:
                info = self._normalize_index_record(pid, self._local_index[pid])
                out[pid] = info
                self._cache_put(pid, info)
            else:
                remaining_pid.append(pid)

        # Local hits for EID (local index may be keyed by eid)
        remaining_eid: List[str] = []
        for eid in clean_eid:
            if eid in self._local_index:
                li = self._local_index[eid]
                info = self._normalize_index_record(li.get("person_id"), li, prefer_eid=eid)
                out[eid] = info
            else:
                remaining_eid.append(eid)

        # No DB? Return bare for the remaining
        if not self._have_psnaccount:
            for pid in remaining_pid:
                out[pid] = self._bare(pid)
            for eid in remaining_eid:
                out[eid] = self._bare(None, eid)
            return out

        # Fetch by PERSONID
        fetched_by_pid: Dict[str, Dict[str, Optional[str]]] = {}
        if remaining_pid:
            for row in self._fetch_from_psnaccount_by_key("PERSONID", remaining_pid):
                pid = row.get("PERSONID")
                fetched_by_pid[pid] = self._row_to_person_info(row)

        # Fetch by EMPLOYEEID
        fetched_by_eid: Dict[str, Dict[str, Optional[str]]] = {}
        if remaining_eid:
            for row in self._fetch_from_psnaccount_by_key("EMPLOYEEID", remaining_eid):
                eid = row.get("EMPLOYEEID")
                fetched_by_eid[eid] = self._row_to_person_info(row)

        # Collect branch ids to batch-resolve department names
        branch_ids: List[str] = []
        for info in list(fetched_by_pid.values()) + list(fetched_by_eid.values()):
            bid = info.get("department_id")
            if bid:
                branch_ids.append(bid)
        self._populate_org_cache(branch_ids)

        # Stitch PID results (keyed by PID)
        for pid in remaining_pid:
            info = fetched_by_pid.get(pid)
            if info:
                self._attach_org(info)
                out[pid] = info
                self._cache_put(pid, info)  # cache by PERSONID
            else:
                out[pid] = self._bare(pid)

        # Stitch EID results (keyed by EID)
        for eid in remaining_eid:
            info = fetched_by_eid.get(eid)
            if info:
                self._attach_org(info)
                out[eid] = info
                if info.get("person_id"):
                    # cache the same record by PERSONID for future PID lookups
                    self._cache_put(info["person_id"], info)
            else:
                out[eid] = self._bare(None, eid)

        return out

    # ---------- public: DEPARTMENT ----------

    def resolve_departments(self, unit_ids: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
        """
        UNITID/BRANCHID list → { id: {department_name, department_code} }
        """
        ids = [str(u).strip() for u in (unit_ids or []) if u is not None and str(u).strip()]
        self._populate_org_cache(ids)
        out: Dict[str, Dict[str, Optional[str]]] = {}
        for u in ids:
            rec = self._org_cache.get(u)
            if rec:
                out[u] = {"department_name": rec.get("department_name"), "department_code": rec.get("department_code")}
        return out

    # ---------- public: LEAVE TYPES ----------

    def resolve_leave_types(
        self,
        *,
        type_ids: Optional[List[str]] = None,
        class_codes: Optional[List[str]] = None,
        raw_names: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Resolve leave type identifiers to CLASSNAME.
        Keys returned match the inputs so callers can map easily.
        """
        type_ids = [str(v).strip() for v in (type_ids or []) if v is not None and str(v).strip()]
        class_codes = [str(v).strip() for v in (class_codes or []) if v is not None and str(v).strip()]
        raw_names = [str(v).strip() for v in (raw_names or []) if v is not None and str(v).strip()]

        # Ensure cache has these keys (load missing)
        missing_ids = [k for k in type_ids if k not in self._leave_cache]
        missing_codes = [k.upper() for k in class_codes if k.upper() not in self._leave_cache]

        if (missing_ids or missing_codes) and self._have_leave:
            self._populate_leave_cache(missing_ids, missing_codes)

        out: Dict[str, str] = {}
        # Map by id
        for k in type_ids:
            out[k] = self._leave_cache.get(k, k)
        # Map by CLASSCODE (UPPER)
        for k in class_codes:
            out[k] = self._leave_cache.get(k.upper(), k)
        # Names pass-through (also normalize via cache if present)
        for n in raw_names:
            out[n] = self._leave_cache.get(n, n)
        return out

    # ---------- status ----------

    def status(self) -> Dict[str, Any]:
        ok = False
        try:
            ok = bool(self.db.test_connection())
        except Exception:
            ok = False
        return {
            "cache_size": len(self.cache),
            "cache_cap": self.cache_cap,
            "local_index_size": len(self._local_index),
            "storage_dir": self.storage_dir,
            "db_connected": ok,
            "psnaccount_present": self._have_psnaccount,
            "org_present": self._have_org,
            "leave_class_present": self._have_leave,
            "org_cache_size": len(self._org_cache),
            "leave_cache_size": len(self._leave_cache),
        }

    # ---------- internals (people) ----------

    def _bare(self, pid: Optional[str], eid: Optional[str] = None) -> Dict[str, Optional[str]]:
        display = pid or eid
        return {
            "person_id": pid,
            "name": display,
            "employee_id": eid,
            "email": None,
            "cardnum": None,
            "department_id": None,
            "department_name": None,
            "department_code": None,
        }

    def _cache_put(self, pid: str, info: Dict[str, Optional[str]]):
        if not pid:
            return
        if len(self.cache) >= self.cache_cap:
            # Evict ~20% oldest keys
            for k in list(self.cache.keys())[: max(1, self.cache_cap // 5)]:
                self.cache.pop(k, None)
        self.cache[pid] = info

    def _normalize_index_record(
        self,
        pid: Optional[str],
        li: Dict[str, Optional[str]],
        *,
        prefer_eid: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        info = {
            "person_id": (pid or li.get("person_id") or None),
            "name": li.get("name"),
            "employee_id": prefer_eid or li.get("employee_id"),
            "email": li.get("email"),
            "cardnum": li.get("cardnum"),
            "department_id": self._norm_id(li.get("department_id") or li.get("branch_id")),
            "department_name": li.get("department_name") or li.get("branch_name"),
            "department_code": li.get("department_code") or li.get("branch_code"),
        }
        return info

    def _norm_id(self, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    def _load_local_index(self):
        try:
            idx_path = os.path.join(self.storage_dir, "people_index.json")
            if os.path.exists(idx_path):
                with open(idx_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    normalized = {}
                    for k, v in data.items():
                        if not isinstance(v, dict):
                            continue
                        pid = self._norm_id(v.get("person_id"))
                        eid = self._norm_id(v.get("employee_id"))
                        key = self._norm_id(k)  # file key may be pid or eid or arbitrary
                        if not key and not pid and not eid:
                            continue
                        normalized[key or (pid or eid)] = {
                            "person_id": pid,
                            "name": v.get("name"),
                            "employee_id": eid,
                            "email": v.get("email"),
                            "cardnum": v.get("cardnum"),
                            "department_id": self._norm_id(v.get("department_id") or v.get("branch_id")),
                            "department_name": v.get("department_name") or v.get("branch_name"),
                            "department_code": v.get("department_code") or v.get("branch_code"),
                        }
                    self._local_index = normalized
                    logger.info("Loaded people_index.json with %d entries", len(self._local_index))
        except Exception as e:
            logger.warning("Failed loading people_index.json: %s", e)

    def _table_exists(self, schema: str, name: str) -> bool:
        key = f"{schema}.{name}".lower()
        try:
            if key in self._table_exists_cache:
                return self._table_exists_cache[key]
            sql = """
            SELECT 1
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            """
            rows, _ = self.db.run_select(sql, params=(schema, name), max_rows=1)
            exists = bool(rows)
        except Exception:
            exists = False
        self._table_exists_cache[key] = exists
        return exists

    def _fq_table_exists(self, fq: str) -> bool:
        def _strip(x: str) -> str:
            x = x.strip().strip("[]").strip('"')
            return x
        parts = [p for p in re.split(r"\.(?![^\[]*\])", fq) if p]
        parts = [_strip(p) for p in parts]
        if len(parts) == 1:
            schema, table = "dbo", parts[0]
            dbname = None
        elif len(parts) == 2:
            schema, table = parts
            dbname = None
        else:
            dbname, schema, table = parts[-3], parts[-2], parts[-1]
        key = f"{dbname or '(current)'}.{schema}.{table}".lower()
        if key in self._table_exists_cache:
            return self._table_exists_cache[key]
        try:
            if dbname:
                sql = f"""
                    SELECT 1
                    FROM [{dbname}].INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                """
            else:
                sql = """
                    SELECT 1
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                """
            rows, _ = self.db.run_select(sql, params=(schema, table), max_rows=1)
            exists = bool(rows)
        except Exception:
            exists = False
        self._table_exists_cache[key] = exists
        return exists

    def _fetch_from_psnaccount_by_key(self, key_col: str, key_vals: List[str]) -> List[Dict[str, Optional[str]]]:
        if not key_vals:
            return []
        cols = [
            "PERSONID","TRUENAME","CARDNUM","EMPLOYEEID","COMPANYEMAIL",
            "FIRSTNAME","MIDDLENAME","LASTNAME","ENGNAME","BRANCHID",
        ]
        out: List[Dict[str, Optional[str]]] = []
        for i in range(0, len(key_vals), self._BATCH_SIZE):
            chunk = key_vals[i : i + self._BATCH_SIZE]
            placeholders = ",".join(["?"] * len(chunk))
            sql = f"SELECT {', '.join(cols)} FROM dbo.PSNACCOUNT WHERE {key_col} IN ({placeholders})"
            try:
                rows, headers = self.db.run_select(sql, params=tuple(chunk), max_rows=10_000)
                for r in rows:
                    out.append(dict(zip(headers, r)))
            except DatabaseQueryError as e:
                logger.warning("PSNACCOUNT lookup failed (%s): %s", key_col, e)
            except Exception as e:
                logger.error("PSNACCOUNT lookup unexpected error (%s): %s", key_col, e)
        return out

    def _row_to_person_info(self, row: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        pid = self._norm_id(row.get("PERSONID"))
        eid = self._norm_id(row.get("EMPLOYEEID"))
        bid_raw = row.get("BRANCHID")
        bid = self._norm_id(bid_raw)
        info = {
            "person_id": pid,
            "name": _format_name(row),
            "employee_id": eid,
            "email": row.get("COMPANYEMAIL") or None,
            "cardnum": row.get("CARDNUM") or None,
            "department_id": bid,
            "department_name": None,
            "department_code": None,
        }
        return info

    # ---------- internals (ORG) ----------

    def _populate_org_cache(self, unit_ids: Iterable[str]) -> None:
        if not self._fq_table_exists(self._org_table):
            return
        missing = []
        for raw in unit_ids:
            if raw is None:
                continue
            uid = str(raw).strip()
            if not uid:
                continue
            if uid not in self._org_cache:
                missing.append(uid)
        if not missing:
            return
        for i in range(0, len(missing), self._BATCH_SIZE):
            chunk = missing[i : i + self._BATCH_SIZE]
            placeholders = ",".join(["?"] * len(chunk))
            sql = f"""
                SELECT 
                  CAST(UNITID AS NVARCHAR(100)) AS unit_id,
                  COALESCE(UNITDISPLAYNAME, UNITNAME) AS branch_name,
                  UNITCODE AS branch_code
                FROM {self._org_table}
                WHERE CAST(UNITID AS NVARCHAR(100)) IN ({placeholders})
            """
            try:
                rows, headers = self.db.run_select(sql, params=tuple(chunk), max_rows=10_000)
                for r in rows:
                    rec = dict(zip(headers, r))
                    uid = self._norm_id(rec.get("unit_id"))
                    if uid:
                        self._org_cache[uid] = {
                            "department_name": rec.get("branch_name"),
                            "department_code": rec.get("branch_code"),
                        }
            except DatabaseQueryError as e:
                logger.warning("ORG lookup failed: %s", e)
            except Exception as e:
                logger.error("ORG unexpected error: %s", e)

    def _attach_org(self, info: Dict[str, Optional[str]]) -> None:
        dep_id = self._norm_id(info.get("department_id"))
        if not dep_id:
            return
        rec = self._org_cache.get(dep_id)
        if not rec:
            return
        # Only set if not already present
        info.setdefault("department_name", rec.get("department_name"))
        info.setdefault("department_code", rec.get("department_code"))

    # ---------- internals (LEAVE TYPES) ----------

    def _populate_leave_cache(self, ids: List[str], codes_upper: List[str]) -> None:
        if not self._fq_table_exists(self._leave_table):
            return
        need_ids = [k for k in ids if k not in self._leave_cache]
        need_codes = [k for k in codes_upper if k not in self._leave_cache]
        if not need_ids and not need_codes:
            return

        # Build dynamic WHERE for IDs and CLASSCODE
        parts = []
        params: List[Any] = []
        if need_ids:
            parts.append(f"CAST(ID AS NVARCHAR(100)) IN ({','.join(['?']*len(need_ids))})")
            params.extend(need_ids)
        if need_codes:
            parts.append(f"UPPER(CAST(CLASSCODE AS NVARCHAR(100))) IN ({','.join(['?']*len(need_codes))})")
            params.extend(need_codes)

        where = " OR ".join(parts) if parts else "1=0"
        sql = f"""
            SELECT 
              CAST(ID AS NVARCHAR(100)) AS id_norm,
              UPPER(CAST(CLASSCODE AS NVARCHAR(100))) AS code_norm,
              CAST(CLASSNAME AS NVARCHAR(400)) AS class_name
            FROM {self._leave_table}
            WHERE {where}
        """
        try:
            rows, headers = self.db.run_select(sql, params=tuple(params), max_rows=10_000)
            for r in rows or []:
                rec = dict(zip(headers, r))
                id_norm = self._norm_id(rec.get("id_norm"))
                code_norm = self._norm_id(rec.get("code_norm"))
                name = rec.get("class_name") or ""
                if id_norm:
                    self._leave_cache[id_norm] = name
                if code_norm:
                    self._leave_cache[code_norm] = name
                if name:
                    # also map name→name as identity for convenience
                    self._leave_cache[name] = name
        except DatabaseQueryError as e:
            logger.warning("LEAVE CLASS lookup failed: %s", e)
        except Exception as e:
            logger.error("LEAVE CLASS unexpected error: %s", e)
