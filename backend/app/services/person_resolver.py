# backend/app/services/person_resolver.py
from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional, Any, Iterable, Tuple

from app.services.db_service import SQLServerDatabaseService, DatabaseQueryError

logger = logging.getLogger(__name__)

# ------------------------------ #
# Small helpers                   #
# ------------------------------ #

def _coalesce_str(*vals) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _format_name(row: Dict[str, Optional[str]]) -> str:
    """
    Display name priority (PSNACCOUNT):
      1) TRUENAME
      2) FIRSTNAME + MIDDLENAME + LASTNAME
      3) ENGNAME
      4) EMPLOYEEID
      5) PERSONID
    """
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


# ------------------------------ #
# Resolver (PSNACCOUNT + ORG)     #
# ------------------------------ #

class PersonResolver:
    """
    Resolve PERSONID **or** EMPLOYEEID to display info using dbo.PSNACCOUNT
    and department/branch info using dbo.ORGStdStruct.

    Resolution order:
      1) in-memory cache (keyed by PERSONID)
      2) local JSON index (optional) — may include department fields
      3) DB batch lookup from dbo.PSNACCOUNT (by PERSONID and/or EMPLOYEEID)
         -> Collect BRANCHID -> batch-lookup dbo.ORGStdStruct for names/codes

    Returned dict per input key:
      {
        "person_id": str | None,
        "name": str | None,
        "employee_id": str | None,
        "email": str | None,
        "cardnum": str | None,
        "department_id": str | None,
        "department_name": str | None,
        "department_code": str | None
      }
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
        self.cache: Dict[str, Dict[str, Optional[str]]] = {}  # keyed by PERSONID
        self.storage_dir = storage_dir or os.getenv("STORAGE_DIR", "./storage")
        self._local_index: Dict[str, Dict[str, Optional[str]]] = {}

        self._table_exists_cache: Dict[str, bool] = {}
        self._have_psnaccount = self._table_exists("dbo", "PSNACCOUNT")
        self._have_org = self._table_exists("dbo", "ORGStdStruct")
        if not self._have_psnaccount:
            logger.warning("dbo.PSNACCOUNT not found. PersonResolver will serve only from cache/local index.")
        if not self._have_org:
            logger.warning("dbo.ORGStdStruct not found. Department fields will be None unless present in local index.")

        # allow overriding fully-qualified org table if needed (e.g., eHRAntung_DB.dbo.ORGStdStruct)
        self._org_table = os.getenv("ORG_TABLE", "dbo.ORGStdStruct")

        self._org_cache: Dict[str, Dict[str, Optional[str]]] = {}  # keyed by UNITID (NVARCHAR)
        self._load_local_index()

    # ---------- public ----------

    def resolve(self, person_id: str) -> Dict[str, Optional[str]]:
        """Resolve a single PERSONID. (Kept for compatibility.)"""
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
        Batch resolve by PERSONID and (optionally) EMPLOYEEID.

        Returns a dict mapping **the same keys you passed in** (PID or EID)
        to a normalized info payload (including department fields when available).
        """
        clean_pid = [str(p).strip() for p in (person_ids or []) if p and str(p).strip()]
        clean_eid = [str(e).strip() for e in (employee_ids or []) if e and str(e).strip()]

        out: Dict[str, Dict[str, Optional[str]]] = {}

        # 1) cache + local index hits for PERSONID keys
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

        # 2) local index hits for EMPLOYEEID keys (cache is PID-based, so we only alias)
        remaining_eid: List[str] = []
        for eid in clean_eid:
            if eid in self._local_index:
                li = self._local_index[eid]
                info = self._normalize_index_record(li.get("person_id"), li, prefer_eid=eid)
                out[eid] = info
            else:
                remaining_eid.append(eid)

        # If no DB, fill misses with bare fallbacks
        if not self._have_psnaccount:
            for pid in remaining_pid:
                out[pid] = self._bare(pid)
            for eid in remaining_eid:
                out[eid] = self._bare(None, eid)
            return out

        # 3) DB fetch (by PERSONID)
        fetched_by_pid: Dict[str, Dict[str, Optional[str]]] = {}
        if remaining_pid:
            for row in self._fetch_from_psnaccount_by_key("PERSONID", remaining_pid):
                pid = row.get("PERSONID")
                fetched_by_pid[pid] = self._row_to_person_info(row)

        # 4) DB fetch (by EMPLOYEEID)
        fetched_by_eid: Dict[str, Dict[str, Optional[str]]] = {}
        if remaining_eid:
            for row in self._fetch_from_psnaccount_by_key("EMPLOYEEID", remaining_eid):
                eid = row.get("EMPLOYEEID")
                fetched_by_eid[eid] = self._row_to_person_info(row)

        # 5) Gather all branch_ids to resolve departments in one shot
        branch_ids: List[str] = []
        for info in list(fetched_by_pid.values()) + list(fetched_by_eid.values()):
            bid = info.get("department_id")
            if bid:
                branch_ids.append(bid)
        self._populate_org_cache(branch_ids)

        # 6) consolidate + cache; keep caller’s key
        for pid in remaining_pid:
            info = fetched_by_pid.get(pid)
            if info:
                self._attach_org(info)
                out[pid] = info
                self._cache_put(pid, info)
            else:
                out[pid] = self._bare(pid)

        for eid in remaining_eid:
            info = fetched_by_eid.get(eid)
            if info:
                self._attach_org(info)
                out[eid] = info
                if info.get("person_id"):
                    self._cache_put(info["person_id"], info)
            else:
                out[eid] = self._bare(None, eid)

        return out

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
            "org_cache_size": len(self._org_cache),
        }

    # ---------- internals ----------

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
            # simple trim: drop ~20% oldest by insertion order
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
        """Normalize a record loaded from the local JSON index."""
        info = {
            "person_id": (pid or li.get("person_id") or None),
            "name": li.get("name"),
            "employee_id": prefer_eid or li.get("employee_id"),
            "email": li.get("email"),
            "cardnum": li.get("cardnum"),
            "department_id": li.get("department_id") or li.get("branch_id"),
            "department_name": li.get("department_name") or li.get("branch_name"),
            "department_code": li.get("department_code") or li.get("branch_code"),
        }
        return info

    def _load_local_index(self):
        """
        Optional JSON structure (keys may be PERSONID or EMPLOYEEID):
          {
            "P000123": {
              "person_id": "P000123", "name": "王小明", "employee_id": "E123",
              "email": "...", "cardnum": "...",
              "department_id": "2001", "department_name": "HR", "department_code": "HR01"
            },
            "E123": { ...same shape... }
          }
        """
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
                        pid = (v.get("person_id") or "").strip() or None
                        eid = (v.get("employee_id") or "").strip() or None
                        key = (k or "").strip()
                        if not key and not pid and not eid:
                            continue
                        normalized[key or (pid or eid)] = {
                            "person_id": pid,
                            "name": v.get("name"),
                            "employee_id": eid,
                            "email": v.get("email"),
                            "cardnum": v.get("cardnum"),
                            "department_id": v.get("department_id") or v.get("branch_id"),
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

    def _fetch_from_psnaccount_by_key(self, key_col: str, key_vals: List[str]) -> List[Dict[str, Optional[str]]]:
        """Generic batch fetch by PERSONID or EMPLOYEEID. Includes BRANCHID for department resolution."""
        if not key_vals:
            return []
        cols = [
            "PERSONID",
            "TRUENAME",
            "CARDNUM",
            "EMPLOYEEID",
            "COMPANYEMAIL",
            "FIRSTNAME",
            "MIDDLENAME",
            "LASTNAME",
            "ENGNAME",
            "BRANCHID",  # <-- critical for department join
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
        """Map a PSNACCOUNT row into our normalized payload (department_id from BRANCHID)."""
        pid = row.get("PERSONID")
        eid = row.get("EMPLOYEEID")
        bid = (row.get("BRANCHID") or None)
        info = {
            "person_id": pid,
            "name": _format_name(row),
            "employee_id": eid,
            "email": row.get("COMPANYEMAIL") or None,
            "cardnum": row.get("CARDNUM") or None,
            "department_id": str(bid).strip() if bid is not None and str(bid).strip() else None,
            "department_name": None,
            "department_code": None,
        }
        return info

    # ---------- ORG lookups & attach ----------

    def _populate_org_cache(self, unit_ids: Iterable[str]) -> None:
        """Fetch missing UNITIDs from ORGStdStruct in batches and populate _org_cache."""
        if not self._have_org:
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
            # Support fully-qualified override via ORG_TABLE env var
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
                    uid = rec.get("unit_id")
                    if uid:
                        self._org_cache[uid] = {
                            "department_name": rec.get("branch_name"),
                            "department_code": rec.get("branch_code"),
                        }
            except DatabaseQueryError as e:
                logger.warning("ORGStdStruct lookup failed: %s", e)
            except Exception as e:
                logger.error("ORGStdStruct unexpected error: %s", e)

    def _attach_org(self, info: Dict[str, Optional[str]]) -> None:
        """Attach org details to an info dict if department_id exists."""
        dep_id = info.get("department_id")
        if not dep_id:
            return
        rec = self._org_cache.get(str(dep_id).strip())
        if not rec:
            return
        # Only fill if missing (respect any prefilled values)
        info.setdefault("department_name", rec.get("department_name"))
        info.setdefault("department_code", rec.get("department_code"))
