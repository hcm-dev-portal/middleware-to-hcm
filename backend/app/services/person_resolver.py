# backend/app/services/person_resolver.py
from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional, Any

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
        ]
        if p
    )
    return _coalesce_str(
        row.get("TRUENAME"),
        full_en if full_en else None,
        row.get("ENGNAME"),
        row.get("EMPLOYEEID"),
        row.get("PERSONID"),
    ) or (row.get("PERSONID") or "")


# ------------------------------ #
# Resolver (PSNACCOUNT + cache)   #
# ------------------------------ #

class PersonResolver:
    """
    Resolve PERSONID **or** EMPLOYEEID to display info using dbo.PSNACCOUNT.

    Resolution order:
      1) in-memory cache (keyed by PERSONID)
      2) local JSON index (optional) — supports: person_id, name, employee_id, email, cardnum
      3) DB batch lookup from dbo.PSNACCOUNT (by PERSONID and/or EMPLOYEEID)

    Returned dict per input key (keeps the caller’s key for mapping back):
      {
        "person_id": str | None,
        "name": str | None,
        "employee_id": str | None,
        "email": str | None,
        "cardnum": str | None
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
        if not self._have_psnaccount:
            logger.warning("dbo.PSNACCOUNT not found. PersonResolver will serve only from cache/local index.")
        self._load_local_index()

    # ---------- public ----------

    def resolve(self, person_id: str) -> Dict[str, Optional[str]]:
        """Resolve a single PERSONID. (Kept for compatibility.)"""
        pid = (person_id or "").strip()
        if not pid:
            return {"person_id": person_id, "name": None, "employee_id": None, "email": None, "cardnum": None}

        hit = self.cache.get(pid)
        if hit:
            return hit

        li = self._local_index.get(pid)
        if li:
            info = {
                "person_id": pid,
                "name": li.get("name"),
                "employee_id": li.get("employee_id"),
                "email": li.get("email"),
                "cardnum": li.get("cardnum"),
            }
            self._cache_put(pid, info)
            return info

        results = self.resolve_many([pid])
        return results.get(pid, {"person_id": pid, "name": pid, "employee_id": None, "email": None, "cardnum": None})

    def resolve_many(
        self,
        person_ids: List[str],
        *,
        employee_ids: List[str] | None = None
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Batch resolve by PERSONID and (optionally) EMPLOYEEID.

        Returns a dict mapping **the same keys you passed in** (PID or EID)
        to a normalized info payload.
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
                li = self._local_index[pid]
                info = {
                    "person_id": pid,
                    "name": li.get("name"),
                    "employee_id": li.get("employee_id"),
                    "email": li.get("email"),
                    "cardnum": li.get("cardnum"),
                }
                out[pid] = info
                self._cache_put(pid, info)
            else:
                remaining_pid.append(pid)

        # 2) local index hits for EMPLOYEEID keys (cache is PID-based, so we only alias)
        remaining_eid: List[str] = []
        for eid in clean_eid:
            if eid in self._local_index:
                li = self._local_index[eid]
                info = {
                    "person_id": li.get("person_id") or None,
                    "name": li.get("name"),
                    "employee_id": eid,
                    "email": li.get("email"),
                    "cardnum": li.get("cardnum"),
                }
                out[eid] = info
            else:
                remaining_eid.append(eid)

        # If no DB, fill misses with bare fallbacks
        if not self._have_psnaccount:
            for pid in remaining_pid:
                out[pid] = {"person_id": pid, "name": pid, "employee_id": None, "email": None, "cardnum": None}
            for eid in remaining_eid:
                out[eid] = {"person_id": None, "name": eid, "employee_id": eid, "email": None, "cardnum": None}
            return out

        # 3) DB fetch (by PERSONID)
        fetched_by_pid: Dict[str, Dict[str, Optional[str]]] = {}
        if remaining_pid:
            for row in self._fetch_from_psnaccount_by_key("PERSONID", remaining_pid):
                pid = row.get("PERSONID")
                info = {
                    "person_id": pid,
                    "name": _format_name(row),
                    "employee_id": row.get("EMPLOYEEID"),
                    "email": row.get("COMPANYEMAIL") or None,
                    "cardnum": row.get("CARDNUM") or None,
                }
                fetched_by_pid[pid] = info

        # 4) DB fetch (by EMPLOYEEID)
        fetched_by_eid: Dict[str, Dict[str, Optional[str]]] = {}
        if remaining_eid:
            for row in self._fetch_from_psnaccount_by_key("EMPLOYEEID", remaining_eid):
                eid = row.get("EMPLOYEEID")
                info = {
                    "person_id": row.get("PERSONID"),
                    "name": _format_name(row),
                    "employee_id": eid,
                    "email": row.get("COMPANYEMAIL") or None,
                    "cardnum": row.get("CARDNUM") or None,
                }
                fetched_by_eid[eid] = info

        # 5) consolidate + cache; keep caller’s key
        for pid in remaining_pid:
            info = fetched_by_pid.get(pid)
            if info:
                out[pid] = info
                self._cache_put(pid, info)
            else:
                out[pid] = {"person_id": pid, "name": pid, "employee_id": None, "email": None, "cardnum": None}

        for eid in remaining_eid:
            info = fetched_by_eid.get(eid)
            if info:
                out[eid] = info
                # cache by PERSONID if present
                if info.get("person_id"):
                    self._cache_put(info["person_id"], info)
            else:
                out[eid] = {"person_id": None, "name": eid, "employee_id": eid, "email": None, "cardnum": None}

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
        }

    # ---------- internals ----------

    def _cache_put(self, pid: str, info: Dict[str, Optional[str]]):
        if len(self.cache) >= self.cache_cap:
            # simple trim: drop ~20% oldest by insertion order
            for k in list(self.cache.keys())[: max(1, self.cache_cap // 5)]:
                self.cache.pop(k, None)
        self.cache[pid] = info

    def _load_local_index(self):
        """
        Optional JSON structure:
          {
            "P000123": { "person_id": "P000123", "name": "王小明", "employee_id": "E123", "email": "...", "cardnum": "..." },
            "E123":    { "person_id": "P000123", "name": "王小明", "employee_id": "E123", "email": "...", "cardnum": "..." }
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
        """Generic batch fetch by PERSONID or EMPLOYEEID."""
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
