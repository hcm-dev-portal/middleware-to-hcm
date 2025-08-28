# backend/app/services/person_resolver.py
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
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
    """
    Priority:
      1) TRUENAME
      2) FIRST + MIDDLE + LAST (collapsed)
      3) ENGNAME
      4) EMPLOYEEID
      5) PERSONID
    """
    full_en = " ".join([p for p in [
        (row.get("FIRSTNAME") or "").strip(),
        (row.get("MIDDLENAME") or "").strip(),
        (row.get("LASTNAME") or "").strip(),
    ] if p])
    return _coalesce_str(
        row.get("TRUENAME"),
        full_en if full_en else None,
        row.get("ENGNAME"),
        row.get("EMPLOYEEID"),
        row.get("PERSONID"),
    ) or (row.get("PERSONID") or "")

class PersonResolver:
    """
    Resolve PERSONID -> display name (+ optional extras) with 3 layers:
      1) in-memory cache
      2) local JSON index (optional)
      3) DB fallback (batch or single)
    """
    def __init__(
        self,
        db_service: SQLServerDatabaseService,
        storage_dir: Optional[str] = None,
        cache_cap: int = 5000
    ):
        self.db = db_service
        self.cache_cap = cache_cap
        self.cache: Dict[str, Dict[str, Optional[str]]] = {}
        self.storage_dir = storage_dir or os.getenv("STORAGE_DIR", "./storage")
        self._local_index: Dict[str, Dict[str, Optional[str]]] = {}
        self._tried_fallback_table = False
        self._fallback_table_exists = False
        self._load_local_index()

    # ---------- public ----------

    def resolve(self, person_id: str) -> Dict[str, Optional[str]]:
        """Resolve one PERSONID to {'person_id','name','employee_id','email'}."""
        pid = (person_id or "").strip()
        if not pid:
            return {"person_id": person_id, "name": None, "employee_id": None, "email": None}

        # 1) cache
        hit = self.cache.get(pid)
        if hit:
            return hit

        # 2) local index
        if pid in self._local_index:
            info = self._local_index[pid]
            self._cache_put(pid, info)
            return info

        # 3) DB
        results = self.resolve_many([pid])
        return results.get(pid, {"person_id": pid, "name": pid, "employee_id": None, "email": None})

    def resolve_many(self, person_ids: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
        """Batch resolve. Returns {person_id: {...}} for only the input IDs."""
        clean = [p.strip() for p in person_ids if p and str(p).strip()]
        if not clean:
            return {}

        out: Dict[str, Dict[str, Optional[str]]] = {}

        # 1) cache hits
        remaining: List[str] = []
        for pid in clean:
            if pid in self.cache:
                out[pid] = self.cache[pid]
            elif pid in self._local_index:
                out[pid] = self._local_index[pid]
                self._cache_put(pid, out[pid])
            else:
                remaining.append(pid)

        if not remaining:
            return out

        # 2) DB batch
        db_rows = self._fetch_from_db(remaining)
        for r in db_rows:
            pid = r.get("PERSONID")
            if not pid:
                continue
            info = {
                "person_id": pid,
                "name": _format_name(r),
                "employee_id": r.get("EMPLOYEEID"),
                "email": r.get("COMPANYEMAIL") or r.get("EMAIL") or None,
            }
            out[pid] = info
            self._cache_put(pid, info)

        # 3) fill any misses with a bare fallback
        for pid in remaining:
            if pid not in out:
                out[pid] = {"person_id": pid, "name": pid, "employee_id": None, "email": None}
                self._cache_put(pid, out[pid])

        return out

    def status(self) -> Dict[str, any]:
        return {
            "cache_size": len(self.cache),
            "cache_cap": self.cache_cap,
            "local_index_size": len(self._local_index),
            "storage_dir": self.storage_dir,
            "db_connected": self.db.test_connection(),
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
        Optionally load a lightweight people index:
            { "P000123": {"person_id":"P000123","name":"王小明","employee_id":"E123","email":"x@corp"} , ... }
        If you already dumped PSNACCOUNT into JSONL/CSV you can prebuild this file.
        """
        try:
            idx_path = os.path.join(self.storage_dir, "people_index.json")
            if os.path.exists(idx_path):
                with open(idx_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._local_index = data
                    logger.info(f"Loaded people_index.json with {len(self._local_index)} entries")
        except Exception as e:
            logger.warning(f"Failed loading people_index.json: {e}")

    def _fetch_from_db(self, person_ids: List[str]) -> List[Dict[str, Optional[str]]]:
        """
        Batch fetch from dbo.PSNACCOUNT first; optionally fallback to dbo.PSNACCOUNT_D (if present).
        """
        cols = [
            "PERSONID","TRUENAME","FIRSTNAME","MIDDLENAME","LASTNAME","ENGNAME",
            "EMPLOYEEID","COMPANYEMAIL"
        ]
        in_clause = ",".join(["?"] * len(person_ids))

        # Try main table
        sql = f"""
        SELECT {", ".join(cols)}
        FROM dbo.PSNACCOUNT
        WHERE PERSONID IN ({in_clause})
        """
        try:
            rows, headers = self.db.run_select(sql, params=tuple(person_ids), max_rows=10_000)
            return [dict(zip(headers, r)) for r in rows]
        except DatabaseQueryError as e:
            logger.warning(f"PSNACCOUNT lookup failed, trying fallback: {e}")

        # Optional fallback (only check once)
        if not self._tried_fallback_table:
            self._tried_fallback_table = True
            try:
                chk_sql = """
                SELECT 1 FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME='PSNACCOUNT_D'
                """
                chk_rows, _ = self.db.run_select(chk_sql)
                self._fallback_table_exists = bool(chk_rows)
            except Exception:
                self._fallback_table_exists = False

        if self._fallback_table_exists:
            sql2 = f"""
            SELECT {", ".join(cols)}
            FROM dbo.PSNACCOUNT_D
            WHERE PERSONID IN ({in_clause})
            """
            try:
                rows, headers = self.db.run_select(sql2, params=tuple(person_ids), max_rows=10_000)
                return [dict(zip(headers, r)) for r in rows]
            except DatabaseQueryError as e:
                logger.error(f"PSNACCOUNT_D fallback failed: {e}")

        return []
