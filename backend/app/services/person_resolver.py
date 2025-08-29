# backend/app/services/person_resolver.py
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
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
    Resolve PERSONID -> display info via:
      1) in-memory cache
      2) local JSON index (optional)
      3) DB fallback (batch)
    Tries these person sources (if present): dbo.PSNACCOUNT, dbo.PSNACCOUNT_D, dbo.BIPSNACCOUNTSP
    """

    # keep generous but under SQL Server param limit (2100) with buffer
    _BATCH_SIZE = 1000

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

        # detect once per process
        self._table_exists_cache: Dict[str, bool] = {}
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
        clean = [str(p).strip() for p in person_ids if p and str(p).strip()]
        if not clean:
            return {}

        out: Dict[str, Dict[str, Optional[str]]] = {}

        # 1) cache / local-index hits
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

        # 2) DB batch in chunks to avoid 2100 param limit
        fetched: Dict[str, Dict[str, Optional[str]]] = {}
        for i in range(0, len(remaining), self._BATCH_SIZE):
            chunk = remaining[i:i + self._BATCH_SIZE]
            for row in self._fetch_from_db(chunk):
                pid = row.get("PERSONID")
                if not pid:
                    continue
                info = {
                    "person_id": pid,
                    "name": _format_name(row),
                    "employee_id": row.get("EMPLOYEEID"),
                    "email": row.get("COMPANYEMAIL") or row.get("EMAIL") or None,
                }
                fetched[pid] = info

        # cache + consolidate
        for pid, info in fetched.items():
            out[pid] = info
            self._cache_put(pid, info)

        # 3) fill misses with bare fallback
        for pid in remaining:
            if pid not in out:
                out[pid] = {"person_id": pid, "name": pid, "employee_id": None, "email": None}
                self._cache_put(pid, out[pid])

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
        """
        try:
            idx_path = os.path.join(self.storage_dir, "people_index.json")
            if os.path.exists(idx_path):
                with open(idx_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._local_index = data
                    logger.info("Loaded people_index.json with %d entries", len(self._local_index))
        except Exception as e:
            logger.warning("Failed loading people_index.json: %s", e)

    def _table_exists(self, schema: str, name: str) -> bool:
        key = f"{schema}.{name}".lower()
        if key in self._table_exists_cache:
            return self._table_exists_cache[key]
        try:
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

    def _fetch_from_db(self, person_ids: List[str]) -> List[Dict[str, Optional[str]]]:
        """
        Batch fetch using any available person table.
        We try in this order and fill only missing IDs at each step:
          1) dbo.PSNACCOUNT
          2) dbo.PSNACCOUNT_D
          3) dbo.BIPSNACCOUNTSP
        """
        targets: List[Tuple[str, List[str]]] = []

        # Common columns across sources (safe set)
        base_cols = ["PERSONID", "TRUENAME", "EMPLOYEEID"]
        extraname_cols = ["FIRSTNAME", "MIDDLENAME", "LASTNAME", "ENGNAME"]
        email_cols = ["COMPANYEMAIL"]

        have_psn = self._table_exists("dbo", "PSNACCOUNT")
        have_psn_d = self._table_exists("dbo", "PSNACCOUNT_D")
        have_bi = self._table_exists("dbo", "BIPSNACCOUNTSP")

        if have_psn:
            targets.append(("dbo.PSNACCOUNT", base_cols + extraname_cols + email_cols))
        if have_psn_d:
            targets.append(("dbo.PSNACCOUNT_D", base_cols + extraname_cols + email_cols))
        if have_bi:
            # BI snapshot usually lacks email/name parts beyond TRUENAME
            targets.append(("dbo.BIPSNACCOUNTSP", base_cols))  # keep minimal and safe

        remaining = set(person_ids)
        results: Dict[str, Dict[str, Optional[str]]] = {}

        for full_table, cols in targets:
            if not remaining:
                break
            # Only query what's still missing
            todo = list(remaining)
            placeholders = ",".join(["?"] * len(todo))
            sql = f"SELECT {', '.join(cols)} FROM {full_table} WHERE PERSONID IN ({placeholders})"
            try:
                rows, headers = self.db.run_select(sql, params=tuple(todo), max_rows=10_000)
                for r in rows:
                    row = dict(zip(headers, r))
                    pid = row.get("PERSONID")
                    if pid:
                        results[pid] = row
                        if pid in remaining:
                            remaining.remove(pid)
            except DatabaseQueryError as e:
                logger.warning("%s lookup failed: %s", full_table, e)
            except Exception as e:
                logger.error("%s lookup unexpected error: %s", full_table, e)

        return list(results.values())
