# ================================================================================
# backend/app/services/data_processing/sql_executor.py
from __future__ import annotations

import logging, re
from typing import List, Tuple, Optional

from app.services.db_service import DatabaseQueryError

logger = logging.getLogger(__name__)


class SQLExecutor:
    
    def __init__(self, db_service):
        self.db_service = db_service
    
    def _friendly_sql_error(self, err: str) -> Optional[str]:
        low = (err or "").lower()
        if "invalid column name" in low or "無效的資料行名稱" in low:
            m = re.search(r"[\"'`\[]\s*([A-Za-z0-9_]+)\s*[\]\"'`]", err) or re.search(r"name '([A-Za-z0-9_]+)'", err)
            col = m.group(1) if m else "a requested field"
            return (
                f"The field `{col}` isn’t available in the current schema. "
                f"I can return the closest valid result without that breakdown."
            )
        if "near 'limit'" in low or "接近 'limit'" in low or "limit" in low:
            return "This is SQL Server; use TOP (N) instead of LIMIT."
        return None

    def guard_and_execute_sql(self, sql: str, rid: Optional[str] = None) -> Tuple[List[Tuple], List[str], Optional[str]]:
        guarded_sql = self._guard_sql(sql)
        if not guarded_sql or guarded_sql.strip().lower() == "select 1 where 1=0":
            return [], [], None
        try:
            rows, cols = self.db_service.run_select(guarded_sql, max_rows=200)
            logger.info("rid=%s SQL ok rows=%d cols=%d", rid, len(rows), len(cols))
            return rows, cols, None
        except DatabaseQueryError as e:
            msg = self._friendly_sql_error(str(e)) or str(e)
            logger.error("rid=%s SQL failed: %s", rid, e)
            return [], [], msg
        except Exception as e:
            msg = self._friendly_sql_error(str(e)) or str(e)
            logger.error("rid=%s SQL unexpected: %s", rid, e, exc_info=True)
            return [], [], msg
    
    def _guard_sql(self, sql: str) -> str:
        """
        Be permissive here and delegate safety checks to db_service._sanitize_sql(),
        which already handles SQL injection protection.
        """
        s = (sql or "").strip()
        if not s:
            return "SELECT 1 WHERE 1=0"
        
        try:
            # returns a cleaned version or raises DatabaseQueryError
            return self.db_service._sanitize_sql(s)
        except Exception as e:
            logger.debug("guard_sql: rejected generated SQL (%s); falling back.", e)
            return "SELECT 1 WHERE 1=0"
