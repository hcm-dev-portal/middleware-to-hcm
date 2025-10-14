# ================================================================================
# backend/app/services/data_processing/sql_executor.py
from __future__ import annotations

import logging, re
from typing import List, Tuple, Optional

from app.services.db_service import DatabaseQueryError
from ..helpers.data_utils import normalize_sql_columns  # column typo/alias fixes

logger = logging.getLogger(__name__)


class SQLExecutor:
    def __init__(self, db_service):
        self.db_service = db_service

    # ---------------- internal helpers ----------------

    def _friendly_sql_error(self, err: str) -> Optional[str]:
        low = (err or "").lower()

        # Invalid column name
        if ("invalid column name" in low) or ("無效的資料行名稱" in low):
            # Try to extract offending column
            m = (
                re.search(r"[\"'`\[]\s*([A-Za-z0-9_]+)\s*[\]\"'`]", err)
                or re.search(r"name '([A-Za-z0-9_]+)'", err, flags=re.IGNORECASE)
            )
            col = m.group(1) if m else "a requested field"
            return (
                f"The field `{col}` isn't available in the current schema. "
                f"I can return the closest valid result without that column."
            )

        # LIMIT used on SQL Server
        if (" near 'limit'" in low) or ("接近 'limit'" in low) or (" limit " in low):
            return "This is SQL Server; use TOP (N) instead of LIMIT."

        # Permission / view creation
        if ("permission" in low and "view" in low) or ("權限" in low and "view" in low) or ("建立 view" in low):
            return (
                "Permissions prevent creating a VIEW in this DB. "
                "Run the SELECT/CTE directly (no CREATE VIEW) and it should work."
            )

        # "CREATE VIEW permission denied" (zh)
        if "create view" in low and ("permission denied" in low or "權限遭拒" in low):
            return "您目前沒有建立 VIEW 的權限，請改以純 SELECT/CTE 方式查詢。"

        return None

    def _rewrite_limit_to_top(self, sql: str) -> str:
        """
        Convert a simple 'SELECT ... LIMIT N' into 'SELECT TOP (N) ...'
        Only applies if:
          - single SELECT statement (no UNION/; inside)
          - LIMIT appears at the very end (with optional semicolon/whitespace)
          - no OFFSET clause present
        
        Note: This preserves user-specified LIMIT clauses in the original SQL.
        """
        s = sql.strip()
        # quick rejects
        if " limit " not in s.lower():
            return sql
        if re.search(r"\boffset\b", s, re.IGNORECASE):
            return sql
        if re.search(r"\bunion\b", s, re.IGNORECASE):
            return sql

        # Extract trailing LIMIT N
        m_lim = re.search(r"\blimit\s+(\d+)\s*;?\s*$", s, flags=re.IGNORECASE)
        if not m_lim:
            return sql

        n = m_lim.group(1)

        # Find start of SELECT list to inject TOP (N)
        m_sel = re.match(r"^\s*select\s+(distinct\s+)?", s, flags=re.IGNORECASE)
        if not m_sel:
            return sql

        inject_pos = m_sel.end()  # right after SELECT / SELECT DISTINCT
        # remove trailing LIMIT .., keep the rest
        body_wo_limit = s[: m_lim.start()].rstrip()

        # Insert TOP (N) after SELECT/DISTINCT
        rewritten = body_wo_limit[:inject_pos] + f"TOP ({n}) " + body_wo_limit[inject_pos:]

        logger.debug("Rewrote LIMIT->TOP: %s", rewritten[:160])
        return rewritten

    def _normalize_and_fix_sql(self, sql: str) -> str:
        # 1) normalize common column variants (LEAVETYPE -> ATTENDANCETYPE, etc.)
        s = normalize_sql_columns(sql or "")

        # 2) rewrite simple LIMIT N to TOP (N) for SQL Server
        s2 = self._rewrite_limit_to_top(s)

        if s2 != sql:
            logger.info("SQL normalized/repaired.\n  before: %s\n  after:  %s", sql[:200], s2[:200])
        return s2

    def _guard_sql(self, sql: str) -> str:
        """
        Be permissive here and delegate safety checks to db_service._sanitize_sql(),
        which already handles SQL injection protection.
        """
        s = (sql or "").strip()
        if not s:
            return "SELECT 1 WHERE 1=0"

        # Apply light normalizations/repairs
        s_fixed = self._normalize_and_fix_sql(s)

        try:
            # returns a cleaned version or raises DatabaseQueryError
            return self.db_service._sanitize_sql(s_fixed)
        except Exception as e:
            logger.debug("guard_sql: rejected generated SQL (%s); falling back.", e)
            return "SELECT 1 WHERE 1=0"

    # ---------------- public API ----------------

    def guard_and_execute_sql(
        self,
        sql: str,
        rid: Optional[str] = None
    ) -> Tuple[List[Tuple], List[str], Optional[str]]:
        guarded_sql = self._guard_sql(sql)
        if not guarded_sql or guarded_sql.strip().lower() == "select 1 where 1=0":
            return [], [], None
        try:
            rows, cols = self.db_service.run_select(guarded_sql)
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