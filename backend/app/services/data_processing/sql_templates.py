# backend/app/services/data_processing/sql_templates.py
from __future__ import annotations

import re
from typing import Optional

from ..helpers.data_utils import get_today_sql_date


class SQLTemplateService:
    """Provides fallback SQL templates for common queries."""

    @staticmethod
    def get_fallback_sql(english_or_local_query: str) -> Optional[str]:
        """
        Return predefined SQL for common query patterns.

        NOTE: Date anchoring is handled upstream by DateProcessor.rewrite_sql_dates(),
        which will replace GETDATE() usage inside get_today_sql_date().
        """
        q = (english_or_local_query or "").lower()

        # Detect "who is out today/currently" in EN or zh (今天/目前/現在/當前)
        is_leave_intent = ("leave" in q) or any(tok in q for tok in ("請假", "休假", "在休假", "谁在休假", "誰在休假"))
        is_todayish = any(tok in q for tok in (
            "today", "now", "currently", "current",
            "今天", "今日", "現在", "目前", "當前", "当前"
        ))

        if is_leave_intent and is_todayish:
            return (
                f"DECLARE @today date = {get_today_sql_date()};\n"
                "SELECT DISTINCT\n"
                "  COALESCE(P.TRUENAME, PD.TRUENAME)       AS Name,\n"
                "  COALESCE(P.EMPLOYEEID, PD.EMPLOYEEID)   AS EMPLOYEEID,\n"
                "  L.PERSONID,\n"
                "  CAST(L.STARTDATE AS date) AS StartDate, L.STARTTIME,\n"
                "  CAST(L.ENDDATE   AS date) AS EndDate,   L.ENDTIME,\n"
                "  CAST(L.WORKDATE  AS date) AS WorkDate,\n"
                "  L.HOURS, L.ATTENDANCETYPE\n"
                "FROM dbo.ATDLEAVEDATA AS L\n"
                "LEFT JOIN dbo.PSNACCOUNT   AS P  ON P.PERSONID  = L.PERSONID\n"
                "LEFT JOIN dbo.PSNACCOUNT_D AS PD ON PD.PERSONID = L.PERSONID AND P.PERSONID IS NULL\n"
                "WHERE (\n"
                "   @today BETWEEN CAST(L.STARTDATE AS date) AND CAST(L.ENDDATE AS date)\n"
                ") OR (\n"
                "   CAST(L.WORKDATE AS date) = @today\n"
                ")\n"
            )

        return None
