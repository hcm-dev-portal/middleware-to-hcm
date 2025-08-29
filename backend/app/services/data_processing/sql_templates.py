# ================================================================================
# backend/app/services/data_processing/sql_templates.py
from __future__ import annotations

import re
from typing import Optional

from ..helpers.data_utils import get_today_sql_date


class SQLTemplateService:
    """Provides fallback SQL templates for common queries."""
    
    @staticmethod
    def get_fallback_sql(english_query: str) -> Optional[str]:
        """Return predefined SQL for common query patterns."""
        q = english_query.lower()
        
        # Daily who-is-out query
        if "leave" in q and any(w in q for w in ("today", "now", "currently", "current")):
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