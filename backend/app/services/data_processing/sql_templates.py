# ================================================================================
# backend/app/services/data_processing/sql_templates.py
from __future__ import annotations

import re
from typing import Optional

from ..helpers.data_utils import get_today_sql_date


def _has_any(q: str, words: list[str]) -> bool:
    ql = q.lower()
    return any(w.lower() in ql for w in words)


def _validated_clause(q: str, default_for_operational: bool = False) -> str:
    """
    Return a 'AND L.VALIDATED = 1' clause when the query implies approved/validated/current.
    """
    if _has_any(q, ["已核准", "已批准", "已驗證", "已验证", "approved", "validated"]):
        return "  AND L.VALIDATED = 1\n"
    if default_for_operational:
        # For 'today/currently/upcoming', default to approved if not explicitly asked otherwise.
        return "  AND L.VALIDATED = 1\n"
    return ""


class SQLTemplateService:
    """Provides fallback SQL templates for common queries (no VIEWs, 5 tables only)."""

    @staticmethod
    def get_fallback_sql(english_or_local_query: str) -> Optional[str]:
        """
        Return predefined SQL for common query patterns.

        NOTE: Date anchoring of GETDATE() is handled upstream by DateProcessor.rewrite_sql_dates(),
        which can replace the result of get_today_sql_date().
        """
        q = (english_or_local_query or "").lower().strip()
        if not q:
            return None

        # --- INTENT FLAGS ---
        is_leave_intent = _has_any(q, ["leave", "請假", "休假", "在休假", "誰在休假", "谁在休假"])
        is_cancel_intent = _has_any(q, ["取消", "cancel", "cancellation", "cancelled", "canceled"])
        is_todayish = _has_any(q, ["today", "now", "currently", "current", "今天", "今日", "現在", "目前", "當前", "当前"])
        is_upcoming = _has_any(q, ["upcoming", "next", "未來", "未来", "接下來", "接下来"])
        is_week_token = _has_any(q, ["this week", "last week", "next week", "本週", "這週", "本周", "上週", "下週", "上周", "下周"])
        is_month_token = _has_any(q, ["this month", "last month", "next month", "本月", "上月", "下月", "這個月", "这个月", "上個月", "下個月", "上个月", "下个月"])
        has_between_dates = bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b.*\b\d{4}-\d{2}-\d{2}\b", q))

        # Helper snippets (always 5 tables max; nullable joins; no VIEWs)
        def _select_core() -> str:
            return (
                "SELECT\n"
                "  COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS department_name,\n"
                "  p.EMPLOYEEID AS employee_id,\n"
                "  p.TRUENAME   AS person_name,\n"
                "  l.PERSONID,\n"
                "  cls.CLASSCODE       AS leave_code,\n"
                "  cls.CLASSNAME       AS leave_type_name,\n"
                "  l.ATTENDANCETYPE,\n"
                "  CAST(l.STARTDATE AS date) AS STARTDATE,\n"
                "  l.STARTTIME,\n"
                "  CAST(l.ENDDATE   AS date) AS ENDDATE,\n"
                "  l.ENDTIME,\n"
                "  CAST(l.WORKDATE  AS date) AS WORKDATE,\n"
                "  l.HOURS\n"
                "FROM dbo.ATDLEAVEDATA l\n"
                "LEFT JOIN dbo.PSNACCOUNT p\n"
                "  ON p.PERSONID = l.PERSONID\n"
                "LEFT JOIN dbo.ATDATTENDANCECLASS cls\n"
                "  ON CAST(cls.ID AS NVARCHAR(100)) = CAST(l.LEAVEID AS NVARCHAR(100))\n"
                "LEFT JOIN dbo.ORGStdStruct org\n"
                "  ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
            )

        # --------------------------------------------------------------------------------
        # 1) TODAY / CURRENTLY on-leave people (default VALIDATED=1)
        # --------------------------------------------------------------------------------
        if is_leave_intent and is_todayish:
            return (
                f"DECLARE @today date = {get_today_sql_date()};\n"
                + _select_core()
                + "WHERE (\n"
                "   @today BETWEEN CAST(l.STARTDATE AS date) AND CAST(l.ENDDATE AS date)\n"
                ") OR (\n"
                "   CAST(l.WORKDATE AS date) = @today\n"
                ")\n"
                + _validated_clause(q, default_for_operational=True) +
                "ORDER BY department_name, person_name;\n"
            )

        # --------------------------------------------------------------------------------
        # 2) UPCOMING window (e.g., next 7 days) – rely on DateProcessor to rewrite query
        #    If the NLP doesn’t rewrite, we still show a sensible default (next 7 days).
        # --------------------------------------------------------------------------------
        if is_leave_intent and is_upcoming:
            return (
                f"DECLARE @base date = {get_today_sql_date()};\n"
                "DECLARE @start date = @base;\n"
                "DECLARE @end   date = DATEADD(day, 6, @base); -- default 7-day window\n"
                + _select_core() +
                "WHERE CAST(l.STARTDATE AS date) <= @end\n"
                "  AND CAST(l.ENDDATE   AS date) >= @start\n"
                + _validated_clause(q, default_for_operational=True) +
                "ORDER BY STARTDATE, department_name, person_name;\n"
            )

        # --------------------------------------------------------------------------------
        # 3) This/Last/Next Week/Month – let DateProcessor rewrite tokens to explicit ranges
        #    Here we just include generic BETWEEN placeholders that will already be replaced
        #    upstream if the processor ran; otherwise they still work if caller fills them.
        # --------------------------------------------------------------------------------
        if is_leave_intent and (is_week_token or is_month_token):
            return (
                "-- Expect DateProcessor to rewrite 'this week/month' etc. into explicit dates.\n"
                + _select_core() +
                "WHERE CAST(l.WORKDATE AS date) BETWEEN /*start_date*/ CAST(GETDATE() AS date) AND /*end_date*/ CAST(GETDATE() AS date)\n"
                + _validated_clause(q, default_for_operational=False) +
                "ORDER BY WORKDATE, department_name, person_name;\n"
            )

        # --------------------------------------------------------------------------------
        # 4) Explicit BETWEEN dates present in the query text
        # --------------------------------------------------------------------------------
        if is_leave_intent and has_between_dates:
            # We don’t parse the dates here; DateProcessor/extractor should supply them upstream.
            # This template simply expects the caller to replace /*start_date*/ and /*end_date*/.
            return (
                _select_core() +
                "WHERE CAST(l.WORKDATE AS date) BETWEEN /*start_date*/ CAST(GETDATE() AS date)\n"
                "                                 AND /*end_date*/   CAST(GETDATE() AS date)\n"
                + _validated_clause(q, default_for_operational=False) +
                "ORDER BY WORKDATE, department_name, person_name;\n"
            )

        # --------------------------------------------------------------------------------
        # 5) Leave cancellations (ATDLEAVECANCELDATA) for a recent window (default last 30 days)
        # --------------------------------------------------------------------------------
        if is_cancel_intent:
            return (
                f"DECLARE @end   date = {get_today_sql_date()};\n"
                "DECLARE @start date = DATEADD(day, -29, @end); -- last 30 days default\n"
                "SELECT\n"
                "  COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS department_name,\n"
                "  p.EMPLOYEEID AS employee_id,\n"
                "  p.TRUENAME   AS person_name,\n"
                "  c.PERSONID,\n"
                "  c.ATTENDANCETYPE AS cancel_type,\n"
                "  CAST(c.WORKDATE AS date) AS WORKDATE,\n"
                "  CAST(c.STARTDATE AS date) AS STARTDATE, c.STARTTIME,\n"
                "  CAST(c.ENDDATE   AS date) AS ENDDATE,   c.ENDTIME,\n"
                "  c.HOURS,\n"
                "  c.REASON         AS cancel_reason,\n"
                "  c.LEAVEREASON    AS cancel_leave_reason,\n"
                "  c.CREATEDATE     AS cancel_created,\n"
                "  c.LASTEDITTIME   AS cancel_lastedit\n"
                "FROM dbo.ATDLEAVECANCELDATA c\n"
                "LEFT JOIN dbo.PSNACCOUNT p\n"
                "  ON p.PERSONID = c.PERSONID\n"
                "LEFT JOIN dbo.ORGStdStruct org\n"
                "  ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
                "WHERE CAST(c.WORKDATE AS date) BETWEEN @start AND @end\n"
                "ORDER BY c.WORKDATE DESC, department_name, person_name;\n"
            )

        # --------------------------------------------------------------------------------
        # 6) Generic “who is on leave” fallback (today as reasonable default)
        # --------------------------------------------------------------------------------
        if is_leave_intent:
            return (
                f"DECLARE @today date = {get_today_sql_date()};\n"
                + _select_core() +
                "WHERE @today BETWEEN CAST(l.STARTDATE AS date) AND CAST(l.ENDDATE AS date)\n"
                + _validated_clause(q, default_for_operational=True) +
                "ORDER BY department_name, person_name;\n"
            )

        return None
