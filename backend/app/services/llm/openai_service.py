# backend/app/services/llm/openai_service.py
from __future__ import annotations

import re
import os
import json
import logging
import time
import hashlib
from typing import List, Optional, Dict, Any, Tuple, Literal

logger = logging.getLogger(__name__)

# Optional OpenAI / LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.schema import BaseMessage, HumanMessage
    from langchain.memory import ConversationBufferMemory
except ImportError:
    ChatOpenAI = None
    ChatPromptTemplate = None
    SystemMessagePromptTemplate = None
    HumanMessagePromptTemplate = None
    BaseMessage = None
    HumanMessage = None
    ConversationBufferMemory = None

# Typed DB exceptions
from app.services.db_service import (
    DatabaseQueryError as DBServiceQueryError,
    DatabaseSyntaxError as DBServiceSyntaxError,
    TableNotFoundError as DBServiceTableNotFoundError,
    ColumnNotFoundError as DBServiceColumnNotFoundError,
    DatabaseDataError as DBServiceDataError,
    DatabaseIntegrityError as DBServiceIntegrityError,
    DatabaseOperationalError as DBServiceOperationalError,
    DatabaseTimeoutError as DBServiceTimeoutError,
    DatabaseConnectionError as DBServiceConnectionError,
    PermissionDeniedError as DBServicePermissionDeniedError,
    DeadlockError as DBServiceDeadlockError,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ORG_TABLE = os.getenv("ORG_TABLE", "[eHRAntung_DB].[dbo].[ORGStdStruct]")
VAC_RESULT_TABLE = os.getenv("VAC_RESULT_TABLE", "[eHRAntung_DB].[dbo].[ATDCALCUVACATIONRESULT]")

# ────────────────────────────────────────────────────────────────────────────────
# Table whitelist (3 leave facts + 2 dims + 1 balance snapshot) and utilities
# ────────────────────────────────────────────────────────────────────────────────
def _norm_ident(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\[\]`\"]", "", s)
    return s.lower()

def _schema_table_suffix(fullish: str) -> str:
    p = _norm_ident(fullish).split(".")
    return ".".join(p[-2:]) if len(p) >= 2 else _norm_ident(fullish)

# Build whitelist suffixes (accept DB-qualified prefixes at runtime)
_ORG_SUFFIX = _schema_table_suffix(ORG_TABLE)
_VAC_SUFFIX = _schema_table_suffix(VAC_RESULT_TABLE)

# add to ALLOWED_TABLE_SUFFIXES
ALLOWED_TABLE_SUFFIXES = {
    "dbo.atdleavedata",
    "dbo.atdleavecanceldata",
    "dbo.atdattendanceclass",
    "dbo.psnaccount",
    "dbo.atdcalcuvacationresult",   # ← NEW
    _ORG_SUFFIX,
}

# in TABLE_WHITELIST_TEXT join, include the exact name:
TABLE_WHITELIST_TEXT = ", ".join(sorted({
    "dbo.ATDLEAVEDATA",
    "dbo.ATDLEAVECANCELDATA",
    "dbo.ATDATTENDANCECLASS",
    "dbo.PSNACCOUNT",
    "dbo.ATDCALCUVACATIONRESULT",   # ← NEW
    re.sub(r"[\[\]]", "", ORG_TABLE),
}))

def _extract_sql_tables(sql: str) -> list[str]:
    toks = re.findall(r"(?i)\bfrom\s+([^\s\(\),]+)|\bjoin\s+([^\s\(\),]+)", sql or "")
    raw = [t[0] or t[1] for t in toks if (t[0] or t[1])]
    cleaned = []
    for t in raw:
        t = t.rstrip(",")
        t = t.split("\n")[0]
        cleaned.append(_norm_ident(t))
    return [t for t in cleaned if "." in t]

def _tables_with_bad_whitelist(sql: str) -> list[str]:
    bad = []
    for t in _extract_sql_tables(sql):
        suffix = _schema_table_suffix(t)
        if suffix not in ALLOWED_TABLE_SUFFIXES:
            bad.append(t)
    return bad

def _prune_schema_text(schema_text: str) -> str:
    if not schema_text:
        return ""
    keep = []
    for line in schema_text.splitlines():
        ln = _norm_ident(line)
        if any(suf in ln for suf in ALLOWED_TABLE_SUFFIXES):
            keep.append(line)
    return "\n".join(keep) if len(keep) >= 5 else schema_text


# ────────────────────────────────────────────────────────────────────────────────
# Language detection (lightweight & robust for mixed zh/en)
# ────────────────────────────────────────────────────────────────────────────────
def detect_query_language(text: str) -> Literal["zh-tw", "en"]:
    if not text or not text.strip():
        return "en"
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin_num = sum(1 for c in text if c.isascii() and (c.isalpha() or c.isdigit()))
    if chinese_chars >= 2 and chinese_chars >= latin_num:
        return "zh-tw"
    if any(k in text for k in ["請假", "考勤", "部門", "員工", "今天", "現在", "統計", "趨勢"]):
        return "zh-tw"
    return "en"


class UnifiedBilingualOpenAIService:
    """
    Bilingual T-SQL generation/repair/explanation with strict SELECT-only guardrails,
    now intent-aware (template_ref + slots + tables) and balance-snapshot aware.
    """

    # ────────────────────────────────────────────────────────────────────────
    # Few-shot templates keyed by template_ref (bilingual, alias-safe)
    # ────────────────────────────────────────────────────────────────────────
    FEW_SHOT_TEMPLATES: Dict[str, Dict[str, str]] = {
        # Current on-leave
        "current_on_leave_by_dept": {
            "en": (
                "-- intent: current_on_leave_by_dept\n"
                "WITH x AS (\n"
                "    SELECT fact.PERSONID\n"
                "    FROM dbo.ATDLEAVEDATA AS fact\n"
                "    WHERE fact.VALIDATED = 1\n"
                "      AND CAST(@today AS date) BETWEEN CAST(fact.STARTDATE AS date) AND CAST(fact.ENDDATE AS date)\n"
                ")\n"
                "SELECT COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS department_name,\n"
                "       p.EMPLOYEEID AS employee_id,\n"
                "       p.TRUENAME   AS person_name,\n"
                "       fact.ATTENDANCETYPE,\n"
                "       CAST(fact.STARTDATE AS date) AS STARTDATE,\n"
                "       CAST(fact.ENDDATE   AS date) AS ENDDATE\n"
                "FROM dbo.ATDLEAVEDATA AS fact\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = fact.PERSONID\n"
                f"LEFT JOIN {ORG_TABLE} AS org\n"
                "  ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
                "WHERE fact.VALIDATED = 1\n"
                "  AND CAST(@today AS date) BETWEEN CAST(fact.STARTDATE AS date) AND CAST(fact.ENDDATE AS date)"
            ),
            "zh": (
                "-- 意圖: current_on_leave_by_dept\n"
                "WITH x AS (\n"
                "    SELECT fact.PERSONID\n"
                "    FROM dbo.ATDLEAVEDATA AS fact\n"
                "    WHERE fact.VALIDATED = 1\n"
                "      AND CAST(@today AS date) BETWEEN CAST(fact.STARTDATE AS date) AND CAST(fact.ENDDATE AS date)\n"
                ")\n"
                "SELECT COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS 部門,\n"
                "       p.EMPLOYEEID AS 員編,\n"
                "       p.TRUENAME   AS 姓名,\n"
                "       fact.ATTENDANCETYPE,\n"
                "       CAST(fact.STARTDATE AS date) AS STARTDATE,\n"
                "       CAST(fact.ENDDATE   AS date) AS ENDDATE\n"
                "FROM dbo.ATDLEAVEDATA AS fact\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = fact.PERSONID\n"
                f"LEFT JOIN {ORG_TABLE} AS org\n"
                "  ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
                "WHERE fact.VALIDATED = 1\n"
                "  AND CAST(@today AS date) BETWEEN CAST(fact.STARTDATE AS date) AND CAST(fact.ENDDATE AS date)"
            ),
        },
        # Cancellations
        "cancellations_detail": {
            "en": (
                "-- intent: cancellations_detail\n"
                "WITH cancel AS (\n"
                "  SELECT CAST(c.PERSONID AS NVARCHAR(100)) AS person_id_norm,\n"
                "         CAST(c.FORM_NO  AS NVARCHAR(100)) AS form_no_norm,\n"
                "         CAST(c.RECORD_ID AS NVARCHAR(100)) AS record_id_norm,\n"
                "         c.OID, c.ATTENDANCETYPE, c.WORKDATE, c.STARTDATE, c.ENDDATE,\n"
                "         c.STARTTIME, c.ENDTIME, c.HOURS, c.REASON AS cancel_reason,\n"
                "         c.LEAVEREASON, c.CREATEDATE AS cancel_createdate, c.LASTEDITTIME\n"
                "  FROM dbo.ATDLEAVECANCELDATA c\n"
                "  WHERE (@from IS NULL OR CAST(c.WORKDATE AS date) >= CAST(@from AS date))\n"
                "    AND (@to   IS NULL OR CAST(c.WORKDATE AS date) <  CAST(@to   AS date))\n"
                "), leave_data AS (\n"
                "  SELECT CAST(ld.FORM_NO AS NVARCHAR(100)) AS form_no_norm,\n"
                "         CAST(ld.RECORD_ID AS NVARCHAR(100)) AS record_id_norm,\n"
                "         CAST(ld.PERSONID AS NVARCHAR(100)) AS person_id_norm,\n"
                "         CAST(ld.LEAVEID AS NVARCHAR(100)) AS leave_id_norm,\n"
                "         ld.ATTENDANCETYPE, ld.STARTDATE, ld.ENDDATE, ld.STARTTIME, ld.ENDTIME, ld.HOURS, ld.LEAVEREASON\n"
                "  FROM dbo.ATDLEAVEDATA ld\n"
                ")\n"
                "SELECT p.PERSONID AS person_id,\n"
                "       p.EMPLOYEEID AS employee_id,\n"
                "       p.TRUENAME AS person_name,\n"
                "       c.OID AS cancel_oid,\n"
                "       c.form_no_norm AS cancel_form_no,\n"
                "       c.cancel_reason,\n"
                "       c.LEAVEREASON AS cancel_leave_reason,\n"
                "       c.cancel_createdate,\n"
                "       c.LASTEDITTIME AS cancel_lastedit_time,\n"
                "       ld.ATTENDANCETYPE AS original_leave_type,\n"
                "       ld.STARTDATE AS original_start_date,\n"
                "       ld.ENDDATE   AS original_end_date,\n"
                "       ld.HOURS     AS original_leave_hours,\n"
                "       ld.LEAVEREASON AS original_leave_reason\n"
                "FROM cancel c\n"
                "LEFT JOIN leave_data ld ON (c.form_no_norm = ld.form_no_norm OR c.record_id_norm = ld.record_id_norm)\n"
                "LEFT JOIN dbo.PSNACCOUNT p ON c.person_id_norm = CAST(p.PERSONID AS NVARCHAR(100))"
            ),
            "zh": (
                "-- 意圖: cancellations_detail\n"
                "WITH cancel AS (\n"
                "  SELECT CAST(c.PERSONID AS NVARCHAR(100)) AS person_id_norm,\n"
                "         CAST(c.FORM_NO  AS NVARCHAR(100)) AS form_no_norm,\n"
                "         CAST(c.RECORD_ID AS NVARCHAR(100)) AS record_id_norm,\n"
                "         c.OID, c.ATTENDANCETYPE, c.WORKDATE, c.STARTDATE, c.ENDDATE,\n"
                "         c.STARTTIME, c.ENDTIME, c.HOURS, c.REASON AS cancel_reason,\n"
                "         c.LEAVEREASON, c.CREATEDATE AS cancel_createdate, c.LASTEDITTIME\n"
                "  FROM dbo.ATDLEAVECANCELDATA c\n"
                "  WHERE (@from IS NULL OR CAST(c.WORKDATE AS date) >= CAST(@from AS date))\n"
                "    AND (@to   IS NULL OR CAST(c.WORKDATE AS date) <  CAST(@to   AS date))\n"
                "), leave_data AS (\n"
                "  SELECT CAST(ld.FORM_NO AS NVARCHAR(100)) AS form_no_norm,\n"
                "         CAST(ld.RECORD_ID AS NVARCHAR(100)) AS record_id_norm,\n"
                "         CAST(ld.PERSONID AS NVARCHAR(100)) AS person_id_norm,\n"
                "         CAST(ld.LEAVEID AS NVARCHAR(100)) AS leave_id_norm,\n"
                "         ld.ATTENDANCETYPE, ld.STARTDATE, ld.ENDDATE, ld.STARTTIME, ld.ENDTIME, ld.HOURS, ld.LEAVEREASON\n"
                "  FROM dbo.ATDLEAVEDATA ld\n"
                ")\n"
                "SELECT p.PERSONID AS person_id,\n"
                "       p.EMPLOYEEID AS employee_id,\n"
                "       p.TRUENAME AS person_name,\n"
                "       c.OID AS cancel_oid,\n"
                "       c.form_no_norm AS cancel_form_no,\n"
                "       c.cancel_reason,\n"
                "       c.LEAVEREASON AS cancel_leave_reason,\n"
                "       c.cancel_createdate,\n"
                "       c.LASTEDITTIME AS cancel_lastedit_time,\n"
                "       ld.ATTENDANCETYPE AS original_leave_type,\n"
                "       ld.STARTDATE AS original_start_date,\n"
                "       ld.ENDDATE   AS original_end_date,\n"
                "       ld.HOURS     AS original_leave_hours,\n"
                "       ld.LEAVEREASON AS original_leave_reason\n"
                "FROM cancel c\n"
                "LEFT JOIN leave_data ld ON (c.form_no_norm = ld.form_no_norm OR c.record_id_norm = ld.record_id_norm)\n"
                "LEFT JOIN dbo.PSNACCOUNT p ON c.person_id_norm = CAST(p.PERSONID AS NVARCHAR(100))"
            ),
        },
        # Resolve class for leave_id → readable type
        "resolve_leave_class": {
            "en": (
                "-- intent: resolve_leave_class\n"
                "SELECT p.PERSONID AS person_id,\n"
                "       p.EMPLOYEEID AS employee_id,\n"
                "       p.TRUENAME AS person_name,\n"
                "       cls.ID AS leave_id,\n"
                "       cls.CLASSCODE AS leave_code,\n"
                "       cls.CLASSNAME AS leave_type_name,\n"
                "       cls.CLASSTYPE AS leave_class_type,\n"
                "       fact.ATTENDANCETYPE,\n"
                "       fact.WORKDATE, fact.STARTDATE, fact.ENDDATE, fact.STARTTIME, fact.ENDTIME,\n"
                "       fact.HOURS, fact.DEPARTMENTID, fact.LEAVEREASON\n"
                "FROM dbo.ATDLEAVEDATA AS fact\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = fact.PERSONID\n"
                "LEFT JOIN dbo.ATDATTENDANCECLASS AS cls\n"
                "  ON CAST(fact.LEAVEID AS NVARCHAR(100)) = CAST(cls.ID AS NVARCHAR(100))"
            ),
            "zh": (
                "-- 意圖: resolve_leave_class\n"
                "SELECT p.PERSONID AS person_id,\n"
                "       p.EMPLOYEEID AS employee_id,\n"
                "       p.TRUENAME AS person_name,\n"
                "       cls.ID AS leave_id,\n"
                "       cls.CLASSCODE AS 假別代碼,\n"
                "       cls.CLASSNAME AS 假別名稱,\n"
                "       cls.CLASSTYPE AS 假別類型,\n"
                "       fact.ATTENDANCETYPE,\n"
                "       fact.WORKDATE, fact.STARTDATE, fact.ENDDATE, fact.STARTTIME, fact.ENDTIME,\n"
                "       fact.HOURS, fact.DEPARTMENTID, fact.LEAVEREASON\n"
                "FROM dbo.ATDLEAVEDATA AS fact\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = fact.PERSONID\n"
                "LEFT JOIN dbo.ATDATTENDANCECLASS AS cls\n"
                "  ON CAST(fact.LEAVEID AS NVARCHAR(100)) = CAST(cls.ID AS NVARCHAR(100))"
            ),
        },
        # Usage by type/when/who
        "usage_by_type_when_who": {
            "en": (
                "-- intent: usage_by_type_when_who\n"
                "SELECT\n"
                "  COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS department_name,\n"
                "  p.EMPLOYEEID AS employee_id,\n"
                "  p.TRUENAME   AS person_name,\n"
                "  cls.CLASSCODE AS leave_code,\n"
                "  cls.CLASSNAME AS leave_type_name,\n"
                "  CAST(fact.WORKDATE AS date) AS work_date,\n"
                "  SUM(ISNULL(fact.HOURS,0)) AS total_hours\n"
                "FROM dbo.ATDLEAVEDATA AS fact\n"
                "LEFT JOIN dbo.ATDATTENDANCECLASS AS cls\n"
                "  ON CAST(fact.LEAVEID AS NVARCHAR(100)) = CAST(cls.ID AS NVARCHAR(100))\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = fact.PERSONID\n"
                f"LEFT JOIN {ORG_TABLE} AS org\n"
                "  ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
                "WHERE fact.VALIDATED = 1\n"
                "  AND (@from IS NULL OR CAST(fact.WORKDATE AS date) >= CAST(@from AS date))\n"
                "  AND (@to   IS NULL OR CAST(fact.WORKDATE AS date) <  CAST(@to   AS date))\n"
                "GROUP BY COALESCE(org.UNITDISPLAYNAME, org.UNITNAME), p.EMPLOYEEID, p.TRUENAME, cls.CLASSCODE, cls.CLASSNAME, CAST(fact.WORKDATE AS date)\n"
                "ORDER BY work_date, department_name, person_name, leave_type_name"
            ),
            "zh": (
                "-- 意圖: usage_by_type_when_who\n"
                "SELECT\n"
                "  COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS 部門,\n"
                "  p.EMPLOYEEID AS 員編,\n"
                "  p.TRUENAME   AS 姓名,\n"
                "  cls.CLASSCODE AS 假別代碼,\n"
                "  cls.CLASSNAME AS 假別名稱,\n"
                "  CAST(fact.WORKDATE AS date) AS 工作日,\n"
                "  SUM(ISNULL(fact.HOURS,0)) AS 總時數\n"
                "FROM dbo.ATDLEAVEDATA AS fact\n"
                "LEFT JOIN dbo.ATDATTENDANCECLASS AS cls\n"
                "  ON CAST(fact.LEAVEID AS NVARCHAR(100)) = CAST(cls.ID AS NVARCHAR(100))\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = fact.PERSONID\n"
                f"LEFT JOIN {ORG_TABLE} AS org\n"
                "  ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
                "WHERE fact.VALIDATED = 1\n"
                "  AND (@from IS NULL OR CAST(fact.WORKDATE AS date) >= CAST(@from AS date))\n"
                "  AND (@to   IS NULL OR CAST(fact.WORKDATE AS date) <  CAST(@to   AS date))\n"
                "GROUP BY COALESCE(org.UNITDISPLAYNAME, org.UNITNAME), p.EMPLOYEEID, p.TRUENAME, cls.CLASSCODE, cls.CLASSNAME, CAST(fact.WORKDATE AS date)\n"
                "ORDER BY 工作日, 部門, 姓名, 假別名稱"
            ),
        },
        # ✅ Authoritative: remaining balance by person (snapshot table)
        "remaining_balance_by_person": {
            "en": (
                "-- intent: remaining_balance_by_person (authoritative balance source)\n"
                "SELECT p.PERSONID        AS person_id,\n"
                "       p.EMPLOYEEID      AS employee_id,\n"
                "       p.TRUENAME        AS person_name,\n"
                "       bal.VACAYEAR      AS year,\n"
                "       bal.VACAMONTH     AS month,\n"
                "       bal.VACATIONTYPE  AS vacation_type_code,\n"
                "       bal.VACDAYS       AS entitlement_days,\n"
                "       bal.USEDAYS       AS used_days,\n"
                "       bal.REMAINDAYS    AS remaining_days,\n"
                "       bal.CANUSEDATE    AS can_use_from,\n"
                "       bal.DISABLEDDATE  AS disable_on,\n"
                "       bal.LASTYEARREMAINDAYS AS carry_over_days\n"
                "FROM dbo.ATDCALCUVACATIONRESULT AS bal\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = bal.PERSONID\n"
                "WHERE (@year IS NULL OR bal.VACAYEAR = @year)\n"
                "  AND bal.REMAINDAYS > 0\n"
                "  AND (@today IS NULL OR (bal.CANUSEDATE <= CAST(@today AS date))\n"
                "                        AND (bal.DISABLEDDATE IS NULL OR bal.DISABLEDDATE >= CAST(@today AS date)))\n"
            ),
            "zh": (
                "-- 意圖: remaining_balance_by_person（權威餘額來源）\n"
                "SELECT p.PERSONID        AS 人員ID,\n"
                "       p.EMPLOYEEID      AS 員工編號,\n"
                "       p.TRUENAME        AS 姓名,\n"
                "       bal.VACAYEAR      AS 年度,\n"
                "       bal.VACAMONTH     AS 月份,\n"
                "       bal.VACATIONTYPE  AS 假別代碼,\n"
                "       bal.VACDAYS       AS 給定天數,\n"
                "       bal.USEDAYS       AS 已用天數,\n"
                "       bal.REMAINDAYS    AS 剩餘天數,\n"
                "       bal.CANUSEDATE    AS 可用起日,\n"
                "       bal.DISABLEDDATE  AS 失效日,\n"
                "       bal.LASTYEARREMAINDAYS AS 去年遞延天數\n"
                "FROM dbo.ATDCALCUVACATIONRESULT AS bal\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = bal.PERSONID\n"
                "WHERE (@year IS NULL OR bal.VACAYEAR = @year)\n"
                "  AND bal.REMAINDAYS > 0\n"
                "  AND (@today IS NULL OR (bal.CANUSEDATE <= CAST(@today AS date))\n"
                "                        AND (bal.DISABLEDDATE IS NULL OR bal.DISABLEDDATE >= CAST(@today AS date)))\n"
            ),
            "zh": (
                f"-- 意圖: remaining_balance_by_person（權威來源：{VAC_RESULT_TABLE}）\n"
                "WITH latest AS (\n"
                "  SELECT r.PERSONID, r.VACAYEAR, r.VACAMONTH, r.VACATIONTYPE,\n"
                "         r.VACDAYS, r.USEDAYS, r.REMAINDAYS, r.CANUSEDATE, r.DISABLEDDATE,\n"
                "         r.LASTEDITTIME, r.CREATIONTIME,\n"
                "         ROW_NUMBER() OVER (\n"
                "           PARTITION BY r.PERSONID, r.VACAYEAR, r.VACATIONTYPE\n"
                "           ORDER BY ISNULL(r.LASTEDITTIME, r.CREATIONTIME) DESC,\n"
                "                    ISNULL(r.DISABLEDDATE, '9999-12-31') DESC,\n"
                "                    r.VACAMONTH DESC\n"
                "         ) AS rn\n"
                f"  FROM {VAC_RESULT_TABLE} AS r\n"
                "  WHERE (@year IS NULL OR r.VACAYEAR = @year)\n"
                "    AND (@vacationtype IS NULL OR r.VACATIONTYPE = @vacationtype)\n"
                "    AND (r.CANUSEDATE IS NULL OR CAST(@today AS date) >= CAST(r.CANUSEDATE AS date))\n"
                "    AND (r.DISABLEDDATE IS NULL OR CAST(@today AS date) <= CAST(r.DISABLEDDATE AS date))\n"
                ")\n"
                "SELECT COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS 部門,\n"
                "       p.EMPLOYEEID AS 員編,\n"
                "       p.TRUENAME   AS 姓名,\n"
                "       l.VACAYEAR, l.VACATIONTYPE, l.VACDAYS, l.USEDAYS, l.REMAINDAYS,\n"
                "       l.CANUSEDATE, l.DISABLEDDATE\n"
                "FROM latest AS l\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = l.PERSONID\n"
                f"LEFT JOIN {ORG_TABLE} AS org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
                "WHERE l.rn = 1\n"
                "ORDER BY 部門, 姓名"
            ),
        },
        # Person → Branch map
        "person_branch_map": {
            "en": (
                "-- intent: person_branch_map\n"
                "SELECT p.PERSONID AS person_id,\n"
                "       p.TRUENAME  AS person_name,\n"
                "       CAST(p.BRANCHID AS NVARCHAR(100)) AS branch_id,\n"
                "       COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS branch_name,\n"
                "       org.UNITCODE AS branch_code,\n"
                "       ISNULL(org.ISDELETE, 0) AS branch_is_deleted_flag\n"
                "FROM dbo.PSNACCOUNT AS p\n"
                f"LEFT JOIN {ORG_TABLE} AS org\n"
                "  ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))"
            ),
            "zh": (
                "-- 意圖: person_branch_map\n"
                "SELECT p.PERSONID AS person_id,\n"
                "       p.TRUENAME  AS person_name,\n"
                "       CAST(p.BRANCHID AS NVARCHAR(100)) AS branch_id,\n"
                "       COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS branch_name,\n"
                "       org.UNITCODE AS branch_code,\n"
                "       ISNULL(org.ISDELETE, 0) AS branch_is_deleted_flag\n"
                "FROM dbo.PSNACCOUNT AS p\n"
                f"LEFT JOIN {ORG_TABLE} AS org\n"
                "  ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))"
            ),
        },
    }

    def __init__(self, model_name: str = OPENAI_MODEL, temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.llm_enabled = bool(OPENAI_API_KEY) and ChatOpenAI is not None
        self.memory = None

        self.generation_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "repair_attempts": 0,
            "successful_repairs": 0,
            "total_tokens_used": 0,
            "avg_generation_time": 0.0,
        }

        self._explain_cache: Dict[str, str] = {}
        self._explain_cache_max = 128

        self.sql_prompt_en = None
        self.sql_prompt_zh = None
        self.repair_sql_prompt_en = None
        self.repair_sql_prompt_zh = None
        self.explanation_prompt_en = None
        self.explanation_prompt_zh = None

        self._initialize_llm()
        self._initialize_all_prompts()
        

    # ────────────────────────────────────────────────────────────────────────────
    # LLM init
    # ────────────────────────────────────────────────────────────────────────────
    def _initialize_llm(self):
        if not self.llm_enabled:
            logger.warning("LLM DISABLED: No API key or langchain_openai missing.")
            return
        t0 = time.perf_counter()
        try:
            try:
                self.llm = ChatOpenAI(  # type: ignore
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=OPENAI_API_KEY,  # type: ignore
                )
            except TypeError:
                self.llm = ChatOpenAI(  # type: ignore
                    model_name=self.model_name,  # type: ignore
                    temperature=self.temperature,  # type: ignore
                    openai_api_key=OPENAI_API_KEY,  # type: ignore
                )
            self.memory = ConversationBufferMemory(return_messages=True) if ConversationBufferMemory else None
            logger.info("LLM INITIALIZED: model=%s temp=%.2f memory=%s init_ms=%d",
                        self.model_name, self.temperature, bool(self.memory),
                        int((time.perf_counter() - t0) * 1000))
        except Exception as e:
            logger.error("LLM INIT FAILED: %s: %s", type(e).__name__, e, exc_info=True)
            self.llm = None
            self.llm_enabled = False

    def _explain_cache_key(self, question: str, row_count: int, columns: List[str],
                       aggregates: Dict[str, Any], sample_text: str, language: str) -> str:
        payload = json.dumps({
            "q": question,
            "rc": row_count,
            "cols": columns,
            "aggs": aggregates,
            "s": sample_text,
            "lang": language,
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _explain_cache_get(self, k: str) -> Optional[str]:
        return self._explain_cache.get(k)

    def _explain_cache_put(self, k: str, v: str):
        if len(self._explain_cache) >= self._explain_cache_max:
            # simple FIFO trim
            old_key = next(iter(self._explain_cache))
            self._explain_cache.pop(old_key, None)
        self._explain_cache[k] = v


    # ────────────────────────────────────────────────────────────────────────────
    # Prompts (EN + ZH) – intent-aware, balance-snapshot aware
    # ────────────────────────────────────────────────────────────────────────────
    def _initialize_all_prompts(self):
        if not ChatPromptTemplate:
            logger.warning("PROMPTS DISABLED: ChatPromptTemplate not available.")
            return

        self.sql_prompt_en = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "You are an expert T-SQL analyst for HR leave & attendance data.\n"
                "Return exactly ONE safe **SELECT-only** Microsoft SQL Server (T-SQL) query.\n\n"
                "INTENT:\n{intent_debug}\n\n"
                "FEW-SHOT (follow structure & aliases if applicable):\n{few_shot}\n\n"
                "DATE HANDLING:\n"
                "- If the user implies 'today/current', your caller will substitute @today; do not call GETDATE().\n"
                "- Date filters use CAST(column AS date) and BETWEEN where appropriate.\n\n"
                "BUSINESS RULES:\n"
                f"- For balances/entitlement, prefer {VAC_RESULT_TABLE}; respect @today within CANUSEDATE..DISABLEDDATE.\n"
                "- Use VALIDATED = 1 when counting approved leave.\n"
                "- WORKDATE is the occurrence date; STARTDATE/ENDDATE is the request span.\n"
                "- Join person/org only when needed.\n\n"
                "T-SQL RULES (STRICT):\n"
                "- Only SELECT (CTEs allowed). No DML/DDL.\n"
                "- Declare every alias in FROM/JOIN before use; never reference undeclared aliases.\n"
                "- Prefer fully-qualified tables (schema.table) and qualified columns.\n"
                "- No LIMIT; use TOP (N) with ORDER BY if needed. Pagination via ORDER BY ... OFFSET ... FETCH.\n"
                "- GROUP BY must include all non-aggregates.\n\n"
                "Available schema:\n{schema}\n"
                "Table whitelist (MUST use only these): {table_whitelist}\n"
                "Suggested joins:\n{join_hints}"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "User question: {query}\n\n"
                "Slots (JSON): {slots_json}\n"
                "Return only the SQL query (no markdown, no comments outside SQL)."
            ),
        ])

        self.sql_prompt_zh = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "你是人資請假/考勤資料的 T-SQL 專家。\n"
                "請只回傳一個安全的 **僅限 SELECT** 的 Microsoft SQL Server (T-SQL) 查詢。\n\n"
                "意圖：\n{intent_debug}\n\n"
                "Few-shot（盡量遵循結構與別名）：\n{few_shot}\n\n"
                "日期處理：\n"
                "- 若使用者暗示「今天/目前」，呼叫端會提供 @today；不要使用 GETDATE()。\n"
                "- 日期過濾採用 CAST(column AS date) 與 BETWEEN。\n\n"
                "業務規則：\n"
                f"- 餘額/給予請以 {VAC_RESULT_TABLE} 為主；於有效期（CANUSEDATE..DISABLEDDATE）加入 @today 條件。\n"
                "- 統計已批准請假請加 VALIDATED = 1。\n"
                "- WORKDATE 為發生日；STARTDATE/ENDDATE 為申請區間。\n"
                "- 需要時再關聯人員/部門。\n\n"
                "T-SQL 規範（嚴格）：\n"
                "- 只允許 SELECT（可用 CTE），禁止 DML/DDL。\n"
                "- 別名必須先在 FROM/JOIN 宣告，禁止引用未宣告別名。\n"
                "- 優先使用完整表名與限定欄位（table.column）。\n"
                "- 不可用 LIMIT；如需取前 N 筆，用 TOP (N) 並搭配 ORDER BY。分頁用 ORDER BY ... OFFSET ... FETCH。\n"
                "- GROUP BY 必須包含所有非聚合欄位。\n\n"
                "可用架構：\n{schema}\n"
                "允許使用之資料表（僅限）：{table_whitelist}\n"
                "建議關聯：\n{join_hints}"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "使用者問題：{query}\n\n"
                "Slots (JSON)：{slots_json}\n"
                "只回傳 SQL 本體（不要 markdown，也不要 SQL 外註解）。"
            ),
        ])

        self._initialize_repair_prompts()
        self._initialize_explanation_prompts()
        logger.info("PROMPTS INITIALIZED (intent-aware, balance-snapshot aware).")

    def _initialize_repair_prompts(self):
        self.repair_sql_prompt_en = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "You fix failing Microsoft SQL Server (T-SQL) queries.\n"
                "Output exactly one corrected **SELECT-only** T-SQL statement.\n"
                "Keep original intent; use only schema columns; respect GROUP BY; declare aliases before use.\n\n"
                "INTENT:\n{intent_debug}\n\nFEW-SHOT:\n{few_shot}\n\n"
                "...\nAvailable schema:\n{schema}\n"
                "Table whitelist (MUST use only these): {table_whitelist}\n"
                "Join hints:\n{join_hints}"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "Database error:\n{error_summary}\n\nFailed SQL:\n{failed_sql}\n\nReturn only the corrected SQL."
            ),
        ])
        self.repair_sql_prompt_zh = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "你要修復失敗的 Microsoft SQL Server (T-SQL) 查詢。\n"
                "請輸出一個修正後的 **僅限 SELECT** 的 T-SQL 語句，維持原意且僅用架構欄位；別名須先宣告；不得有註解。\n\n"
                "意圖：\n{intent_debug}\n\nFew-shot：\n{few_shot}\n\n"
                "...\n可用架構：\n{schema}\n"
                "允許使用之資料表（僅限）：{table_whitelist}\n"
                "關聯提示：\n{join_hints}"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "資料庫錯誤：\n{error_summary}\n\n失敗的SQL：\n{failed_sql}\n\n只回傳修正後的SQL。"
            ),
        ])

    def _initialize_explanation_prompts(self):
        self.explanation_prompt_en = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "You are a senior data analyst writing for executives.\n"
                "STRICT RULES:\n"
                "- Use ONLY the data provided in Columns, Aggregates, and Sample rows; do NOT invent columns or values.\n"
                "- If the data is insufficient, say so briefly.\n"
                "- Be concise, precise, and neutral; no SQL or code.\n"
                "FORMAT (markdown):\n"
                "### Executive Summary\n"
                "• 2–3 bullets with the headline numbers (totals, counts), directly tied to the question.\n"
                "### Key Observations\n"
                "• 2–4 bullets on patterns, distributions, outliers, time windows, or concentration by categories present in Columns.\n"
                "### Risks & Actions\n"
                "• 1–3 bullets with specific recommended next steps for managers (e.g., follow-ups, thresholds, policy checks).\n"
                "### Data Quality Notes\n"
                "• 1–2 bullets on caveats (e.g., row_count=0, missing columns, truncated samples, effective-date windows).\n"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "Question: {question}\n"
                "Row count: {row_count}\n"
                "Columns: {columns}\n"
                "Aggregates (JSON): {aggregates_json}\n"
                "Sample rows (truncated):\n{sample_text}\n"
                "\nWrite the brief in English."
            ),
        ])

        self.explanation_prompt_zh = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(  # type: ignore
                "你是一位服務高階主管的資深資料分析師。\n"
                "嚴格規則：\n"
                "- 只可使用【欄位、統計摘要、樣本資料】中提供的資訊；不可臆測或新增欄位/數值。\n"
                "- 若資料不足請明確指出。\n"
                "- 簡潔、準確、中立；不得包含 SQL 或程式碼。\n"
                "輸出格式（Markdown）：\n"
                "### 摘要\n"
                "• 2–3 點重點數字（總數、筆數等），需與題目直接相關。\n"
                "### 主要觀察\n"
                "• 2–4 點關於分布、集中度、異常值、時間區間或類別（以實際欄位為準）。\n"
                "### 風險與行動建議\n"
                "• 1–3 點給主管的具體建議（例如追蹤、門檻、政策確認）。\n"
                "### 資料品質說明\n"
                "• 1–2 點備註（例如筆數=0、欄位缺漏、樣本截斷、生效期間限制）。\n"
            ),
            HumanMessagePromptTemplate.from_template(  # type: ignore
                "問題：{question}\n"
                "資料筆數：{row_count}\n"
                "欄位：{columns}\n"
                "統計摘要 (JSON)：{aggregates_json}\n"
                "資料樣本（截斷）：\n{sample_text}\n"
                "\n請以繁體中文撰寫上述格式的說明。"
            ),
        ])


    # ────────────────────────────────────────────────────────────────────────────
    # Utilities (intent block, few-shot rendering, extraction, sanitation, fixes)
    # ────────────────────────────────────────────────────────────────────────────
    def _render_few_shot(self, template_ref: Optional[str], slots: Dict[str, Any],
                         language: Literal["zh-tw", "en"]) -> str:
        if not template_ref:
            return ""
        tpl = self.FEW_SHOT_TEMPLATES.get(template_ref)
        if not tpl:
            return ""
        raw = tpl["zh" if language == "zh-tw" else "en"]
        # Basic interpolation (keep @year/@today as bind params; only replace {year} if present)
        year = slots.get("year") or slots.get("Year") or ""
        rendered = raw.replace("{year}", str(year) if year else "2024")
        return rendered

    def _intent_debug(self, intent_context: Optional[Dict[str, Any]]) -> str:
        if not intent_context:
            return "(no intent)"
        try:
            tpl = intent_context.get("template_ref")
            slots = intent_context.get("slots", {})
            tables = intent_context.get("tables", [])
            cands = intent_context.get("candidates", [])
            top = cands[0] if cands else {}
            as_lines = [
                f"template_ref={tpl or top.get('template_ref')}",
                f"slots={json.dumps(slots, ensure_ascii=False)}",
                f"tables_hint={','.join(tables or top.get('tables', []))}",
            ]
            if top:
                as_lines.append(f"intent_title={top.get('title')}")
                as_lines.append(f"intent_score={top.get('score')}")
            return "\n".join(as_lines)
        except Exception:
            return json.dumps(intent_context, ensure_ascii=False)

    _FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    _FIRST_SELECT_RE = re.compile(r"(?is)\bwith\b[\s\S]+?\bselect\b|\bselect\b")
    _PROHIBITED_RE = re.compile(
        r"(?is)\b(insert|update|delete|merge|drop|alter|create|truncate|exec|execute|grant|revoke)\b"
    )

    def _extract_sql_from_text(self, text: str) -> str:
        if not text:
            return ""
        m = self._FENCE_RE.search(text)
        sql = m.group(1) if m else text
        sql = sql.strip()
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.I)
        sql = re.sub(r"\s*```$", "", sql)
        m2 = self._FIRST_SELECT_RE.search(sql)
        if m2:
            sql = sql[m2.start():].strip()
        return sql

    def _normalize_id_quotes(self, sql: str) -> str:
        s = re.sub(r"`([^`]+)`", r"[\1]", sql)
        s = re.sub(r'(?<!")"([A-Za-z_][\w]*)"(?!")', r"[\1]", s)
        return s

    def _tsql_limit_fix(self, sql: str) -> str:
        if not sql:
            return sql
        s = sql.strip().rstrip(";")
        m = re.search(r"\blimit\s+(\d+)\s*$", s, flags=re.I)
        if m and re.search(r"^\s*select\b", s, flags=re.I):
            n = m.group(1)
            s = re.sub(r"\blimit\s+\d+\s*$", "", s, flags=re.I).strip()
            s = re.sub(r"(?i)^\s*select", f"SELECT TOP ({n})", s, count=1)
            logger.debug("TSQL_FIX: LIMIT→TOP(%s)", n)
        return s

    def _ensure_select_only(self, sql: str) -> str:
        if not sql:
            return ""
        s = sql.strip()
        if self._PROHIBITED_RE.search(s):
            logger.warning("SANITIZE: prohibited keyword detected; returning safe empty SELECT.")
            return "SELECT 1 WHERE 1=0"
        parts = [p.strip() for p in re.split(r";\s*(?=WITH\b|SELECT\b|$)", s, flags=re.I)]
        first_valid = next((p for p in parts if re.match(r"(?is)^(with\b|select\b)", p)), "")
        if not first_valid:
            return ""
        return first_valid

    def _finalize_sql(self, sql: str) -> str:
        s = self._extract_sql_from_text(sql)
        s = self._normalize_id_quotes(s)
        s = self._tsql_limit_fix(s)
        s = self._ensure_select_only(s)
        return s.strip()

    def _create_query_signature(self, query: str, language: str) -> str:
        normalized = re.sub(r"\s+", " ", (query or "").lower().strip())
        return hashlib.md5(f"{language}:{normalized}".encode()).hexdigest()[:12]

    def _invoke_llm(self, messages: List[BaseMessage], context: str = "") -> str:  # type: ignore
        if not (self.llm_enabled and self.llm and messages):
            logger.warning("LLM_INVOKE: unavailable (enabled=%s, llm=%s, msgs=%s)",
                           self.llm_enabled, bool(self.llm), bool(messages))
            return ""
        t0 = time.perf_counter()
        self.generation_stats["total_requests"] += 1
        try:
            user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)  # type: ignore
            prev = (user_msg.content[:120] + "...") if user_msg and len(user_msg.content) > 120 else (user_msg.content if user_msg else "")
            logger.debug("LLM_INVOKE_START: ctx=%s user_preview=%r", context, prev)
            resp = self.llm.invoke(messages)
            content = str(resp.content)
            if self.memory and user_msg:
                self.memory.save_context({"input": user_msg.content}, {"output": content})
            dt = time.perf_counter() - t0
            self.generation_stats["successful_generations"] += 1
            n = self.generation_stats["successful_generations"]
            self.generation_stats["avg_generation_time"] = (
                (self.generation_stats["avg_generation_time"] * (n - 1)) + dt
            ) / n
            logger.info("LLM_OK: ctx=%s time=%.2fs len=%d avg=%.2fs",
                        context, dt, len(content), self.generation_stats["avg_generation_time"])
            return content
        except Exception as e:
            dt = time.perf_counter() - t0
            self.generation_stats["failed_generations"] += 1
            logger.error("LLM_FAIL: ctx=%s time=%.2fs %s: %s", context, dt, type(e).__name__, e, exc_info=True)
            return ""

    # ────────────────────────────────────────────────────────────────────────────
    # Error classification
    # ────────────────────────────────────────────────────────────────────────────
    def _is_repairable_error(self, e: DBServiceQueryError) -> bool:
        repairable = (
            DBServiceSyntaxError,
            DBServiceTableNotFoundError,
            DBServiceColumnNotFoundError,
            DBServiceDataError,
            DBServiceIntegrityError,
            DBServiceOperationalError,
            DBServiceQueryError,
        )
        non_repairable = (
            DBServiceTimeoutError,
            DBServiceConnectionError,
            DBServicePermissionDeniedError,
        )
        ok = isinstance(e, repairable) and not isinstance(e, non_repairable)
        logger.debug("ERROR_CLASS: %s repairable=%s", type(e).__name__, ok)
        return ok

    # ────────────────────────────────────────────────────────────────────────────
    # SQL generation & repair (INTENT-AWARE)
    # ────────────────────────────────────────────────────────────────────────────
    def generate_sql(
        self,
        query: str,
        schema: str,
        join_hints: str,
        *,
        intent_context: Optional[Dict[str, Any]] = None,
        language: Optional[Literal["zh-tw", "en"]] = None,
    ) -> str:
        t0 = time.perf_counter()
        if not self.llm_enabled:
            logger.warning("SQL_GEN: LLM disabled → fallback stub.")
            return "SELECT 1 WHERE 1=0"

        language = language or detect_query_language(query)
        sig = self._create_query_signature(query, language)
        logger.info("SQL_GEN_START: sig=%s lang=%s q='%s'", sig, language, query[:120])

        try:
            prompt = self.sql_prompt_zh if language == "zh-tw" else self.sql_prompt_en
            if not prompt:
                logger.error("SQL_GEN: prompt missing for lang=%s", language)
                return "SELECT 1 WHERE 1=0"

            intent_debug = self._intent_debug(intent_context)
            slots = (intent_context or {}).get("slots", {}) or {}
            few_shot = self._render_few_shot((intent_context or {}).get("template_ref"), slots, language)

            sanitized_schema = _prune_schema_text(schema)

            messages = prompt.format_messages(
                query=query,
                schema=sanitized_schema,
                join_hints=join_hints,
                intent_debug=intent_debug,
                few_shot=few_shot,
                slots_json=json.dumps(slots, ensure_ascii=False),
                table_whitelist=TABLE_WHITELIST_TEXT,
            )
            raw = self._invoke_llm(messages, f"sql_gen_{'zh' if language=='zh-tw' else 'en'}")
            if not raw:
                logger.warning("SQL_GEN_EMPTY: sig=%s", sig)
                return "SELECT 1 WHERE 1=0"

            final_sql = self._finalize_sql(raw)
            dt = time.perf_counter() - t0
            logger.info("SQL_GEN_OK: sig=%s time=%.2fs len=%d", sig, dt, len(final_sql))
            logger.debug("SQL_GEN_SQL: sig=%s\n%s", sig, final_sql)

            bad = _tables_with_bad_whitelist(final_sql)
            if bad:
                logger.error("WHITELIST_VIOLATION: %s", bad)
                return "SELECT 1 WHERE 1=0"

            return final_sql or "SELECT 1 WHERE 1=0"
        except Exception as e:
            dt = time.perf_counter() - t0
            logger.error("SQL_GEN_FAIL: sig=%s time=%.2fs %s: %s", sig, dt, type(e).__name__, e, exc_info=True)
            return "SELECT 1 WHERE 1=0"
    
    def generate_sql_with_repair(
        self,
        question: str,
        schema: str,
        join_hints: str,
        *,
        intent_context: Optional[Dict[str, Any]] = None,
        language: Optional[Literal["zh-tw", "en"]] = None,
        failed_sql: Optional[str] = None,
        error_summary: Optional[str] = None,
        max_attempts: int = 3,
    ) -> Tuple[str, int]:
        t0 = time.perf_counter()
        language = language or detect_query_language(question)
        sig = self._create_query_signature(question, language)
        attempts = 0
        sql = ""

        logger.info("SQL_REPAIR_START: sig=%s lang=%s max_attempts=%d has_failed=%s",
                    sig, language, max_attempts, bool(failed_sql))

        sanitized_schema = _prune_schema_text(schema)

        while attempts < max_attempts:
            attempts += 1
            a0 = time.perf_counter()

            if attempts == 1 and not failed_sql:
                logger.debug("SQL_REPAIR_ATTEMPT: sig=%s attempt=%d fresh-gen", sig, attempts)
                sql = self.generate_sql(
                    question, sanitized_schema, join_hints,
                    intent_context=intent_context, language=language
                )
            else:
                self.generation_stats["repair_attempts"] += 1
                if not self.llm_enabled:
                    logger.warning("SQL_REPAIR_ABORT: LLM disabled.")
                    break
                repair_prompt = self.repair_sql_prompt_zh if language == "zh-tw" else self.repair_sql_prompt_en
                if not repair_prompt:
                    logger.warning("SQL_REPAIR_ABORT: missing repair prompt for %s", language)
                    break

                intent_debug = self._intent_debug(intent_context)
                slots = (intent_context or {}).get("slots", {}) or {}
                few_shot = self._render_few_shot((intent_context or {}).get("template_ref"), slots, language)

                messages = repair_prompt.format_messages(
                    failed_sql=failed_sql or sql,
                    error_summary=error_summary or "(no error message)",
                    schema=sanitized_schema,
                    join_hints=join_hints,
                    intent_debug=intent_debug,
                    few_shot=few_shot,
                    table_whitelist=TABLE_WHITELIST_TEXT,
                )
                raw = self._invoke_llm(messages, f"sql_repair_{'zh' if language=='zh-tw' else 'en'}")
                sql = self._finalize_sql(raw)
                if sql and sql != "SELECT 1 WHERE 1=0":
                    self.generation_stats["successful_repairs"] += 1

            logger.debug("SQL_REPAIR_ATTEMPT_DONE: sig=%s attempt=%d time=%.2fs len=%d",
                        sig, attempts, time.perf_counter() - a0, len(sql))

            if sql.strip():
                bad = _tables_with_bad_whitelist(sql)
                if bad:
                    logger.warning("REPAIR_WHITELIST_BLOCK: %s (retrying...)", bad)
                    failed_sql = sql
                    continue

                logger.info("SQL_REPAIR_OK: sig=%s attempts=%d total=%.2fs",
                            sig, attempts, time.perf_counter() - t0)
                return sql, attempts

        logger.warning("SQL_REPAIR_FAIL: sig=%s attempts=%d total=%.2fs",
                    sig, attempts or 1, time.perf_counter() - t0)
        return ("SELECT 1 WHERE 1=0", attempts or 1)



    # ────────────────────────────────────────────────────────────────────────────
    # Execution + repair loop
    # ────────────────────────────────────────────────────────────────────────────
    def run_query_with_llm_repair(
        self,
        db_service,
        user_question: str,
        schema: str,
        join_hints: str,
        *,
        intent_context: Optional[Dict[str, Any]] = None,
        params: Optional[Tuple[Any, ...]] = None,
        max_rows: int = 1000,
        query_timeout: Optional[int] = 10,
        max_attempts: int = 3,
    ) -> Tuple[List[Tuple], List[str], str, int]:
        t0 = time.perf_counter()
        language = detect_query_language(user_question)
        sig = self._create_query_signature(user_question, language)

        logger.info("QUERY_START: sig=%s lang=%s rows<=%d timeout=%s attempts<=%d",
                    sig, language, max_rows, query_timeout, max_attempts)

        sql, attempts = self.generate_sql_with_repair(
            question=user_question,
            schema=schema,
            join_hints=join_hints,
            intent_context=intent_context,
            language=language,
            max_attempts=1,
        )

        try:
            a0 = time.perf_counter()
            rows, cols = db_service.run_select(sql, params=params, max_rows=max_rows, query_timeout=query_timeout)
            logger.info("QUERY_OK: sig=%s attempts=%d rows=%d cols=%d exec=%.2fs total=%.2fs",
                        sig, attempts, len(rows), len(cols),
                        time.perf_counter() - a0, time.perf_counter() - t0)
            logger.info("QUERY_SQL_OK: sig=%s sql=%s", sig, sql[:300])
            return rows, cols, sql, attempts
        except DBServiceQueryError as e:
            logger.warning("QUERY_FAIL: sig=%s attempt=1 err=%s: %s", sig, type(e).__name__, str(e)[:240])
            if not self._is_repairable_error(e):
                logger.error("QUERY_ABORT: non-repairable %s", type(e).__name__)
                raise

            error_details = self._build_error_summary(e, language)
            last_sql = sql

            while attempts < max_attempts:
                attempts += 1
                logger.debug("QUERY_REPAIR: sig=%s attempt=%d", sig, attempts)
                sql, _ = self.generate_sql_with_repair(
                    question=user_question,
                    schema=schema,
                    join_hints=join_hints,
                    intent_context=intent_context,
                    language=language,
                    failed_sql=last_sql,
                    error_summary=error_details,
                    max_attempts=1,
                )
                try:
                    a0 = time.perf_counter()
                    rows, cols = db_service.run_select(sql, params=params, max_rows=max_rows, query_timeout=query_timeout)
                    logger.info("QUERY_REPAIR_OK: sig=%s attempts=%d rows=%d exec=%.2fs total=%.2fs",
                                sig, attempts, len(rows), time.perf_counter() - a0, time.perf_counter() - t0)
                    logger.info("QUERY_SQL_REPAIRED: sig=%s sql=%s", sig, sql[:300])
                    return rows, cols, sql, attempts
                except DBServiceQueryError as e2:
                    logger.warning("QUERY_REPAIR_FAIL: sig=%s attempt=%d %s: %s",
                                   sig, attempts, type(e2).__name__, str(e2)[:240])
                    if not self._is_repairable_error(e2):
                        logger.error("QUERY_REPAIR_ABORT: non-repairable at attempt %d", attempts)
                        raise
                    last_sql = sql
                    error_details = self._build_error_summary(e2, language)

            logger.error("QUERY_EXHAUSTED: sig=%s attempts=%d total=%.2fs", sig, attempts, time.perf_counter() - t0)
            raise

    def _build_error_summary(self, e: DBServiceQueryError, language: Literal["zh-tw", "en"]) -> str:
        parts = []
        if getattr(e, "category", None):
            parts.append(f"category={e.category}")
        if getattr(e, "db_code", None) is not None:
            parts.append(f"db_code={e.db_code}")
        if getattr(e, "sqlstate", None):
            parts.append(f"sqlstate={e.sqlstate}")
        meta = "; ".join(parts) if parts else ""
        msg = f"{type(e).__name__}: {str(e)}"
        if language == "zh-tw":
            msg = f"錯誤類型: {type(e).__name__}: {str(e)}"
        return f"{msg} ({meta})" if meta else msg

    # ────────────────────────────────────────────────────────────────────────────
    # Explanations
    # ────────────────────────────────────────────────────────────────────────────
    def generate_explanation(self, question: str, row_count: int, columns: List[str],
                             aggregates: Dict[str, Any], sample_text: str) -> str:
        return self._generate_explanation_internal(question, row_count, columns, aggregates, sample_text, "en")

    def generate_explanation_chinese(self, question: str, row_count: int, columns: List[str],
                                     aggregates: Dict[str, Any], sample_text: str) -> str:
        return self._generate_explanation_internal(question, row_count, columns, aggregates, sample_text, "zh-tw")
    

    def _generate_explanation_internal(self, question: str, row_count: int, columns: List[str],
                                   aggregates: Dict[str, Any], sample_text: str,
                                   language: Literal["zh-tw", "en"]) -> str:
        # 0) Zero-row fast path (no LLM needed)
        if row_count <= 0:
            if language == "zh-tw":
                return (
                    "### 摘要\n"
                    "• 查詢結果為 0 筆，無可供分析的資料。\n\n"
                    "### 資料品質說明\n"
                    "• 請確認日期區間、條件或權限是否正確；若為有效期/快照表，亦需注意生效日期條件。"
                )
            else:
                return (
                    "### Executive Summary\n"
                    "• The query returned 0 rows; no analyzable data is available.\n\n"
                    "### Data Quality Notes\n"
                    "• Verify date range, filters, or permissions. If this involves effective-dated snapshots, check validity windows."
                )

        # 1) Build cache key and try cache
        key = self._explain_cache_key(question, row_count, columns or [], aggregates or {}, sample_text or "", language)
        cached = self._explain_cache_get(key)
        if cached:
            return cached

        # 2) If LLM disabled / prompt missing → fallback
        if not self.llm_enabled or not ChatPromptTemplate:
            text = self._fallback_explanation(aggregates, language)
            self._explain_cache_put(key, text)
            return text

        prompt = self.explanation_prompt_zh if language == "zh-tw" else self.explanation_prompt_en
        if not prompt:
            text = self._fallback_explanation(aggregates, language)
            self._explain_cache_put(key, text)
            return text

        # 3) RAW-only safeguards in content (columns list is authoritative)
        cols_joined = ", ".join(columns) if columns else "(none)"
        aggs_json = json.dumps(aggregates or {}, ensure_ascii=False)

        msgs = prompt.format_messages(
            question=question,
            row_count=row_count,
            columns=cols_joined,
            aggregates_json=aggs_json,
            sample_text=sample_text or "(no sample)",
        )
        resp = self._invoke_llm(msgs, f"explain_{'zh' if language=='zh-tw' else 'en'}")
        text = (resp or "").strip() or self._fallback_explanation(aggregates, language)

        # 4) Post-trim: keep it compact and safe
        if len(text) > 2400:
            text = text[:2400].rstrip() + "…"

        self._explain_cache_put(key, text)
        return text


    def _fallback_explanation(self, aggregates: Dict[str, Any], language: Literal["zh-tw", "en"] = "en") -> str:
        rc = int(aggregates.get("row_count", 0) or 0)
        up = aggregates.get("unique_people")
        bt = aggregates.get("by_leave_type") or {}
        th = aggregates.get("total_hours")

        if language == "zh-tw":
            parts = [f"{rc} 筆記錄。"]
            if up is not None: parts.append(f"{up} 位不重複人員。")
            if bt:
                total = sum(bt.values())
                if total:
                    top = sorted(bt.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    parts.append("主要請假類型：" + "、".join(f"{k}（{v}，{round(v/total*100,1)}%）" for k, v in top))
            if th: parts.append(f"總時數：{th}")
            return " ".join(parts)
        else:
            parts = [f"{rc} records."]
            if up is not None: parts.append(f"{up} unique people.")
            if bt:
                total = sum(bt.values())
                if total:
                    top = sorted(bt.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    parts.append("Top leave types: " + ", ".join(f"{k} ({v}, {round(v/total*100,1)}%)" for k, v in top))
            if th: parts.append(f"Total hours: {th}")
            return " ".join(parts)

    # ────────────────────────────────────────────────────────────────────────────
    # Convenience & stats
    # ────────────────────────────────────────────────────────────────────────────
    def run_query_with_llm_repair_language_aware(
        self,
        db_service,
        user_question: str,
        original_language: Literal["zh-tw", "en"],
        schema: str,
        join_hints: str,
        *,
        intent_context: Optional[Dict[str, Any]] = None,
        max_rows: int = 1000,
        query_timeout: int = 10,
        max_attempts: int = 3,
        **kwargs,
    ) -> Tuple[List, List[str], str, int]:
        logger.debug("LANGUAGE_AWARE_QUERY: original_lang=%s → main path", original_language)
        return self.run_query_with_llm_repair(
            db_service=db_service,
            user_question=user_question,
            schema=schema,
            join_hints=join_hints,
            intent_context=intent_context,
            params=None,
            max_rows=max_rows,
            query_timeout=query_timeout,
            max_attempts=max_attempts,
        )

    def get_service_stats(self) -> Dict[str, Any]:
        total = self.generation_stats["total_requests"]
        succ = self.generation_stats["successful_generations"]
        repair_attempts = self.generation_stats["repair_attempts"]
        return {
            "service_enabled": self.llm_enabled,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "total_requests": total,
            "successful_generations": succ,
            "failed_generations": self.generation_stats["failed_generations"],
            "success_rate_percent": round((succ / max(total, 1)) * 100, 2),
            "repair_attempts": repair_attempts,
            "successful_repairs": self.generation_stats["successful_repairs"],
            "repair_rate_percent": round((repair_attempts / max(total, 1)) * 100, 2),
            "repair_success_rate_percent": round((self.generation_stats["successful_repairs"] / max(repair_attempts, 1)) * 100, 2),
            "avg_generation_time_seconds": round(self.generation_stats["avg_generation_time"], 3),
            "has_memory": bool(self.memory),
            "prompts_initialized": all([
                self.sql_prompt_en, self.sql_prompt_zh,
                self.repair_sql_prompt_en, self.repair_sql_prompt_zh,
                self.explanation_prompt_en, self.explanation_prompt_zh,
            ]),
        }

    def reset_stats(self):
        logger.info("SERVICE_STATS_RESET")
        self.generation_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "repair_attempts": 0,
            "successful_repairs": 0,
            "total_tokens_used": 0,
            "avg_generation_time": 0.0,
        }

    def _simple_completion(self, system_prompt: str, user_prompt: str) -> str:
        if not self.llm_enabled or not ChatPromptTemplate:
            return ""
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_prompt) if SystemMessagePromptTemplate else "",
                HumanMessagePromptTemplate.from_template("{u}")  # type: ignore
            ])
            msgs = prompt.format_messages(u=user_prompt)
            return self._invoke_llm(msgs, "simple_completion") or ""
        except Exception as e:
            logger.error("SIMPLE_COMPLETION_FAIL: %s: %s", type(e).__name__, e)
            return ""


# Backward compatibility aliases
OpenAIService = UnifiedBilingualOpenAIService
LanguageAwareOpenAIService = UnifiedBilingualOpenAIService
