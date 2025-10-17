# backend/app/reports/service.py
"""
Report service with:
- Ask-then-refine state machine support for the frontend
- Dynamic section builder (user chooses what to include)
- Chart/visualization configuration
- Statistical aggregations
- Real data integration via SQL queries (sync DB executed safely from async endpoints)
"""

import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Iterable, Tuple
from datetime import datetime

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field
from anyio import to_thread

logger = logging.getLogger(__name__)

# ============================================
# Models & Enums
# ============================================

class ChartType(str, Enum):
    BAR = "bar"
    PIE = "pie"
    LINE = "line"
    TABLE = "table"
    HEATMAP = "heatmap"
    GAUGE = "gauge"

class SectionType(str, Enum):
    EXECUTIVE_SUMMARY   = "executive_summary"
    LEAVE_BY_DEPARTMENT = "leave_by_department"
    LEAVE_BY_TYPE       = "leave_by_type"
    LEAVE_TRENDS        = "leave_trends"
    TOP_EMPLOYEES       = "top_employees_by_leave"
    ATTENDANCE_RATE     = "attendance_rate"
    SICK_LEAVE_ANALYSIS = "sick_leave_analysis"
    BALANCE_SNAPSHOT    = "balance_snapshot"
    EXPIRING_SOON       = "expiring_soon"
    OVERTIME_SUMMARY    = "overtime_summary"
    DEPARTMENT_COMPARISON = "department_comparison"
    RISK_ASSESSMENT     = "risk_assessment"
    RECOMMENDATIONS     = "recommendations"
    DATA_QUALITY        = "data_quality"

@dataclass
class ChartConfig:
    chart_type: ChartType
    title: str
    description: str = ""
    x_axis: str = ""
    y_axis: str = ""
    data_source: str = ""  # key in data_bucket
    filters: Dict[str, Any] = field(default_factory=dict)
    include_legend: bool = True
    include_labels: bool = True
    sort_by: Optional[str] = None
    limit: Optional[int] = None

@dataclass
class SectionConfig:
    section_type: SectionType
    title: str
    description: str = ""
    include: bool = True
    charts: List[ChartConfig] = field(default_factory=list)
    statistics: List[str] = field(default_factory=list)
    depth: str = "standard"  # "high_level", "standard", "detailed"
    include_table: bool = False
    table_rows: int = 50

@dataclass
class ReportConfigSchema:
    sections: List[SectionConfig] = field(default_factory=list)
    include_appendix: bool = True
    include_risks: bool = True
    include_recommendations: bool = True
    export_format: str = "docx"
    color_scheme: str = "professional"
    language: str = "en-US"

# ============================================
# Request payloads (used by routes)
# ============================================

class ConfigureReportRequest(BaseModel):
    """Frontend calls this first to get available sections + clarifying questions."""
    query: str
    analysis: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Comprehensive leave analysis with department breakdown and charts",
                "analysis": {"report_type": "leave_analysis", "time_period": "2025-Q1"}
            }
        }

class SectionSelectionRequest(BaseModel):
    """Frontend calls this with user selection."""
    request_id: str
    selected_sections: List[SectionType]            # Pydantic will coerce string -> enum
    chart_preferences: Dict[str, List[str]] = Field(default_factory=dict)  # will normalize
    depth_level: str = "standard"                   # "high_level" | "standard" | "detailed"
    language: str = "en-US"

# ============================================
# SQL builders
# ============================================

class DataQueryBuilder:
    @staticmethod
    def query_leave_by_department(limit: int = 100) -> str:
        return f"""
SELECT TOP ({limit})
    d.dept_display_name AS department,
    d.dept_code,
    COUNT(DISTINCT lf.PERSONID) AS num_employees,
    COUNT(*) AS total_leaves,
    SUM(CAST(lf.total_hours AS FLOAT)) AS total_hours,
    AVG(CAST(lf.total_hours AS FLOAT)) AS avg_hours_per_leave,
    SUM(CAST(lf.calculated_days AS FLOAT)) AS total_days,
    AVG(CAST(lf.calculated_days AS FLOAT)) AS avg_days_per_employee,
    SUM(CASE WHEN lf.VALIDATED = 1 THEN 1 ELSE 0 END) AS approved_count,
    SUM(CASE WHEN lf.VALIDATED = 0 THEN 1 ELSE 0 END) AS pending_count
FROM [eHRAntung_DB].[dbo].[ATDLEAVEDATA] lf
LEFT JOIN [eHRAntung_DB].[dbo].[PSNACCOUNT] p ON lf.PERSONID = p.PERSONID
LEFT JOIN [eHRAntung_DB].[dbo].[ORGStdStruct] d 
    ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(d.UNITID AS NVARCHAR(100))
WHERE lf.VALIDATED = 1
  AND YEAR(lf.WORKDATE) = YEAR(GETDATE())
GROUP BY d.dept_display_name, d.dept_code, d.UNITID
ORDER BY total_hours DESC;
        """

    @staticmethod
    def query_leave_by_type(limit: int = 100) -> str:
        return f"""
SELECT TOP ({limit})
    lt.leave_type_name,
    lt.CLASSCODE,
    COUNT(*) AS count,
    SUM(CAST(lf.total_hours AS FLOAT)) AS total_hours,
    AVG(CAST(lf.total_hours AS FLOAT)) AS avg_hours,
    COUNT(DISTINCT lf.PERSONID) AS unique_employees,
    SUM(CASE WHEN lf.VALIDATED = 1 THEN 1 ELSE 0 END) AS approved,
    SUM(CASE WHEN lf.VALIDATED = 0 THEN 1 ELSE 0 END) AS pending
FROM [eHRAntung_DB].[dbo].[ATDLEAVEDATA] lf
LEFT JOIN [eHRAntung_DB].[dbo].[ATDATTENDANCECLASS] lt 
    ON CAST(lf.LEAVEID AS NVARCHAR(100)) = CAST(lt.ID AS NVARCHAR(100))
WHERE YEAR(lf.WORKDATE) = YEAR(GETDATE())
GROUP BY lt.leave_type_name, lt.CLASSCODE, lt.ID
ORDER BY total_hours DESC;
        """

    @staticmethod
    def query_top_employees_by_leave(limit: int = 50) -> str:
        return f"""
SELECT TOP ({limit})
    p.EMPLOYEEID,
    p.person_name,
    d.dept_display_name,
    COUNT(*) AS leave_count,
    SUM(CAST(lf.total_hours AS FLOAT)) AS total_hours,
    SUM(CAST(lf.calculated_days AS FLOAT)) AS total_days,
    AVG(CAST(lf.total_hours AS FLOAT)) AS avg_per_leave
FROM [eHRAntung_DB].[dbo].[ATDLEAVEDATA] lf
LEFT JOIN [eHRAntung_DB].[dbo].[PSNACCOUNT] p ON lf.PERSONID = p.PERSONID
LEFT JOIN [eHRAntung_DB].[dbo].[ORGStdStruct] d 
    ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(d.UNITID AS NVARCHAR(100))
WHERE lf.VALIDATED = 1
  AND YEAR(lf.WORKDATE) = YEAR(GETDATE())
GROUP BY p.EMPLOYEEID, p.person_name, d.dept_display_name, p.PERSONID
ORDER BY total_hours DESC;
        """

    @staticmethod
    def query_balance_snapshot(limit: int = 100) -> str:
        return f"""
SELECT TOP ({limit})
    p.EMPLOYEEID,
    p.person_name,
    d.dept_display_name,
    r.VACAYEAR,
    lt.leave_type_name,
    r.VACDAYS AS entitlement_days,
    r.USEDAYS AS used_days,
    r.REMAINDAYS AS remaining_days,
    CAST(r.CANUSEDATE AS DATE) AS available_from,
    CAST(r.DISABLEDDATE AS DATE) AS expires_on,
    CASE 
        WHEN r.REMAINDAYS <= 2 THEN 'Critical'
        WHEN r.REMAINDAYS <= 5 THEN 'Low'
        WHEN DATEDIFF(DAY, GETDATE(), r.DISABLEDDATE) <= 30 THEN 'Expiring Soon'
        ELSE 'Healthy'
    END AS balance_status
FROM [eHRAntung_DB].[dbo].[ATDCALCUVACATIONRESULT] r
LEFT JOIN [eHRAntung_DB].[dbo].[PSNACCOUNT] p ON r.PERSONID = p.PERSONID
LEFT JOIN [eHRAntung_DB].[dbo].[ORGStdStruct] d 
    ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(d.UNITID AS NVARCHAR(100))
LEFT JOIN [eHRAntung_DB].[dbo].[ATDATTENDANCECLASS] lt 
    ON CAST(r.VACATIONTYPE AS NVARCHAR(100)) = CAST(lt.ID AS NVARCHAR(100))
WHERE r.VACAYEAR = YEAR(GETDATE())
ORDER BY r.REMAINDAYS ASC;
        """

    @staticmethod
    def query_expiring_soon(days_threshold: int = 30, limit: int = 50) -> str:
        return f"""
SELECT TOP ({limit})
    p.EMPLOYEEID,
    p.person_name,
    d.dept_display_name,
    r.REMAINDAYS,
    CAST(r.DISABLEDDATE AS DATE) AS expires_on,
    DATEDIFF(DAY, GETDATE(), r.DISABLEDDATE) AS days_until_expiry,
    lt.leave_type_name
FROM [eHRAntung_DB].[dbo].[ATDCALCUVACATIONRESULT] r
LEFT JOIN [eHRAntung_DB].[dbo].[PSNACCOUNT] p ON r.PERSONID = p.PERSONID
LEFT JOIN [eHRAntung_DB].[dbo].[ORGStdStruct] d 
    ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(d.UNITID AS NVARCHAR(100))
LEFT JOIN [eHRAntung_DB].[dbo].[ATDATTENDANCECLASS] lt 
    ON CAST(r.VACATIONTYPE AS NVARCHAR(100)) = CAST(lt.ID AS NVARCHAR(100))
WHERE r.VACAYEAR = YEAR(GETDATE())
  AND DATEDIFF(DAY, GETDATE(), r.DISABLEDDATE) BETWEEN 0 AND {days_threshold}
  AND r.REMAINDAYS > 0
ORDER BY r.DISABLEDDATE ASC;
        """

# ============================================
# Section builders (lightweight examples)
# ============================================

class SectionBuilder:
    @staticmethod
    def build_leave_by_department_section(config: SectionConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        rows = data.get("leave_by_department", [])
        return {
            "title": config.title,
            "description": config.description,
            "depth": config.depth,
            "charts": [{
                "type": "bar", "title": "Leave Hours by Department",
                "data_key": "leave_by_department", "x_axis": "department", "y_axis": "total_hours"
            }] if any(c.chart_type == ChartType.BAR for c in config.charts) else [],
            "tables": [{
                "title": "Department Summary",
                "columns": ["department", "num_employees", "total_leaves", "total_hours", "avg_hours_per_leave"],
            }] if config.include_table else [],
            "statistics": {
                "total_departments": len(rows),
                "total_hours": sum((r.get("total_hours") or 0) for r in rows),
            }
        }

    @staticmethod
    def build_leave_by_type_section(config: SectionConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        rows = data.get("leave_by_type", [])
        return {
            "title": config.title,
            "description": config.description,
            "depth": config.depth,
            "charts": [
                {"type": "pie", "title": "Leave Distribution by Type", "data_key": "leave_by_type"},
                {"type": "bar", "title": "Leave Type Details", "data_key": "leave_by_type"},
            ],
            "statistics": {"total_types": len(rows)}
        }

    @staticmethod
    def build_balance_snapshot_section(config: SectionConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        rows = data.get("balance_snapshot", [])
        return {
            "title": config.title,
            "description": config.description,
            "charts": [{"type": "gauge", "title": "Organizational Leave Health", "data_key": "balance_snapshot_gauge"}],
            "tables": [{
                "title": "Balance Status by Employee",
                "columns": ["EMPLOYEEID", "person_name", "remaining_days", "expires_on", "balance_status"]
            }] if config.include_table else [],
            "statistics": {
                "healthy": sum(1 for r in rows if r.get("balance_status") == "Healthy"),
                "low":     sum(1 for r in rows if r.get("balance_status") == "Low"),
                "critical":sum(1 for r in rows if r.get("balance_status") == "Critical"),
            }
        }

# ============================================
# Helpers: DB execution & normalization
# ============================================

def _lang_hint_from(payload_lang: str) -> str:
    lang = (payload_lang or "").lower()
    return "zh-tw" if lang.startswith("zh") else "en"

def _rows_to_dicts(rows: Iterable[tuple], columns: Iterable[str]) -> List[Dict[str, Any]]:
    cols = list(columns or [])
    out: List[Dict[str, Any]] = []
    for tup in rows or []:
        out.append({cols[i]: tup[i] for i in range(min(len(cols), len(tup)))})
    return out

async def _db_fetch_all(
    request: Request,
    sql: str,
    *,
    params: Optional[tuple] = None,
    max_rows: int = 1000,
    language_hint: str = "en",
) -> List[Dict[str, Any]]:
    """
    Execute a SELECT via sync DB service in a worker thread.
    Expects request.app.state.db to be LanguageAwareSQLServerDatabaseService.
    """
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=500, detail="Database unavailable")

    def _run() -> List[Dict[str, Any]]:
        rows, columns = db.run_select(
            sql,
            params=params,
            max_rows=max_rows,
            query_timeout=None,
            language_hint=language_hint,
            enable_query_hints=True,
        )
        return _rows_to_dicts(rows, columns)

    try:
        return await to_thread.run_sync(_run)
    except Exception as e:
        logger.exception("SQL execution failed")
        raise HTTPException(status_code=500, detail=f"SQL error: {e}")

def _normalize_chart_prefs(raw: Dict[str, List[str]]) -> Dict[SectionType, List[ChartType]]:
    """
    Frontend sends { "leave_by_type": ["pie","bar"], ... }.
    Convert to { SectionType.LEAVE_BY_TYPE: [ChartType.PIE, ChartType.BAR], ... }
    """
    out: Dict[SectionType, List[ChartType]] = {}
    for k, v in (raw or {}).items():
        try:
            sec = SectionType(k) if not isinstance(k, SectionType) else k
        except Exception:
            continue
        pref: List[ChartType] = []
        for c in v or []:
            try:
                pref.append(ChartType(c) if not isinstance(c, ChartType) else c)
            except Exception:
                pass
        out[sec] = pref or [ChartType.BAR]
    return out

# ============================================
# Public service functions (used by routes)
# ============================================

async def get_report_options(payload: ConfigureReportRequest, request: Request) -> Dict[str, Any]:
    """
    Step 1 in the frontend state machine:
    - Analyze intent (basic heuristics here)
    - Return available sections, depth options, chart types
    - Optionally return clarifying questions (frontend will ask user)
    """
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    report_type = payload.analysis.get("report_type", "leave_analysis")
    available_sections = _get_available_sections_for_type(report_type)

    # Example: add a clarifying question if user mentioned expiring/expiry anywhere
    clarifying_questions: List[Dict[str, Any]] = []
    q_lower = (payload.query or "").lower()
    if ("expire" in q_lower or "expiring" in q_lower) and SectionType.EXPIRING_SOON not in available_sections:
        clarifying_questions.append({
            "id": "include_expiring",
            "prompt": "Include a section for leave expiring soon?",
            "options": [{"label": "Yes", "value": True}, {"label": "No", "value": False}]
        })

    logger.info("rid=%s get_report_options type=%s sections=%d", rid, report_type, len(available_sections))

    return {
        "request_id": rid,
        "analysis": {
            "report_type": report_type,
            "received_at": datetime.utcnow().isoformat() + "Z",
        },
        "available_sections": [
            {
                "id": s.value,
                "name": s.value.replace("_", " ").title(),
                "description": _get_section_description(s),
                "chart_options": [c.value for c in ChartType],
                "include_table": True,
                "can_customize": True,
            }
            for s in available_sections
        ],
        "depth_options": ["high_level", "standard", "detailed"],
        "chart_types": [c.value for c in ChartType],
        "clarifying_questions": clarifying_questions,
        "defaults": {
            "depth_level": "standard",
            "preselected_sections": [SectionType.EXECUTIVE_SUMMARY.value],
            "chart_preferences": {}
        }
    }

# Optional back-compat for old frontend that calls /analyze
async def analyze(payload: ConfigureReportRequest, request: Request) -> Dict[str, Any]:
    return await get_report_options(payload, request)

async def execute_report_with_config(payload: SectionSelectionRequest, request: Request) -> Dict[str, Any]:
    """
    Step 3 in the frontend state machine:
    - Execute the SQL for each selected section
    - Build section payloads (titles/charts/statistics)
    - Return a report_id and filename so the frontend can download
    """
    rid = payload.request_id or uuid.uuid4().hex
    logger.info("rid=%s execute_report sections=%d depth=%s", rid, len(payload.selected_sections), payload.depth_level)

    # Normalize chart preferences
    chart_prefs = _normalize_chart_prefs(payload.chart_preferences)

    # Build internal config
    config = ReportConfigSchema(
        sections=[
            SectionConfig(
                section_type=sec,
                title=sec.value.replace("_", " ").title(),
                depth=payload.depth_level,
                charts=[
                    ChartConfig(
                        chart_type=ct,
                        title=f"{sec.value} - {ct.value}",
                        data_source=sec.value,
                    ) for ct in (chart_prefs.get(sec) or [ChartType.BAR])
                ],
                include_table=True,
                table_rows=50
            )
            for sec in payload.selected_sections
        ],
        language=payload.language
    )

    # Execute SQL per section
    data_bucket: Dict[str, Any] = {}
    db_lang = _lang_hint_from(payload.language)

    try:
        for section_cfg in config.sections:
            sec = section_cfg.section_type

            if sec == SectionType.LEAVE_BY_DEPARTMENT:
                sql = DataQueryBuilder.query_leave_by_department(limit=200)
                data_bucket["leave_by_department"] = await _db_fetch_all(
                    request, sql, max_rows=200, language_hint=db_lang
                )

            elif sec == SectionType.LEAVE_BY_TYPE:
                sql = DataQueryBuilder.query_leave_by_type(limit=200)
                data_bucket["leave_by_type"] = await _db_fetch_all(
                    request, sql, max_rows=200, language_hint=db_lang
                )

            elif sec == SectionType.TOP_EMPLOYEES:
                sql = DataQueryBuilder.query_top_employees_by_leave(limit=100)
                data_bucket["top_employees_by_leave"] = await _db_fetch_all(
                    request, sql, max_rows=100, language_hint=db_lang
                )

            elif sec == SectionType.BALANCE_SNAPSHOT:
                sql = DataQueryBuilder.query_balance_snapshot(limit=300)
                rows = await _db_fetch_all(request, sql, max_rows=300, language_hint=db_lang)
                data_bucket["balance_snapshot"] = rows
                # derive a simple gauge score
                total = len(rows) or 1
                healthy = sum(1 for r in rows if r.get("balance_status") == "Healthy")
                data_bucket["balance_snapshot_gauge"] = {"score": round(100 * healthy / total, 1)}

            elif sec == SectionType.EXPIRING_SOON:
                sql = DataQueryBuilder.query_expiring_soon(days_threshold=30, limit=200)
                data_bucket["expiring_soon"] = await _db_fetch_all(
                    request, sql, max_rows=200, language_hint=db_lang
                )

            # Extend with more sections as needed...
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("rid=%s SQL execution failed", rid)
        raise HTTPException(status_code=500, detail=f"Failed to execute SQL: {e}")

    # Build section payloads
    built_sections: List[Dict[str, Any]] = []
    for section_cfg in config.sections:
        sec = section_cfg.section_type
        try:
            if sec == SectionType.LEAVE_BY_DEPARTMENT:
                built_sections.append(SectionBuilder.build_leave_by_department_section(section_cfg, data_bucket))
            elif sec == SectionType.LEAVE_BY_TYPE:
                built_sections.append(SectionBuilder.build_leave_by_type_section(section_cfg, data_bucket))
            elif sec == SectionType.BALANCE_SNAPSHOT:
                built_sections.append(SectionBuilder.build_balance_snapshot_section(section_cfg, data_bucket))
            else:
                built_sections.append({
                    "title": section_cfg.title,
                    "description": section_cfg.description,
                    "depth": section_cfg.depth,
                    "charts": [asdict(c) for c in section_cfg.charts],
                    "tables": [],
                    "statistics": {}
                })
        except Exception as e:
            logger.exception("rid=%s section build error for %s", rid, sec.value)
            built_sections.append({
                "title": section_cfg.title,
                "error": f"Failed to build section: {e}"
            })

    # Produce a report artifact identity (your /download/{report_id} route should use it)
    report_id = uuid.uuid4().hex
    filename = f"leave-report-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.docx"

    result = {
        "status": "report_ready",
        "request_id": rid,
        "report_id": report_id,
        "filename": filename,
        "analysis": {"report_type": "leave_analysis"},
        "configuration": asdict(config),
        "data_bucket": data_bucket,
        "sections_built": built_sections
    }

    logger.info("rid=%s execute_report done sections=%d report_id=%s", rid, len(config.sections), report_id)
    return result

# ============================================
# Internal helpers
# ============================================

def _get_available_sections_for_type(report_type: str) -> List[SectionType]:
    type_sections = {
        "leave_analysis": [
            SectionType.EXECUTIVE_SUMMARY,
            SectionType.LEAVE_BY_DEPARTMENT,
            SectionType.LEAVE_BY_TYPE,
            SectionType.TOP_EMPLOYEES,
            SectionType.BALANCE_SNAPSHOT,
            SectionType.EXPIRING_SOON,
            SectionType.RISK_ASSESSMENT,
            SectionType.RECOMMENDATIONS,
        ],
        "overtime_analysis": [
            SectionType.EXECUTIVE_SUMMARY,
            SectionType.OVERTIME_SUMMARY,
            SectionType.DEPARTMENT_COMPARISON,
            SectionType.RISK_ASSESSMENT,
        ],
        "attendance_analysis": [
            SectionType.EXECUTIVE_SUMMARY,
            SectionType.ATTENDANCE_RATE,
            SectionType.SICK_LEAVE_ANALYSIS,
            SectionType.DEPARTMENT_COMPARISON,
        ],
        "balance_report": [
            SectionType.EXECUTIVE_SUMMARY,
            SectionType.BALANCE_SNAPSHOT,
            SectionType.EXPIRING_SOON,
            SectionType.RISK_ASSESSMENT,
        ],
    }
    return type_sections.get(report_type, [SectionType.EXECUTIVE_SUMMARY, SectionType.DATA_QUALITY])

def _get_section_description(section: SectionType) -> str:
    descriptions = {
        SectionType.EXECUTIVE_SUMMARY: "High-level overview and key takeaways",
        SectionType.LEAVE_BY_DEPARTMENT: "Leave analysis by department with comparisons",
        SectionType.LEAVE_BY_TYPE: "Distribution of leave types",
        SectionType.LEAVE_TRENDS: "Historical trends and forecasting",
        SectionType.TOP_EMPLOYEES: "Top leave takers",
        SectionType.ATTENDANCE_RATE: "Attendance metrics and patterns",
        SectionType.SICK_LEAVE_ANALYSIS: "Sick leave usage patterns",
        SectionType.BALANCE_SNAPSHOT: "Current leave balance status",
        SectionType.EXPIRING_SOON: "Leave expiring in the near term",
        SectionType.OVERTIME_SUMMARY: "Overtime hours and cost analysis",
        SectionType.DEPARTMENT_COMPARISON: "Comparative analysis between departments",
        SectionType.RISK_ASSESSMENT: "Identified risks and compliance issues",
        SectionType.RECOMMENDATIONS: "Actionable recommendations",
        SectionType.DATA_QUALITY: "Data integrity and completeness assessment",
    }
    return descriptions.get(section, "Report section")
