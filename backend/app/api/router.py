# ================================================================================
# backend/app/api/router.py
from __future__ import annotations

# =========================
# Standard Library
# =========================
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# =========================
# Third-Party
# =========================
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# =========================
# Internal Services
# =========================
from app.services.data_processing.data_analyzer import DataAnalyzer
from app.services.db_service import SQLServerDatabaseService
from app.services.nlp_service_2 import LanguageNativeNLPService
from app.services.person_resolver import PersonResolver
from app.services.helpers.data_utils import (
    _apply_resolved,
    _collect_ids_from_rows,
    anchor_today,  # CRITICAL: unified date anchor (respects NLP_TODAY_OVERRIDE)
)
from app.services.factory import create_enhanced_nlp_service

from app.services.memory.simple_query_memory import SimpleQueryMemoryService

# Reports service
from app.reports.service import (
    ReportAnalysisRequest,
    ReportGenerationRequest,
    analyze_report,
    download_report_response,
    generate_report,
)

# Leave service (new service-code-first + legacy helpers)
from app.leave.service import (
    HCMServiceCallRequest,
    LeaveBalanceRequest,
    LeaveBalanceResponse,
    LeaveRequest,
    LeaveResponse,
    get_employee_leave_balance,
    hcm_call,
    submit_leave_request,
    validate_leave_request,
)

from datetime import datetime as _dt, date as _date, timedelta as _td

from app.services.helpers.data_utils import anchor_today
from app.services.utils.date_fill import fill_trend_to_today, build_meta

logger = logging.getLogger(__name__)


# Shared analyzer
_type_analyzer = DataAnalyzer()

# Routers
router_main = APIRouter()
router_leave = APIRouter(prefix="/api/leave", tags=["leave"])

# =========================
# Models (local)
# =========================
class UserInfo(BaseModel):
    id: str
    account: str
    user_name: str | None = None
    email: str | None = None
    mobile: str | None = None
    is_active: bool | int | None = None
    site_code: str | None = None
    active_start_on: str | None = None
    active_end_on: str | None = None
    changed_on: str | None = None


# =========================
# Utilities
# =========================
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-\.@]+$")
_ALLOWED_LEAVE_KINDS = {"metrics", "trend"}  # strict guard for kind


def _sanitize_user_id(user_id: str) -> str:
    if not _SAFE_ID_RE.match(user_id or ""):
        raise HTTPException(status_code=400, detail="Invalid user_id")
    return user_id


def _frontend_paths() -> tuple[Path, Path, Path, Path]:
    base_dir = Path(__file__).resolve().parents[2]
    project_root = base_dir.parent
    frontend_dir = project_root / "frontend"
    index_file = frontend_dir / "index.html"
    return base_dir, project_root, frontend_dir, index_file


def _decode_json_field(val):
    if not val:
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except Exception:
        return []


def _apply_type_labels_to_metrics(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload

    def _label(code):
        try:
            return _type_analyzer.label_leave_type(code)
        except Exception:
            return str(code) if code is not None else "(unknown)"

    for arr_key in ("on_leave_details", "upcoming_leave"):
        arr = payload.get(arr_key)
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, dict):
                    code = item.get("type_code")
                    item["type_label"] = _label(code)
    return payload


def _get_db(request: Request) -> SQLServerDatabaseService:
    """
    Unified accessor for DB service; supports both legacy `app.state.db`
    and newer `app.state.db_service`.
    """
    db = getattr(request.app.state, "db", None) or getattr(request.app.state, "db_service", None)
    if not isinstance(db, SQLServerDatabaseService):
        raise HTTPException(status_code=500, detail="Database service not initialized")
    return db


# =========================
# Users (tags: users)
# =========================
@router_main.get("/api/user/{user_id}", response_model=UserInfo, tags=["users"])
async def get_user(user_id: str, request: Request):
    """
    Read-only user profile for display in UI.
    Returns a safe subset of columns. No password/salt fields.
    """
    user_id = _sanitize_user_id(user_id)
    db = _get_db(request)

    sql = """
    SELECT TOP (1)
        [id],
        [account],
        [user_name],
        [email],
        [mobile],
        [is_active],
        [site_code],
        CONVERT(varchar(10), [active_start_on], 23) AS active_start_on,
        CONVERT(varchar(10), [active_end_on], 23)   AS active_end_on,
        CONVERT(varchar(19), [changed_on], 120)     AS changed_on
    FROM [eHRAntung_DB].[gcore].[om_user]
    WHERE [id] = ?
    """

    try:
        rows, cols = db.run_select(sql, params=[user_id])
    except TypeError:
        sql_fmt = sql.replace("WHERE [id] = ?", f"WHERE [id] = '{user_id}'")
        rows, cols = db.run_select(sql_fmt)

    if not rows:
        raise HTTPException(status_code=404, detail="User not found")

    row = dict(zip(cols, rows[0]))
    if "is_active" in row and isinstance(row["is_active"], (int, bool)):
        row["is_active"] = int(row["is_active"])
    return UserInfo(**row)


@router_main.get("/api/user/me", response_model=UserInfo, tags=["users"])
async def get_me(request: Request):
    # In real auth, derive user_id from the session/JWT; for now use 'chiuzu'
    return await get_user("chiuzu", request)


# =========================
# HCM Dynamic Service (tags: hcm)
# =========================
@router_main.post("/api/hcm/call", tags=["hcm"])
async def api_hcm_call(
    req: HCMServiceCallRequest,
    request: Request,
    x_idempotency_key: Optional[str] = Header(default=None, convert_underscores=False),
):
    """
    Generic HCM caller used by the dynamic form.
    Backend constructs fresh LogonInfo with a current ExpiredDate
    and forwards 'data' as-is using 'service_code'.
    """
    try:
        result = await hcm_call(req, idempotency_key=x_idempotency_key)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("HCM dynamic call failed")
        raise HTTPException(status_code=500, detail=f"HCM call failed: {str(e)}")


# =========================
# Legacy Leave API (tags: leave)
# =========================
@router_leave.post("/submit", response_model=LeaveResponse)
async def submit_leave(
    request: LeaveRequest,
    x_idempotency_key: Optional[str] = Header(default=None, convert_underscores=False),
):
    """
    Legacy: Submit a leave request via mapped leave_type.
    Prefer /api/hcm/call going forward.
    """
    try:
        validation = await validate_leave_request(request)
        if not validation["valid"]:
            return LeaveResponse(
                success=False,
                message=f"Validation failed: {', '.join(validation['errors'])}",
                data={"warnings": validation.get("warnings", [])},
            )
        result = await submit_leave_request(request, idempotency_key=x_idempotency_key)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error submitting leave request")
        raise HTTPException(status_code=500, detail=f"Failed to submit leave request: {str(e)}")


@router_leave.post("/balance", response_model=LeaveBalanceResponse)
async def get_leave_balance(request: LeaveBalanceRequest):
    """Legacy: Get employee leave balance."""
    try:
        return await get_employee_leave_balance(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting leave balance")
        raise HTTPException(status_code=500, detail=f"Failed to get leave balance: {str(e)}")


@router_leave.post("/validate")
async def validate_leave(request: LeaveRequest):
    """Legacy: Validate a leave request without submitting."""
    try:
        validation = await validate_leave_request(request)
        return {"success": True, "validation": validation}
    except Exception as e:
        logger.exception("Error validating leave request")
        raise HTTPException(status_code=500, detail=f"Failed to validate leave request: {str(e)}")


# =========================
# Static / SPA (tags: static)
# =========================
@router_main.get("/", include_in_schema=False, tags=["static"])
async def serve_index():
    _, _, frontend_dir, index_file = _frontend_paths()
    logger.info("Frontend dir=%s index exists=%s", frontend_dir, index_file.exists())
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse("/docs")


@router_main.get("/dashboard", include_in_schema=False, tags=["static"])
async def serve_dashboard():
    base_dir = Path(__file__).resolve().parents[2]
    frontend_dir = base_dir / "frontend"
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse("/docs")


@router_main.get("/api/ping", include_in_schema=False, tags=["static"])
async def ping():
    return {"ok": True}


@router_main.get("/leave_page.html", include_in_schema=False, tags=["static"])
async def serve_leave_page():
    _, _, frontend_dir, _ = _frontend_paths()
    leave_page_file = frontend_dir / "leave_page.html"
    if leave_page_file.exists():
        return FileResponse(str(leave_page_file))
    return RedirectResponse("/docs")


@router_main.get("/translations.js", include_in_schema=False, tags=["static"])
async def serve_translations():
    _, _, frontend_dir, _ = _frontend_paths()
    translations_file = frontend_dir / "translations.js"
    if translations_file.exists():
        return FileResponse(str(translations_file))
    return PlainTextResponse("// translations.js not found", media_type="application/javascript")


@router_main.get("/generate_report.html", include_in_schema=False, tags=["static"])
async def serve_generate_report():
    _, _, frontend_dir, _ = _frontend_paths()
    generate_report_file = frontend_dir / "generate_report.html"
    if generate_report_file.exists():
        return FileResponse(str(generate_report_file))
    return RedirectResponse("/docs")


# =========================
# Reports API (tags: reports)
# =========================
@router_main.post("/api/reports/analyze", tags=["reports"])
async def api_reports_analyze(req: ReportAnalysisRequest):
    return await analyze_report(req)


@router_main.post("/api/reports/generate", tags=["reports"])
async def api_reports_generate(req: ReportGenerationRequest):
    return await generate_report(req)


@router_main.get("/api/reports/download/{report_id}", tags=["reports"])
async def api_reports_download(report_id: str):
    return await download_report_response(report_id)


# =========================
# Leave Dashboard Data (tags: leave-dashboard)
# =========================



# If your project already defines this elsewhere, feel free to remove this fallback.
_ALLOWED_LEAVE_KINDS = {"metrics", "trend"}


def _decode_json_field(val: Any) -> list[dict]:
    if not val:
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except Exception:
        return []


def _collect_ids_from_rows(*rows_groups: list[list[dict]]) -> tuple[list[str], list[str]]:
    """Collect PERSONID and EMPLOYEEID (if present) from nested rows."""
    pids, eids = set(), set()
    for group in rows_groups:
        if not group:
            continue
        for row in group:
            pid = (row.get("person_id") or row.get("PERSONID") or "").strip()
            if pid:
                pids.add(pid)
            eid = (row.get("employee_id") or row.get("EMPLOYEEID") or "").strip()
            if eid:
                eids.add(eid)
    return list(pids), list(eids)


def _apply_resolved(rows: list[dict], resolved: dict[str, dict]) -> list[dict]:
    """Apply resolver map to rows in-place (safe copy)."""
    out: list[dict] = []
    for r in rows or []:
        rr = dict(r)
        pid = str(rr.get("person_id") or rr.get("PERSONID") or "").strip()
        if pid and pid in resolved:
            meta = resolved[pid]
            # Prefer existing values, only backfill if missing
            rr.setdefault("person_name", meta.get("person_name"))
            rr.setdefault("employee_id", meta.get("employee_id"))
            rr.setdefault("display_name", meta.get("display_name") or meta.get("person_name"))
            # if resolver has org data, backfill department fields (but don't overwrite SQL output)
            rr.setdefault("department_id", meta.get("department_id"))
            rr.setdefault("department_code", meta.get("department_code"))
            rr.setdefault("department_name", meta.get("department_name"))
        out.append(rr)
    return out


@router_main.get("/api/leave_data", tags=["leave-dashboard"])
async def leave_data(
    request: Request,
    kind: str = "metrics",
    as_of: Optional[str] = None,
    days: int = 7,
) -> Dict[str, Any]:
    """
    Unified, date-safe leave dashboard data endpoint.

    Guarantees:
    - `effective_as_of`: last data day (clamped to dataset window; never future).
    - `current_date`: shared anchor from `anchor_today()` (respects any override).
    - Trend is padded forward to `current_date` with zero-count days (via fill_trend_to_today).
    - Adds `meta` with freshness info (via build_meta; safe fallback if unavailable).
    """
    db = _get_db(request)

    # ---- Guard 'kind'
    k = (kind or "").lower().strip()
    if k not in _ALLOWED_LEAVE_KINDS:
        raise HTTPException(status_code=400, detail=f"Invalid kind '{kind}'. Use one of: metrics, trend.")

    # ---- Anchor "today" (shared across system)
    anchor: _date = anchor_today()  # may be overridden centrally
    anchor_str = anchor.strftime("%Y-%m-%d")

    # ---- Parse as_of (optional). Accept YYYY-MM-DD or YYYY/MM/DD; clamp to <= anchor.
    as_of_dt: _date = anchor
    if as_of:
        token = as_of.strip().replace("/", "-")
        try:
            parsed = _dt.strptime(token, "%Y-%m-%d").date()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'as_of' format. Use YYYY-MM-DD.")
        if parsed > anchor:
            parsed = anchor
        as_of_dt = parsed

    # ---- Fetch dataset window
    try:
        win_rows, _ = db.run_select(
            """
            SELECT
              CONVERT(varchar(10), MIN(CAST(WORKDATE AS date)), 23) AS min_date,
              CONVERT(varchar(10), MAX(CAST(WORKDATE AS date)), 23) AS max_date
            FROM dbo.ATDLEAVEDATA
            """
        )
        min_date_str, max_date_str = (win_rows[0][0], win_rows[0][1]) if win_rows else (None, None)
    except Exception as e:
        logger.warning("Failed to fetch data window: %s", e)
        min_date_str, max_date_str = None, None

    if not min_date_str or not max_date_str:
        # Empty shell while still exposing anchor/meta for UI
        base = {
            "success": True,
            "data_window": {"min_date": min_date_str, "max_date": max_date_str},
            "effective_as_of": None,
            "current_date": anchor_str,
        }
        # try build_meta, fall back
        try:
            base["meta"] = build_meta(
                effective_as_of=None,
                current_date=anchor_str,
                min_date=min_date_str,
                max_date=max_date_str,
                requested_days=days if k == "trend" else None,
            )
        except Exception:
            base["meta"] = {"is_stale": True, "stale_days": None, "note": "no data window"}
        return {**base, ("trend" if k == "trend" else "metrics"): ([] if k == "trend" else {})}

    min_dt = _dt.strptime(min_date_str, "%Y-%m-%d").date()
    max_dt = _dt.strptime(max_date_str, "%Y-%m-%d").date()

    # ---- Clamp as_of to data window
    if as_of_dt < min_dt:
        as_of_dt = min_dt
    elif as_of_dt > max_dt:
        as_of_dt = max_dt
    as_of_str = as_of_dt.strftime("%Y-%m-%d")

    # ---- Compute effective trend start if needed
    effective_start_str, effective_days = None, days
    if k == "trend":
        try:
            days = max(1, min(int(days), 31))  # clamp defensively (UI only needs recent)
        except Exception:
            days = 7
        start_dt = max(min_dt, as_of_dt - _td(days=days - 1))
        effective_days = (as_of_dt - start_dt).days + 1
        effective_start_str = start_dt.strftime("%Y-%m-%d")

    # ---- Execute SQL (metrics/trend)
    try:
        from app.home_page_metrics.leave_metrics import _sql_leave_metrics, _sql_leave_trend

        sql = _sql_leave_trend(as_of_str, effective_days) if k == "trend" else _sql_leave_metrics(as_of_str)
        rows, columns = db.run_select(sql, language_hint="zh-tw", enable_query_hints=True)

        extra_ctx: Dict[str, Any] = {
            "success": True,
            "data_window": {"min_date": min_date_str, "max_date": max_date_str},
            "effective_as_of": as_of_str,
            "current_date": anchor_str,  # always expose today's anchor
        }
        if k == "trend" and effective_start_str:
            extra_ctx["effective_range"] = {"start": effective_start_str, "end": as_of_str}

        # Attach meta (with safe fallback if util signature differs)
        try:
            extra_ctx["meta"] = build_meta(
                effective_as_of=as_of_str,
                current_date=anchor_str,
                min_date=min_date_str,
                max_date=max_date_str,
                requested_days=effective_days if k == "trend" else None,
            )
        except Exception:
            extra_ctx["meta"] = {
                "is_stale": (as_of_dt < anchor),
                "stale_days": (anchor - as_of_dt).days if as_of_dt < anchor else 0,
                "window": {"min": min_date_str, "max": max_date_str},
            }

        if not rows:
            return ({"trend": [], **extra_ctx} if k == "trend" else {"metrics": {}, **extra_ctx})

        row = dict(zip(columns, rows[0]))

        # ---------- metrics ----------
        if k == "metrics" and isinstance(row.get("metrics"), str):
            payload = json.loads(row["metrics"])

            # Person enrichment
            resolver = PersonResolver(db_service=db)
            details = _decode_json_field(payload.get("on_leave_details"))
            upcoming = _decode_json_field(payload.get("upcoming_leave"))
            pid_list, eid_list = _collect_ids_from_rows(details, upcoming)
            resolved = resolver.resolve_many(pid_list, employee_ids=eid_list)
            payload["on_leave_details"] = _apply_resolved(details, resolved)
            payload["upcoming_leave"] = _apply_resolved(upcoming, resolved)

            # Leave type labels (keep your existing helper)
            try:
                from app.home_page_metrics.leave_metrics import _apply_type_labels_to_metrics as _labels_fn
                payload = _labels_fn(payload)
            except Exception:
                pass  # if helper missing, continue silently

            return {"metrics": payload, **extra_ctx}

        # ---------- trend ----------
        if k == "trend" and isinstance(row.get("trend"), str):
            trend_list = json.loads(row["trend"])

            # 1) pad to TODAY (anchor) so charts always show the current date
            try:
                trend_list = fill_trend_to_today(
                    trend_list,
                    effective_as_of=as_of_str,            # last data day
                    today=anchor_str,                     # UI 'today'
                    date_key="date",
                    count_key="count",
                    people_key="people_on_leave",
                )
            except Exception as e:
                logger.warning("fill_trend_to_today failed: %s", e)

            # 2) enrich person names/departments for each day's people list
            resolver = PersonResolver(db_service=db)
            all_people_arrays = [_decode_json_field(d.get("people_on_leave")) for d in trend_list]
            pid_list, eid_list = _collect_ids_from_rows(*all_people_arrays)
            resolved = resolver.resolve_many(pid_list, employee_ids=eid_list)

            for d in trend_list:
                ppl = _decode_json_field(d.get("people_on_leave"))
                d["people_on_leave"] = _apply_resolved(ppl, resolved)

            return {"trend": trend_list, **extra_ctx}

        # Fallback passthrough
        return {**row, **extra_ctx}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("/api/leave_data failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"leave_data query failed: {str(e)}")



# =========================
# Assistant - New
# =========================

def get_query_memory_service(request: Request) -> SimpleQueryMemoryService:
    """
    Lazy-instantiated, process-local query memory.
    TTL and cache size can be tuned via env vars without code changes.
    """
    mem = getattr(request.app.state, "query_memory", None)
    if not isinstance(mem, SimpleQueryMemoryService):
        ttl = int(os.getenv("MEM_TTL_MIN", "45"))
        cap = int(os.getenv("MEM_CACHE_SIZE", "300"))
        mem = SimpleQueryMemoryService(cache_ttl_minutes=ttl, max_cache_size=cap)
        request.app.state.query_memory = mem
    return mem


def get_enhanced_nlp_service(request: Request):
    """
    Get the enhanced NLP service with visualization capabilities.
    Lazily creates and caches it on app.state.nlp_enhanced if needed.
    Also sets legacy alias `app.state.nlp` for backward compatibility.
    """
    if hasattr(request.app.state, "nlp_enhanced"):
        svc = request.app.state.nlp_enhanced
        # mirror to legacy alias if missing
        if not hasattr(request.app.state, "nlp"):
            request.app.state.nlp = svc
        return svc

    db_service = getattr(request.app.state, "db_service", None) or getattr(request.app.state, "db", None)
    if isinstance(db_service, SQLServerDatabaseService):
        service = create_enhanced_nlp_service(db_service)
        request.app.state.nlp_enhanced = service
        # keep legacy alias to avoid breaking existing admin tools
        request.app.state.nlp = service
        return service

    return None


def _ok(payload: Dict[str, Any], status: int = 200) -> JSONResponse:
    payload.setdefault("success", True)
    return JSONResponse(payload, status_code=status)


def _err(rid: str, msg: str, status: int = 500, extra: Optional[Dict[str, Any]] = None) -> JSONResponse:
    body: Dict[str, Any] = {"success": False, "error": msg, "rid": rid}
    if extra:
        body.update(extra)
    return JSONResponse(body, status_code=status)


@router_main.post("/api/assistant/query", tags=["assistant"])
async def assistant_query(payload: dict, request: Request):
    """
    Standard query endpoint – visualization disabled.
    Expects: { query: str, schema?: str, lang?: "en"|"zh-tw" }
    Memory: rewrites follow-ups and learns from results.
    """
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    # Best-effort session id (client can send X-Session-ID)
    session_id = (
        request.headers.get("X-Session-ID")
        or request.headers.get("X-User-Id")
        or request.headers.get("X-Account")
        or "default"
    )

    q = (payload or {}).get("query", "").strip()
    schema = (payload or {}).get("schema", "dbo")
    lang = (payload or {}).get("lang")  # optional override

    if not q:
        return _err(rid, "Query is required", status=400)

    t0 = time.perf_counter()
    request.app.state.last_request_id = rid

    nlp: LanguageNativeNLPService = get_enhanced_nlp_service(request)
    if nlp is None:
        return _err(rid, "NLP service not available", status=503)

    # ---- Memory: follow-up rewrite -----------------------------------------
    mem = get_query_memory_service(request)
    rewritten_q, applied_ctx = mem.rewrite_with_context(session_id, q)
    if rewritten_q != q:
        logger.info("ASSIST_MEM_REWRITE rid=%s sid=%s applied=%s", rid, session_id, applied_ctx)

    try:
        # Run the query (no viz)
        result = await run_in_threadpool(
            nlp.process_complete_query,
            rewritten_q,  # user_input (possibly rewritten)
            schema,       # schema_name
            rid,          # rid
            False,        # include_visualization (OFF)
            None,         # force_chart_type
            lang,         # lang override
        )
        if not isinstance(result, dict):
            result = {}

        # Normalize for FE
        result.setdefault("success", True)
        result.setdefault("rid", rid)
        result["visualization"] = None

        # ---- Memory: learn from results ------------------------------------
        # Extract robustly from NLP result (keep graceful fallbacks)
        generated_sql = (
            result.get("sql")
            or result.get("generated_sql")
            or result.get("final_sql")
            or result.get("query_sql")
            or ""
        )
        relevant_tables = (
            result.get("relevant_tables")
            or result.get("tables")
            or result.get("table_names")
            or []
        )
        columns = result.get("columns") or result.get("column_names") or []
        rows = result.get("rows") or result.get("data") or []

        # Any time meta we can anchor for later follow-ups (optional)
        meta_time = {}
        for k in ("effective_as_of", "effective_range"):
            if k in result:
                meta_time[k] = result[k]

        exec_ms = int((time.perf_counter() - t0) * 1000)
        ok = bool(result.get("success", True))

        # Persist memory (safe even if some fields are empty)
        try:
            mem.learn_from_query(
                query=rewritten_q,
                relevant_tables=relevant_tables if isinstance(relevant_tables, list) else [],
                generated_sql=generated_sql or "",
                success=ok,
                execution_time=exec_ms,
                session_id=session_id,
                time_anchor=str(meta_time.get("effective_as_of")) if meta_time else None,
                time_range=(
                    str((meta_time.get("effective_range") or {}).get("start")),
                    str((meta_time.get("effective_range") or {}).get("end")),
                ) if meta_time.get("effective_range") else None,
            )
            mem.record_success(
                session_id=session_id,
                query=rewritten_q,
                generated_sql=generated_sql or "",
                columns=columns if isinstance(columns, list) else [],
                rows=rows if isinstance(rows, list) else [],
                relevant_tables=relevant_tables if isinstance(relevant_tables, list) else [],
                schema_ctx=schema,
                meta_time=meta_time or None,
            )
        except Exception as me:
            logger.warning("ASSIST_MEM_LEARN_FAIL rid=%s sid=%s err=%s", rid, session_id, me)

        # Light memory echo (non-breaking; FE can ignore)
        result.setdefault("memory", {})
        result["memory"].update({
            "applied_context": applied_ctx,
            "rewritten_query": rewritten_q if rewritten_q != q else None,
            "session_id": session_id,
        })

        return _ok(result)

    except Exception as e:
        return _err(rid, str(e), status=500)
    finally:
        ms = int((time.perf_counter() - t0) * 1000)
        request.app.state.last_request_ms = ms



@router_main.post("/api/assistant/query_with_viz", tags=["assistant"])
async def assistant_query_with_viz(payload: dict, request: Request):
    """
    Enhanced query endpoint with visualization generation.
    Expects: { query, schema?, include_visualization?: bool=true, chart_type?: string, lang?: "en"|"zh-tw" }
    Memory: rewrites follow-ups and learns from results.
    """
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    session_id = (
        request.headers.get("X-Session-ID")
        or request.headers.get("X-User-Id")
        or request.headers.get("X-Account")
        or "default"
    )

    q = (payload or {}).get("query", "").strip()
    schema = (payload or {}).get("schema", "dbo")
    include_viz = bool((payload or {}).get("include_visualization", True))
    force_chart_type = (payload or {}).get("chart_type") or None
    lang = (payload or {}).get("lang")

    if not q:
        logger.warning("VIZ_API_REJECT: rid=%s reason=empty_query", rid)
        return _err(rid, "Query is required", status=400)

    t0 = time.perf_counter()
    request.app.state.last_request_id = rid
    nlp: LanguageNativeNLPService = get_enhanced_nlp_service(request)
    if nlp is None:
        logger.error("VIZ_API_REJECT: rid=%s reason=nlp_not_available", rid)
        return _err(rid, "NLP service not available", status=503)

    # ---- Memory: follow-up rewrite -----------------------------------------
    mem = get_query_memory_service(request)
    rewritten_q, applied_ctx = mem.rewrite_with_context(session_id, q)
    if rewritten_q != q:
        logger.info("VIZ_MEM_REWRITE rid=%s sid=%s applied=%s", rid, session_id, applied_ctx)

    try:
        logger.info(
            "VIZ_API_START: rid=%s include_viz=%s chart=%s lang=%s q=%r",
            rid, include_viz, force_chart_type, lang, rewritten_q[:200]
        )

        # Do the heavy work in a threadpool
        result: Dict[str, Any] = await run_in_threadpool(
            nlp.process_complete_query,
            rewritten_q, schema, rid,
            include_viz,       # include_visualization
            force_chart_type,  # force_chart_type
            lang               # lang override
        )

        if not isinstance(result, dict):
            logger.warning("VIZ_API_WARN: rid=%s got_non_dict_result type=%s", rid, type(result))
            result = {}

        # Normalize success flags for frontend
        result["success"] = True
        result["ok"] = True
        result["rid"] = rid

        # ---- Memory: learn from results (before viz normalization logging) ---
        generated_sql = (
            result.get("sql")
            or result.get("generated_sql")
            or result.get("final_sql")
            or result.get("query_sql")
            or ""
        )
        relevant_tables = (
            result.get("relevant_tables")
            or result.get("tables")
            or result.get("table_names")
            or []
        )
        columns = result.get("columns") or result.get("column_names") or []
        rows = result.get("rows") or result.get("data") or []

        meta_time = {}
        for k in ("effective_as_of", "effective_range"):
            if k in result:
                meta_time[k] = result[k]

        try:
            exec_ms = int((time.perf_counter() - t0) * 1000)
            mem.learn_from_query(
                query=rewritten_q,
                relevant_tables=relevant_tables if isinstance(relevant_tables, list) else [],
                generated_sql=generated_sql or "",
                success=True,
                execution_time=exec_ms,
                session_id=session_id,
                time_anchor=str(meta_time.get("effective_as_of")) if meta_time else None,
                time_range=(
                    str((meta_time.get("effective_range") or {}).get("start")),
                    str((meta_time.get("effective_range") or {}).get("end")),
                ) if meta_time.get("effective_range") else None,
            )
            mem.record_success(
                session_id=session_id,
                query=rewritten_q,
                generated_sql=generated_sql or "",
                columns=columns if isinstance(columns, list) else [],
                rows=rows if isinstance(rows, list) else [],
                relevant_tables=relevant_tables if isinstance(relevant_tables, list) else [],
                schema_ctx=schema,
                meta_time=meta_time or None,
            )
        except Exception as me:
            logger.warning("VIZ_MEM_LEARN_FAIL rid=%s sid=%s err=%s", rid, session_id, me)

        # ---- Normalize visualization payload --------------------------------
        viz = (result.get("visualization") or result.get("viz") or {}) or {}
        result["viz"] = viz

        raw_url  = viz.get("url") or viz.get("image_url")
        raw_b64  = viz.get("base64") or viz.get("data_base64")
        raw_path = viz.get("path") or viz.get("file_path")
        enabled  = bool(viz.get("enabled"))

        logger.info(
            "VIZ_API_META: rid=%s enabled=%s url=%r b64=%s path=%r",
            rid, enabled, raw_url, bool(raw_b64 and raw_b64.strip()), raw_path
        )

        viz_generated = bool(enabled or raw_url or (raw_b64 and raw_b64.strip()) or raw_path)
        result["visualization_generated"] = viz_generated

        public_url: Optional[str] = None
        debug_reason: Optional[str] = None

        if raw_url and isinstance(raw_url, str):
            public_url = raw_url
            logger.info("VIZ_API_URL_USE: rid=%s url=%s", rid, public_url)
        elif raw_b64 and isinstance(raw_b64, str) and raw_b64.strip():
            b64 = raw_b64.split(",", 1)[-1].strip()
            public_url = f"data:image/png;base64,{b64}"
            logger.info("VIZ_API_URL_FROM_BASE64: rid=%s size_chars=%d", rid, len(b64))
        elif raw_path and isinstance(raw_path, str):
            try:
                import base64, mimetypes, os as _os
                if _os.path.exists(raw_path):
                    mime = mimetypes.guess_type(raw_path)[0] or "image/png"
                    with open(raw_path, "rb") as f:
                        enc = base64.b64encode(f.read()).decode("ascii")
                    public_url = f"data:{mime};base64,{enc}"
                    logger.info("VIZ_API_URL_FROM_PATH: rid=%s path=%s mime=%s bytes=%d",
                                rid, raw_path, mime, len(enc) // 4 * 3)
                else:
                    debug_reason = f"file_not_found:{raw_path}"
                    logger.warning("VIZ_API_FILE_MISSING: rid=%s path=%s", rid, raw_path)
            except Exception as e:
                debug_reason = f"path_read_error:{type(e).__name__}:{e}"
                logger.exception("VIZ_API_PATH_READ_FAIL: rid=%s path=%s err=%s", rid, raw_path, e)

        if viz_generated and public_url:
            result["visualization_url"] = public_url
            result["image_url"] = public_url
            result["visualization_type"] = viz.get("type")
            result["visualization_title"] = viz.get("title")
            result["visualization_insights"] = viz.get("insights") or []
            result["visualization_reason"] = viz.get("reasoning") or viz.get("reason") or ""
            logger.info("VIZ_API_OK: rid=%s type=%s title=%r", rid, viz.get("type"), viz.get("title"))
        else:
            result["visualization_url"] = None
            result["image_url"] = None
            result["visualization_type"] = viz.get("type")
            result["visualization_title"] = viz.get("title")
            result["visualization_insights"] = viz.get("insights") or []
            result["visualization_reason"] = (
                viz.get("reason")
                or viz.get("reasoning")
                or debug_reason
                or "Visualization not available (no public URL or data URI)."
            )
            logger.warning(
                "VIZ_API_NO_URL: rid=%s enabled=%s reason=%r type=%s title=%r",
                rid, viz_generated, result["visualization_reason"], viz.get("type"), viz.get("title")
            )

        # Light memory echo (non-breaking; FE can ignore)
        result.setdefault("memory", {})
        result["memory"].update({
            "applied_context": applied_ctx,
            "rewritten_query": rewritten_q if rewritten_q != q else None,
            "session_id": session_id,
        })

        ms = int((time.perf_counter() - t0) * 1000)
        request.app.state.last_request_ms = ms
        logger.info("VIZ_API_DONE: rid=%s ms=%d", rid, ms)
        return _ok(result)

    except Exception as e:
        logger.exception("VIZ_API_FATAL: rid=%s err=%s", rid, e)
        return _err(rid, str(e), status=500, extra={"visualization_generated": False})



@router_main.get("/api/assistant/visualization_status", tags=["assistant"])
async def get_visualization_status(request: Request):
    """
    Lightweight capability probe for the client (to decide whether to show the button).
    """
    nlp = get_enhanced_nlp_service(request)
    if nlp and getattr(nlp, "viz", None):
        return _ok({
            "available": True,
            "auto_visualization": getattr(nlp, "enable_auto_visualization", False),
            "supported_languages": ["en", "zh-tw"],
            "chart_types": [
                "line_chart", "bar_chart", "pie_chart",
                "scatter_plot", "histogram", "heatmap", "box_plot", "table", "area_chart",
            ],
        })
    return _ok({"available": False, "reason": "Visualization extension not initialized"})


# =========================
# Vector Admin / Health (tags: assistant, vector, health)
# =========================
@router_main.post("/api/vector/reload", tags=["vector"])
async def vector_reload(request: Request):
    vb = getattr(request.app.state, "vector_bootstrap", None)
    if vb is None:
        raise HTTPException(status_code=500, detail="Vector bootstrapper missing")
    try:
        result = await vb.start()  # idempotent
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector reload failed: {e}")


@router_main.get("/api/health", tags=["health"])
async def health(
    request: Request,
    no_db: bool = Query(False),
    no_vector: bool = Query(False),
    warm: bool = Query(False),
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out: Dict[str, Any] = {}

    # DB check
    if not no_db:
        try:
            t = time.perf_counter()
            db = getattr(request.app.state, "db", None)
            if not isinstance(db, SQLServerDatabaseService):
                db = getattr(request.app.state, "db_service", None)
            if not isinstance(db, SQLServerDatabaseService):
                raise RuntimeError("DB service not initialized")
            db_ok = bool(db.test_connection(login_timeout=2))
            out["database_connection"] = db_ok
            out["database_ms"] = int((time.perf_counter() - t) * 1000)
            logger.info("health: db ok=%s dur=%dms", db_ok, out["database_ms"])
        except BaseException as e:
            out["database_connection"] = False
            out["database_error"] = f"{type(e).__name__}: {e}"
            logger.exception("health: db check raised %s", type(e).__name__)
    else:
        out["database_connection"] = None
        out["database_skipped"] = True

    # Vector bootstrap status
    vb = getattr(request.app.state, "vector_bootstrap", None)
    out["vector_bootstrap"] = vb.status if vb else {"available": False}

    if warm and vb:
        try:
            warm_result = await vb.start()
            out["vector_bootstrap_after_warm"] = warm_result
            # 👇 CRITICAL: Expose the detailed summary
            if "summary" in warm_result:
                out["vector_bootstrap_summary"] = warm_result["summary"]
        except Exception as e:
            out["vector_bootstrap_after_warm_error"] = f"{type(e).__name__}: {e}"

    # Vector service status
    if not no_vector:
        try:
            t = time.perf_counter()
            nlp: LanguageNativeNLPService = getattr(request.app.state, "nlp", None) or getattr(request.app.state, "nlp_enhanced", None)
            if nlp:
                vector_status = nlp.vector_status()
                out["vector_db"] = {"ready": vector_status.get("ready", False), "service_type": "language_native", **vector_status}
            else:
                out["vector_db"] = {"ready": False, "error": "NLP service not available"}

            out["vector_ms"] = int((time.perf_counter() - t) * 1000)
            vec_ready = bool(out["vector_db"].get("ready"))
            logger.info("health: vector ok=%s dur=%dms", vec_ready, out["vector_ms"])
        except BaseException as e:
            out["vector_db"] = {"ready": False, "error": f"{type(e).__name__}: {e}"}
            logger.exception("health: vector_status raised %s", type(e).__name__)
    else:
        out["vector_db"] = {"skipped": True}

    db_ok = out.get("database_connection") is True if "database_connection" in out else True
    vec_ok = bool(out.get("vector_db", {}).get("ready", True))
    out["ready_for_queries"] = db_ok and vec_ok
    out["total_ms"] = int((time.perf_counter() - t0) * 1000)
    out["services"] = {"nlp_mode": "language_native", "language_native_processing": True}

    return out


# =========================
# Debug (tags: debug)
# =========================
@router_main.get("/debug/leave/health", tags=["debug"])
def leave_health(request: Request):
    """Health check for leave system."""
    result = {}

    nlp: LanguageNativeNLPService = getattr(request.app.state, "nlp", None) or getattr(request.app.state, "nlp_enhanced", None)
    if nlp and getattr(nlp, "vector_search", None):
        try:
            result["nlp_service"] = nlp.vector_status()
        except Exception as e:
            result["nlp_service"] = {"ready": False, "error": str(e)}
    else:
        result["nlp_service"] = {"ready": False, "error": "Service not available"}

    sanity = {}
    try:
        if nlp:
            backend = getattr(nlp.vector_search, "_db", None) or getattr(nlp.vector_search, "db", None)
            if backend and hasattr(backend, "relationships_sanity_check"):
                sanity = backend.relationships_sanity_check()
    except Exception as e:
        sanity = {"error": f"{type(e).__name__}: {e}"}

    result["sanity_check"] = sanity
    result["service_type"] = "language_native"
    return result


@router_main.get("/debug/leave/join-hints", tags=["debug"])
def leave_join_hints(request: Request, tables: List[str] = Query(..., alias="tables")):
    """Get join hints using NLP service."""
    nlp: LanguageNativeNLPService = getattr(request.app.state, "nlp", None) or getattr(request.app.state, "nlp_enhanced", None)
    if not nlp or not getattr(nlp, "vector_search", None):
        raise HTTPException(status_code=500, detail="NLP/Vector service not initialized")
    try:
        hints = nlp.vector_search.get_join_hints(tables)
        return {"tables": tables, "join_hints": hints, "service_type": "language_native"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"join-hints failed: {e}")


# =========================
# Export combined router
# =========================
router = APIRouter()
router.include_router(router_main)
router.include_router(router_leave)
