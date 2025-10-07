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
from app.services.helpers.data_utils import _apply_resolved, _collect_ids_from_rows
from app.services.factory import create_enhanced_nlp_service



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


def _collect_person_ids(*arrays):
    ids = set()
    for arr in arrays:
        if not isinstance(arr, list):
            continue
        for r in arr:
            pid = r.get("person_id") or r.get("PERSONID")
            if pid:
                ids.add(str(pid).strip())
    return list(ids)


def _patch_rows(rows, resolved_map):
    out = []
    for r in rows or []:
        pid = str(r.get("person_id") or r.get("PERSONID") or "").strip()
        info = resolved_map.get(pid, {})
        r["person_id"] = pid or None
        r["person_name"] = info.get("name") or pid or None
        if not r.get("employee_id"):
            r["employee_id"] = info.get("employee_id")
        if not r.get("email"):
            r["email"] = info.get("email")
        r["cardnum"] = info.get("cardnum")
        out.append(r)
    return out


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
    db: SQLServerDatabaseService = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=500, detail="Database service not initialized")

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
@router_main.get("/api/leave_data", tags=["leave-dashboard"])
async def leave_data(
    request: Request,
    kind: str = "metrics",
    as_of: Optional[str] = None,
    days: int = 7,
) -> Dict[str, Any]:
    from datetime import date, datetime, timedelta
    import json
    import logging

    logger = logging.getLogger(__name__)

    # Validate/normalize as_of
    if as_of:
        try:
            as_of_dt = datetime.strptime(as_of.replace("/", "-"), "%Y-%m-%d").date()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'as_of' format. Use YYYY-MM-DD.")
    else:
        as_of_dt = date.today()

    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=500, detail="Database service not initialized")

    # Look up data window
    min_date_str = None
    max_date_str = None
    try:
        win_rows, _ = db.run_select(
            """
            SELECT
              CONVERT(varchar(10), MIN(CAST(WORKDATE AS date)), 23) AS min_date,
              CONVERT(varchar(10), MAX(CAST(WORKDATE AS date)), 23) AS max_date
            FROM dbo.ATDLEAVEDATA
            """
        )
        if win_rows and len(win_rows[0]) >= 2:
            min_date_str, max_date_str = win_rows[0][0], win_rows[0][1]
    except Exception as e:
        logger.warning("Failed to fetch data window: %s", e)

    if not min_date_str or not max_date_str:
        return {
            "success": True,
            "data_window": {"min_date": min_date_str, "max_date": max_date_str},
            "metrics": {} if kind.lower() == "metrics" else [],
        }

    # Clamp as_of to dataset window
    min_dt = datetime.strptime(min_date_str, "%Y-%m-%d").date()
    max_dt = datetime.strptime(max_date_str, "%Y-%m-%d").date()
    if as_of_dt < min_dt:
        as_of_dt = min_dt
    if as_of_dt > max_dt:
        as_of_dt = max_dt
    as_of_str = as_of_dt.strftime("%Y-%m-%d")

    # For trend: compute start and effective days
    effective_start_str = None
    effective_days = days
    if kind.lower() == "trend":
        try:
            days = max(1, min(int(days), 365))
        except Exception:
            days = 7

        start_dt = as_of_dt - timedelta(days=max(0, days - 1))
        if start_dt < min_dt:
            start_dt = min_dt
        if as_of_dt > max_dt:
            as_of_dt = max_dt

        effective_days = max(1, (as_of_dt - start_dt).days + 1)
        as_of_str = as_of_dt.strftime("%Y-%m-%d")
        effective_start_str = start_dt.strftime("%Y-%m-%d")

    # Query SQL
    try:
        from app.home_page_metrics.leave_metrics import _sql_leave_metrics, _sql_leave_trend

        if kind.lower() == "trend":
            sql = _sql_leave_trend(as_of_str, effective_days)
        else:
            sql = _sql_leave_metrics(as_of_str)

        rows, columns = db.run_select(sql, language_hint="zh-tw", enable_query_hints=True)

        extra_ctx = {
            "success": True,
            "data_window": {"min_date": min_date_str, "max_date": max_date_str},
            "effective_as_of": as_of_str,
        }
        if kind.lower() == "trend" and effective_start_str:
            extra_ctx["effective_range"] = {"start": effective_start_str, "end": as_of_str}

        if not rows:
            if kind.lower() == "trend":
                return {"trend": [], **extra_ctx}
            return {"metrics": {}, **extra_ctx}

        row = dict(zip(columns, rows[0]))

        # ---------- metrics branch ----------
        if kind.lower() == "metrics" and "metrics" in row and isinstance(row["metrics"], str):
            payload = json.loads(row["metrics"])

            # Enrich with PersonResolver (supports PERSONID and EMPLOYEEID)
            from app.services.person_resolver import PersonResolver

            resolver = PersonResolver(db_service=db)
            details = _decode_json_field(payload.get("on_leave_details"))
            upcoming = _decode_json_field(payload.get("upcoming_leave"))

            pid_list, eid_list = _collect_ids_from_rows(details, upcoming)
            resolved = resolver.resolve_many(pid_list, employee_ids=eid_list)

            payload["on_leave_details"] = _apply_resolved(details, resolved)
            payload["upcoming_leave"] = _apply_resolved(upcoming, resolved)

            # (optional) If you have type label normalization, keep it:
            try:
                from app.home_page_metrics.leave_metrics import _apply_type_labels_to_metrics
                payload = _apply_type_labels_to_metrics(payload)
            except Exception:
                pass

            return {"metrics": payload, **extra_ctx}

        # ---------- trend branch ----------
        if kind.lower() == "trend" and "trend" in row and isinstance(row["trend"], str):
            trend_list = json.loads(row["trend"])

            from app.services.person_resolver import PersonResolver
            resolver = PersonResolver(db_service=db)

            all_people_arrays = []
            for day in trend_list:
                ppl = _decode_json_field(day.get("people_on_leave"))
                all_people_arrays.append(ppl)

            pid_list, eid_list = _collect_ids_from_rows(*all_people_arrays)
            resolved = resolver.resolve_many(pid_list, employee_ids=eid_list)

            for day in trend_list:
                ppl = _decode_json_field(day.get("people_on_leave"))
                day["people_on_leave"] = _apply_resolved(ppl, resolved)

            return {"trend": trend_list, **extra_ctx}

        # Fallback – return raw row (already includes extra_ctx)
        return {**row, **extra_ctx}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("/api/leave_data failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"leave_data query failed: {str(e)}")



# # =========================
# # Assistant - OLD / Vector Admin / Health (tags: assistant, vector, health) 
# # =========================
# @router_main.post("/api/assistant/query", tags=["assistant"])
# async def assistant_query(payload: dict, request: Request):
#     rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
#     q = (payload or {}).get("query") or ""

#     t0 = time.perf_counter()
#     logger.info("rid=%s /assistant/query start len=%d", rid, len(q))

#     nlp: LanguageNativeNLPService = getattr(request.app.state, "nlp", None)
#     if nlp is None:
#         logger.error("rid=%s /assistant/query error: No NLP service available", rid)
#         return JSONResponse({"success": False, "error": "NLP service not available"}, status_code=200)

#     try:
#         data = await run_in_threadpool(nlp.process_complete_query, q, "dbo", rid)
#         if isinstance(data, dict):
#             data["service_info"] = {"service_type": "language_native", "language_native_processing": True}
#         return JSONResponse({"success": True, **(data or {})})
#     except Exception as e:
#         logger.exception("rid=%s /assistant/query error: %s: %s", rid, type(e).__name__, e)
#         return JSONResponse(
#             {
#                 "success": False,
#                 "error": str(e),
#                 "service_info": {"service_type": "language_native", "error_in_service": "language_native"},
#             },
#             status_code=200,
#         )
#     finally:
#         ms = int((time.perf_counter() - t0) * 1000)
#         logger.info("rid=%s /assistant/query done ms=%d", rid, ms)



# =========================
# Assistant - New
# =========================



router_main = APIRouter()

def get_enhanced_nlp_service(request: Request):
    """
    Get the enhanced NLP service with visualization capabilities.
    Lazily creates and caches it on app.state.nlp_enhanced if needed.
    """
    if hasattr(request.app.state, "nlp_enhanced"):
        return request.app.state.nlp_enhanced

    db_service = getattr(request.app.state, "db_service", None)
    if db_service is not None:
        service = create_enhanced_nlp_service(db_service)
        request.app.state.nlp_enhanced = service
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
    """
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
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

    try:
        result = await run_in_threadpool(
            nlp.process_complete_query,
            q,          # user_input
            schema,     # schema_name
            rid,        # rid
            False,      # include_visualization (OFF)
            None,       # force_chart_type
            lang,       # lang override
        )
        if not isinstance(result, dict):
            result = {}
        result.setdefault("success", True)
        result.setdefault("rid", rid)
        # ensure visualization is absent here
        result["visualization"] = None
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
    """
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    q = (payload or {}).get("query", "").strip()
    schema = (payload or {}).get("schema", "dbo")
    include_viz = bool((payload or {}).get("include_visualization", True))
    force_chart_type = (payload or {}).get("chart_type") or None  # e.g., "bar_chart"
    lang = (payload or {}).get("lang")

    if not q:
        return _err(rid, "Query is required", status=400)

    t0 = time.perf_counter()
    request.app.state.last_request_id = rid
    nlp: LanguageNativeNLPService = get_enhanced_nlp_service(request)
    if nlp is None:
        return _err(rid, "NLP service not available", status=503)

    result: Dict[str, Any] = {}
    try:
        result = await run_in_threadpool(
            nlp.process_complete_query,
            q, schema, rid,
            include_viz,       # include_visualization (ON when button is pressed)
            force_chart_type,  # force_chart_type
            lang               # lang override
        )

        if not isinstance(result, dict):
            result = {}
        result.setdefault("success", True)
        result.setdefault("rid", rid)

        # Normalize viz meta for the UI
        viz = result.get("visualization") or {}
        viz_enabled = bool(viz.get("enabled"))
        result["visualization_generated"] = viz_enabled
        if viz_enabled:
            result["visualization_url"] = viz.get("url") or viz.get("image_url")
            result["visualization_type"] = viz.get("type")
            result["visualization_title"] = viz.get("title")
            result["visualization_insights"] = viz.get("insights") or []
            result["visualization_reason"] = viz.get("reasoning") or ""
        else:
            result["visualization_reason"] = viz.get("reason", "Not generated")

        return _ok(result)
    except Exception as e:
        return _err(rid, str(e), status=500, extra={"visualization_generated": False})
    finally:
        ms = int((time.perf_counter() - t0) * 1000)
        request.app.state.last_request_ms = ms


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
            out["vector_bootstrap_after_warm"] = await vb.start()
        except Exception as e:
            out["vector_bootstrap_after_warm_error"] = f"{type(e).__name__}: {e}"

    # Vector service status
    if not no_vector:
        try:
            t = time.perf_counter()
            nlp: LanguageNativeNLPService = getattr(request.app.state, "nlp", None)
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

    nlp: LanguageNativeNLPService = getattr(request.app.state, "nlp", None)
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
    nlp: LanguageNativeNLPService = getattr(request.app.state, "nlp", None)
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
