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
import io
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# =========================
# Third-Party
# =========================
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, HTMLResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# =========================
# Internal Services
# =========================
from app.services.db_service import SQLServerDatabaseService
from app.services.nlp_service import LeaveNLPPipeline

from app.home_page_metrics.leave_metrics import _sql_leave_metrics, _sql_leave_trend
from app.services.data_processing.data_analyzer import DataAnalyzer

from app.services.person_resolver import PersonResolver
from app.services.helpers.data_utils import _apply_resolved, _collect_ids_from_rows


# Shared analyzer
_type_analyzer = DataAnalyzer()

logger = logging.getLogger(__name__)

# Routers
router_main = APIRouter()                             # no prefix
router_leave = APIRouter(prefix="/api/leave", tags=["leave"])
static_router = APIRouter(tags=["static"])


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


def _get_db(request: Request) -> SQLServerDatabaseService:
    """
    Unified accessor for DB service; supports both legacy `app.state.db`
    and newer `app.state.db_service`.
    """
    db = getattr(request.app.state, "db", None) or getattr(request.app.state, "db_service", None)
    if not isinstance(db, SQLServerDatabaseService):
        raise HTTPException(status_code=500, detail="Database service not initialized")
    return db


def _ok(payload: Dict[str, Any], status: int = 200) -> JSONResponse:
    payload.setdefault("success", True)
    return JSONResponse(payload, status_code=status)


def _err(rid: str, msg: str, status: int = 500, extra: Optional[Dict[str, Any]] = None) -> JSONResponse:
    body: Dict[str, Any] = {"success": False, "error": msg, "rid": rid}
    if extra:
        body.update(extra)
    return JSONResponse(body, status_code=status)


def get_leave_nlp_pipeline(request: Request) -> LeaveNLPPipeline:
    """
    Lazily create and cache the LeaveNLPPipeline on app.state.leave_nlp.
    This is the central chat/LLM service for leave analytics.
    """
    svc = getattr(request.app.state, "leave_nlp", None)
    if isinstance(svc, LeaveNLPPipeline):
        return svc

    svc = LeaveNLPPipeline()
    request.app.state.leave_nlp = svc
    return svc


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
        # Fallback if driver doesn't support params
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
    _, _, frontend_dir, index_file = _frontend_paths()
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse("/docs")


@router_main.get("/api/ping", include_in_schema=False, tags=["static"])
async def ping():
    return {"ok": True}


# =========================
# Assistant (chat-only, no charts)
# =========================
@router_main.post("/api/assistant/query", tags=["assistant"])
async def assistant_query(payload: dict, request: Request):
    """
    Leave AI Assistant – chat-style query.

    Expects JSON body:
    {
      "query": "...",              # required natural language question (zh-tw or en)
      "schema": "...",             # optional schema description text (for LLM)
      "join_hints": "...",         # optional join hints text
      "lang": "zh-tw" | "en"       # optional language override (normally auto-detected)
    }

    Returns:
      {
        "question": "...",
        "language_detected": "zh-tw" | "en",
        "sql": "...",
        "rows": [...],
        "columns": [...],
        "attempts": n,
        "aggregates": {...},
        "explanation_zh": "### ...",
        "intent_context": {...},
        "success": true,
        "rid": "..."
      }
    """
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    q = (payload or {}).get("query", "").strip()
    schema_text = (payload or {}).get("schema", "") or ""  # you can send full schema from frontend
    join_hints = (payload or {}).get("join_hints", "") or ""
    _lang_override = (payload or {}).get("lang")  # currently unused; LLM auto-detects

    if not q:
        return _err(rid, "Query is required", status=400)

    t0 = time.perf_counter()
    request.app.state.last_request_id = rid

    db_service = _get_db(request)
    nlp = get_leave_nlp_pipeline(request)

    try:
        result = await run_in_threadpool(
            nlp.answer_question,
            db_service,
            q,
            schema_text,
            join_hints,
        )
        if not isinstance(result, dict):
            result = {}

        result.setdefault("success", True)
        result.setdefault("rid", rid)
        return _ok(result)
    except Exception as e:
        logger.exception("assistant_query failed")
        return _err(rid, str(e), status=500)
    finally:
        ms = int((time.perf_counter() - t0) * 1000)
        request.app.state.last_request_ms = ms


# =========================
# Health (tags: health)
# =========================
@router_main.get("/api/health", tags=["health"])
async def health(
    request: Request,
    no_db: bool = Query(False),
    warm: bool = Query(False),  # kept for compatibility, but only affects NLP warmup now
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out: Dict[str, Any] = {}

    # DB check
    if not no_db:
        try:
            t = time.perf_counter()
            db = _get_db(request)
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

    # LLM / NLP service status
    try:
        t = time.perf_counter()
        nlp = get_leave_nlp_pipeline(request)
        llm_stats = nlp.llm.get_service_stats() if getattr(nlp, "llm", None) else {}
        out["llm"] = {
            "service_enabled": llm_stats.get("service_enabled", False),
            "model_name": llm_stats.get("model_name", None),
            "success_rate_percent": llm_stats.get("success_rate_percent", None),
        }
        out["nlp_ms"] = int((time.perf_counter() - t) * 1000)

        # Optional "warm" behaviour: you could run a tiny no-op query here if desired.
        if warm and getattr(nlp, "llm", None) and hasattr(nlp.llm, "reset_stats"):
            nlp.llm.reset_stats()
            out["llm_warm_reset"] = True
    except BaseException as e:
        out["llm"] = {"service_enabled": False, "error": f"{type(e).__name__}: {e}"}
        logger.exception("health: llm status raised %s", type(e).__name__)

    db_ok = out.get("database_connection") is True if "database_connection" in out else True
    llm_ok = out.get("llm", {}).get("service_enabled", False)
    out["ready_for_queries"] = bool(db_ok and llm_ok)
    out["total_ms"] = int((time.perf_counter() - t0) * 1000)
    out["services"] = {"nlp_mode": "leave_pipeline", "leave_chat_enabled": llm_ok}

    return out


# =========================
# Debug (tags: debug)
# =========================
@router_main.get("/debug/leave/health", tags=["debug"])
def leave_health(request: Request):
    """
    Lightweight debug endpoint for the leave chat pipeline.
    Shows LLM stats and basic DB health.
    """
    result: Dict[str, Any] = {}

    # LLM stats
    try:
        nlp = get_leave_nlp_pipeline(request)
        if getattr(nlp, "llm", None):
            result["llm_stats"] = nlp.llm.get_service_stats()
        else:
            result["llm_stats"] = {"service_enabled": False, "error": "LLM service not initialized"}
    except Exception as e:
        result["llm_stats"] = {"service_enabled": False, "error": str(e)}

    # DB check
    try:
        db = _get_db(request)
        result["db_connection"] = bool(db.test_connection(login_timeout=2))
    except Exception as e:
        result["db_connection"] = False
        result["db_error"] = str(e)

    result["service_type"] = "leave_nlp_pipeline"
    return result


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

    # Validate/normalize as_of
    if as_of:
        try:
            as_of_dt = datetime.strptime(as_of.replace("/", "-"), "%Y-%m-%d").date()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'as_of' format. Use YYYY-MM-DD.")
    else:
        as_of_dt = date.today()

    db = _get_db(request)

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
            resolver = PersonResolver(db_service=db)
            details = _decode_json_field(payload.get("on_leave_details"))
            upcoming = _decode_json_field(payload.get("upcoming_leave"))

            pid_list, eid_list = _collect_ids_from_rows(details, upcoming)
            resolved = resolver.resolve_many(pid_list, employee_ids=eid_list)

            payload["on_leave_details"] = _apply_resolved(details, resolved)
            payload["upcoming_leave"] = _apply_resolved(upcoming, resolved)

            # optional: type label normalization
            try:
                from app.home_page_metrics.leave_metrics import _apply_type_labels_to_metrics as _labels_fn
                payload = _labels_fn(payload)
            except Exception:
                payload = _apply_type_labels_to_metrics(payload)

            return {"metrics": payload, **extra_ctx}

        # ---------- trend branch ----------
        if kind.lower() == "trend" and "trend" in row and isinstance(row["trend"], str):
            trend_list = json.loads(row["trend"])

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




# =========================
# Export combined router (IMPORTANT)
# =========================
router = APIRouter()
router.include_router(router_main)
router.include_router(router_leave)
router.include_router(static_router)
