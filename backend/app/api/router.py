# backend/app/api/router.py
import os
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.services.db_service import SQLServerDatabaseService
from app.services.nlp_service import NLPService
from app.services.leave_vector import build_leave_index
from app.home_page_metrics.leave_metrics import _sql_leave_metrics, _sql_leave_trend

# ⬇️ Import report logic & models from the new service module
from app.reports.service import (
    ReportAnalysisRequest,
    ReportGenerationRequest,
    analyze_report,          # async
    generate_report,         # async
    download_report_response # sync
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Lifespan: initialize services ONCE and attach to app.state
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        db = SQLServerDatabaseService()
        nlp = NLPService(
            db_service=db,
            model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
            temperature=0.1,
        )
        app.state.db = db
        app.state.nlp = nlp
        logger.info("App services initialized: db, nlp")
    except Exception as e:
        logger.exception(f"Service init failed: {type(e).__name__}: {e}")

    try:
        yield
    finally:
        logger.info("App services shutting down")

# FastAPI app + router
app = FastAPI(lifespan=lifespan)
router = APIRouter()

# ------------------------------------------------------------------
# Debug: Leave vector endpoints
# ------------------------------------------------------------------
_db = build_leave_index()

@router.get("/debug/leave/health")
def leave_health():
    report = _db.relationships_sanity_check()
    return {"health": _db.health_check(), "sanity": report}

@router.get("/debug/leave/join-hints")
def leave_join_hints(tables: List[str] = Query(..., alias="tables")):
    return {"tables": tables, "join_hints": _db.join_hints(tables)}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _frontend_paths():
    base_dir = Path(__file__).resolve().parents[2]     # backend/
    project_root = base_dir.parent                     # project root
    frontend_dir = project_root / "frontend"
    index_file = frontend_dir / "index.html"
    return base_dir, project_root, frontend_dir, index_file

# ------------------------------------------------------------------
# Static / SPA
# ------------------------------------------------------------------
@router.get("/", include_in_schema=False)
async def serve_index():
    base_dir, project_root, frontend_dir, index_file = _frontend_paths()
    logger.info(f"Router location: {Path(__file__).resolve()}")
    logger.info(f"Base dir (backend): {base_dir}")
    logger.info(f"Project root:       {project_root}")
    logger.info(f"Frontend dir:       {frontend_dir}")
    logger.info(f"Looking for index:  {index_file} (exists={index_file.exists()})")

    if index_file.exists():
        logger.info("Serving index.html")
        return FileResponse(str(index_file))
    logger.warning("index.html not found; redirecting to /docs")
    return RedirectResponse("/docs")

@router.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    base_dir = Path(__file__).resolve().parents[2]
    frontend_dir = base_dir / "frontend"
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse("/docs")

# Simple ping
@router.get("/api/ping", include_in_schema=False)
async def ping():
    return {"ok": True}

# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------
@router.get("/api/health")
async def health(
    request: Request,
    no_db: bool = Query(False),
    no_index: bool = Query(False),   # kept for compatibility
    no_vector: bool = Query(False),
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out: Dict[str, Any] = {}

    # DB
    if not no_db:
        try:
            t = time.perf_counter()
            db: SQLServerDatabaseService = getattr(request.app.state, "db", None)  # type: ignore
            if db is None:
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

    # Vector
    if not no_vector:
        logger.info("health: entering vector check")
        try:
            t = time.perf_counter()
            nlp: NLPService = getattr(request.app.state, "nlp", None) # type: ignore
            vec = nlp.vector_status() if nlp else {"ready": False, "error": "NLP missing"}
            if not isinstance(vec, dict):
                vec = {"info": str(vec)}
            out["vector_db"] = vec
            out["vector_ms"] = int((time.perf_counter() - t) * 1000)
            logger.info(
                "health: completed vector check ok=%s dur=%dms",
                bool(vec.get("ready")), out["vector_ms"]
            )
        except BaseException as e:
            out["vector_db"] = {"ready": False, "error": f"{type(e).__name__}: {e}"}
            logger.exception("health: vector_status raised %s", type(e).__name__)
    else:
        out["vector_db"] = {"skipped": True}

    # Backward-compat index flag (not computed here)
    db_ok = out.get("database_connection") is True if "database_connection" in out else True
    idx_ok = out.get("index_present") is True if "index_present" in out else True
    vec_ok = bool(out.get("vector_db", {}).get("ready", True))
    out["ready_for_queries"] = db_ok and (idx_ok or vec_ok)
    out["total_ms"] = int((time.perf_counter() - t0) * 1000)
    logger.info("health: ready=%s total=%dms", out["ready_for_queries"], out["total_ms"])
    return out

# ------------------------------------------------------------------
# Assistant Query (LLM)
# ------------------------------------------------------------------
@router.post("/api/assistant/query")
async def assistant_query(payload: dict, request: Request):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    q = (payload or {}).get("query") or ""
    lang = (payload or {}).get("lang") or "en-US"

    t0 = time.perf_counter()
    logger.info(f"rid={rid} /assistant/query start qlen={len(q)} lang={lang}")

    nlp: NLPService = getattr(request.app.state, "nlp", None) # type: ignore
    if nlp is None:
        logger.error("rid=%s /assistant/query error: NLP service not initialized on app.state", rid)
        return JSONResponse({"success": False, "error": "NLP service not initialized"}, status_code=200)

    try:
        data = await run_in_threadpool(nlp.process_complete_query, q, "dbo", rid)
        return JSONResponse({"success": True, **(data or {})})
    except Exception as e:
        logger.exception("rid=%s /assistant/query error: %s: %s", rid, type(e).__name__, e)
        return JSONResponse({"success": False, "error": str(e)}, status_code=200)
    finally:
        ms = int((time.perf_counter() - t0) * 1000)
        logger.info(f"rid={rid} /assistant/query done ms={ms}")

# ------------------------------------------------------------------
# Dashboard Data (uses DB on app.state)
# ------------------------------------------------------------------
@router.get("/api/leave_data")
async def leave_data(
    request: Request,
    kind: str = "metrics",
    as_of: Optional[str] = None,
    days: int = 7,
) -> Dict[str, Any]:
    from datetime import datetime, date, timedelta

    # Validate/normalize as_of
    if as_of:
        try:
            as_of_dt = datetime.strptime(as_of.replace("/", "-"), "%Y-%m-%d").date()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'as_of' format. Use YYYY-MM-DD.")
    else:
        as_of_dt = date.today()

    as_of_str = as_of_dt.strftime("%Y-%m-%d")

    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=500, detail="Database service not initialized")

    # --- Look up data window (min/max WORKDATE) ---
    min_date_str = None
    max_date_str = None
    try:
        win_rows, win_cols = db.run_select(
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

    # Clamp as_of to known window
    effective_as_of_str = as_of_str
    if max_date_str and effective_as_of_str > max_date_str:
        effective_as_of_str = max_date_str
    if min_date_str and effective_as_of_str < min_date_str:
        effective_as_of_str = min_date_str

    # Trend window
    effective_days = days
    effective_start_str = None
    if kind.lower() == "trend":
        try:
            end_dt = datetime.strptime(effective_as_of_str, "%Y-%m-%d").date()
            start_dt = end_dt - timedelta(days=max(0, days - 1))
            if min_date_str:
                min_dt = datetime.strptime(min_date_str, "%Y-%m-%d").date()
                if start_dt < min_dt:
                    start_dt = min_dt
            if max_date_str:
                max_dt = datetime.strptime(max_date_str, "%Y-%m-%d").date()
                if end_dt > max_dt:
                    end_dt = max_dt
            effective_days = max(1, (end_dt - start_dt).days + 1)
            effective_as_of_str = end_dt.strftime("%Y-%m-%d")
            effective_start_str = start_dt.strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning("Failed to clamp trend window: %s", e)

    try:
        # Build SQL with clamped dates
        if kind.lower() == "trend":
            sql = _sql_leave_trend(effective_as_of_str, effective_days)
        else:
            sql = _sql_leave_metrics(effective_as_of_str)

        rows, columns = db.run_select(sql)
        if not rows:
            base = {
                "success": True,
                "data_window": {"min_date": min_date_str, "max_date": max_date_str},
                "effective_as_of": effective_as_of_str,
            }
            if kind.lower() == "trend":
                base["trend"] = []
                if effective_start_str:
                    base["effective_range"] = {"start": effective_start_str, "end": effective_as_of_str}
            else:
                base["metrics"] = {}
            return base

        row = dict(zip(columns, rows[0]))
        extra_ctx = {
            "data_window": {"min_date": min_date_str, "max_date": max_date_str},
            "effective_as_of": effective_as_of_str,
        }
        if kind.lower() == "trend" and effective_start_str:
            extra_ctx["effective_range"] = {"start": effective_start_str, "end": effective_as_of_str}

        if "metrics" in row and isinstance(row["metrics"], str):
            payload = json.loads(row["metrics"])
            return {"success": True, "metrics": payload, **extra_ctx}

        if "trend" in row and isinstance(row["trend"], str):
            payload = json.loads(row["trend"])
            return {"success": True, "trend": payload, **extra_ctx}

        return {"success": True, **row, **extra_ctx}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/leave_data failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"leave_data query failed: {str(e)}")

# ------------------------------------------------------------------
# Reports API — thin endpoints calling report service
# ------------------------------------------------------------------
@router.post("/api/reports/analyze")
async def reports_analyze(payload: ReportAnalysisRequest, request: Request):
    return await analyze_report(payload, request)

@router.post("/api/reports/generate")
async def reports_generate(payload: ReportGenerationRequest, request: Request):
    return await generate_report(payload, request)

@router.get("/api/reports/download/{report_id}")
async def reports_download(report_id: str, request: Request):
    # Returns a FileResponse or raises HTTPException
    return download_report_response(report_id, request)

# Attach router to app
app.include_router(router)
