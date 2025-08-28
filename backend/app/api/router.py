# backend/app/api/router.py
import os
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.services.db_service import SQLServerDatabaseService
from app.services.nlp_service import NLPService

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
        # keep the app running; endpoints will return structured errors
        logger.exception(f"Service init failed: {type(e).__name__}: {e}")

    try:
        yield
    finally:
        logger.info("App services shutting down")

# The single FastAPI app
app = FastAPI(lifespan=lifespan)

# One router for all routes
router = APIRouter()


# If your SQL helpers live elsewhere, import them here:
# from app.api.commands import _sql_leave_metrics, _sql_leave_trend
# Otherwise keep their definitions below this file.

# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------
# backend/app/api/router.py
import os, json, time, uuid, logging
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.services.db_service import SQLServerDatabaseService

from app.services.nlp_service import NLPService

logger = logging.getLogger(__name__)
router = APIRouter()

# ------- Helpers -------
def _frontend_paths():
    # backend/app/api/router.py -> parents[2] = backend/
    base_dir = Path(__file__).resolve().parents[2]
    # project root = backend.parent
    project_root = base_dir.parent
    frontend_dir = project_root / "frontend"
    index_file = frontend_dir / "index.html"
    return base_dir, project_root, frontend_dir, index_file

# ------- Routes -------
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

@router.get("/api/health")
async def health(request: Request,
                 no_db: bool = Query(False),
                 no_index: bool = Query(False),
                 no_vector: bool = Query(False)) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out: Dict[str, Any] = {}

    # DB
    if not no_db:
        try:
            t = time.perf_counter()
            db: SQLServerDatabaseService = getattr(request.app.state, "db", None)
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
            nlp: NLPService = getattr(request.app.state, "nlp", None)
            vec = nlp.vector_status() if nlp else {"ready": False, "error": "NLP missing"}
            if not isinstance(vec, dict):
                vec = {"info": str(vec)}
            out["vector_db"] = vec
            out["vector_ms"] = int((time.perf_counter() - t) * 1000)
            logger.info("health: completed vector check ok=%s dur=%dms",
                        bool(vec.get("ready")), out["vector_ms"])
        except BaseException as e:
            out["vector_db"] = {"ready": False, "error": f"{type(e).__name__}: {e}"}
            logger.exception("health: vector_status raised %s", type(e).__name__)
    else:
        out["vector_db"] = {"skipped": True}

    db_ok = out.get("database_connection") is True if "database_connection" in out else True
    idx_ok = out.get("index_present") is True if "index_present" in out else True
    vec_ok = bool(out.get("vector_db", {}).get("ready", True))
    out["ready_for_queries"] = db_ok and (idx_ok or vec_ok)
    out["total_ms"] = int((time.perf_counter() - t0) * 1000)
    logger.info("health: ready=%s total=%dms", out["ready_for_queries"], out["total_ms"])
    return out

@router.post("/api/assistant/query")
async def assistant_query(payload: dict, request: Request):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    q = (payload or {}).get("query") or ""
    lang = (payload or {}).get("lang") or "en-US"

    t0 = time.perf_counter()
    logger.info(f"rid={rid} /assistant/query start qlen={len(q)} lang={lang}")

    nlp: NLPService = getattr(request.app.state, "nlp", None)
    if nlp is None:
        logger.error(f"rid={rid} /assistant/query error: NLP service not initialized on app.state")
        return JSONResponse({"success": False, "error": "NLP service not initialized"}, status_code=200)

    try:
        data = await run_in_threadpool(nlp.process_complete_query, q, "dbo", rid)
        return JSONResponse({"success": True, **(data or {})})
    except Exception as e:
        logger.exception(f"rid={rid} /assistant/query error: {type(e).__name__}: {e}")
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
    days: int = 7
) -> Dict[str, Any]:
    from datetime import datetime, date

    # Validate/normalize as_of
    if as_of:
        try:
            as_of_dt = datetime.strptime(as_of.replace('/', '-'), "%Y-%m-%d").date()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'as_of' format. Use YYYY-MM-DD.")
    else:
        as_of_dt = date.today()

    as_of_str = as_of_dt.strftime("%Y-%m-%d")

    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=500, detail="Database service not initialized")

    try:
        if kind.lower() == "trend":
            sql = _sql_leave_trend(as_of_str, days)   # keep your existing helper
        else:
            sql = _sql_leave_metrics(as_of_str)       # keep your existing helper

        rows, columns = db.run_select(sql)
        if not rows:
            return {"success": False, "error": "No data returned from database."}

        # Expect single-row JSON projection
        row = dict(zip(columns, rows[0]))
        if "metrics" in row and isinstance(row["metrics"], str):
            payload = json.loads(row["metrics"])
            return {"success": True, **({"metrics": payload} if kind != "trend" else payload)}
        if "trend" in row and isinstance(row["trend"], str):
            payload = json.loads(row["trend"])
            return {"success": True, "trend": payload}

        # Fallback (already materialized)
        return {"success": True, **row}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/leave_data failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"leave_data query failed: {str(e)}")

# ------------------------------------------------------------------
# Static / SPA
# ------------------------------------------------------------------
@router.get("/", include_in_schema=False)
async def serve_index():
    base_dir = Path(__file__).resolve().parents[2]   # backend/
    frontend_dir = base_dir / "frontend"
    index_file = frontend_dir / "index.html"

    logger.info(f"Router file location: {Path(__file__).resolve()}")
    logger.info(f"Base dir: {base_dir}")
    logger.info(f"Frontend dir: {frontend_dir}")
    logger.info(f"Looking for index.html at: {index_file}")
    logger.info(f"File exists: {index_file.exists()}")

    if index_file.exists():
        logger.info("Serving index.html")
        return FileResponse(str(index_file))
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

# Attach router to app
app.include_router(router)

"""
DO NOT MODIFY - Chiuzu 08/27/2025
"""
# ---------- Dashboard Data SQL builders (metrics / trend) ----------

def _sql_leave_metrics(as_of: str) -> str:
    return f"""
WITH params AS (
  SELECT CAST('{as_of}' AS DATE) AS asOf
),
wk AS (
  SELECT
    asOf,
    ((DATEPART(weekday, asOf) + 5) % 7) AS w,
    DATEADD(day, -((DATEPART(weekday, asOf)+5)%7), asOf) AS weekStart,
    DATEADD(day,  6-((DATEPART(weekday, asOf)+5)%7), asOf) AS weekEnd
  FROM params
),
leave_src AS (
  SELECT
    PERSONID,
    DEPARTMENTID,
    ATTENDANCETYPE,
    COALESCE(
      COALESCE(TRY_CONVERT(date, STARTDATE, 112), TRY_CONVERT(date, STARTDATE, 23), TRY_CONVERT(date, STARTDATE)),
      COALESCE(TRY_CONVERT(date, WORKDATE, 112),  TRY_CONVERT(date, WORKDATE, 23),  TRY_CONVERT(date, WORKDATE))
    ) AS SDATE,
    COALESCE(
      COALESCE(TRY_CONVERT(date, ENDDATE, 112), TRY_CONVERT(date, ENDDATE, 23), TRY_CONVERT(date, ENDDATE)),
      COALESCE(TRY_CONVERT(date, WORKDATE, 112), TRY_CONVERT(date, WORKDATE, 23), TRY_CONVERT(date, WORKDATE))
    ) AS EDATE,
    VALIDATED
  FROM dbo.ATDLEAVEDATA
),
on_leave_day AS (
  SELECT l.PERSONID, l.DEPARTMENTID, l.ATTENDANCETYPE AS [type], l.EDATE
  FROM leave_src l
  CROSS JOIN params p
  WHERE l.SDATE <= p.asOf AND l.EDATE >= p.asOf
),
pending_reqs AS (
  SELECT COUNT(*) AS cnt
  FROM leave_src
  WHERE (VALIDATED IS NULL OR VALIDATED = 0)
),
upcoming_next7 AS (
  SELECT
    l.PERSONID AS person_id,
    l.ATTENDANCETYPE AS [type],
    l.SDATE AS start_date,
    l.EDATE AS end_date
  FROM leave_src l
  CROSS JOIN params p
  WHERE l.SDATE BETWEEN DATEADD(day, 1, p.asOf) AND DATEADD(day, 7, p.asOf)
),
dept_summary AS (
  SELECT DEPARTMENTID AS department_id, COUNT(*) AS [count]
  FROM on_leave_day
  GROUP BY DEPARTMENTID
),
overtime_week AS (
  SELECT
    SUM(CAST(HOURS AS DECIMAL(10,2))) AS total_hours,
    COUNT(DISTINCT PERSONID)          AS people
  FROM dbo.ATDHISOVERTIME
  CROSS JOIN wk
  WHERE COALESCE(TRY_CONVERT(date, OVERTIMEDATE, 112), TRY_CONVERT(date, OVERTIMEDATE, 23), TRY_CONVERT(date, OVERTIMEDATE))
        BETWEEN wk.weekStart AND wk.weekEnd
),
low_balance AS (
  SELECT COUNT(*) AS low_cnt
  FROM (
    SELECT PERSONID, MIN(REMAINDAYS) AS rem
    FROM (
      SELECT PERSONID, REMAINDAYS FROM dbo.ATDNONCALCULATEDVACATION
      UNION ALL
      SELECT PERSONID, REMAINDAYS FROM dbo.ATDHISNONCALCULATEDVACATION
    ) X
    GROUP BY PERSONID
  ) Y
  WHERE TRY_CAST(rem AS DECIMAL(10,2)) < 5
)
SELECT
  1 AS success,
  (
    SELECT
      (SELECT COUNT(*) FROM on_leave_day)               AS employees_on_leave_today,
      (SELECT cnt FROM pending_reqs)                    AS pending_leave_requests,
      (SELECT low_cnt FROM low_balance)                 AS low_balance_count,
      (SELECT ISNULL(total_hours,0) FROM overtime_week) AS overtime_hours,
      (SELECT ISNULL(people,0) FROM overtime_week)      AS overtime_people,
      (SELECT TOP (50)
         PERSONID AS person_id, [type], CONVERT(date, EDATE) AS end_date
       FROM on_leave_day
       ORDER BY PERSONID
       FOR JSON PATH)                                   AS on_leave_details,
      (SELECT
         person_id, CONVERT(date, start_date) AS start_date,
         CONVERT(date, end_date)   AS end_date, [type]
       FROM upcoming_next7
       ORDER BY start_date, person_id
       FOR JSON PATH)                                   AS upcoming_leave,
      (SELECT department_id, [count]
       FROM dept_summary
       ORDER BY [count] DESC
       FOR JSON PATH)                                   AS department_summary
    FOR JSON PATH, WITHOUT_ARRAY_WRAPPER
  ) AS metrics;
"""


def _sql_leave_trend(as_of: str, days: int) -> str:
    days = max(1, min(int(days or 7), 31))
    return f"""
WITH params AS (
  SELECT CAST('{as_of}' AS DATE) AS asOf
),
s(d) AS (
  SELECT DATEADD(day, -({days}-1), asOf) FROM params
  UNION ALL
  SELECT DATEADD(day, 1, d)
  FROM s CROSS JOIN params
  WHERE d < (SELECT asOf FROM params)
),
leave_src AS (
  SELECT
    COALESCE(
      COALESCE(TRY_CONVERT(date, STARTDATE, 112), TRY_CONVERT(date, STARTDATE, 23), TRY_CONVERT(date, STARTDATE)),
      COALESCE(TRY_CONVERT(date, WORKDATE, 112),  TRY_CONVERT(date, WORKDATE, 23),  TRY_CONVERT(date, WORKDATE))
    ) AS SDATE,
    COALESCE(
      COALESCE(TRY_CONVERT(date, ENDDATE, 112), TRY_CONVERT(date, ENDDATE, 23), TRY_CONVERT(date, ENDDATE)),
      COALESCE(TRY_CONVERT(date, WORKDATE, 112), TRY_CONVERT(date, WORKDATE, 23), TRY_CONVERT(date, WORKDATE))
    ) AS EDATE
  FROM dbo.ATDLEAVEDATA
),
counts AS (
  SELECT s.d AS [date],
         (SELECT COUNT(*) FROM leave_src l WHERE l.SDATE <= s.d AND l.EDATE >= s.d) AS [count]
  FROM s
)
SELECT 1 AS success,
       (SELECT CONVERT(date, [date]) AS [date], [count]
        FROM counts ORDER BY [date]
        FOR JSON PATH) AS trend
OPTION (MAXRECURSION 200);
"""