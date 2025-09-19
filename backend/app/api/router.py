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
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from app.services.data_processing.data_analyzer import DataAnalyzer
from app.services.db_service import SQLServerDatabaseService

# Import the language-native NLP service (v2)
from app.services.nlp_service_2 import LanguageNativeNLPService

# Report service
from app.reports.service import (
    ReportAnalysisRequest,
    ReportGenerationRequest,
    analyze_report,
    generate_report,
    download_report_response
)

logger = logging.getLogger(__name__)

# Initialize once per process
_type_analyzer = DataAnalyzer()

# ------------------------------------------------------------------
# Lifespan: initialize language-native NLP service only
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        db = SQLServerDatabaseService()
        model_default = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        
        # Initialize language-native NLP service
        nlp_service = LanguageNativeNLPService(
            db_service=db,
            model_name=model_default,
            temperature=0.0,
        )
        
        app.state.nlp = nlp_service
        app.state.db = db
        
        logger.info("App services initialized: db, language-native nlp (model=%s)", model_default)
        
    except Exception as e:
        logger.exception("Service init failed: %s: %s", type(e).__name__, e)
        raise

    try:
        yield
    finally:
        logger.info("App services shutting down")

# FastAPI app + router
app = FastAPI(lifespan=lifespan)
router = APIRouter()

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _frontend_paths():
    base_dir = Path(__file__).resolve().parents[2]
    project_root = base_dir.parent
    frontend_dir = project_root / "frontend"
    index_file = frontend_dir / "index.html"
    return base_dir, project_root, frontend_dir, index_file

# ------------------------------------------------------------------
# Static / SPA
# ------------------------------------------------------------------
@router.get("/", include_in_schema=False)
async def serve_index():
    base_dir, project_root, frontend_dir, index_file = _frontend_paths()
    logger.info("Router=%s", Path(__file__).resolve())
    logger.info("Frontend dir=%s index exists=%s", frontend_dir, index_file.exists())

    if index_file.exists():
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

@router.get("/api/ping", include_in_schema=False)
async def ping():
    return {"ok": True}

@router.get("/leave_page.html", include_in_schema=False)
async def serve_leave_page():
    base_dir, project_root, frontend_dir, index_file = _frontend_paths()
    leave_page_file = frontend_dir / "leave_page.html"
    if leave_page_file.exists():
        return FileResponse(str(leave_page_file))
    return RedirectResponse("/docs")

@router.get("/translations.js", include_in_schema=False)
async def serve_translations():
    base_dir, project_root, frontend_dir, index_file = _frontend_paths()
    translations_file = frontend_dir / "translations.js"
    if translations_file.exists():
        return FileResponse(str(translations_file))
    return PlainTextResponse("// translations.js not found", media_type="application/javascript")

# ------------------------------------------------------------------
# Reports API 
# ------------------------------------------------------------------
@router.post("/api/reports/analyze")
async def reports_analyze(payload: ReportAnalysisRequest, request: Request):
    return await analyze_report(payload, request)

@router.post("/api/reports/generate")
async def reports_generate(payload: ReportGenerationRequest, request: Request):
    return await generate_report(payload, request)

@router.get("/api/reports/download/{report_id}")
async def reports_download(report_id: str, request: Request):
    return download_report_response(report_id, request)

@router.get("/generate_report.html", include_in_schema=False)
async def serve_generate_report():
    base_dir, project_root, frontend_dir, index_file = _frontend_paths()
    generate_report_file = frontend_dir / "generate_report.html"
    if generate_report_file.exists():
        return FileResponse(str(generate_report_file))
    return RedirectResponse("/docs")

# ------------------------------------------------------------------
# Leave API Routes
# ------------------------------------------------------------------
from app.leave.service import (
    LeaveRequest,
    LeaveBalanceRequest, 
    LeaveResponse,
    LeaveBalanceResponse,
    submit_leave_request,
    get_employee_leave_balance,
    validate_leave_request
)

@router.post("/api/leave/submit", response_model=LeaveResponse)
async def submit_leave(request: LeaveRequest):
    """Submit a leave request to HCM system"""
    try:
        # Validate the request first
        validation = await validate_leave_request(request)
        if not validation["valid"]:
            return LeaveResponse(
                success=False,
                message=f"Validation failed: {', '.join(validation['errors'])}"
            )
        
        # Submit the leave request
        result = await submit_leave_request(request)
        return result
        
    except Exception as e:
        logger.error(f"Error submitting leave request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit leave request: {str(e)}")

@router.post("/api/leave/balance", response_model=LeaveBalanceResponse)
async def get_leave_balance(request: LeaveBalanceRequest):
    """Get employee leave balance"""
    try:
        result = await get_employee_leave_balance(request)
        return result
        
    except Exception as e:
        logger.error(f"Error getting leave balance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get leave balance: {str(e)}")

@router.post("/api/leave/validate")
async def validate_leave(request: LeaveRequest):
    """Validate a leave request without submitting"""
    try:
        validation = await validate_leave_request(request)
        return {
            "success": True,
            "validation": validation
        }
        
    except Exception as e:
        logger.error(f"Error validating leave request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to validate leave request: {str(e)}")

@router.get("/api/leave/types")
async def get_leave_types():
    """Get available leave types"""
    return {
        "success": True,
        "leave_types": [
            {"id": "annual", "name": "Annual Leave", "hcm_id": "19"},
            {"id": "sick", "name": "Sick Leave", "hcm_id": "20"},
            {"id": "personal", "name": "Personal Leave", "hcm_id": "21"},
            {"id": "emergency", "name": "Emergency Leave", "hcm_id": "22"}
        ]
    }

# ------------------------------------------------------------------
# Leave Assistant API
# ------------------------------------------------------------------
@router.post("/api/leave/assistant/query")
async def leave_assistant_query(request: dict):
    """Process natural language leave requests through AI assistant"""
    try:
        query = request.get("query", "").strip()
        user_id = request.get("user_id", "sa")  # Default user for testing
        employee_id = request.get("employee_id", "A011")  # Default employee for testing
        
        if not query:
            return {
                "success": False,
                "message": "Query cannot be empty"
            }
        
        # Process the query and determine the response
        response = await process_leave_assistant_query(query, user_id, employee_id)
        return response
        
    except Exception as e:
        logger.error(f"Error processing assistant query: {str(e)}")
        return {
            "success": False,
            "message": f"Error processing request: {str(e)}"
        }

async def process_leave_assistant_query(query: str, user_id: str, employee_id: str) -> dict:
    """Process natural language queries for leave requests"""
    
    query_lower = query.lower()
    
    # Check balance requests
    if any(word in query_lower for word in ["balance", "remaining", "left", "available"]):
        try:
            balance_request = LeaveBalanceRequest(user_id=user_id, employee_id=employee_id)
            balance_response = await get_employee_leave_balance(balance_request)
            
            if balance_response.success:
                balance_text = "\n".join([
                    f"• {leave_type.title()} Leave: {days} days"
                    for leave_type, days in balance_response.balances.items()
                ])
                
                return {
                    "success": True,
                    "type": "info",
                    "message": f"Here are your current leave balances:\n\n{balance_text}\n\nWhat type of leave would you like to request?",
                    "data": balance_response.balances
                }
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
    
    # Parse leave requests
    leave_request_data = parse_leave_request_from_query(query, user_id, employee_id)
    
    if leave_request_data["needs_clarification"]:
        return {
            "success": True,
            "type": "clarification",
            "message": leave_request_data["clarification_message"],
            "suggested_form": leave_request_data.get("form_data")
        }
    elif leave_request_data["can_create_form"]:
        return {
            "success": True,
            "type": "form_ready",
            "message": leave_request_data["message"],
            "form_data": leave_request_data["form_data"]
        }
    else:
        return {
            "success": True,
            "type": "general",
            "message": "I can help you with your leave request. Could you provide more details about:\n\n• The dates you need off\n• Type of leave (annual, sick, personal, emergency)\n• Duration (full day, half day, multiple days)\n• Reason for leave (optional)"
        }

def parse_leave_request_from_query(query: str, user_id: str, employee_id: str) -> dict:
    """Parse natural language query into leave request parameters"""
    from datetime import datetime, timedelta
    import re
    
    query_lower = query.lower()
    result = {
        "needs_clarification": False,
        "can_create_form": False,
        "clarification_message": "",
        "message": "",
        "form_data": None
    }
    
    # Detect leave type
    leave_type = "annual"  # default
    if any(word in query_lower for word in ["sick", "illness", "medical", "doctor"]):
        leave_type = "sick"
    elif any(word in query_lower for word in ["personal", "family", "appointment"]):
        leave_type = "personal"
    elif any(word in query_lower for word in ["emergency", "urgent", "sudden"]):
        leave_type = "emergency"
    elif any(word in query_lower for word in ["vacation", "holiday", "annual", "pto"]):
        leave_type = "annual"
    
    # Detect dates
    start_date = None
    end_date = None
    duration_type = "full-day"
    
    # Tomorrow
    if "tomorrow" in query_lower:
        tomorrow = datetime.now() + timedelta(days=1)
        start_date = tomorrow.strftime("%Y-%m-%d")
        end_date = start_date
        duration_type = "full-day"
        
        result.update({
            "can_create_form": True,
            "message": f"I understand you need tomorrow ({start_date}) off. Let me prepare a leave request form:",
            "form_data": {
                "user_id": user_id,
                "employee_id": employee_id,
                "leave_type": leave_type,
                "start_date": start_date,
                "end_date": end_date,
                "duration_type": duration_type,
                "reason": ""
            }
        })
        return result
    
    # Today
    if "today" in query_lower:
        today = datetime.now()
        start_date = today.strftime("%Y-%m-%d")
        end_date = start_date
        duration_type = "full-day"
        
        result.update({
            "can_create_form": True,
            "message": f"I understand you need today ({start_date}) off. Let me prepare a leave request form:",
            "form_data": {
                "user_id": user_id,
                "employee_id": employee_id,
                "leave_type": leave_type,
                "start_date": start_date,
                "end_date": end_date,
                "duration_type": duration_type,
                "reason": "Same-day leave request" if leave_type != "sick" else "Sick leave"
            }
        })
        return result
    
    # Next week
    if "next week" in query_lower:
        result.update({
            "needs_clarification": True,
            "clarification_message": "I'd be happy to help you request time off next week! Could you please specify:\n\n• Which specific days (e.g., Monday-Wednesday)?\n• How many days total?\n• Is this for vacation/annual leave?\n• Any specific reason?"
        })
        return result
    
    # Half day
    if any(phrase in query_lower for phrase in ["half day", "half-day", "morning", "afternoon"]):
        if "morning" in query_lower or "am" in query_lower:
            duration_type = "half-day-am"
        elif "afternoon" in query_lower or "pm" in query_lower:
            duration_type = "half-day-pm"
        else:
            duration_type = "half-day-am"  # default to morning
    
    # Multiple days pattern
    days_match = re.search(r'(\d+)\s*days?', query_lower)
    if days_match:
        num_days = int(days_match.group(1))
        if num_days > 1:
            duration_type = "multiple-days"
            result.update({
                "needs_clarification": True,
                "clarification_message": f"I understand you need {num_days} days off. To prepare your request, please let me know:\n\n• Starting date (YYYY-MM-DD)\n• Is this consecutive days?\n• Type of leave: {leave_type.title()} Leave\n• Any specific reason?"
            })
            return result
    
    # If we can't determine specific dates, ask for clarification
    result.update({
        "needs_clarification": True,
        "clarification_message": f"I'd be happy to help you request {leave_type} leave! To prepare the best request, please provide:\n\n• Specific dates needed\n• Duration (full day, half day, multiple days)\n• Brief reason (optional)"
    })
    
    return result

# ------------------------------------------------------------------
# Vector Admin
# ------------------------------------------------------------------
@router.post("/api/vector/reload")
async def vector_reload(request: Request):
    vb = getattr(request.app.state, "vector_bootstrap", None)
    if vb is None:
        raise HTTPException(status_code=500, detail="Vector bootstrapper missing")
    try:
        result = await vb.start()  # idempotent: if already started/finished, returns status
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector reload failed: {e}")

# ------------------------------------------------------------------
# Health Check
# ------------------------------------------------------------------
@router.get("/api/health")
async def health(
    request: Request,
    no_db: bool = Query(False),
    no_vector: bool = Query(False),
    warm: bool = Query(False),
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out: Dict[str, Any] = {}

    # Database check
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

    # Optional: trigger warmup from health if warm=1
    if warm and vb:
        try:
            out["vector_bootstrap_after_warm"] = await vb.start()
        except Exception as e:
            out["vector_bootstrap_after_warm_error"] = f"{type(e).__name__}: {e}"

    # Vector service status
    if not no_vector:
        try:
            t = time.perf_counter()
            nlp = getattr(request.app.state, "nlp", None)
            
            if nlp:
                vector_status = nlp.vector_status()
                out["vector_db"] = {
                    "ready": vector_status.get("ready", False),
                    "service_type": "language_native",
                    **vector_status
                }
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

    # Ready flag
    db_ok = out.get("database_connection") is True if "database_connection" in out else True
    vec_ok = bool(out.get("vector_db", {}).get("ready", True))
    out["ready_for_queries"] = db_ok and vec_ok
    out["total_ms"] = int((time.perf_counter() - t0) * 1000)
    out["services"] = {
        "nlp_mode": "language_native",
        "language_native_processing": True
    }

    return out

# ------------------------------------------------------------------
# Assistant Query (Simplified - Single Service)
# ------------------------------------------------------------------
@router.post("/api/assistant/query")
async def assistant_query(payload: dict, request: Request):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    q = (payload or {}).get("query") or ""
    
    t0 = time.perf_counter()
    logger.info("rid=%s /assistant/query start len=%d", rid, len(q))

    nlp = getattr(request.app.state, "nlp", None)
    if nlp is None:
        logger.error("rid=%s /assistant/query error: No NLP service available", rid)
        return JSONResponse({"success": False, "error": "NLP service not available"}, status_code=200)

    try:
        data = await run_in_threadpool(nlp.process_complete_query, q, "dbo", rid)
        
        # Add service metadata to response
        if isinstance(data, dict):
            data["service_info"] = {
                "service_type": "language_native",
                "language_native_processing": True
            }
        
        return JSONResponse({"success": True, **(data or {})})
    except Exception as e:
        logger.exception("rid=%s /assistant/query error: %s: %s", rid, type(e).__name__, e)
        return JSONResponse({
            "success": False, 
            "error": str(e),
            "service_info": {
                "service_type": "language_native",
                "error_in_service": "language_native"
            }
        }, status_code=200)
    finally:
        ms = int((time.perf_counter() - t0) * 1000)
        logger.info("rid=%s /assistant/query done ms=%d", rid, ms)

# ------------------------------------------------------------------
# Debug Endpoints
# ------------------------------------------------------------------
@router.get("/debug/leave/health")
def leave_health(request: Request):
    """Health check for leave system."""
    result = {}
    
    nlp = getattr(request.app.state, "nlp", None)
    if nlp and getattr(nlp, "vector_search", None):
        try:
            result["nlp_service"] = nlp.vector_status()
        except Exception as e:
            result["nlp_service"] = {"ready": False, "error": str(e)}
    else:
        result["nlp_service"] = {"ready": False, "error": "Service not available"}
    
    # Try to get sanity check
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

@router.get("/debug/leave/join-hints")
def leave_join_hints(request: Request, tables: List[str] = Query(..., alias="tables")):
    """Get join hints using NLP service."""
    nlp = getattr(request.app.state, "nlp", None)
    if not nlp or not getattr(nlp, "vector_search", None):
        raise HTTPException(status_code=500, detail="NLP/Vector service not initialized")
    try:
        hints = nlp.vector_search.get_join_hints(tables)
        return {
            "tables": tables, 
            "join_hints": hints,
            "service_type": "language_native"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"join-hints failed: {e}")

# ------------------------------------------------------------------
# Dashboard Data Helper Functions
# ------------------------------------------------------------------
from app.services.person_resolver import PersonResolver
import json

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
        # Keep existing employee_id/email if already set; otherwise fill
        if not r.get("employee_id"):
            r["employee_id"] = info.get("employee_id")
        if not r.get("email"):
            r["email"] = info.get("email")
        # New: cardnum
        r["cardnum"] = info.get("cardnum")
        out.append(r)
    return out

def _apply_type_labels_to_metrics(payload: dict) -> dict:
    """
    Adds 'type_label' next to 'type_code' for arrays in metrics payload.
    Silently no-ops if arrays/fields are missing.
    """
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

# ------------------------------------------------------------------
# Dashboard Data API
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

    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=500, detail="Database service not initialized")

    # Look up data window (min/max WORKDATE) from live data
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

    # If no data window, return empty (but successful) payload
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

    # For trend: compute start=end-(days-1), clamp to min/max
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

    # Build SQL using helpers
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

        if kind.lower() == "metrics" and "metrics" in row and isinstance(row["metrics"], str):
            payload = json.loads(row["metrics"])

            # Enrich with PersonResolver
            resolver = PersonResolver(db_service=db)
            # decode arrays the SQL produced (strings) into lists
            details = _decode_json_field(payload.get("on_leave_details"))
            upcoming = _decode_json_field(payload.get("upcoming_leave"))

            # collect unique PERSONIDs and resolve in batch
            pid_list = _collect_person_ids(details, upcoming)
            resolved = resolver.resolve_many(pid_list)  # {pid: {...}}

            # write back person_name/cardnum/etc
            payload["on_leave_details"] = _patch_rows(details, resolved)
            payload["upcoming_leave"]   = _patch_rows(upcoming, resolved)

            # add type labels last (now that arrays are dicts with type_code)
            payload = _apply_type_labels_to_metrics(payload)
            return {"metrics": payload, **extra_ctx}

        if kind.lower() == "trend" and "trend" in row and isinstance(row["trend"], str):
            trend_list = json.loads(row["trend"])

            # Enrich with PersonResolver
            resolver = PersonResolver(db_service=db)

            # Gather all person_ids across all days
            all_people_arrays = []
            for day in trend_list:
                ppl = _decode_json_field(day.get("people_on_leave"))
                all_people_arrays.append(ppl)
            pid_list = _collect_person_ids(*all_people_arrays)
            resolved = resolver.resolve_many(pid_list)

            # Patch each day's people_on_leave
            for day in trend_list:
                ppl = _decode_json_field(day.get("people_on_leave"))
                day["people_on_leave"] = _patch_rows(ppl, resolved)

            return {"trend": trend_list, **extra_ctx}

        return {**row, **extra_ctx}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("/api/leave_data failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"leave_data query failed: {str(e)}")

# Attach router to app
app.include_router(router)