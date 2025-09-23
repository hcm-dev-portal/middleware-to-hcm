# app/leave/service.py
from __future__ import annotations

# =========================
# Standard Imports
# =========================
import json
import logging
import os
from datetime import datetime, timedelta, date, time
from typing import Dict, Any, Optional, List

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from app.leave.error_helper import (
    classify_hcm_message,
    default_explainer,          # (import kept if referenced elsewhere)
    ai_enhance_explanation,
)

logger = logging.getLogger(__name__)

# =========================
# Configuration (env-first)
# =========================
HCM_BASE_URL = os.getenv("HCM_BASE_URL", "https://metaguruoa.com/HRM/eHRExternalService/service.ashx") # "https://qgaia.royal.club.tw/eHR/eHRExternalService/service.ashx")
HCM_ACCESS_TOKEN = os.getenv("HCM_ACCESS_TOKEN", "8D337A53-CE50-40AB-BDEE-0FB645D69FEC") # "1DA1FAD6-6183-4174-8321-E8B853EA8D2D")
HCM_BUSINESS_UNIT = os.getenv("HCM_BUSINESS_UNIT", "0")
HCM_LOGON_REGION = os.getenv("HCM_LOGON_REGION", os.getenv("HCM_REGION", "zh-CN"))  # unify naming
HCM_DEFAULT_LOGIN = os.getenv("HCM_DEFAULT_LOGIN", "chiuzu") # "sa")

HTTP_TIMEOUT_SECS = float(os.getenv("HTTP_TIMEOUT_SECS", "30"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "2"))

# ExpiredDate freshness
HCM_EXPIRE_OFFSET_SECS = int(os.getenv("HCM_EXPIRE_OFFSET_SECS", "600"))  # 10 minutes

if not HCM_ACCESS_TOKEN:
    logger.warning("HCM_ACCESS_TOKEN missing; HCM calls will fail until configured.")

# =========================
# Helpers
# =========================
def _expired_at_str(offset_secs: int = HCM_EXPIRE_OFFSET_SECS) -> str:
    """Return naive local time 'YYYY-MM-DD HH:MM:SS' now + offset_secs."""
    return (datetime.now() + timedelta(seconds=max(1, int(offset_secs)))).strftime("%Y-%m-%d %H:%M:%S")


async def _make_explanation(
    *,
    vendor_result: Dict[str, Any] | Any,
    message: str,
    service_code: str,
    login: str,
    data: Dict[str, Any],
    app_code: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a structured explanation:
    - start from deterministic default_explainer(...)
    - optionally enhance via ai_enhance_explanation (with nlp=None here)
    """
    raw_payload = vendor_result if isinstance(vendor_result, dict) else {"raw": vendor_result}

    # Context for explainers (default_explainer expects end-user leave fields; we provide what we have)
    user_ctx = {
        # leave-ish hints if present in 'data' (safe best-effort)
        "employee_id": (data.get("detail", [{}])[0] or {}).get("employeeid"),
        "leave_type": (data.get("detail", [{}])[0] or {}).get("leavetypeid"),
        "start_date": (data.get("detail", [{}])[0] or {}).get("begindate"),
        "end_date":   (data.get("detail", [{}])[0] or {}).get("enddate"),
        "duration_type": None,
    }

    base = default_explainer(app_code, user_ctx)
    # add why/message echo from vendor
    base["why"] = message or base.get("why") or ""
    base["message"] = message or base.get("message") or ""
    base["code"] = app_code or base.get("code") or "HCM_ERROR"

    context = {
        "service_code": service_code,
        "login_name": login,
        "data": data,
        "vendor_message_id": raw_payload.get("MessageID"),
        "extra": extra or {},
        **user_ctx,
    }

    # We don't have NLP here; pass None so helper falls back gracefully
    try:
        enhanced = await ai_enhance_explanation(None, base, raw_payload, context)
        return enhanced or base
    except Exception:
        return base



# =========================
# Pydantic Models
# =========================
class HCMServiceCallRequest(BaseModel):
    """
    Generic request to call HCM by service code.
    We pass 'data' exactly as the vendor expects inside the 'Data' block.
    """
    service_code: str = Field(..., description="Vendor service code, e.g. applyformleave, createformleave, checkovertime")
    user_id: str = Field(HCM_DEFAULT_LOGIN, description="LoginName to use for LogonInfo")
    data: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("service_code")
    @classmethod
    def _clean_service_code(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("service_code is required")
        return v


def _parse_hhmm(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


class LeaveRequest(BaseModel):
    user_id: str = Field(..., description="Login name for authentication")
    employee_id: str = Field(..., description="Employee ID from HCM system")
    # Label accepted for legacy flow; we map to a default HCM leave type id.
    leave_type: str = Field(..., description="Label only; legacy flow maps to HCM leavetypeid")
    start_date: str = Field(..., description="YYYY-MM-DD")
    start_time: str = Field(default="09:00", description="HH:MM")
    end_date: str = Field(..., description="YYYY-MM-DD")
    end_time: str = Field(default="18:00", description="HH:MM")
    reason: Optional[str] = Field(default="", description="Reason for leave")
    duration_type: str = Field(default="full-day", description="full-day | half-day-am | half-day-pm | multiple-days")

    # cached fields for validation
    _start_date_dt: Optional[date] = None
    _end_date_dt: Optional[date] = None
    _start_time_t: Optional[time] = None
    _end_time_t: Optional[time] = None

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_dates(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")
        return v

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_times(cls, v: str) -> str:
        try:
            _parse_hhmm(v)
        except ValueError:
            raise ValueError("Times must be in HH:MM 24h format")
        return v

    @model_validator(mode="after")
    def normalize(self) -> "LeaveRequest":
        self._start_date_dt = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        self._end_date_dt = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        self._start_time_t = _parse_hhmm(self.start_time)
        self._end_time_t = _parse_hhmm(self.end_time)

        # Duration short-hands
        if self.duration_type in ("full-day", "half-day-am", "half-day-pm"):
            mapping = {
                "full-day": ("09:00", "18:00"),
                "half-day-am": ("09:00", "12:00"),
                "half-day-pm": ("13:00", "18:00"),
            }
            st, et = mapping[self.duration_type]
            self.start_time, self.end_time = st, et
            self._start_time_t = _parse_hhmm(st)
            self._end_time_t = _parse_hhmm(et)

        if self._end_date_dt < self._start_date_dt:
            raise ValueError("end_date cannot be before start_date")
        if self._start_date_dt == self._end_date_dt and self._start_time_t > self._end_time_t:
            raise ValueError("For same-day leave, start_time cannot be after end_time")
        return self


class LeaveBalanceRequest(BaseModel):
    user_id: str
    employee_id: str


class LeaveResponse(BaseModel):
    success: bool
    message: str
    request_id: Optional[str] = None
    form_number: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    code: Optional[str] = None


class LeaveBalanceResponse(BaseModel):
    success: bool
    balances: Dict[str, float]
    employee_id: str


# =========================
# HCM Client (service-code-first)
# =========================
class HCMLeaveService:
    def __init__(self) -> None:
        self.base_url = HCM_BASE_URL
        self.access_token = HCM_ACCESS_TOKEN
        self.business_unit = HCM_BUSINESS_UNIT
        self.region = HCM_LOGON_REGION

        # Default form numbers for legacy helpers
        self.form_numbers = {
            "createformleave": "10003",
            "applyformleave": "10003",
        }

    def _build_logon_info(self, *, service_code: str, login_name: str, expire_offset_secs: Optional[int] = None) -> str:
        expire_date = _expired_at_str(expire_offset_secs or HCM_EXPIRE_OFFSET_SECS)
        return (
            f"LoginName={login_name}"
            f"&BusinessUnit={self.business_unit}"
            f"&LogonRegion={self.region}"
            f"&ExpiredDate={expire_date}"
            f"&ServiceCode={service_code}"
        )

    def _build_payload(self, *, service_code: str, login_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "AccessToken": self.access_token,
            "LogonInfo": self._build_logon_info(service_code=service_code, login_name=login_name),
            "Data": data,
        }

    async def _post_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(HTTP_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
                    r = await client.post(self.base_url, json=payload, headers={"Content-Type": "application/json"})
                    r.raise_for_status()
                    # Try JSON; fallback to text if needed
                    try:
                        return r.json()
                    except Exception:
                        return {"raw": r.text}
            except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exc = e
                logger.warning("HCM request attempt %s failed: %s", attempt + 1, repr(e))
        raise HTTPException(status_code=502, detail=f"HCM upstream error: {last_exc}")

    async def call_service(
        self,
        *,
        service_code: str,
        login_name: Optional[str],
        data: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generic entry-point matching your Postman collection.
        e.g., service_code='applyformleave', data={...}
        """
        login = (login_name or HCM_DEFAULT_LOGIN).strip() or HCM_DEFAULT_LOGIN

        # ---- Normalize incoming data a bit (tolerate dict/str/None for detail) ----
        data = data or {}
        if isinstance(data, dict) and "detail" in data:
            det = data.get("detail")
            if isinstance(det, dict):
                data["detail"] = [det]
            elif isinstance(det, list):
                # keep as-is
                pass
            elif det in ("", None):
                # drop empty detail to avoid vendor deserialization error
                data.pop("detail", None)
            else:
                # unknown type – best effort: wrap nothing
                data["detail"] = []


        payload = self._build_payload(service_code=service_code, login_name=login, data=data)
        if idempotency_key:
            payload["IdempotencyKey"] = idempotency_key

        logger.info("HCM call_service [%s] payload: %s", service_code, json.dumps(payload, ensure_ascii=False))
        result = await self._post_with_retries(payload)
        logger.info("HCM call_service [%s] result: %s", service_code, result)

        # Normalize
        is_success = bool(isinstance(result, dict) and result.get("IsSuccess") is True)
        message = str(result.get("Message") or "").strip() if isinstance(result, dict) else ""

        if not is_success:
            # ----- Vendor-specific hints -----
            vendor_code = None
            if isinstance(result, dict):
                mid = str(result.get("MessageID") or "").strip()
                msg = message or ""
                msg_lower = msg.lower()

                if mid == "9900" or ("已存在" in msg) or ("already exists" in msg_lower):
                    vendor_code = "FORM_ALREADY_EXISTS"
                if ("沒有需要簽核" in msg) or ("no form to approve" in msg_lower) or ("nothing to approve" in msg_lower):
                    vendor_code = "NOTHING_TO_APPROVE"

            # ----- Map message -> app code (string) -----
            # classify_hcm_message returns a single string code in your error_helper.py
            classified_code = classify_hcm_message(message or "")

            # Final app code preference: vendor-specific > classified > default
            app_code = vendor_code or classified_code or "HCM_ERROR"

            # Build structured explanation (deterministic base + optional AI enhance)
            explanation = await _make_explanation(
                vendor_result=result,
                message=message,
                service_code=service_code,
                login=login,
                data=data,
                app_code=app_code,
                extra={},
            )

            # Add actionable guidance per vendor_code
            if app_code == "FORM_ALREADY_EXISTS":
                explanation.setdefault("title", "Form already exists")
                explanation.setdefault("probable_causes", []).extend([
                    "Using a fixed 'formno' that refers to a template/instance",
                    "Re-submission of the same create call without uniqueness",
                ])
                explanation.setdefault("checks", []).extend([
                    "Confirm if 'formno' is a template ID vs an instance number",
                    "Check vendor docs for the correct unique field (e.g., instance form id)",
                ])
                explanation.setdefault("next_steps", []).extend([
                    "Skip create and proceed to apply using the existing instance form number",
                    "If uniqueness is required, generate a unique reference per request",
                ])

            if app_code == "NOTHING_TO_APPROVE":
                explanation.setdefault("title", "No form to approve")
                explanation.setdefault("probable_causes", []).extend([
                    "Provided 'formno' points to a template, not a created instance",
                    "The form was not created, or it was already applied/approved",
                ])
                explanation.setdefault("checks", []).extend([
                    "Capture the instance form number/id from the create response",
                    "Verify employee/date/time match the intended record",
                ])
                explanation.setdefault("next_steps", []).extend([
                    "Re-run create, capture the instance identifier, then call apply with that value",
                    "If an instance already exists, look it up and use its instance form number in apply",
                ])

            return {
                "success": False,
                "code": app_code,
                "message": message or "HCM reported failure",
                "data": {
                    "context": {"service_code": service_code, "login_name": login},
                    "explanation": explanation,
                    "hcm": {"raw": result if isinstance(result, dict) else {"raw": result}},
                },
            }




    # -------- Legacy helpers (still used by /api/leave/submit) ----------
    async def _create_form_leave(
        self,
        *,
        employee_id: str,
        leave_type_id: str,
        start_date: str,
        start_time: str,
        end_date: str,
        end_time: str,
        login_name: str,
        idempotency_key: Optional[str],
    ) -> Dict[str, Any]:
        data = {
            "formno": self.form_numbers["createformleave"],
            "detail": [
                {
                    "employeeid": employee_id,
                    "leavetypeid": leave_type_id,
                    "begindate": start_date,
                    "begintime": start_time,
                    "enddate": end_date,
                    "endtime": end_time,
                }
            ],
        }
        return await self.call_service(
            service_code="createformleave",
            login_name=login_name,
            data=data,
            idempotency_key=idempotency_key,
        )

    async def _apply_form_leave(
        self,
        *,
        employee_id: str,
        leave_type_id: str,
        start_date: str,
        start_time: str,
        end_date: str,
        end_time: str,
        reason: str,
        login_name: str,
        idempotency_key: Optional[str],
    ) -> Dict[str, Any]:
        data = {
            "formno": self.form_numbers["applyformleave"],
            "reason": reason or "Leave request submitted via AI assistant",
            "detail": [
                {
                    "employeeid": employee_id,
                    "leavetypeid": leave_type_id,
                    "begindate": start_date,
                    "begintime": start_time,
                    "enddate": end_date,
                    "endtime": end_time,
                }
            ],
        }
        return await self.call_service(
            service_code="applyformleave",
            login_name=login_name,
            data=data,
            idempotency_key=idempotency_key,
        )

    async def create_leave_request(self, request: LeaveRequest, *, idempotency_key: Optional[str]) -> LeaveResponse:
        """
        Legacy flow retained: createformleave -> applyformleave.
        """
        try:
            # Map legacy label to default HCM leave type id (adjust if you add mapping later)
            leave_type_id = "19"

            # Step 1: create
            create_res = await self._create_form_leave(
                employee_id=request.employee_id,
                leave_type_id=leave_type_id,
                start_date=request.start_date,
                start_time=request.start_time,
                end_date=request.end_date,
                end_time=request.end_time,
                login_name=request.user_id or HCM_DEFAULT_LOGIN,
                idempotency_key=idempotency_key,
            )
            if not create_res.get("success"):
                # If the only problem was "already exists", try to apply anyway
                if create_res.get("code") == "FORM_ALREADY_EXISTS":
                    logger.info("createformleave reported already-exists; continuing to apply step.")
                else:
                    msg = create_res.get("message", "Failed to create leave form")
                    code = create_res.get("code")
                    return LeaveResponse(success=False, message=msg, code=code, data=create_res.get("data"))


            # Step 2: apply
            apply_res = await self._apply_form_leave(
                employee_id=request.employee_id,
                leave_type_id=leave_type_id,
                start_date=request.start_date,
                start_time=request.start_time,
                end_date=request.end_date,
                end_time=request.end_time,
                reason=request.reason or "",
                login_name=request.user_id or HCM_DEFAULT_LOGIN,
                idempotency_key=idempotency_key,
            )
            if not apply_res.get("success"):
                msg = apply_res.get("message", "Failed to apply leave")
                code = apply_res.get("code")
                return LeaveResponse(success=False, message=msg, code=code, data=apply_res.get("data"))

            req_id = f"LR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            return LeaveResponse(
                success=True,
                message=apply_res.get("message", "Leave request submitted successfully"),
                request_id=req_id,
                form_number=self.form_numbers["applyformleave"],
                data={
                    "hcm": apply_res.get("hcm"),
                    "context": {
                        "employee_id": request.employee_id,
                        "start_date": request.start_date,
                        "end_date": request.end_date,
                        "duration_type": request.duration_type,
                    },
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error creating leave request")
            return LeaveResponse(success=False, message=f"System error: {str(e)}", code="SYSTEM_ERROR")

    async def get_leave_balance(self, request: LeaveBalanceRequest) -> LeaveBalanceResponse:
        try:
            # TODO: replace with real HCM call when available
            mock_balances = {"annual": 15.5, "sick": 8.0, "personal": 3.0, "emergency": 0.0}
            return LeaveBalanceResponse(success=True, balances=mock_balances, employee_id=request.employee_id)
        except Exception as e:
            logger.exception("Error getting leave balance")
            raise HTTPException(status_code=500, detail=f"Failed to get leave balance: {str(e)}")


# =========================
# Singleton instance
# =========================
_hcm = HCMLeaveService()

# =========================
# Public service functions
# =========================
async def hcm_call(req: HCMServiceCallRequest, *, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Primary service-code-first entry point.
    """
    return await _hcm.call_service(
        service_code=req.service_code,
        login_name=req.user_id or HCM_DEFAULT_LOGIN,
        data=req.data,
        idempotency_key=idempotency_key,
    )


async def submit_hcm_service(req: HCMServiceCallRequest, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Alias kept for compatibility with existing router/controller code that calls submit_hcm_service.
    """
    return await hcm_call(req, idempotency_key=idempotency_key)


# ---- Legacy flows kept for existing UI routes (/api/leave/submit etc.) ----
async def submit_leave_request(request: LeaveRequest, idempotency_key: Optional[str] = None) -> LeaveResponse:
    return await _hcm.create_leave_request(request, idempotency_key=idempotency_key)


async def get_employee_leave_balance(request: LeaveBalanceRequest) -> LeaveBalanceResponse:
    return await _hcm.get_leave_balance(request)


async def validate_leave_request(request: LeaveRequest) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    start_d = request._start_date_dt
    end_d = request._end_date_dt

    today = date.today()
    if start_d and start_d < today and request.leave_type != "sick":
        warnings.append("Leave start date is in the past (non-sick)")

    if start_d == end_d and request.duration_type == "multiple-days":
        warnings.append("Duration type is 'multiple-days' but start/end are the same day")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
