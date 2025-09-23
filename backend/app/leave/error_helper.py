# app/leave/error_helper.py
from typing import Dict, Any, List, Optional

# 1) lightweight message -> code mapping
KNOWN_ERROR_PATTERNS = [
    ("AccessToken不存在", "AUTH_TOKEN_MISSING"),
    ("已經過期", "AUTH_TOKEN_EXPIRED"),
    ("時數不足", "INSUFFICIENT_BALANCE"),
    ("小時不足", "INSUFFICIENT_BALANCE"),
    ("請假時數小於", "DURATION_TOO_SHORT"),
    ("請假時數大於", "DURATION_TOO_LONG"),
    ("日期格式", "DATE_FORMAT_INVALID"),
    ("找不到員工", "EMPLOYEE_NOT_FOUND"),
]

def classify_hcm_message(msg: str) -> str:
    m = (msg or "").strip()
    for key, code in KNOWN_ERROR_PATTERNS:
        if key in m:
            return code
    return "HCM_ERROR"

def default_explainer(code: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a structured, human-friendly explanation + suggestions.
    This is deterministic and fast. We can also overlay an AI explanation if available.
    """
    employee_id = context.get("employee_id")
    leave_type = context.get("leave_type")
    start_date = context.get("start_date")
    end_date   = context.get("end_date")
    duration   = context.get("duration_type")

    # sensible defaults
    out = {
        "title": "There was a problem submitting your leave.",
        "why": "The system rejected the request.",
        "suggestions": [],
        "next_steps": [],
        "code": code,
    }

    if code == "AUTH_TOKEN_MISSING":
        out["why"] = "Our connection to HCM is missing an access token."
        out["suggestions"] = [
            "Refresh the HCM access token (backend setting).",
            "Try again after the service is re-authenticated."
        ]
        out["next_steps"] = ["Please notify HRIS admin or click 'Retry' once this is fixed."]
    elif code == "AUTH_TOKEN_EXPIRED":
        out["why"] = "The HCM access token has expired."
        out["suggestions"] = [
            "Renew or refresh the HCM access token.",
            "If this keeps happening, increase the token refresh frequency."
        ]
        out["next_steps"] = ["We’ll retry automatically once the token is refreshed."]
    elif code == "INSUFFICIENT_BALANCE":
        out["why"] = "Your balance is not enough for the requested leave."
        out["suggestions"] = [
            "Reduce the number of days or choose a different leave type.",
            "Split the request (e.g., take fewer days now).",
            "Check your current balance from the left panel."
        ]
        out["next_steps"] = [
            f"Try changing duration or leave type and resubmit.",
            "Ask the assistant to propose the closest allowed request."
        ]
    elif code == "DURATION_TOO_SHORT":
        out["why"] = "Requested duration is below the minimum unit allowed by policy."
        out["suggestions"] = [
            "Change to a valid unit (e.g., half-day or full-day).",
            "If you need hours, check whether hourly leave is enabled for your site."
        ]
        out["next_steps"] = ["Update the duration and submit again."]
    elif code == "DURATION_TOO_LONG":
        out["why"] = "Requested duration exceeds your balance or policy limit."
        out["suggestions"] = [
            "Shorten the date range.",
            "Consider combining different leave types if allowed."
        ]
        out["next_steps"] = ["Reduce the number of days and resubmit."]
    elif code == "DATE_FORMAT_INVALID":
        out["why"] = "The date format or range looks invalid for HCM."
        out["suggestions"] = [
            "Use YYYY-MM-DD format.",
            "Ensure the end date is not before the start date."
        ]
        out["next_steps"] = ["Fix the dates and try again."]
    elif code == "EMPLOYEE_NOT_FOUND":
        out["why"] = f"The employee ID ({employee_id}) was not found in HCM."
        out["suggestions"] = [
            "Verify your employee ID in the profile panel.",
            "Contact HR if the ID is correct but still not recognized."
        ]
        out["next_steps"] = ["Correct the employee ID then resubmit."]
    else:  # HCM_ERROR
        out["why"] = "HCM returned an error we didn’t recognize."
        out["suggestions"] = [
            "Try again shortly.",
            "If it persists, contact HRIS with the Request ID."
        ]
        out["next_steps"] = ["Use the 'Why?' panel to view full details."]

    return out

async def ai_enhance_explanation(nlp, base: Dict[str, Any], raw: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional: call your LanguageNativeNLPService to rewrite 'why/suggestions/next_steps'
    into friendlier, site-aware language. If NLP is unavailable, just return base.
    """
    if not nlp:
        return base
    try:
        prompt = (
            "You are an HR leave assistant. Given the structured error context below, write a short, friendly "
            "explanation for the end user and 3–5 actionable suggestions.\n\n"
            f"Base: {base}\n\n"
            f"RawHCM: {raw}\n\n"
            f"Context: {context}\n\n"
            "Return JSON with keys: title, why, suggestions (array), next_steps (array)."
        )
        out = await nlp.simple_json(prompt)  # implement simple_json in your service or use an existing method
        # Merge conservatively
        for k in ("title", "why", "suggestions", "next_steps"):
            if out.get(k):
                base[k] = out[k]
        return base
    except Exception:
        return base
