# app/leave/ai_explainer.py
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# OPTIONAL OpenAI client (or your wrapper); fall back to rule-based if not present
try:
    # If you already have a service wrapper, prefer importing that.
    from app.reports.llm_client import OpenAIService  # your existing wrapper (if any)
except Exception:
    OpenAIService = None  # type: ignore


def _safe(obj: Any, default: str = "") -> str:
    try:
        s = str(obj or "").strip()
        return s
    except Exception:
        return default


def _get_locale_strings(locale: str = "en") -> Dict[str, Any]:
    """Return localized strings for UI messages."""
    loc = (locale or "en").lower()
    if loc.startswith("zh"):
        return {
            "title_prefix": "提交休假申請時發生問題",
            "why_default": "請求無法被處理",
            "auth_expired": {
                "why": "HCM 存取權杖已過期或無效",
                "probable_causes": [
                    "權杖生命週期已結束或時鐘偏移",
                    "部署後權杖未更新"
                ],
                "checks": [
                    "確認 HCM_ACCESS_TOKEN 環境變數與供應商提供的權杖相符",
                    "確認權杖有效期限和伺服器時間"
                ],
                "next_steps": [
                    "重新整理/輪換權杖並重試",
                    "如果經常發生，自動化權杖重新整理"
                ],
                "auto_actions": [
                    {"id": "retry_after_token_refresh", "label": "重新整理權杖後重試"}
                ]
            },
            "bad_time": {
                "why": "開始/結束時間或日期範圍無效",
                "probable_causes": [
                    "同日請假的結束時間早於開始時間",
                    "跨日範圍但半日選擇不一致"
                ],
                "checks": [
                    "同日休假請確保 開始時間 ≤ 結束時間",
                    "半日休假使用 上午(09:00–12:00) 或 下午(13:00–18:00)"
                ],
                "next_steps": [
                    "調整時間範圍後重新提交",
                    "如政策要求最少時數，請增加時長"
                ],
                "auto_actions": [
                    {"id": "fix_times_to_default", "label": "自動修正為政策時間"}
                ]
            },
            "insufficient_balance": {
                "why": "所選休假類型的餘額可能不足",
                "probable_causes": [
                    "餘額低於所需時數/天數",
                    "選擇了錯誤的休假類型"
                ],
                "checks": [
                    "檢查所選類型的目前餘額",
                    "嘗試不同的休假類型（如個人假 vs 年假）"
                ],
                "next_steps": [
                    "提交較短的時長",
                    "如允許，申請例外休假核准"
                ],
                "auto_actions": [
                    {"id": "open_balance_panel", "label": "顯示我的休假餘額"}
                ]
            },
            "null_reference": {
                "why": "HCM 系統發生內部錯誤（空物件參考）",
                "probable_causes": [
                    "使用了固定的表單編號，參考到範本/實例",
                    "重複提交相同的建立請求，缺乏唯一性",
                    "員工資料或休假類型設定不完整"
                ],
                "checks": [
                    "確認員工ID在HCM系統中存在且完整",
                    "檢查休假類型ID對應（19=年假, 20=病假, 21=個人假, 22=緊急假）",
                    "確認日期格式和時間範圍符合系統要求"
                ],
                "next_steps": [
                    "使用「為什麼？」面板查看完整詳細資訊",
                    "跳過建立，使用現有實例表單編號進行申請",
                    "如需唯一性，為每個請求生成唯一參考號",
                    "聯絡HR系統管理員確認設定"
                ],
                "auto_actions": [
                    {"id": "generate_unique_ref", "label": "生成唯一參考號"},
                    {"id": "check_employee_setup", "label": "檢查員工設定"},
                    {"id": "retry_with_validation", "label": "使用驗證重試"}
                ]
            },
            "generic": {
                "why": "HCM 服務返回錯誤",
                "probable_causes": [
                    "HCM 暫時中斷或負載格式錯誤",
                    "休假類型ID映射不正確"
                ],
                "checks": [
                    "檢查原始回應（下方）中的供應商提示字串",
                    "確認休假類型ID映射（19=年假, 20=病假, 21=個人假, 22=緊急假）"
                ],
                "next_steps": [
                    "稍後重試，或如持續發生請聯絡HR系統負責人"
                ]
            }
        }
    else:  # English
        return {
            "title_prefix": "There was a problem submitting your leave",
            "why_default": "The request could not be applied",
            "auth_expired": {
                "why": "The HCM access token has expired or is invalid",
                "probable_causes": [
                    "Token lifetime elapsed or clock skew",
                    "Token not rotated after deployment"
                ],
                "checks": [
                    "Verify HCM_ACCESS_TOKEN env var matches vendor-supplied token",
                    "Confirm token validity window and server time"
                ],
                "next_steps": [
                    "Refresh/rotate the token and retry",
                    "Automate token refresh if this happens frequently"
                ],
                "auto_actions": [
                    {"id": "retry_after_token_refresh", "label": "Retry after refreshing token"}
                ]
            },
            "bad_time": {
                "why": "The begin/end time or date range is not valid",
                "probable_causes": [
                    "End time is earlier than start time for a same-day request",
                    "Cross-day range with inconsistent half-day selection"
                ],
                "checks": [
                    "For same day leave, ensure start_time <= end_time",
                    "For half-day, use AM (09:00–12:00) or PM (13:00–18:00)"
                ],
                "next_steps": [
                    "Adjust time range and resubmit",
                    "If policy requires minimum hours, increase duration"
                ],
                "auto_actions": [
                    {"id": "fix_times_to_default", "label": "Auto-fix to policy times"}
                ]
            },
            "insufficient_balance": {
                "why": "You may not have enough balance for the selected leave type",
                "probable_causes": [
                    "Balance below required hours/days",
                    "Wrong leave type chosen"
                ],
                "checks": [
                    "Check current balances for the selected type",
                    "Try a different leave type (e.g., personal vs annual)"
                ],
                "next_steps": [
                    "Submit a shorter duration",
                    "Request approval for exceptional leave if allowed"
                ],
                "auto_actions": [
                    {"id": "open_balance_panel", "label": "Show my leave balances"}
                ]
            },
            "null_reference": {
                "why": "HCM system encountered an internal error (null object reference)",
                "probable_causes": [
                    "Using a fixed form number that refers to a template/instance",
                    "Re-submission of the same create call without uniqueness",
                    "Incomplete employee data or leave type configuration"
                ],
                "checks": [
                    "Confirm employee ID exists and is complete in HCM system",
                    "Verify leave type ID mapping (19=annual, 20=sick, 21=personal, 22=emergency)",
                    "Ensure date format and time range meet system requirements"
                ],
                "next_steps": [
                    "Use the 'Why?' panel to view full details",
                    "Skip create and proceed to apply using the existing instance form number",
                    "If uniqueness is required, generate a unique reference per request",
                    "Contact HR systems owner to verify configuration"
                ],
                "auto_actions": [
                    {"id": "generate_unique_ref", "label": "Generate unique reference"},
                    {"id": "check_employee_setup", "label": "Check employee setup"},
                    {"id": "retry_with_validation", "label": "Retry with validation"}
                ]
            },
            "generic": {
                "why": "The HCM service returned an error",
                "probable_causes": [
                    "Temporary HCM outage or malformed payload",
                    "Incorrect mapping for leave type IDs"
                ],
                "checks": [
                    "Inspect raw response (below) for vendor hint strings",
                    "Confirm leavetypeid mapping (19=annual, 20=sick, 21=personal, 22=emergency)"
                ],
                "next_steps": [
                    "Retry later or contact HR systems owner if it persists"
                ]
            }
        }


def _default_playbook(code: str, message: str, ctx: Dict[str, Any], locale: str = "en") -> Dict[str, Any]:
    """
    Deterministic guidance we can always return, even if the LLM is offline.
    Now supports bilingual responses.
    """
    code = _safe(code).upper()
    message = _safe(message)
    message_id = _safe(ctx.get("message_id"))
    strings = _get_locale_strings(locale)

    base: Dict[str, Any] = {
        "title": strings["title_prefix"],
        "why": strings["why_default"],
        "probable_causes": [],
        "checks": [],
        "next_steps": [],
        "auto_actions": [],
        "message": message,
        "code": code or "UNKNOWN",
        "message_id": message_id,
        "locale": locale,
    }

    # Specific patterns
    msg_lower = message.lower()
    if code in ("AUTH_TOKEN_EXPIRED", "AUTH_EXPIRED", "AUTH_TOKEN_INVALID"):
        base.update(strings["auth_expired"])
    elif code in ("BAD_TIME_RANGE", "INVALID_TIME", "START_AFTER_END"):
        base.update(strings["bad_time"])
    elif code in ("INSUFFICIENT_BALANCE",):
        base.update(strings["insufficient_balance"])
    elif message_id == "9900" or ("object reference not set" in msg_lower) or ("null reference" in msg_lower):
        base.update(strings["null_reference"])
    else:
        base.update(strings["generic"])

    # Context echo (trimmed)
    base["context"] = {
        "employee_id": ctx.get("employee_id"),
        "leave_type": ctx.get("leave_type"),
        "start_date": ctx.get("start_date"),
        "end_date": ctx.get("end_date"),
        "duration_type": ctx.get("duration_type"),
        "message_id": message_id,
    }
    return base


def llm_enhance(
    *,
    code: str,
    message: str,
    context: Dict[str, Any],
    raw_hcm: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    locale: str = "en",
) -> Dict[str, Any]:
    """
    If an LLM is available, produce a richer, structured explanation.
    Always returns a safe dict (falls back to rule-based).
    Now supports bilingual enhancement.
    """
    try:
        # Extract message_id from raw_hcm for better error classification
        if raw_hcm and isinstance(raw_hcm, dict):
            context = dict(context or {})
            context["message_id"] = _safe(raw_hcm.get("MessageID"))
    except Exception:
        # do not fail if raw_hcm is odd
        pass

    # Start with deterministic playbook
    baseline = _default_playbook(code, message, context, locale)

    if OpenAIService is None:
        return baseline

    try:
        client = OpenAIService(
            model_name=model_name or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

        # Adjust system prompt based on locale
        loc = (locale or "en").lower()
        if loc.startswith("zh"):
            system = (
                "你是一個HR休假助理。"
                "根據HCM錯誤與上下文，輸出嚴格的JSON物件，鍵："
                "title, why, probable_causes(陣列), checks(陣列), next_steps(陣列), "
                "auto_actions(包含{id,label}的陣列), message, code, context"
                "(包含employee_id, leave_type, start_date, end_date, duration_type)。"
                "使用清晰、非技術性的繁體中文，簡潔有用。只輸出JSON。"
            )
        else:
            system = (
                "You are an HR leave assistant. "
                "Given an HCM error and context, output a STRICT JSON object with keys: "
                "title, why, probable_causes (array), checks (array), next_steps (array), "
                "auto_actions (array of {id,label}), message, code, context "
                "(object with employee_id, leave_type, start_date, end_date, duration_type). "
                "Use clear, non-technical language. Keep it concise. Output JSON only."
            )

        user_payload = {
            "code": code,
            "message": message,
            "context": context,
            "raw_hcm": raw_hcm or {},
            "locale": locale,
        }

        # Use your wrapper's low-level call to get a string response.
        raw = client.simple_json_chat(system_prompt=system, user_payload=user_payload)
        s = _safe(raw)

        # Normalize fenced code responses if any:
        if s.startswith("```"):
            s = s.strip("`").split("\n", 1)[-1]
            if not s.strip().startswith("{"):
                s = s.split("```", 1)[0]

        obj: Optional[Dict[str, Any]] = None
        try:
            obj = json.loads(s)
        except Exception:
            obj = None

        # Merge with baseline to guarantee required keys
        if isinstance(obj, dict):
            merged = dict(baseline)
            merged.update(obj)  # model output can override defaults
            merged["locale"] = locale
            # Ensure required keys exist
            for key in ("title", "why", "probable_causes", "checks", "next_steps", "auto_actions", "message", "code", "context"):
                merged.setdefault(key, baseline.get(key))
            return merged

    except Exception as e:
        logger.warning("LLM enhancement failed: %s", e)

    return baseline


def get_explanation(
    error_code: str,
    error_message: str,
    context: Dict[str, Any],
    raw_hcm_response: Optional[Dict[str, Any]] = None,
    user_locale: str = "en",
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Main entry point for getting error explanations.

    Args:
        error_code: Error code from the system
        error_message: Human readable error message
        context: Request context (employee_id, dates, etc.)
        raw_hcm_response: Raw HCM API response
        user_locale: User's locale preference ("en", "zh", "zh-TW", etc.)
        use_llm: Whether to use LLM enhancement

    Returns:
        Structured explanation dictionary
    """
    if use_llm:
        return llm_enhance(
            code=error_code,
            message=error_message,
            context=context,
            raw_hcm=raw_hcm_response,
            locale=user_locale,
        )
    else:
        return _default_playbook(error_code, error_message, context, user_locale)
