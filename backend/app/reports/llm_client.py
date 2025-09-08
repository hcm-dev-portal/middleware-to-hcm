# backend/app/reports/llm_client.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Reuse your existing services if available
from app.services.llm.openai_service import OpenAIService
from app.services.aws.translation_service import AWSTranslationService

logger = logging.getLogger(__name__)


# =========================
# Public data structures
# =========================
@dataclass
class Intent:
    """What the user wants us to generate."""
    report_type: str = "unknown"          # leave_analysis | overtime_analysis | attendance_analysis | balance_report | unknown
    time_period: str = "monthly"          # weekly | monthly | quarterly | yearly
    time_range: Optional[Tuple[str, str]] = None  # (iso start, iso end) if known
    departments: List[str] = field(default_factory=list)  # ['all'] or list of depts
    metrics: List[str] = field(default_factory=list)      # keys the backend understands
    confidence: float = 0.75
    needs_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)


@dataclass
class Narrative:
    """Structured content to place into the DOCX (or show in UI)."""
    language: str
    title: str
    executive_summary: str
    methodology: str
    key_insights: List[str]
    risks: List[str]
    recommendations: List[str]
    appendix_notes: List[str] = field(default_factory=list)


# =========================
# LLM Client
# =========================
class LLMClient:
    """
    High-level client that:
      1) Analyzes a user query into a formal Intent.
      2) Builds a well-structured Narrative for the report.
      3) Optionally translates to/from zh-TW using your AWS service.

    Safe to use when OpenAI/AWS are not available (falls back to templates).
    """

    def __init__(
        self,
        llm: Optional[OpenAIService] = None,
        translator: Optional[AWSTranslationService] = None,
        default_lang: str = "en-US",
    ):
        self.llm = llm or OpenAIService(model_name="gpt-4o-mini", temperature=0.2)
        self.translator = translator or AWSTranslationService()
        self.default_lang = default_lang

    # -----------------------------
    # Public API
    # -----------------------------
    def analyze_intent(self, user_query: str, preferred_lang: Optional[str] = None) -> Intent:
        """
        Turn a raw user query into a normalized Intent. If the LLM is available,
        we extract a richer, JSON-typed intent. Otherwise we fall back to rules.
        """
        lang = self._normalize_lang(preferred_lang or self.default_lang)

        # Optional: detect + translate to English for better intent extraction
        detected_lang, _ = self.translator.detect_language(user_query)
        source_lang = self._normalize_lang(detected_lang)
        english_query = (
            self.translator.translate_to_english(user_query, source_lang)
            if source_lang != "en-US"
            else user_query
        )

        if self.llm.llm_enabled and self._prompt_layer_ready():
            try:
                intent = self._llm_extract_intent(english_query)
                # Keep user language preference on departments/labels if needed
                return intent
            except Exception as e:
                logger.warning("LLM intent extraction failed, falling back. %s", e)

        # Fallback: deterministic heuristics (keeps your existing behavior aligned)
        return self._rule_based_intent(english_query)

    def build_narrative(
        self,
        *,
        query: str,
        intent: Intent,
        data_bucket: Dict[str, Any],
        title: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> Narrative:
        """
        Produce a formal Narrative: executive summary, methodology, insights, risks, recommendations.
        Uses OpenAI when available, otherwise deterministic templates.
        """
        lang = self._normalize_lang(target_language or self.default_lang)
        title = title or self._title_from_intent(intent) #type: ignore

        # Build a compact context object for the LLM / fallback
        context = {
            "query": query,
            "intent": intent.__dict__,
            "data": data_bucket,  # whatever your tools produced
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

        if self.llm.llm_enabled and self._prompt_layer_ready():
            try:
                narrative = self._llm_narrative(context, lang, title)
                return narrative
            except Exception as e:
                logger.warning("LLM narrative generation failed, falling back. %s", e)

        # Fallback: deterministic narrative
        narrative = self._template_narrative(context, lang, title)

        # If user wants zh-TW and our text is English, translate end product once
        if lang == "zh-TW":
            narrative = self._maybe_translate_narrative(narrative, "zh-TW")

        return narrative

    # -----------------------------
    # LLM-powered internals
    # -----------------------------
    def _prompt_layer_ready(self) -> bool:
        # LangChain prompt objects are initialized in OpenAIService
        return bool(getattr(self.llm, "explanation_prompt", None))

    def _llm_extract_intent(self, english_query: str) -> Intent:
        """
        Ask the LLM to return a strict JSON intent. We reuse the existing LLM to keep deps minimal.
        """
        system = (
            "You are an HR analytics planner. Extract a strict JSON object describing the intent.\n"
            "Fields: report_type(one of: leave_analysis, overtime_analysis, attendance_analysis, balance_report, unknown),"
            " time_period(one of: weekly, monthly, quarterly, yearly),"
            " time_range: {start: ISO8601|null, end: ISO8601|null},"
            " departments: string[],"
            " metrics: string[],"
            " confidence: number in [0,1],"
            " needs_clarification: boolean,"
            " clarification_questions: string[].\n"
            "Default conservatively if unspecified; do not hallucinate dates."
        )

        # Use the same chat client with a minimal custom call
        try:
            from langchain.schema import SystemMessage, HumanMessage  # type: ignore
        except Exception:
            # If langchain schema isn't available (older env) do deterministic
            return self._rule_based_intent(english_query)

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Query: {english_query}\nReturn ONLY JSON."),
        ]
        raw = self.llm._invoke_llm(messages)  # returns str
        if not raw:
            return self._rule_based_intent(english_query)

        # Be robust to code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").split("\n", 1)[-1]
            if cleaned.strip().startswith("{") is False:
                cleaned = cleaned.split("```", 1)[0]
        try:
            obj = json.loads(cleaned)
        except Exception:
            # Try to locate first {...} block
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                obj = json.loads(cleaned[start : end + 1])
            else:
                return self._rule_based_intent(english_query)

        # Normalize → Intent
        tr = obj.get("time_range") or {}
        start = tr.get("start")
        end = tr.get("end")
        intent = Intent(
            report_type=(obj.get("report_type") or "unknown"),
            time_period=(obj.get("time_period") or "monthly"),
            time_range=(start, end) if (start and end) else None,
            departments=list(obj.get("departments") or []),
            metrics=list(obj.get("metrics") or []),
            confidence=float(obj.get("confidence") or 0.75),
            needs_clarification=bool(obj.get("needs_clarification") or False),
            clarification_questions=list(obj.get("clarification_questions") or []),
        )
        return intent

    def _llm_narrative(self, context: Dict[str, Any], lang: str, title: str) -> Narrative:
        """
        Use the LLM to write a formal narrative. We leverage the 'explanation' pathway for consistency,
        but provide a precise instruction to output a structured JSON block of sections.
        """
        # Compose a compact pseudo-aggregate to fit the OpenAIService API
        aggregates = {
            "row_count": _safe_int(context, ["data", "row_count"], 0),
            "unique_people": _safe_int(context, ["data", "unique_people"], None),
            "by_leave_type": _safe_dict(context, ["data", "leave_metrics", "by_type"]),
            "total_hours": _safe_int(context, ["data", "ot_metrics", "total_hours"], None),
        }

        # We piggyback on generate_explanation (which already handles LLM call),
        # but we pass a special instruction asking it to produce JSON sections.
        instruction = (
            "Return a JSON with keys: "
            "executive_summary (string, 2–4 sentences, formal), "
            "methodology (string, what we did and any assumptions), "
            "key_insights (array of 4–7 concise bullets), "
            "risks (array of 2–4 bullets), "
            "recommendations (array of 3–6 action bullets). "
            "Do not include code fences."
        )
        columns = ["field", "value"]  # Placeholder, unused but required
        sample_text = json.dumps(_safe_dict(context, ["data"]), ensure_ascii=False)[:1500]

        # Use the same function to generate content; prepend instruction to question
        question = f"{instruction}\n\nOriginal request: {context.get('query','')}"
        raw = self.llm.generate_explanation(
            question=question,
            row_count=int(aggregates.get("row_count") or 0),
            columns=columns,
            aggregates=aggregates,
            sample_text=sample_text,
        )

        # Try to parse JSON; if not, fallback
        payload = _try_extract_json(raw)
        if not payload:
            return self._template_narrative(context, lang, title)

        narrative = Narrative(
            language=lang,
            title=title,
            executive_summary=str(payload.get("executive_summary") or "").strip(),
            methodology=str(payload.get("methodology") or "").strip(),
            key_insights=[str(x).strip() for x in payload.get("key_insights") or []],
            risks=[str(x).strip() for x in payload.get("risks") or []],
            recommendations=[str(x).strip() for x in payload.get("recommendations") or []],
            appendix_notes=[
                f"Generated at {context.get('generated_at')}",
                f"Time period: {context.get('intent', {}).get('time_period', 'monthly')}",
            ],
        )

        # Translate when target is zh-TW
        if lang == "zh-TW":
            narrative = self._maybe_translate_narrative(narrative, "zh-TW")

        return narrative

    # -----------------------------
    # Fallbacks (no-LLM paths)
    # -----------------------------
    def _rule_based_intent(self, english_query: str) -> Intent:
        q = english_query.lower()
        # Simple classification (aligned with your backend)
        if any(w in q for w in ["leave", "vacation", "time off", "absence"]):
            rt = "leave_analysis"
            metrics = ["total_leave_days", "leave_by_type", "department_breakdown"]
        elif any(w in q for w in ["overtime", "extra hours", "ot"]):
            rt = "overtime_analysis"
            metrics = ["total_overtime_hours", "overtime_by_department", "employee_overtime"]
        elif any(w in q for w in ["attendance", "present", "absent"]):
            rt = "attendance_analysis"
            metrics = ["attendance_rate", "absence_patterns", "department_comparison"]
        elif any(w in q for w in ["balance", "remaining", "accrued"]):
            rt = "balance_report"
            metrics = ["vacation_balance", "sick_leave_balance", "low_balance_alerts"]
        else:
            rt = "unknown"
            metrics = ["general_metrics"]

        # Period
        if "week" in q:
            tp = "weekly"
        elif "quarter" in q or "q1" in q or "q2" in q or "q3" in q or "q4" in q:
            tp = "quarterly"
        elif "year" in q or "annual" in q:
            tp = "yearly"
        else:
            tp = "monthly"

        needs = rt in ("unknown",)  # ask when unsure
        clarify = []
        if needs:
            clarify.append(
                "What time period should I use? (weekly, monthly, quarterly, yearly)"
            )

        return Intent(
            report_type=rt,
            time_period=tp,
            time_range=None,
            departments=[],
            metrics=metrics,
            confidence=0.8 if rt != "unknown" else 0.6,
            needs_clarification=needs,
            clarification_questions=clarify,
        )

    def _template_narrative(self, context: Dict[str, Any], lang: str, title: str) -> Narrative:
        intent = context.get("intent", {}) or {}
        rt = intent.get("report_type", "unknown").replace("_", " ")
        tp = intent.get("time_period", "monthly")
        # Heuristic bullets from available data
        bullets: List[str] = []

        # Leave
        lm = _safe_dict(context, ["data", "leave_metrics"])
        if lm:
            totals = lm.get("totals", {})
            by_type = lm.get("by_type", {})
            top_type = max(by_type, key=by_type.get) if by_type else None
            bullets += [
                f"Total leave days: {totals.get('leave_days', '—')}, average per employee: {totals.get('avg_days_per_employee', '—')}.",
            ]
            if top_type:
                bullets.append(f"Leading leave type: {top_type} ({by_type[top_type]} days).")

        # Overtime
        ot = _safe_dict(context, ["data", "ot_metrics"])
        if ot:
            bullets += [
                f"Overtime hours: {ot.get('total_hours', '—')} (avg {ot.get('avg_per_employee', '—')} per employee).",
                f"Estimated overtime cost: ${ot.get('est_cost', 0):,}."
            ]

        # Attendance
        at = _safe_dict(context, ["data", "att_metrics"])
        if at:
            bullets += [
                f"Attendance rate: {at.get('attendance_rate', '—')}%.",
                f"Unplanned absences: {at.get('unplanned_absences', '—')}."
            ]

        # Balance
        bal = _safe_dict(context, ["data", "bal_metrics"])
        if bal:
            bullets += [
                f"Employees with low vacation balance (<5 days): {bal.get('low_balance_count', '—')}.",
                f"Average vacation balance: {bal.get('avg_vacation_balance', '—')} days."
            ]

        if not bullets:
            bullets = [
                "Key metrics were collected and reviewed for the requested scope.",
                "No critical anomalies were observed in the current period."
            ]

        executive = (
            f"This {rt} covering the {tp} period summarizes current performance and patterns. "
            f"The overview below highlights notable signals and areas of interest."
        )
        methodology = (
            "Methodology: We analyzed aggregated HR system records for the selected period, "
            "including department-level breakdowns where applicable. Standard business rules "
            "were applied; outliers and incomplete records were excluded when detected."
        )
        risks = [
            "Potential data gaps due to late submissions or pending approvals.",
            "Small sample sizes in certain departments may skew percentages."
        ]
        recs = [
            "Review departments showing sustained variance versus historical norms.",
            "Confirm policy adherence for teams with elevated leave or overtime.",
            "Schedule a follow-up review next period to track corrective actions."
        ]

        narrative = Narrative(
            language=lang,
            title=title,
            executive_summary=executive,
            methodology=methodology,
            key_insights=bullets[:6],
            risks=risks,
            recommendations=recs,
            appendix_notes=[
                f"Generated at {context.get('generated_at')}",
                f"Time period: {tp}",
            ],
        )

        if lang == "zh-TW":
            narrative = self._maybe_translate_narrative(narrative, "zh-TW")

        return narrative

    # -----------------------------
    # Translation helpers
    # -----------------------------
    def _maybe_translate_narrative(self, n: Narrative, target_lang: str) -> Narrative:
        """Translate fields from English → target_lang using AWS if available."""
        try:
            if target_lang == "en-US":
                return n
            n2 = Narrative(
                language=target_lang,
                title=self.translator.translate_from_english(n.title, target_lang),
                executive_summary=self.translator.translate_from_english(n.executive_summary, target_lang),
                methodology=self.translator.translate_from_english(n.methodology, target_lang),
                key_insights=[self.translator.translate_from_english(x, target_lang) for x in n.key_insights],
                risks=[self.translator.translate_from_english(x, target_lang) for x in n.risks],
                recommendations=[self.translator.translate_from_english(x, target_lang) for x in n.recommendations],
                appendix_notes=[self.translator.translate_from_english(x, target_lang) for x in n.appendix_notes],
            )
            return n2
        except Exception as e:
            logger.warning("Narrative translation failed, keeping source language. %s", e)
            return n

    # -----------------------------
    # Utilities
    # -----------------------------
    @staticmethod
    def _normalize_lang(code: Optional[str]) -> str:
        if not code:
            return "en-US"
        return AWSTranslationService.normalize_language_code(code)


# =========================
# Local helpers (module)
# =========================
def _try_extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from model output."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").split("\n", 1)[-1]
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return None
    return None


def _safe_int(obj: Dict[str, Any], path: List[str], default: Optional[int] = None) -> Optional[int]:
    cur: Any = obj
    for p in path:
        cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            return default
    try:
        return int(cur)
    except Exception:
        return default


def _safe_dict(obj: Dict[str, Any], path: List[str]) -> Dict[str, Any]:
    cur: Any = obj
    for p in path:
        cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            return {}
    return cur if isinstance(cur, dict) else {}
