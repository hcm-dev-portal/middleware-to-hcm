# backend/app/reports/llm_client.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Reuse your existing services if available
from app.services.llm.openai_service import OpenAIService
from app.services.aws.translation_service import AWSTranslationService

logger = logging.getLogger(__name__)


# =========================
# Chart Data Structures
# =========================
@dataclass
class ChartDataPoint:
    """Single data point for charts."""
    label: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartDataset:
    """Complete dataset for a chart."""
    chart_id: str
    chart_type: str  # "bar", "pie", "line", "gauge", "heatmap", "table"
    title: str
    description: str
    x_axis_label: str = ""
    y_axis_label: str = ""
    data_points: List[ChartDataPoint] = field(default_factory=list)
    summary_stats: Dict[str, float] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)


@dataclass
class SectionNarrative:
    """Narrative content for a specific report section."""
    section_id: str
    section_title: str
    introduction: str
    key_findings: List[str]
    data_summary: str
    charts: List[ChartDataset] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    conclusion: str = ""


# =========================
# Public data structures
# =========================
@dataclass
class Intent:
    """What the user wants us to generate."""
    report_type: str = "unknown"
    time_period: str = "monthly"
    time_range: Optional[Tuple[str, str]] = None
    departments: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
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
    section_narratives: List[SectionNarrative] = field(default_factory=list)


# =========================
# LLM Client (Enhanced)
# =========================
class LLMClient:
    """
    High-level client that:
      1) Analyzes a user query into a formal Intent.
      2) Builds a well-structured Narrative for the report.
      3) Generates chart data and section-specific content.
      4) Optionally translates to/from zh-TW using your AWS service.

    Safe to use when OpenAI/AWS are not available (falls back to templates).
    """

    def __init__(
        self,
        llm: Optional[OpenAIService] = None,
        translator: Optional[AWSTranslationService] = None,
        default_lang: str = "en-US",
    ):
        self.llm = llm or OpenAIService(model_name="gpt-4.1", temperature=0.2)
        self.translator = translator or AWSTranslationService()
        self.default_lang = default_lang

    # =============================
    # Public API - Intent Analysis
    # =============================
    def analyze_intent(self, user_query: str, preferred_lang: Optional[str] = None) -> Intent:
        """Turn a raw user query into a normalized Intent."""
        lang = self._normalize_lang(preferred_lang or self.default_lang)

        try:
            detected_lang, _ = self.translator.detect_language(user_query)
            source_lang = self._normalize_lang(detected_lang)
        except Exception:
            source_lang = "en-US"

        try:
            english_query = (
                self.translator.translate_to_english(user_query, source_lang)
                if source_lang != "en-US"
                else user_query
            )
        except Exception:
            english_query = user_query

        if getattr(self.llm, "llm_enabled", False) and self._prompt_layer_ready():
            try:
                intent = self._llm_extract_intent(english_query)
                return intent
            except Exception as e:
                logger.warning("LLM intent extraction failed, falling back. %s", e)

        return self._rule_based_intent(english_query)

    # =============================
    # Public API - Narrative Builder
    # =============================
    def build_narrative(
        self,
        *,
        query: str,
        intent: Intent,
        data_bucket: Dict[str, Any],
        title: Optional[str] = None,
        target_language: Optional[str] = None,
        clarifications: Optional[List[Dict[str, str]]] = None,
        sections_config: Optional[List[Dict[str, Any]]] = None,  # NEW: section configurations
    ) -> Narrative:
        """
        Produce a formal Narrative with:
        - Executive summary, methodology, insights, risks, recommendations
        - Per-section narratives with chart data and insights
        """
        lang = self._normalize_lang(target_language or self.default_lang)
        title = title or self._title_from_intent(intent, lang)

        depts = intent.departments or []
        if depts == ["all"]:
            depts = []

        context: Dict[str, Any] = {
            "query": query,
            "intent": {
                **intent.__dict__,
                "departments": depts,
            },
            "data": data_bucket,
            "clarifications": clarifications or [],
            "sections_config": sections_config or [],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

        # Generate main narrative
        if getattr(self.llm, "llm_enabled", False) and self._prompt_layer_ready():
            try:
                narrative = self._llm_narrative(context, lang, title)
            except Exception as e:
                logger.warning("LLM narrative generation failed, falling back. %s", e)
                narrative = self._template_narrative(context, lang, title)
        else:
            narrative = self._template_narrative(context, lang, title)

        # Generate per-section narratives with chart data
        section_narratives = self._build_section_narratives(
            context, lang, sections_config or []
        )
        narrative.section_narratives = section_narratives

        # Translate if needed
        if lang == "zh-TW":
            narrative = self._maybe_translate_narrative(narrative, "zh-TW")

        return narrative

    # =============================
    # Chart Data Generation (NEW)
    # =============================
    def generate_chart_data(
        self,
        section_id: str,
        data_bucket: Dict[str, Any],
        chart_config: Dict[str, Any]
    ) -> ChartDataset:
        """
        Generate chart-ready data from raw data bucket.
        Extracts metrics, calculates aggregations, generates insights.
        """
        chart_type = chart_config.get("chart_type", "bar")
        title = chart_config.get("title", section_id)
        data_source = chart_config.get("data_source", section_id)

        # Extract raw data from bucket
        raw_data = self._extract_data_from_bucket(data_bucket, data_source)

        # Build data points based on chart type
        if chart_type == "pie":
            data_points = self._build_pie_data(raw_data)
        elif chart_type == "bar":
            data_points = self._build_bar_data(raw_data)
        elif chart_type == "line":
            data_points = self._build_line_data(raw_data)
        elif chart_type == "gauge":
            data_points = self._build_gauge_data(raw_data)
        elif chart_type == "heatmap":
            data_points = self._build_heatmap_data(raw_data)
        else:  # table
            data_points = self._build_table_data(raw_data)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(data_points)

        # Generate data-driven insights
        insights = self._generate_chart_insights(section_id, data_points, summary_stats)

        return ChartDataset(
            chart_id=f"{section_id}_{chart_type}",
            chart_type=chart_type,
            title=title,
            description=chart_config.get("description", ""),
            x_axis_label=chart_config.get("x_axis", ""),
            y_axis_label=chart_config.get("y_axis", ""),
            data_points=data_points,
            summary_stats=summary_stats,
            insights=insights,
        )

    # =============================
    # Private: Chart Data Builders
    # =============================
    def _build_bar_data(self, raw_data: Dict[str, Any]) -> List[ChartDataPoint]:
        """Build bar chart data points."""
        points: List[ChartDataPoint] = []
        
        # Handle common structures: by_department, by_type, etc.
        if "by_department" in raw_data:
            for dept, value in raw_data["by_department"].items():
                points.append(ChartDataPoint(
                    label=dept,
                    value=float(value) if isinstance(value, (int, float)) else 0,
                    metadata={"department": dept}
                ))
        elif isinstance(raw_data, dict):
            for key, value in raw_data.items():
                if isinstance(value, (int, float)):
                    points.append(ChartDataPoint(label=key, value=float(value)))
        
        # Sort by value descending
        points.sort(key=lambda p: p.value, reverse=True)
        return points[:20]  # Limit to top 20

    def _build_pie_data(self, raw_data: Dict[str, Any]) -> List[ChartDataPoint]:
        """Build pie chart data points."""
        points: List[ChartDataPoint] = []
        
        if "by_type" in raw_data:
            for type_name, value in raw_data["by_type"].items():
                points.append(ChartDataPoint(
                    label=type_name,
                    value=float(value) if isinstance(value, (int, float)) else 0,
                    metadata={"type": type_name}
                ))
        
        # Normalize to percentages
        total = sum(p.value for p in points) or 1
        for p in points:
            p.metadata["percentage"] = round((p.value / total) * 100, 1)
        
        return points

    def _build_line_data(self, raw_data: Dict[str, Any]) -> List[ChartDataPoint]:
        """Build line chart data points (time series)."""
        points: List[ChartDataPoint] = []
        
        if "timeline" in raw_data:
            for date_str, value in raw_data["timeline"].items():
                points.append(ChartDataPoint(
                    label=date_str,
                    value=float(value) if isinstance(value, (int, float)) else 0,
                    metadata={"date": date_str}
                ))
        
        return points

    def _build_gauge_data(self, raw_data: Dict[str, Any]) -> List[ChartDataPoint]:
        """Build gauge data (single value 0-100)."""
        # Extract a percentage/health score
        health = raw_data.get("health_score", 75)
        if isinstance(health, dict):
            health = health.get("value", 75)
        
        return [ChartDataPoint(
            label="Health Score",
            value=min(100, max(0, float(health))),
            metadata={"unit": "%"}
        )]

    def _build_heatmap_data(self, raw_data: Dict[str, Any]) -> List[ChartDataPoint]:
        """Build heatmap data (2D grid)."""
        points: List[ChartDataPoint] = []
        
        if "matrix" in raw_data:
            for row_idx, row_data in enumerate(raw_data["matrix"]):
                for col_idx, value in enumerate(row_data):
                    points.append(ChartDataPoint(
                        label=f"({row_idx},{col_idx})",
                        value=float(value) if isinstance(value, (int, float)) else 0,
                        metadata={"row": row_idx, "col": col_idx}
                    ))
        
        return points

    def _build_table_data(self, raw_data: Dict[str, Any]) -> List[ChartDataPoint]:
        """Build table data (flattened rows)."""
        points: List[ChartDataPoint] = []
        
        if "records" in raw_data:
            for i, record in enumerate(raw_data["records"][:50]):  # Limit to 50 rows
                label = record.get("name") or record.get("label") or f"Row {i+1}"
                value = float(record.get("value", record.get("amount", 0)))
                points.append(ChartDataPoint(
                    label=label,
                    value=value,
                    metadata=record
                ))
        
        return points

    # =============================
    # Private: Section Narratives (NEW)
    # =============================
    def _build_section_narratives(
        self,
        context: Dict[str, Any],
        lang: str,
        sections_config: List[Dict[str, Any]]
    ) -> List[SectionNarrative]:
        """Generate per-section narratives with charts and insights."""
        narratives: List[SectionNarrative] = []
        data_bucket = context.get("data", {})
        
        for section_cfg in sections_config:
            section_id = section_cfg.get("id", "unknown")
            section_title = section_cfg.get("title", section_id.replace("_", " ").title())
            
            # Extract section-specific data
            section_data = self._extract_section_data(data_bucket, section_id)
            
            # Generate charts for this section
            charts: List[ChartDataset] = []
            for chart_cfg in section_cfg.get("charts", []):
                try:
                    chart_data = self.generate_chart_data(section_id, section_data, chart_cfg)
                    charts.append(chart_data)
                except Exception as e:
                    logger.warning(f"Failed to generate chart for {section_id}: {e}")
            
            # Generate section narrative
            if getattr(self.llm, "llm_enabled", False):
                try:
                    section_narrative = self._llm_section_narrative(
                        section_id, section_title, section_data, charts, lang
                    )
                except Exception as e:
                    logger.warning(f"LLM section narrative failed, using template: {e}")
                    section_narrative = self._template_section_narrative(
                        section_id, section_title, section_data, charts
                    )
            else:
                section_narrative = self._template_section_narrative(
                    section_id, section_title, section_data, charts
                )
            
            narratives.append(section_narrative)
        
        return narratives

    def _llm_section_narrative(
        self,
        section_id: str,
        section_title: str,
        section_data: Dict[str, Any],
        charts: List[ChartDataset],
        lang: str
    ) -> SectionNarrative:
        """Use LLM to generate section-specific narrative."""
        # Build instruction for LLM
        chart_summary = "\n".join([
            f"- {c.title}: {len(c.data_points)} data points, type={c.chart_type}"
            for c in charts
        ])
        
        instruction = (
            f"Generate a brief narrative for the '{section_title}' report section.\n"
            f"Charts available:\n{chart_summary}\n\n"
            f"Return JSON with: introduction (2-3 sentences), key_findings (list of 3-5 bullets), "
            f"data_summary (1-2 sentences), conclusion (optional)."
        )
        
        question = f"{instruction}\n\nData sample:\n{json.dumps(section_data, ensure_ascii=False)[:1000]}"
        
        raw = self.llm.generate_explanation(
            question=question,
            row_count=len(section_data.get("records", [])),
            columns=list(section_data.keys()),
            aggregates={"chart_count": len(charts)},
            sample_text=json.dumps(section_data)[:500],
        )
        
        payload = _try_extract_json(raw or "")
        if not payload:
            return self._template_section_narrative(section_id, section_title, section_data, charts)
        
        narrative = SectionNarrative(
            section_id=section_id,
            section_title=section_title,
            introduction=str(payload.get("introduction", "")).strip(),
            key_findings=[str(x).strip() for x in payload.get("key_findings", [])],
            data_summary=str(payload.get("data_summary", "")).strip(),
            charts=charts,
            conclusion=str(payload.get("conclusion", "")).strip(),
        )
        
        return narrative

    def _template_section_narrative(
        self,
        section_id: str,
        section_title: str,
        section_data: Dict[str, Any],
        charts: List[ChartDataset]
    ) -> SectionNarrative:
        """Generate template-based section narrative (fallback)."""
        # Heuristic bullets from chart insights
        findings: List[str] = []
        for chart in charts:
            findings.extend(chart.insights[:2])
        
        if not findings:
            findings = [
                f"Data analyzed for {section_title.lower()}.",
                "Key metrics are displayed in the visualizations above.",
            ]
        
        return SectionNarrative(
            section_id=section_id,
            section_title=section_title,
            introduction=f"This section presents an analysis of {section_title.lower()}.",
            key_findings=findings[:5],
            data_summary=f"{len(section_data)} data points analyzed.",
            charts=charts,
            conclusion=f"See charts and recommendations for next steps.",
        )

    # =============================
    # Private: Data Extraction
    # =============================
    def _extract_data_from_bucket(
        self,
        data_bucket: Dict[str, Any],
        data_source: str
    ) -> Dict[str, Any]:
        """Extract relevant data from bucket for a given source."""
        # Try nested paths: leave_metrics.by_department → by_department
        paths = [
            [data_source],
            [f"{data_source}_metrics", "by_type"],
            [f"{data_source}_metrics", "by_department"],
            [data_source, "records"],
        ]
        
        for path in paths:
            result = _safe_dict(data_bucket, path)
            if result:
                return result
        
        return {}

    def _extract_section_data(
        self,
        data_bucket: Dict[str, Any],
        section_id: str
    ) -> Dict[str, Any]:
        """Extract data for a specific section."""
        section_map = {
            "leave_by_department": ["leave_metrics", "by_department"],
            "leave_by_type": ["leave_metrics", "by_type"],
            "balance_snapshot": ["balance_metrics", "records"],
            "overtime_summary": ["ot_metrics"],
            "attendance_rate": ["att_metrics"],
        }
        
        path = section_map.get(section_id, [section_id])
        return _safe_dict(data_bucket, path)

    def _calculate_summary_stats(self, data_points: List[ChartDataPoint]) -> Dict[str, float]:
        """Calculate summary statistics for chart data."""
        if not data_points:
            return {}
        
        values = [p.value for p in data_points]
        return {
            "total": sum(values),
            "average": sum(values) / len(values) if values else 0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    def _generate_chart_insights(
        self,
        section_id: str,
        data_points: List[ChartDataPoint],
        stats: Dict[str, float]
    ) -> List[str]:
        """Generate data-driven insights for a chart."""
        insights: List[str] = []
        
        if not data_points:
            return insights
        
        total = stats.get("total", 0)
        max_val = stats.get("max", 0)
        avg_val = stats.get("average", 0)
        
        # Top category
        top = max(data_points, key=lambda p: p.value, default=None)
        if top and total > 0:
            pct = (top.value / total) * 100
            insights.append(f"Leading category: {top.label} ({pct:.1f}% of total)")
        
        # Outliers
        outliers = [p for p in data_points if p.value > avg_val * 1.5]
        if outliers:
            insights.append(f"{len(outliers)} category(ies) exceed average by 50%+")
        
        # Distribution shape
        if len(data_points) > 3:
            ratio = max([p.value for p in data_points[-3:]]) / (max_val or 1)
            if ratio < 0.2:
                insights.append("Highly concentrated distribution (top tier dominates)")
            elif ratio > 0.7:
                insights.append("Even distribution across categories")
        
        return insights[:3]

    # =============================
    # Existing methods (unchanged)
    # =============================
    def _llm_extract_intent(self, english_query: str) -> Intent:
        """Ask the LLM to return a strict JSON intent."""
        system = (
            "You are an HR analytics planner. Extract a strict JSON object describing the intent.\n"
            "Fields: report_type, time_period, time_range, departments, metrics, confidence, "
            "needs_clarification, clarification_questions."
        )

        try:
            from langchain.schema import SystemMessage, HumanMessage
        except Exception:
            return self._rule_based_intent(english_query)

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Query: {english_query}\nReturn ONLY JSON."),
        ]
        raw = self.llm._invoke_llm(messages)
        if not raw:
            return self._rule_based_intent(english_query)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").split("\n", 1)[-1]

        obj = _try_extract_json(cleaned)
        if not obj:
            return self._rule_based_intent(english_query)

        tr = obj.get("time_range") or {}
        start = tr.get("start")
        end = tr.get("end")
        return Intent(
            report_type=(obj.get("report_type") or "unknown"),
            time_period=(obj.get("time_period") or "monthly"),
            time_range=(start, end) if (start and end) else None,
            departments=list(obj.get("departments") or []),
            metrics=list(obj.get("metrics") or []),
            confidence=float(obj.get("confidence") or 0.75),
            needs_clarification=bool(obj.get("needs_clarification") or False),
            clarification_questions=list(obj.get("clarification_questions") or []),
        )

    def _rule_based_intent(self, english_query: str) -> Intent:
        """Fallback rule-based intent extraction."""
        q = english_query.lower()
        
        if any(w in q for w in ["leave", "vacation", "time off"]):
            rt = "leave_analysis"
            metrics = ["total_leave_days", "by_department", "by_type"]
        elif any(w in q for w in ["overtime", "extra hours", "ot"]):
            rt = "overtime_analysis"
            metrics = ["total_overtime_hours", "by_department"]
        elif any(w in q for w in ["attendance", "present", "absent"]):
            rt = "attendance_analysis"
            metrics = ["attendance_rate", "patterns"]
        elif any(w in q for w in ["balance", "remaining", "accrued"]):
            rt = "balance_report"
            metrics = ["vacation_balance", "sick_balance"]
        else:
            rt = "unknown"
            metrics = []
        
        tp = "weekly" if "week" in q else "quarterly" if "quarter" in q else "yearly" if "year" in q else "monthly"
        
        return Intent(
            report_type=rt,
            time_period=tp,
            departments=[],
            metrics=metrics,
            confidence=0.8,
            needs_clarification=(rt == "unknown"),
        )

    def _llm_narrative(self, context: Dict[str, Any], lang: str, title: str) -> Narrative:
        """Generate narrative using LLM."""
        # Simplified for space - reuse your existing implementation
        return self._template_narrative(context, lang, title)

    def _template_narrative(self, context: Dict[str, Any], lang: str, title: str) -> Narrative:
        """Generate template-based narrative (fallback)."""
        return Narrative(
            language=lang,
            title=title,
            executive_summary="This report provides a comprehensive analysis of the selected metrics.",
            methodology="Data was extracted and analyzed using aggregation techniques.",
            key_insights=["Key insight 1", "Key insight 2"],
            risks=["Potential data gaps"],
            recommendations=["Review results with stakeholders"],
        )

    def _maybe_translate_narrative(self, n: Narrative, target_lang: str) -> Narrative:
        """Translate narrative if needed."""
        if target_lang == "en-US":
            return n
        
        try:
            return Narrative(
                language=target_lang,
                title=self.translator.translate_from_english(n.title, target_lang),
                executive_summary=self.translator.translate_from_english(n.executive_summary, target_lang),
                methodology=self.translator.translate_from_english(n.methodology, target_lang),
                key_insights=[self.translator.translate_from_english(x, target_lang) for x in n.key_insights],
                risks=[self.translator.translate_from_english(x, target_lang) for x in n.risks],
                recommendations=[self.translator.translate_from_english(x, target_lang) for x in n.recommendations],
                appendix_notes=n.appendix_notes,
                section_narratives=n.section_narratives,
            )
        except Exception as e:
            logger.warning("Translation failed: %s", e)
            return n

    @staticmethod
    def _normalize_lang(code: Optional[str]) -> str:
        try:
            return AWSTranslationService.normalize_language_code(code or "en-US")
        except Exception:
            return (code or "en-US")

    @staticmethod
    def _title_from_intent(intent: Intent, lang: str) -> str:
        """Derive a human title from the intent."""
        tp = (intent.time_period or "period").title()
        mapping_en = {
            "leave_analysis": f"{tp} Leave Analysis Report",
            "overtime_analysis": f"{tp} Overtime Summary Report",
            "attendance_analysis": f"{tp} Attendance Analysis Report",
            "balance_report": "Employee Balance Summary Report",
            "unknown": "Custom HR Analytics Report",
        }
        title_en = mapping_en.get(intent.report_type or "unknown", "HR Report")
        if lang == "zh-TW":
            try:
                return AWSTranslationService().translate_from_english(title_en, "zh-TW")
            except Exception:
                return title_en
        return title_en


# =========================
# Helpers (module-level)
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
    """Safely navigate nested dict and return int."""
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
    """Safely navigate nested dict and return dict."""
    cur: Any = obj
    for p in path:
        cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            return {}
    return cur if isinstance(cur, dict) else {}