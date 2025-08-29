# backend/app/services/nlp_service.py
from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional

from app.services.db_service import SQLServerDatabaseService

# Import our specialized services
from .aws.translation_service import AWSTranslationService
from .llm.openai_service import OpenAIService
from .data_processing.data_analyzer import DataAnalyzer
from .data_processing.date_processor import DateProcessor
from .data_processing.sql_templates import SQLTemplateService
from .data_processing.sql_executor import SQLExecutor
from .data_processing.person_enrichment import PersonEnrichmentService
from .retrieval.vector_search_service import VectorSearchService
from .helpers.data_utils import jsonable_value, normalize_sql_columns, format_sample_data

logger = logging.getLogger(__name__)


def _ms(t0: float) -> int:
    """Calculate milliseconds elapsed since timestamp."""
    return int((time.perf_counter() - t0) * 1000)


class NLPService:
    """
    Main orchestrator for natural language processing pipeline.
    """

    def __init__(self, db_service: SQLServerDatabaseService, model_name: str = "gpt-4o-mini",
                 temperature: float = 0.1, **_):
        self.db_service = db_service

        # Component services
        self.translation_service = AWSTranslationService()
        self.llm_service = OpenAIService(model_name, temperature)
        self.data_analyzer = DataAnalyzer()
        self.date_processor = DateProcessor()
        self.sql_template_service = SQLTemplateService()
        self.sql_executor = SQLExecutor(db_service)
        self.person_enrichment = PersonEnrichmentService(db_service)
        self.vector_search = VectorSearchService(db_service)

        self._initialize_data_anchor()

    def _initialize_data_anchor(self):
        """Initialize the data anchor (latest date in dataset)."""
        try:
            rows, cols = self.db_service.run_select(
                "SELECT CONVERT(varchar(10), MAX(CAST(WORKDATE AS date)), 23) FROM dbo.ATDLEAVEDATA"
            )
            if rows and rows[0][0]:
                data_anchor = str(rows[0][0])  # e.g., '2023-10-06'
                self.date_processor.set_data_anchor(data_anchor)
                logger.info("Data anchor (latest WORKDATE) = %s", data_anchor)
        except Exception as e:
            logger.warning("Could not determine data anchor: %s", e)

    @property
    def person_table(self) -> str:
        return self.vector_search.person_table

    def vector_status(self) -> Dict[str, Any]:
        return self.vector_search.health_check()

    # ---------- NEW: pure helper for inline preview table ----------
    @staticmethod
    def _markdown_table(columns, rows, limit: int = 20, keep=None) -> str:
        cols = [c for c in columns or []]
        if not rows or not cols:
            return ""
        # project to selected columns if requested
        proj_rows = []
        if keep:
            low = {c.lower(): i for i, c in enumerate(cols)}
            wanted = []
            for k in keep:
                i = low.get(k.lower())
                if i is not None:
                    wanted.append((cols[i], i))
            if wanted:
                cols = [w[0] for w in wanted]
                idxs = [w[1] for w in wanted]
                for r in rows[:limit]:
                    proj_rows.append([("" if i >= len(r) or r[i] is None else str(r[i])) for i in idxs])
            else:
                for r in rows[:limit]:
                    proj_rows.append([("" if v is None else str(v)) for v in r])
        else:
            for r in rows[:limit]:
                proj_rows.append([("" if v is None else str(v)) for v in r])

        if not proj_rows:
            return ""

        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [header, sep]
        for r in proj_rows:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    def process_complete_query(self, user_input: str, schema_name: Optional[str] = "dbo",
                               rid: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            # 1) Detect language & translate to English
            lang, conf = self.translation_service.detect_language(user_input)
            english = self.translation_service.translate_to_english(user_input, lang)
            english = self.date_processor.rewrite_relative_dates(english)
            logger.info("rid=%s query=%r lang=%s conf=%.2f", rid, user_input, lang, conf)

            # 2) Retrieval
            rel_with_scores = self.vector_search.find_relevant_tables(
                english, schema_filter=schema_name, rid=rid
            )
            rel_tables = [t for (t, _) in rel_with_scores]
            join_hints = self.vector_search.get_join_hints(rel_tables)
            schema_ctx = self.vector_search.get_schema_context(rel_tables)

            # 3) SQL generation (LLM). If no tables => no attempt
            sql_raw = ""
            if rel_tables:
                sql_raw = self.llm_service.generate_sql(english, schema_ctx, join_hints)
            sql_raw = normalize_sql_columns(sql_raw)

            # 4) Fallback template if needed
            # Let SQLExecutor guard; don't call its _guard_sql here again.
            if not sql_raw or sql_raw.strip().lower() == "select 1 where 1=0":
                alt = self.sql_template_service.get_fallback_sql(english)
                sql_raw = alt or "SELECT 1 WHERE 1=0"

            # 5) Execute safely (guarded inside)
            rows, columns, execution_error = self.sql_executor.guard_and_execute_sql(sql_raw, rid)

            # 6) Enrich (mapping only; rows unchanged)
            resolved_people = self.person_enrichment.enrich_people_data(rows, columns)

            # 7) Analysis & explanation
            if execution_error:
                explanation_en = f"Query execution failed: {execution_error}"
                table_md = ""
            else:
                aggregates = self.data_analyzer.compute_aggregates(rows, columns)
                sample_text = format_sample_data(rows, columns)

                # LLM summary (or deterministic fallback inside OpenAIService)
                explanation_en = self.llm_service.generate_explanation(
                    english, len(rows), columns, aggregates, sample_text
                )

                # ---------- NEW: add a human-readable preview when it makes sense ----------
                want_details = any(k in english.lower() for k in (
                    "name", "names", "employee id", "employee ids",
                    "list", "show", "sample", "detail", "details", "who"
                ))
                preferred_cols = [
                    "Name", "EmployeeID",
                    "ATTENDANCETYPE", "LEAVETYPE",
                    "HOURS", "StartDate", "WORKDATE", "EndDate"
                ]
                table_md = self._markdown_table(columns, rows, limit=20,
                                                keep=preferred_cols if want_details else None)

                if table_md:
                    # Append under the bullets so the UI (which prints summary) shows the table.
                    explanation_en = explanation_en.strip()
                    explanation_en += "\n\n**Preview (first 20 rows):**\n\n" + table_md

            # 8) Localize AFTER adding preview so bullets are translated too
            localized_explanation = self.translation_service.translate_from_english(explanation_en, lang)

            # 9) Build response
            response = {
                "original_text": user_input,
                "detected_language": lang,
                "language_confidence": conf,
                "english_text": english,
                "intent": "generic",
                "schema": schema_name,
                "relevant_tables": [{"table": t, "score": round(s, 3)} for (t, s) in rel_with_scores],
                "generated_sql": sql_raw or "SELECT 1 WHERE 1=0",
                "execution_successful": execution_error is None,
                "execution_error": execution_error,
                "columns": columns,
                "results": [[jsonable_value(v) for v in r] for r in rows],
                "row_count": len(rows),
                "resolved_people": resolved_people,
                "columns_enriched": columns,
                "results_enriched": [[jsonable_value(v) for v in r] for r in rows],
                "table_markdown": table_md if execution_error is None else "",
                "explanation_english": explanation_en,
                "explanation_localized": localized_explanation,
                "summary": localized_explanation,
                "success": execution_error is None,
            }

            logger.info("rid=%s pipeline ok ms=%d", rid, _ms(t0))
            return response

        except Exception as e:
            logger.error("rid=%s pipeline failed after %dms: %s: %s",
                         rid, _ms(t0), type(e).__name__, e, exc_info=True)
            msg = "An error occurred while processing your query."
            return {
                "original_text": user_input,
                "detected_language": self.translation_service.last_language or "en-US",
                "language_confidence": self.translation_service.last_confidence or 0.0,
                "execution_successful": False,
                "execution_error": str(e),
                "summary": msg,
                "explanation_localized": msg,
                "success": False,
            }
