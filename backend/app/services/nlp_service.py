# backend/app/services/nlp_service.py
from __future__ import annotations

import re
import time
import logging
from typing import Dict, Any, Optional, List, Tuple

from app.services.db_service import SQLServerDatabaseService

# Import our specialized services
from .aws.translation_service import AWSTranslationService

# from .llm.openai_service import OpenAIService

#from .llm.retry_llm_service import OpenAIService
from app.services.llm.openai_service import OpenAIService

from app.services.db_service import (
    DatabaseQueryError,
    DatabaseConnectionError,
    DatabaseTimeoutError,
    PermissionDeniedError,
    DeadlockError,
)

from .data_processing.data_analyzer import DataAnalyzer
from .data_processing.date_processor import DateProcessor
from .data_processing.sql_templates import SQLTemplateService
from .data_processing.sql_executor import SQLExecutor
from .data_processing.person_enrichment import PersonEnrichmentService
from .retrieval.vector_search_service import VectorSearchService
from .helpers.data_utils import jsonable_value, normalize_sql_columns, format_sample_data

from backend.app.services.memory.simple_query_memory import QueryMemoryService

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
        self.llm_service =  OpenAIService(model_name="gpt-4o-mini", temperature=0.1) # OpenAIService(model_name, temperature)
        self.data_analyzer = DataAnalyzer()
        self.date_processor = DateProcessor()
        self.sql_template_service = SQLTemplateService()
        self.sql_executor = SQLExecutor(db_service)
        self.person_enrichment = PersonEnrichmentService(db_service)
        self.vector_search = VectorSearchService(db_service)
        self.memory = QueryMemoryService(db_service)

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

    def _rewrite_followup_with_context(self, english: str, session_id: str) -> str:
        """
        Replace vague references like 'this department' with concrete values from prior results.
        Falls back to LLM rewrite for tricky cases.
        """
        out = english

        # 1) Heuristic replacement using memory snapshot
        dept = self.memory.get_last_focus_value(session_id, ["Department", "DEPARTMENT", "DeptName", "部門"])
        if dept:
            out = re.sub(r"\b(this|that|the)\s+department\b", dept, out, flags=re.I)

        # 2) Optional: quick LLM rewrite to fully specify pronouns (best-effort)
        try:
            if self.llm_service and getattr(self.llm_service, "llm_enabled", False):
                from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
                rewriter = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        "You rewrite short analytics questions into fully specified English, "
                        "replacing vague pronouns with concrete values from context."
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "Question: {question}\n\n"
                        "Context (last columns and a few preview rows):\n{context}\n\n"
                        "Return only the rewritten question. No commentary."
                    ),
                ])
                ctx_txt = ""
                snap = self.memory.session_cache.get(session_id)
                if snap and snap.successful_results:
                    last = snap.successful_results[-1]
                    ctx_txt = f"Columns: {', '.join(last.get('columns') or [])}\nPreview: {last.get('preview')}"
                msgs = rewriter.format_messages(question=out, context=ctx_txt)
                new_q = self.llm_service._invoke_llm(msgs)
                if new_q:
                    return new_q.strip()
        except Exception:
            pass

        return out


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
        session_id = rid or "default"

        try:
            # 1) Language & dates normalization
            lang, conf = self.translation_service.detect_language(user_input)
            english = self.translation_service.translate_to_english(user_input, lang)
            english = self.date_processor.rewrite_relative_dates(english)
            logger.info("rid=%s query=%r lang=%s conf=%.2f", rid, user_input, lang, conf)

            # 2) Retrieval (tables, hints, schema)
            rel_with_scores = self.vector_search.find_relevant_tables(
                english, schema_filter=schema_name, rid=rid
            )
            rel_tables = [t for (t, _) in rel_with_scores]
            join_hints = self.vector_search.get_join_hints(rel_tables)
            schema_ctx = self.vector_search.get_schema_context(rel_tables)

            # 3) Follow-up grounding
            english_grounded = self._rewrite_followup_with_context(english, session_id)

            # 4) Try memory first
            cached_sql, cached_conf = self.memory.check_memory_for_query(
                english_grounded, rel_tables, session_id=session_id
            )

            final_sql = ""
            llm_attempts = 0
            rows: List[Tuple[Any, ...]] = []
            columns: List[str] = []
            execution_error: Optional[str] = None

            # 5) Execute (cache → llm-repair → fallback)
            exec_t0 = time.perf_counter()
            try:
                if cached_sql:
                    final_sql = normalize_sql_columns(cached_sql)
                    rows, columns = self.db_service.run_select(final_sql, max_rows=1000, query_timeout=10)
                    llm_attempts = 0  # pure cache path
                else:
                    if rel_tables:
                        # LLM with self-healing repair
                        rows, columns, final_sql, llm_attempts = self.llm_service.run_query_with_llm_repair(
                            db_service=self.db_service,
                            user_question=english_grounded,
                            schema=schema_ctx,
                            join_hints=join_hints,
                            params=None,
                            max_rows=1000,
                            query_timeout=10,
                            max_attempts=3,
                        )
                        final_sql = normalize_sql_columns(final_sql)
                    else:
                        # No tables → deterministic template
                        alt = self.sql_template_service.get_fallback_sql(english_grounded)
                        final_sql = normalize_sql_columns(alt or "SELECT 1 WHERE 1=0")
                        rows, columns = self.db_service.run_select(final_sql, max_rows=1000, query_timeout=10)
            except Exception as e:
                execution_error = str(e)

            exec_ms = int((time.perf_counter() - exec_t0) * 1000)

            # 6) Learn + record session snapshot
            if execution_error is None:
                self.memory.learn_from_query(
                    english_query=english_grounded,
                    relevant_tables=rel_tables,
                    generated_sql=final_sql,
                    success=True,
                    execution_time=exec_ms / 1000.0,
                    session_id=session_id,
                )
                self.memory.record_success(
                    session_id=session_id,
                    english_query=english_grounded,
                    generated_sql=final_sql,
                    columns=columns,
                    rows=rows,
                    relevant_tables=rel_tables,
                    schema_ctx=schema_ctx,
                )
            else:
                self.memory.learn_from_query(
                    english_query=english_grounded,
                    relevant_tables=rel_tables,
                    generated_sql=final_sql or "",
                    success=False,
                    execution_time=exec_ms / 1000.0,
                    session_id=session_id,
                )

            # 7) Analysis & explanation (only on success)
            if execution_error:
                explanation_en = f"Query execution failed: {execution_error}"
                table_md = ""
            else:
                aggregates = self.data_analyzer.compute_aggregates(rows, columns)
                sample_text = format_sample_data(rows, columns)

                explanation_en = self.llm_service.generate_explanation(
                    english_grounded, len(rows), columns, aggregates, sample_text
                )

                want_details = any(k in english_grounded.lower() for k in (
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
                    explanation_en = explanation_en.strip() + "\n\n**Preview (first 20 rows):**\n\n" + table_md

            # 8) Localize final summary
            localized_explanation = self.translation_service.translate_from_english(
                explanation_en, lang
            )

            # 9) Response
            stats = self.memory.get_memory_stats()
            response = {
                "original_text": user_input,
                "detected_language": lang,
                "language_confidence": conf,
                "english_text": english_grounded,
                "intent": "generic",
                "schema": schema_name,
                "relevant_tables": [{"table": t, "score": round(s, 3)} for (t, s) in rel_with_scores],
                "generated_sql": final_sql or "SELECT 1 WHERE 1=0",
                "llm_attempts": llm_attempts,
                "execution_successful": execution_error is None,
                "execution_error": execution_error,
                "columns": columns,
                "results": [[jsonable_value(v) for v in r] for r in rows],
                "row_count": len(rows),
                "resolved_people": self.person_enrichment.enrich_people_data(rows, columns),
                "columns_enriched": columns,
                "results_enriched": [[jsonable_value(v) for v in r] for r in rows],
                "table_markdown": table_md if execution_error is None else "",
                "explanation_english": explanation_en,
                "explanation_localized": localized_explanation,
                "summary": localized_explanation,
                "success": execution_error is None,
                # ✅ Memory telemetry
                "memory": {
                    "session_id": session_id,
                    "used_cached_sql": bool(cached_sql),
                    "cached_confidence": float(cached_conf) if cached_sql else 0.0,
                    "cache_hit_rate": stats.get("cache_hit_rate"),
                },
            }

            logger.info("rid=%s pipeline ok ms=%d", rid, int((time.perf_counter() - t0) * 1000))
            return response

        except Exception as e:
            logger.error("rid=%s pipeline failed after %dms: %s: %s",
                        rid, int((time.perf_counter() - t0) * 1000), type(e).__name__, e, exc_info=True)
            msg = "An error occurred while processing your query."
            return {
                "original_text": user_input,
                "execution_successful": False,
                "execution_error": str(e),
                "summary": msg,
                "explanation_localized": msg,
                "success": False,
            }

