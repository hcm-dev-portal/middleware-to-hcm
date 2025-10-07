# backend/app/services/nlp_service_v2.py
from __future__ import annotations

import re
import time
import logging
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Literal

from app.services.db_service import SQLServerDatabaseService

# Import our specialized services
from .aws.translation_service import AWSTranslationService
from app.services.llm.openai_service import UnifiedBilingualOpenAIService

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

from app.services.memory.simple_query_memory import SimpleQueryMemoryService

# NEW: Chart agent and base viz service
from .chart.chart_agent import ChartVisualizationAgent, ChartVisualizationAgentConfig
from .chart.visualization_service import VisualizationService

logger = logging.getLogger(__name__)


def _ms(t0: float) -> int:
    """Calculate milliseconds elapsed since timestamp."""
    return int((time.perf_counter() - t0) * 1000)


def detect_language_simple(text: str) -> Literal["zh-tw", "en"]:
    """
    Simple language detection focusing on Chinese characters.
    Returns 'zh-tw' for Chinese, 'en' for English.
    """
    if not text or not text.strip():
        return "en"

    # Count Chinese characters
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    total_chars = len([c for c in text if c.isalnum()])

    if total_chars == 0:
        return "en"

    # If more than 30% Chinese characters, assume Chinese
    chinese_ratio = chinese_chars / total_chars
    return "zh-tw" if chinese_ratio > 0.3 else "en"


class LanguageAwareDateProcessor:
    """Enhanced date processor that handles both English and Chinese temporal expressions."""

    def __init__(self):
        self.base_processor = DateProcessor()

        # Chinese temporal patterns (expanded)
        self.zh_patterns = {
            r"今天": "today",
            r"昨日|昨天": "yesterday",
            r"明天|隔天": "tomorrow",
            r"這個月|本月": "this month",
            r"上個月|上月": "last month",
            r"下個月|下月": "next month",
            r"這週|本週|本周": "this week",
            r"上週|上周": "last week",
            r"下週|下周": "next week",
            r"本季|這季": "this quarter",
            r"上季": "last quarter",
            r"今年": "this year",
            r"去年": "last year",
            r"明年": "next year",
        }

    def set_data_anchor(self, anchor_date: str):
        """Set the data anchor for both processors."""
        self.base_processor.set_data_anchor(anchor_date)

    def rewrite_relative_dates(self, text: str, lang: Literal["zh-tw", "en"]) -> str:
        """Process relative dates in both English and Chinese."""
        if lang == "zh-tw":
            # First convert Chinese temporal expressions to English
            result = text
            for zh_pattern, en_equivalent in self.zh_patterns.items():
                result = re.sub(zh_pattern, en_equivalent, result)

            # Then apply English date processing
            return self.base_processor.rewrite_relative_dates(result)
        else:
            return self.base_processor.rewrite_relative_dates(text)


class LanguageAwareMemoryService:
    """Memory service that stores and matches queries in their original language."""

    def __init__(self, base_memory: SimpleQueryMemoryService):
        self.base_memory = base_memory

    def check_memory_for_query(
        self,
        original_query: str,
        english_query: str,
        relevant_tables: List[str],
        lang: Literal["zh-tw", "en"],
        session_id: str = "default",
    ) -> Tuple[Optional[str], float]:
        """Check memory using original language query as primary key."""
        # Try original language first
        cached_sql, conf = self.base_memory.check_memory_for_query(
            original_query, relevant_tables, session_id=session_id
        )

        # Fallback to English if no match and languages differ
        if not cached_sql and lang == "zh-tw":
            cached_sql, conf = self.base_memory.check_memory_for_query(
                english_query, relevant_tables, session_id=session_id
            )

        return cached_sql, conf

    def learn_from_query(
        self,
        original_query: str,
        english_query: str,
        relevant_tables: List[str],
        generated_sql: str,
        success: bool,
        execution_time: float,
        lang: Literal["zh-tw", "en"],
        session_id: str = "default",
    ):
        """Learn from both original and English queries."""
        # Always store the original query - FIXED INTERFACE
        self.base_memory.learn_from_query(
            query=original_query,  # Fixed: use 'query' not 'english_query'
            relevant_tables=relevant_tables,
            generated_sql=generated_sql,
            success=success,
            execution_time=execution_time,
            session_id=session_id,
        )

        # Also store English version if different (for cross-language lookup)
        if lang == "zh-tw" and original_query != english_query:
            self.base_memory.learn_from_query(
                query=english_query,  # Fixed: use 'query' not 'english_query'
                relevant_tables=relevant_tables,
                generated_sql=generated_sql,
                success=success,
                execution_time=execution_time,
                session_id=session_id,
            )

    def record_success(
        self,
        session_id: str,
        original_query: str,
        english_query: str,
        generated_sql: str,
        columns: List[str],
        rows: List[Tuple],
        relevant_tables: List[str],
        schema_ctx: str,
        lang: Literal["zh-tw", "en"],
    ):
        """Record successful execution in original language."""
        self.base_memory.record_success(
            session_id=session_id,
            query=original_query,  # Fixed: use 'query' not 'english_query'
            generated_sql=generated_sql,
            columns=columns,
            rows=rows,
            relevant_tables=relevant_tables,
            schema_ctx=schema_ctx,
        )

    def get_last_focus_value(self, session_id: str, column_patterns: List[str]) -> Optional[str]:
        """Delegate to base memory."""
        return self.base_memory.get_last_focus_value(session_id, column_patterns)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Delegate to base memory."""
        return self.base_memory.get_memory_stats()


class LanguageAwareContextRewriter:
    """Handles context rewriting for both Chinese and English followup queries."""

    def __init__(self, memory_service: LanguageAwareMemoryService, llm_service: UnifiedBilingualOpenAIService):
        self.memory_service = memory_service
        self.llm_service = llm_service

    def rewrite_followup_with_context(self, query: str, lang: Literal["zh-tw", "en"], session_id: str) -> str:
        """Replace vague references with concrete values from prior results."""
        result = query

        # Get context from memory
        dept = self.memory_service.get_last_focus_value(session_id, ["Department", "DEPARTMENT", "DeptName", "部門"])

        if dept:
            if lang == "zh-tw":
                # Chinese pronoun replacement
                result = re.sub(r"(這個|那個|該)\s*部門", dept, result)
                result = re.sub(r"(這個|那個|該)\s*單位", dept, result)
            else:
                # English pronoun replacement
                result = re.sub(r"\b(this|that|the)\s+department\b", dept, result, flags=re.I)
                result = re.sub(r"\b(this|that|the)\s+unit\b", dept, result, flags=re.I)

        # Optional LLM-based context resolution for complex cases
        if self._needs_llm_rewrite(result, lang):
            result = self._llm_rewrite_with_context(result, lang, session_id)

        return result

    def _needs_llm_rewrite(self, query: str, lang: Literal["zh-tw", "en"]) -> bool:
        """Determine if query needs LLM-based context resolution."""
        if lang == "zh-tw":
            vague_indicators = ["這個", "那個", "它", "他們", "該", "前面", "剛才"]
        else:
            vague_indicators = ["this", "that", "it", "they", "these", "those", "previous"]

        return any(indicator in query.lower() for indicator in vague_indicators)

    def _llm_rewrite_with_context(self, query: str, lang: Literal["zh-tw", "en"], session_id: str) -> str:
        """Use LLM to resolve context with language-appropriate prompting."""
        try:
            # Get context from recent session
            snap = self.memory_service.base_memory.session_cache.get(session_id)
            context_info = ""

            if snap and getattr(snap, "successful_results", None):
                last_result = snap.successful_results[-1]
                cols = last_result.get("columns", [])
                preview = last_result.get("preview", "")
                context_info = f"Columns: {', '.join(cols[:5])}\nSample: {str(preview)[:200]}"

            if lang == "zh-tw":
                system_prompt = (
                    "你是資料分析助理。請將含糊的查詢改寫為具體明確的問題，"
                    "使用上下文中的具體值替換代詞和模糊引用。只回傳改寫後的問題，不要其他說明。"
                )
                user_prompt = f"問題：{query}\n\n上下文：{context_info}\n\n改寫後的問題："
            else:
                system_prompt = (
                    "You rewrite vague analytics questions into fully specified English, "
                    "replacing pronouns with concrete values from context. "
                    "Return only the rewritten question."
                )
                user_prompt = f"Question: {query}\n\nContext: {context_info}\n\nRewritten:"

            rewritten = self.llm_service._simple_completion(system_prompt, user_prompt)  # type: ignore
            return rewritten.strip() if rewritten else query

        except Exception as e:
            logger.warning(f"LLM context rewrite failed: {e}")
            return query


# -------------------------------
# NEW: Chart intent helpers (zh/en)
# -------------------------------
_ZH_CHART_KEYWORDS = [
    "圖",
    "圖表",
    "視覺化",
    "繪圖",
    "趨勢圖",
    "長條圖",
    "柱狀圖",
    "折線圖",
    "圓餅圖",
    "直方圖",
    "散點圖",
    "箱型圖",
    "箱形圖",
    "熱圖",
    "熱力圖",
    "面積圖",
]
_EN_CHART_KEYWORDS = [
    "chart",
    "plot",
    "graph",
    "visualize",
    "visualisation",
    "visualization",
    "trend",
    "line chart",
    "bar chart",
    "pie",
    "histogram",
    "scatter",
    "box",
    "heatmap",
    "area",
]

_ZH_FORCED_TYPE = {
    "長條圖": "bar_chart",
    "柱狀圖": "bar_chart",
    "折線圖": "line_chart",
    "圓餅圖": "pie_chart",
    "直方圖": "histogram",
    "散點圖": "scatter_plot",
    "箱型圖": "box_plot",
    "箱形圖": "box_plot",
    "熱圖": "heatmap",
    "熱力圖": "heatmap",  # fixed
    "面積圖": "area_chart",
    "趨勢圖": "line_chart",
}
_EN_FORCED_TYPE = {
    "bar chart": "bar_chart",
    "line chart": "line_chart",
    "pie": "pie_chart",
    "pie chart": "pie_chart",
    "histogram": "histogram",
    "scatter": "scatter_plot",
    "scatter plot": "scatter_plot",
    "box": "box_plot",
    "box plot": "box_plot",
    "heatmap": "heatmap",
    "area": "area_chart",
    "area chart": "area_chart",
}


def _infer_forced_chart_type(query: str, lang: Literal["zh-tw", "en"]) -> Optional[str]:
    q = (query or "").lower()
    if lang == "zh-tw":
        for k, v in _ZH_FORCED_TYPE.items():
            if k in query:
                return v
    else:
        for k, v in _EN_FORCED_TYPE.items():
            if k in q:
                return v
    return None


def _should_generate_chart(query: str, lang: Literal["zh-tw", "en"], columns: List[str]) -> bool:
    q = (query or "").lower()
    if lang == "zh-tw":
        if any(kw in query for kw in _ZH_CHART_KEYWORDS):
            return True
    else:
        if any(kw in q for kw in _EN_CHART_KEYWORDS):
            return True
    # fallback heuristic: if we have >=2 columns, charts are often useful
    return len(columns) >= 2


class LanguageNativeNLPService:
    """
    Language-native NLP orchestrator that processes queries in their original language
    without translation bottlenecks. Now augmented with an AI chart visualization agent.
    """

    def __init__(self, db_service: SQLServerDatabaseService, model_name: str = "gpt-4o-mini", temperature: float = 0.1, **_):
        self.db_service = db_service

        # Core services
        self.translation_service = AWSTranslationService()  # Only for fallback
        self.llm_service = UnifiedBilingualOpenAIService(model_name=model_name, temperature=temperature)
        self.data_analyzer = DataAnalyzer()
        self.sql_template_service = SQLTemplateService()
        self.sql_executor = SQLExecutor(db_service)
        self.person_enrichment = PersonEnrichmentService(db_service)

        # Enhanced language-aware services
        self.date_processor = LanguageAwareDateProcessor()
        self.vector_search = VectorSearchService(db_service)
        self.memory = LanguageAwareMemoryService(SimpleQueryMemoryService())
        self.context_rewriter = LanguageAwareContextRewriter(self.memory, self.llm_service)

        # NEW: Visualization services
        self.viz_service = VisualizationService()  # deterministic builder & saver
        self.chart_agent = ChartVisualizationAgent(
            viz_service=self.viz_service,
            cfg=ChartVisualizationAgentConfig(
                model="gpt-4o-mini",
                temperature=0.0,
                theme="plotly_white",
                use_ai=True,  # set False to disable LLM planning
            ),
        )

        # Only render charts when the route asks
        self.enable_auto_visualization = False

        self._initialize_data_anchor()

    @property
    def viz(self):
        return self.viz_service

    def _initialize_data_anchor(self):
        """Initialize the data anchor (latest date in dataset)."""
        try:
            rows, cols = self.db_service.run_select(
                "SELECT CONVERT(varchar(10), MAX(CAST(WORKDATE AS date)), 23) FROM dbo.ATDLEAVEDATA"
            )
            if rows and rows[0][0]:
                data_anchor = str(rows[0][0])
                self.date_processor.set_data_anchor(data_anchor)
                logger.info("Data anchor (latest WORKDATE) = %s", data_anchor)
        except Exception as e:
            logger.warning("Could not determine data anchor: %s", e)

    @property
    def person_table(self) -> str:
        return self.vector_search.person_table

    def vector_status(self) -> Dict[str, Any]:
        return self.vector_search.health_check()

    def _markdown_table(self, columns, rows, limit: int = 20, keep=None) -> str:
        """Generate markdown table for results preview."""
        cols = [c for c in columns or []]
        if not rows or not cols:
            return ""

        # Project to selected columns if requested
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

    def _should_show_details(self, query: str, lang: Literal["zh-tw", "en"]) -> bool:
        """Determine if user wants detailed results based on query language."""
        if lang == "zh-tw":
            detail_indicators = ["姓名", "員工", "員工編號", "列表", "顯示", "樣本", "詳細", "明細", "誰", "哪些人", "具體", "清單"]
        else:
            detail_indicators = [
                "name",
                "names",
                "employee id",
                "employee ids",
                "list",
                "show",
                "sample",
                "detail",
                "details",
                "who",
            ]

        return any(indicator in query.lower() for indicator in detail_indicators)

    def _get_language_aware_explanation(
        self, query: str, lang: Literal["zh-tw", "en"], row_count: int, columns: List[str], aggregates: Dict, sample_text: str
    ) -> str:
        """Generate explanation in the original query language."""
        logger.debug("EXPLANATION_LANGUAGE_ROUTING: query='%s' lang=%s", query[:50], lang)

        if lang == "zh-tw":
            explanation = self.llm_service.generate_explanation_chinese(query, row_count, columns, aggregates, sample_text)
            logger.debug("EXPLANATION_CHINESE_GENERATED: length=%d", len(explanation))
            return explanation
        else:
            explanation = self.llm_service.generate_explanation(query, row_count, columns, aggregates, sample_text)
            logger.debug("EXPLANATION_ENGLISH_GENERATED: length=%d", len(explanation))
            return explanation

    def process_complete_query(
        self,
        user_input: str,
        schema_name: Optional[str] = "dbo",
        rid: Optional[str] = None,
        include_visualization: bool = False,  # NEW
        force_chart_type: Optional[str] = None,  # NEW (e.g., 'bar_chart')
        lang_override: Optional[Literal["zh-tw", "en"]] = None,  # NEW
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        session_id = rid or "default"

        try:
            # 1) Language detection (with optional override)
            lang = lang_override or detect_language_simple(user_input)
            logger.info("rid=%s query=%r lang=%s (override=%s)", rid, user_input, lang, bool(lang_override))

            # 2) Language-aware date processing
            query_with_dates = self.date_processor.rewrite_relative_dates(user_input, lang)

            # 3) Context rewriting for followup queries
            grounded_query = self.context_rewriter.rewrite_followup_with_context(query_with_dates, lang, session_id)

            # 4) Language-aware vector search - use original query for Chinese
            search_query = grounded_query if lang == "zh-tw" else grounded_query

            # CRITICAL: Pass language to vector search
            if hasattr(self.vector_search, "find_relevant_tables_with_language"):
                rel_with_scores = self.vector_search.find_relevant_tables_with_language(
                    search_query, schema_filter=schema_name, language=lang, rid=rid
                )
            else:
                rel_with_scores = self.vector_search.find_relevant_tables(search_query, schema_filter=schema_name, rid=rid)

            rel_tables = [t for (t, _) in rel_with_scores]
            join_hints = self.vector_search.get_join_hints(rel_tables)

            # 5) Language-aware schema context
            if hasattr(self.vector_search, "get_schema_context_with_language"):
                schema_ctx = self.vector_search.get_schema_context_with_language(rel_tables, search_query, language=lang)
            else:
                schema_ctx = self.vector_search.get_schema_context(rel_tables)

            # 6) Check memory with language awareness
            cached_sql, cached_conf = self.memory.check_memory_for_query(
                original_query=grounded_query,
                english_query=grounded_query,  # Will be different if translated
                relevant_tables=rel_tables,
                lang=lang,
                session_id=session_id,
            )

            final_sql = ""
            llm_attempts = 0
            rows: List[Tuple[Any, ...]] = []
            columns: List[str] = []
            execution_error: Optional[str] = None

            # 7) Execute query with language-aware prompting
            exec_t0 = time.perf_counter()
            try:
                if cached_sql:
                    final_sql = normalize_sql_columns(cached_sql)
                    rows, columns = self.db_service.run_select(final_sql, max_rows=1000, query_timeout=10)
                    llm_attempts = 0
                    logger.info("CACHED_QUERY_EXECUTION: query='%s' cached_sql='%s'", grounded_query[:50], final_sql[:100])
                else:
                    if rel_tables:
                        logger.debug("LLM_QUERY_GENERATION: lang=%s tables=%s", lang, rel_tables[:3])
                        rows, columns, final_sql, llm_attempts = self.llm_service.run_query_with_llm_repair(
                            db_service=self.db_service,
                            user_question=grounded_query,
                            schema=schema_ctx,
                            join_hints=join_hints,
                            params=None,
                            max_rows=1000,
                            query_timeout=10,
                            max_attempts=3,
                        )
                        final_sql = normalize_sql_columns(final_sql)
                    else:
                        logger.warning("NO_TABLES_FOUND: using fallback SQL for query='%s'", grounded_query[:50])
                        english_for_template = self.translation_service.translate_to_english(grounded_query, lang)
                        alt = self.sql_template_service.get_fallback_sql(english_for_template)
                        final_sql = normalize_sql_columns(alt or "SELECT 1 WHERE 1=0")
                        rows, columns = self.db_service.run_select(final_sql, max_rows=1000, query_timeout=10)
            except Exception as e:
                execution_error = str(e)
                logger.error("QUERY_EXECUTION_ERROR: query='%s' error=%s", grounded_query[:50], execution_error)

            exec_ms = int((time.perf_counter() - exec_t0) * 1000)

            # 8) Learn and record in memory with language awareness
            english_query_for_fallback = grounded_query if lang == "en" else grounded_query  # Keep original for now

            if execution_error is None:
                self.memory.learn_from_query(
                    original_query=grounded_query,
                    english_query=english_query_for_fallback,
                    relevant_tables=rel_tables,
                    generated_sql=final_sql,
                    success=True,
                    execution_time=exec_ms / 1000.0,
                    lang=lang,
                    session_id=session_id,
                )
                self.memory.record_success(
                    session_id=session_id,
                    original_query=grounded_query,
                    english_query=english_query_for_fallback,
                    generated_sql=final_sql,
                    columns=columns,
                    rows=rows,
                    relevant_tables=rel_tables,
                    schema_ctx=schema_ctx,
                    lang=lang,
                )
            else:
                self.memory.learn_from_query(
                    original_query=grounded_query,
                    english_query=english_query_for_fallback,
                    relevant_tables=rel_tables,
                    generated_sql=final_sql or "",
                    success=False,
                    execution_time=exec_ms / 1000.0,
                    lang=lang,
                    session_id=session_id,
                )

            # 9) Generate language-native explanation and results
            if execution_error:
                explanation = f"查詢執行失敗：{execution_error}" if lang == "zh-tw" else f"Query execution failed: {execution_error}"
                table_md = ""
                visualization_payload: Optional[Dict[str, Any]] = None
            else:
                aggregates = self.data_analyzer.compute_aggregates(rows, columns)
                sample_text = format_sample_data(rows, columns)

                logger.info("EXPLANATION_GENERATION: query='%s' lang=%s rows=%d", grounded_query[:50], lang, len(rows))

                explanation = self._get_language_aware_explanation(
                    grounded_query, lang, len(rows), columns, aggregates, sample_text
                )

                # Generate results table
                want_details = self._should_show_details(grounded_query, lang)
                preferred_cols = ["Name", "EmployeeID", "ATTENDANCETYPE", "LEAVETYPE", "HOURS", "StartDate", "WORKDATE", "EndDate"]
                table_md = self._markdown_table(columns, rows, limit=20, keep=preferred_cols if want_details else None)

                if table_md:
                    preview_header = "**預覽（前20筆）：**" if lang == "zh-tw" else "**Preview (first 20 rows):**"
                    explanation = explanation.strip() + f"\n\n{preview_header}\n\n" + table_md

                # 9b) NEW — Chart Visualization Agent (gated by route flag)
                visualization_payload = None
                if include_visualization:
                    try:
                        # Only attempt if we have some data and (intent or a forced type)
                        if rows and columns and (_should_generate_chart(grounded_query, lang, columns) or force_chart_type):
                            df = pd.DataFrame(rows, columns=columns)

                            # Light date coercion
                            for c in df.columns:
                                lc = str(c).lower()
                                if any(k in lc for k in ["date", "day", "time", "workdate", "startdate", "enddate"]):
                                    with pd.option_context("mode.chained_assignment", None):
                                        try:
                                            df[c] = pd.to_datetime(df[c], errors="ignore")
                                        except Exception:
                                            pass

                            # Respect explicit choice from user or inferred words in query
                            forced = force_chart_type or _infer_forced_chart_type(grounded_query, lang)
                            if forced:
                                visualization_payload = self.viz_service.create_visualization(
                                    df, user_query=grounded_query, force_chart_type=forced, title=None
                                )
                            else:
                                visualization_payload = self.chart_agent.generate_chart(
                                    df=df,
                                    user_query=grounded_query,
                                    sql_meta={"sql": final_sql, "tables": rel_tables, "lang": lang},
                                    force_chart_type=None,
                                    title=None,
                                )

                            if visualization_payload and lang == "zh-tw":
                                reason = visualization_payload.get("reasoning") or ""
                                visualization_payload["reasoning"] = f"圖表推薦：{reason}"
                    except Exception as viz_e:
                        logger.warning("VISUALIZATION_AGENT_FAILED: %s", viz_e)
                        visualization_payload = {"enabled": False, "reason": f"Visualization error: {viz_e}"}

            # 10) Build response
            stats = self.memory.get_memory_stats()

            # Language confidence (simplified since we use rule-based detection)
            chinese_chars = sum(1 for c in user_input if "\u4e00" <= c <= "\u9fff")
            total_chars = len([c for c in user_input if c.isalnum()])
            lang_confidence = min(1.0, (chinese_chars / max(total_chars, 1)) * 2) if lang == "zh-tw" else 0.9

            response = {
                "original_text": user_input,
                "detected_language": lang,
                "language_confidence": lang_confidence,
                "processed_query": grounded_query,
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
                "explanation": explanation,  # Native language explanation
                "summary": explanation,  # Same as explanation for simplicity
                "success": execution_error is None,
                "language_native_processing": True,  # Flag to indicate native processing

                # NEW: Chart visualization payload (present only when requested and no error)
                "visualization": visualization_payload if (execution_error is None and include_visualization) else None,
                "visualization_requested": bool(include_visualization),

                "memory": {
                    "session_id": session_id,
                    "used_cached_sql": bool(cached_sql),
                    "cached_confidence": float(cached_conf) if cached_sql else 0.0,
                    "cache_hit_rate": stats.get("cache_hit_rate"),
                    "language_aware": True,
                },
            }

            logger.info("rid=%s native pipeline ok ms=%d lang=%s explanation_lang=%s", rid, int((time.perf_counter() - t0) * 1000), lang, lang)
            return response

        except Exception as e:
            logger.error(
                "rid=%s native pipeline failed after %dms: %s: %s",
                rid,
                int((time.perf_counter() - t0) * 1000),
                type(e).__name__,
                e,
                exc_info=True,
            )

            # Language-aware error messages
            detected = detect_language_simple(user_input)
            msg = "處理您的查詢時發生錯誤。" if detected == "zh-tw" else "An error occurred while processing your query."

            return {
                "original_text": user_input,
                "detected_language": detected,
                "language_confidence": 0.5,
                "execution_successful": False,
                "execution_error": str(e),
                "summary": msg,
                "explanation": msg,
                "success": False,
                "language_native_processing": True,
                "visualization": None,  # keep shape stable
                "visualization_requested": False,
            }
