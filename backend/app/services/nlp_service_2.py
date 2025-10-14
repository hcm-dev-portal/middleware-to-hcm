# ================================================================================
# backend/app/services/nlp_service_2.py
from __future__ import annotations

import re
import time
import logging
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Literal

from app.services.db_service import SQLServerDatabaseService

# Specialized services
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

# Visualization services
from .chart.chart_agent import ChartVisualizationAgent, ChartVisualizationAgentConfig
from .chart.visualization_service import VisualizationService

logger = logging.getLogger(__name__)


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def detect_language_simple(text: str) -> Literal["zh-tw", "en"]:
    """
    Simple zh/en detector: if >30% alnum chars are CJK, assume zh.
    """
    if not text or not text.strip():
        return "en"
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    total_chars = len([c for c in text if c.isalnum()])
    if total_chars == 0:
        return "en"
    return "zh-tw" if (chinese_chars / total_chars) > 0.3 else "en"


class LanguageAwareDateProcessor:
    """
    Wraps DateProcessor to accept zh tokens by mapping to EN then using base rewriter.
    Also provides SQL date anchoring for generated SQL.
    """
    def __init__(self):
        self.base_processor = DateProcessor()
        self.zh_patterns = {
            r"今天": "today",
            r"昨日|昨天": "yesterday",
            r"明天|隔天": "tomorrow",
            r"這個月|本月|这个月": "this month",
            r"上個月|上月|上个月": "last month",
            r"下個月|下月|下个月": "next month",
            r"這週|本週|本周|这周": "this week",
            r"上週|上周": "last week",
            r"下週|下周": "next week",
            r"本季|這季": "this quarter",
            r"上季": "last quarter",
            r"今年": "this year",
            r"去年": "last year",
            r"明年": "next year",
        }

    def set_data_anchor(self, anchor_date: str):
        self.base_processor.set_data_anchor(anchor_date)

    def get_data_anchor(self) -> Optional[str]:
        return self.base_processor.get_data_anchor()

    def rewrite_relative_dates(self, text: str, lang: Literal["zh-tw", "en"]) -> str:
        if lang == "zh-tw":
            result = text
            for zh_pat, en_tok in self.zh_patterns.items():
                result = re.sub(zh_pat, en_tok, result)
            return self.base_processor.rewrite_relative_dates(result)
        return self.base_processor.rewrite_relative_dates(text)

    def rewrite_sql_dates(self, sql: str) -> str:
        """
        Anchor SQL temporal functions (GETDATE/CURRENT_TIMESTAMP) to current or data anchor.
        """
        return self.base_processor.rewrite_sql_dates(sql)


class LanguageAwareMemoryService:
    """
    Wraps SimpleQueryMemoryService to store/look up by original query,
    with optional English alias for zh queries.
    """
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
        cached_sql, conf = self.base_memory.check_memory_for_query(
            original_query, relevant_tables, session_id=session_id
        )
        if not cached_sql and lang == "zh-tw" and english_query and english_query != original_query:
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
        self.base_memory.learn_from_query(
            query=original_query,
            relevant_tables=relevant_tables,
            generated_sql=generated_sql,
            success=success,
            execution_time=execution_time,
            session_id=session_id,
        )
        if lang == "zh-tw" and english_query and english_query != original_query:
            self.base_memory.learn_from_query(
                query=english_query,
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
        self.base_memory.record_success(
            session_id=session_id,
            query=original_query,
            generated_sql=generated_sql,
            columns=columns,
            rows=rows,
            relevant_tables=relevant_tables,
            schema_ctx=schema_ctx,
        )

    def get_last_focus_value(self, session_id: str, column_patterns: List[str]) -> Optional[str]:
        return self.base_memory.get_last_focus_value(session_id, column_patterns)

    def get_memory_stats(self) -> Dict[str, Any]:
        return self.base_memory.get_memory_stats()


class LanguageAwareContextRewriter:
    """
    Rewrites vague follow-ups (zh/en) using session memory context.
    Falls back to LLM for pronoun resolution when needed.
    """
    def __init__(self, memory_service: LanguageAwareMemoryService, llm_service: UnifiedBilingualOpenAIService):
        self.memory_service = memory_service
        self.llm_service = llm_service

    def rewrite_followup_with_context(self, query: str, lang: Literal["zh-tw", "en"], session_id: str) -> str:
        result = query
        dept = self.memory_service.get_last_focus_value(session_id, ["Department", "DEPARTMENT", "DeptName", "部門"])
        if dept:
            if lang == "zh-tw":
                result = re.sub(r"(這個|那個|該)\s*部門", dept, result)
                result = re.sub(r"(這個|那個|該)\s*單位", dept, result)
            else:
                result = re.sub(r"\b(this|that|the)\s+department\b", dept, result, flags=re.I)
                result = re.sub(r"\b(this|that|the)\s+unit\b", dept, result, flags=re.I)

        if self._needs_llm_rewrite(result, lang):
            return self._llm_rewrite_with_context(result, lang, session_id)
        return result

    def _needs_llm_rewrite(self, query: str, lang: Literal["zh-tw", "en"]) -> bool:
        if lang == "zh-tw":
            vague = ["這個", "那個", "它", "他們", "該", "前面", "剛才"]
        else:
            vague = ["this", "that", "it", "they", "these", "those", "previous"]
        return any(v in query.lower() for v in vague)

    def _llm_rewrite_with_context(self, query: str, lang: Literal["zh-tw", "en"], session_id: str) -> str:
        try:
            snap = self.memory_service.base_memory.session_cache.get(session_id)
            context_info = ""
            if snap and getattr(snap, "successful_results", None):
                last = snap.successful_results[-1]
                cols = last.get("columns", [])
                preview = last.get("preview", "")
                context_info = f"Columns: {', '.join(cols[:5])}\nSample: {str(preview)[:200]}"

            if lang == "zh-tw":
                sys = "你是資料分析助理。請將含糊的查詢改寫為具體明確的問題，只回傳改寫後的問題。"
                usr = f"問題：{query}\n\n上下文：{context_info}\n\n改寫後的問題："
            else:
                sys = "Rewrite the vague analytics question into a fully specified English question. Return only the rewritten question."
                usr = f"Question: {query}\n\nContext: {context_info}\n\nRewritten:"

            rewritten = self.llm_service._simple_completion(sys, usr)  # type: ignore
            return rewritten.strip() if rewritten else query
        except Exception as e:
            logger.warning("LLM context rewrite failed: %s", e)
            return query


# ---------- Chart helpers ----------
_ZH_CHART_KEYWORDS = ["圖", "圖表", "視覺化", "繪圖", "趨勢圖", "長條圖", "柱狀圖", "折線圖", "圓餅圖", "直方圖", "散點圖", "箱型圖", "箱形圖", "熱圖", "熱力圖", "面積圖"]
_EN_CHART_KEYWORDS = ["chart", "plot", "graph", "visualize", "visualisation", "visualization", "trend", "line chart", "bar chart", "pie", "histogram", "scatter", "box", "heatmap", "area"]

_ZH_FORCED_TYPE = {
    "長條圖": "bar_chart", "柱狀圖": "bar_chart", "折線圖": "line_chart", "圓餅圖": "pie_chart",
    "直方圖": "histogram", "散點圖": "scatter_plot", "箱型圖": "box_plot", "箱形圖": "box_plot",
    "熱圖": "heatmap", "熱力圖": "heatmap", "面積圖": "area_chart", "趨勢圖": "line_chart",
}
_EN_FORCED_TYPE = {
    "bar chart": "bar_chart", "line chart": "line_chart", "pie": "pie_chart", "pie chart": "pie_chart",
    "histogram": "histogram", "scatter": "scatter_plot", "scatter plot": "scatter_plot", "box": "box_plot",
    "box plot": "box_plot", "heatmap": "heatmap", "area": "area_chart", "area chart": "area_chart",
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
    return len(columns) >= 2



class LanguageNativeNLPService:
    """
    Orchestrator: zh/en-first pipeline with vector retrieval, memory, date rewrite,
    SQL generation/repair, safe execution, analysis, and optional visualization.
    """

    def __init__(self, db_service: SQLServerDatabaseService, model_name: str = "gpt-4o-mini", temperature: float = 0.1, **_):
        self.db_service = db_service

        # Core services
        self.translation_service = AWSTranslationService()  # Only used for fallback templates
        self.llm_service = UnifiedBilingualOpenAIService(model_name=model_name, temperature=temperature)
        self.data_analyzer = DataAnalyzer()
        self.sql_template_service = SQLTemplateService()
        self.sql_executor = SQLExecutor(db_service)
        self.person_enrichment = PersonEnrichmentService(db_service)

        # Language-aware
        self.date_processor = LanguageAwareDateProcessor()
        self.vector_search = VectorSearchService(db_service)
        self.memory = LanguageAwareMemoryService(SimpleQueryMemoryService())
        self.context_rewriter = LanguageAwareContextRewriter(self.memory, self.llm_service)

        # Visualization
        self.viz_service = VisualizationService()
        self.chart_agent = ChartVisualizationAgent(
            viz_service=self.viz_service,
            cfg=ChartVisualizationAgentConfig(
                model="gpt-4o-mini",
                temperature=0.0,
                theme="plotly_white",
                use_ai=True,
            ),
        )

        # Default off; route can enable
        self.enable_auto_visualization = False

        self._initialize_data_anchor()

    @property
    def viz(self):
        return self.viz_service

    def _initialize_data_anchor(self):
        """
        Use latest WORKDATE as the data anchor if available.
        """
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
    
    # inside LanguageNativeNLPService (near the top of the class)

    @staticmethod
    def _normalize_lang(lang: Optional[str]) -> Literal["zh-tw", "en"]:
        """
        Canonicalize language tags coming from the UI or detectors.
        Accepts: 'zh', 'zh-tw', 'zh_TW', 'ZH-tw', 'zh-Hant', etc.
        """
        if not lang:
            return "en"
        s = str(lang).strip().lower().replace("_", "-")
        if s.startswith("zh"):
            # Treat any zh variant as Traditional Chinese for our prompts
            # If you later need zh-CN vs zh-TW, branch here using extra signals
            return "zh-tw"
        return "en"


    # ---------- UI table helpers ----------
    def _markdown_table(self, columns, rows, limit: int = 20, keep=None) -> str:
        cols = [c for c in columns or []]
        if not rows or not cols:
            return ""
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
        if lang == "zh-tw":
            detail_indicators = ["姓名", "員工", "員工編號", "列表", "顯示", "樣本", "詳細", "明細", "誰", "哪些人", "具體", "清單"]
        else:
            detail_indicators = ["name", "names", "employee id", "employee ids", "list", "show", "sample", "detail", "details", "who"]
        return any(indicator in query.lower() for indicator in detail_indicators)

    def _get_language_aware_explanation(
        self, query: str, lang: Literal["zh-tw", "en"], row_count: int, columns: List[str], aggregates: Dict, sample_text: str
    ) -> str:
        if lang == "zh-tw":
            return self.llm_service.generate_explanation_chinese(query, row_count, columns, aggregates, sample_text)
        return self.llm_service.generate_explanation(query, row_count, columns, aggregates, sample_text)

    # -------------- display enrichment (Department + EmployeeID + Name) --------------
    def _enrich_rows_for_display(
        self, columns: List[str], rows: List[Tuple[Any, ...]]
    ) -> Tuple[List[str], List[List[Any]]]:
        """
        Create a user-friendly projection by ensuring Department, EmployeeID, Name
        are present as the first columns. Uses PersonResolver to fill gaps.
        Returns (enriched_columns, enriched_rows).
        """
        if not columns or not rows:
            return columns or [], [list(r) for r in rows or []]

        # Index lookups (case-insensitive)
        low = {c.lower(): i for i, c in enumerate(columns)}
        idx_pid  = low.get("personid")
        idx_eid  = low.get("employeeid")
        idx_name = low.get("truename") or low.get("name")
        idx_dep_name = low.get("department_name") or low.get("unitname") or low.get("branch_name")
        idx_dep_code = low.get("department_code") or low.get("unitcode") or low.get("branch_code")

        # Collect IDs to resolve
        pids, eids = set(), set()
        for r in rows:
            if idx_pid is not None and idx_pid < len(r) and r[idx_pid]:
                s = str(r[idx_pid]).strip()
                if s:
                    pids.add(s)
            if idx_eid is not None and idx_eid < len(r) and r[idx_eid]:
                s = str(r[idx_eid]).strip()
                if s:
                    eids.add(s)

        # Batch resolve via PersonResolver
        resolved: Dict[str, Dict[str, Optional[str]]] = {}
        try:
            res = self.person_enrichment.person_resolver.resolve_many(list(pids), employee_ids=list(eids))
            resolved.update(res or {})
        except Exception as e:
            logger.warning("Display enrichment resolver failed: %s", e)

        # Decide which friendly headers are needed (avoid duplicates)
        friendly_headers: List[str] = []
        if idx_dep_name is None:
            friendly_headers.append("Department")  # will fill with department_name
        if idx_eid is None:
            friendly_headers.append("EmployeeID")
        # For name, prefer TRUENAME/Name if already present; otherwise add friendly Name
        add_friendly_name = idx_name is None
        if add_friendly_name:
            friendly_headers.append("Name")

        # Build enriched rows
        enriched_rows: List[List[Any]] = []
        for r in rows:
            # pick lookup key (first non-empty among person, employee)
            lk = None
            if idx_pid is not None and idx_pid < len(r) and r[idx_pid]:
                lk = str(r[idx_pid]).strip()
            elif idx_eid is not None and idx_eid < len(r) and r[idx_eid]:
                lk = str(r[idx_eid]).strip()

            info = resolved.get(lk or "", {}) if lk else {}

            # Department logic
            dept = None
            if idx_dep_name is not None and idx_dep_name < len(r) and r[idx_dep_name]:
                dept = r[idx_dep_name]
            else:
                dept = info.get("department_name") or info.get("department_code") or None

            # EmployeeID logic
            eid_val = None
            if idx_eid is not None and idx_eid < len(r) and r[idx_eid]:
                eid_val = r[idx_eid]
            else:
                eid_val = info.get("employee_id")

            # Name logic
            name_val = None
            if idx_name is not None and idx_name < len(r) and r[idx_name]:
                name_val = r[idx_name]
            else:
                name_val = info.get("name") or (lk if lk else None)

            # Assemble new row: [friendly headers values] + original row
            prefix: List[Any] = []
            for hdr in friendly_headers:
                if hdr == "Department":
                    prefix.append(dept)
                elif hdr == "EmployeeID":
                    prefix.append(eid_val)
                elif hdr == "Name":
                    prefix.append(name_val)
                else:
                    prefix.append(None)

            enriched_rows.append(prefix + list(r))

        # Assemble new columns list with friendly headers in front
        enriched_columns = friendly_headers + columns

        return enriched_columns, enriched_rows

    # ---------- main entry ----------
    def process_complete_query(
        self,
        user_input: str,
        schema_name: Optional[str] = "dbo",
        rid: Optional[str] = None,
        include_visualization: bool = False,
        force_chart_type: Optional[str] = None,
        lang_override: Optional[Literal["zh-tw", "en"]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        session_id = rid or "default"

        try:
            # 1) Language detect + normalize (handles overrides like 'zh-TW')
            detected = detect_language_simple(user_input)
            lang = self._normalize_lang(lang_override or detected)
            logger.info("rid=%s query=%r lang=%s override=%s", rid, user_input, lang, bool(lang_override))

            # 2) Date rewrite uses normalized lang
            query_with_dates = self.date_processor.rewrite_relative_dates(user_input, lang)

            # 3) Follow-up context rewrite
            grounded_query = self.context_rewriter.rewrite_followup_with_context(query_with_dates, lang, session_id)

            # 4) Vector retrieval (bilingual path already inside service)
            if hasattr(self.vector_search, "find_relevant_tables_with_language"):
                rel_with_scores = self.vector_search.find_relevant_tables_with_language(
                    grounded_query, schema_filter=schema_name, language=lang, rid=rid
                )
            else:
                rel_with_scores = self.vector_search.find_relevant_tables(grounded_query, schema_filter=schema_name, rid=rid)
            rel_tables = [t for (t, _) in rel_with_scores]
            join_hints = self.vector_search.get_join_hints(rel_tables)

            # 5) Schema context (bilingual)
            if hasattr(self.vector_search, "get_schema_context_with_language"):
                schema_ctx = self.vector_search.get_schema_context_with_language(rel_tables, grounded_query, language=lang)
            else:
                schema_ctx = self.vector_search.get_schema_context(rel_tables)

            # 6) Memory lookup (store by original)
            cached_sql, cached_conf = self.memory.check_memory_for_query(
                original_query=grounded_query,
                english_query=grounded_query,  # we avoid translation for latency
                relevant_tables=rel_tables,
                lang=lang,
                session_id=session_id,
            )

            final_sql = ""
            llm_attempts = 0
            rows: List[Tuple[Any, ...]] = []
            columns: List[str] = []
            execution_error: Optional[str] = None

            exec_t0 = time.perf_counter()
            try:
                if cached_sql:
                    final_sql = normalize_sql_columns(cached_sql)
                    # Anchor dates inside SQL before executing
                    final_sql = self.date_processor.rewrite_sql_dates(final_sql)
                    rows, columns = self.db_service.run_select(final_sql, max_rows=1000, query_timeout=10)
                    llm_attempts = 0
                    logger.info("CACHED_QUERY_EXECUTION: query='%s' cached_sql='%s'", grounded_query[:80], final_sql[:160])
                else:
                    if rel_tables:
                        # LLM generation + repair loop
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
                        # Anchor SQL dates
                        final_sql = self.date_processor.rewrite_sql_dates(final_sql)
                        # Optional re-run after anchoring (only if initial call didn't already run anchored SQL)
                        if not rows and final_sql:
                            rows, columns = self.db_service.run_select(final_sql, max_rows=1000, query_timeout=10)
                    else:
                        # Fallback templates (already view-free)
                        logger.warning("NO_TABLES_FOUND: using fallback SQL")
                        english_for_template = self.translation_service.translate_to_english(grounded_query, lang)
                        alt = self.sql_template_service.get_fallback_sql(english_for_template or grounded_query)
                        final_sql = normalize_sql_columns(alt or "SELECT 1 WHERE 1=0")
                        final_sql = self.date_processor.rewrite_sql_dates(final_sql)
                        rows, columns = self.db_service.run_select(final_sql, max_rows=1000, query_timeout=10)
            except Exception as e:
                execution_error = str(e)
                logger.error("QUERY_EXECUTION_ERROR: %s", execution_error)

            exec_ms = _ms(exec_t0)

            # 7) Memory learn/record
            english_query_for_fallback = grounded_query if lang == "en" else grounded_query
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

            # 8) Explanation + table + optional viz
            if execution_error:
                explanation = "查詢執行失敗：" + execution_error if lang == "zh-tw" else "Query execution failed: " + execution_error
                table_md = ""
                visualization_payload: Optional[Dict[str, Any]] = None
                columns_enriched, rows_enriched = columns, [list(r) for r in rows]
            else:
                # --- Enrich for display (prepend Department, EmployeeID, Name) ---
                columns_enriched, rows_enriched = self._enrich_rows_for_display(columns, rows)

                # Use enriched set for analysis and preview
                aggregates = self.data_analyzer.compute_aggregates(rows_enriched, columns_enriched)
                sample_text = format_sample_data(rows_enriched, columns_enriched)

                explanation = self._get_language_aware_explanation(
                    grounded_query, lang, len(rows_enriched), columns_enriched, aggregates, sample_text
                )

                # Details intent → keep human columns in preview
                want_details = self._should_show_details(grounded_query, lang)
                preferred_cols = [
                    "Department", "EmployeeID", "Name",
                    "ATTENDANCETYPE", "LEAVETYPE", "HOURS",
                    "StartDate", "WORKDATE", "EndDate"
                ]
                table_md = self._markdown_table(
                    columns_enriched, rows_enriched, limit=20, keep=preferred_cols if want_details else None
                )
                if table_md:
                    preview_header = "**預覽（前20筆）：**" if lang == "zh-tw" else "**Preview (first 20 rows):**"
                    explanation = explanation.strip() + f"\n\n{preview_header}\n\n" + table_md

                # Visualization prefers enriched labels
                visualization_payload = None
                if include_visualization and rows_enriched and columns_enriched and (_should_generate_chart(grounded_query, lang, columns_enriched) or force_chart_type):
                    try:
                        df = pd.DataFrame(rows_enriched, columns=columns_enriched)
                        # light datetime coercion
                        for c in df.columns:
                            lc = str(c).lower()
                            if any(k in lc for k in ["date", "day", "time", "workdate", "startdate", "enddate"]):
                                try:
                                    df[c] = pd.to_datetime(df[c], errors="ignore")
                                except Exception:
                                    pass
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

            # 9) Build response
            stats = self.memory.get_memory_stats()
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

                # Raw results preserved for backwards compatibility
                "columns": columns,
                "results": [[jsonable_value(v) for v in r] for r in rows],
                "row_count": len(rows),

                # Programmatic person map (unchanged)
                "resolved_people": self.person_enrichment.enrich_people_data(rows, columns),

                # NEW: enriched outputs drive UI
                "columns_enriched": columns_enriched,
                "results_enriched": [[jsonable_value(v) for v in r] for r in rows_enriched],
                "table_markdown": table_md if execution_error is None else "",

                # Explanation in native language
                "explanation": explanation,
                "summary": explanation,
                "success": execution_error is None,
                "language_native_processing": True,

                # Visualization payload only when requested & successful
                "visualization": visualization_payload if (execution_error is None and include_visualization) else None,
                "visualization_requested": bool(include_visualization),

                # Memory stats
                "memory": {
                    "session_id": session_id,
                    "used_cached_sql": bool(cached_sql),
                    "cached_confidence": float(cached_conf) if cached_sql else 0.0,
                    "cache_hit_rate": stats.get("cache_hit_rate"),
                    "language_aware": True,
                },
            }

            logger.info("rid=%s pipeline ok in %dms (lang=%s)", rid, _ms(t0), lang)
            return response

        except Exception as e:
            logger.error("rid=%s pipeline failed in %dms: %s", rid, _ms(t0), e, exc_info=True)
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
                "visualization": None,
                "visualization_requested": False,
            }
