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

# VAC snapshot + ORG table FQNs (same source used by the vector DB)
from app.services.leave_vector import VAC_RESULT_TABLE, ORG_TABLE

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


def _extract_tables_from_sql(sql: str) -> List[str]:
    """
    Best-effort extractor for [schema].[table] tokens from FROM/JOIN clauses.
    Returns FQN-like names if present; otherwise empty.
    """
    if not sql:
        return []
    # capture things like dbo.Table, [dbo].[Table], eHRAntung_DB.dbo.ATD..., [eHRAntung_DB].[dbo].[ATD...]
    pat = re.compile(r"(?i)(?:from|join)\s+((?:\[[^\]]+\]|\w+)(?:\.(?:\[[^\]]+\]|\w+)){1,2})")
    out = []
    for m in pat.finditer(sql):
        token = m.group(1).strip()
        # normalize brackets → dotted form
        parts = [p.strip("[]") for p in token.split(".")]
        if len(parts) == 2:
            fqn = f"{parts[0]}.{parts[1]}"
        elif len(parts) >= 3:
            fqn = f"[{parts[0]}].[{parts[1]}].[{parts[2]}]"
        else:
            fqn = token
        out.append(fqn)
    # dedup, preserve order
    return list(dict.fromkeys(out))


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


# ---------- preference helpers (VAC snapshot) ----------
def _prefers_vac_snapshot(q: str) -> bool:
    ql = (q or "").lower()
    zh_hit = (("餘" in q or "剩" in q or "餘額" in q or "剩餘" in q or "還有" in q) and ("年假" in q or "特休" in q))
    en_hit = (any(w in ql for w in ["remaining", "unused", "balance"]) and any(w in ql for w in ["annual", "pto", "vacation"]))
    return zh_hit or en_hit


def _sanitize_sql_variables(sql: str) -> str:
    """
    Defensive sanitizer to avoid undeclared @variables from model output.
    - Replace @today-like tokens with GETDATE()
    - Remove trivial DECLARE/SET of such variables if present
    """
    if not sql:
        return sql
    out = sql
    # Replace common date variables with GETDATE()
    out = re.sub(r"@\s*(today|current_date|currdate)\b", "GETDATE()", out, flags=re.I)
    # Nuke simple DECLARE/SET lines for those variables
    out = re.sub(r"(?im)^\s*DECLARE\s+@\s*(today|current_date|currdate)\s+[^;]*;?\s*$", "", out)
    out = re.sub(r"(?im)^\s*SET\s+@\s*(today|current_date|currdate)\s*=\s*[^;]+;?\s*$", "", out)
    # Clean multiple blank lines
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


class LanguageNativeNLPService:
    """
    Orchestrator: zh/en-first pipeline with vector retrieval (intent-first),
    memory, date rewrite, SQL generation/repair, safe execution, analysis,
    and optional visualization.
    """

    def __init__(
        self,
        db_service: SQLServerDatabaseService,
        model_name: str = "gpt-4.1",
        temperature: float = 0.1,
        # Kept for backward-compat but ignored (we don't suppress anything anymore)
        enable_explanation: Optional[bool] = None,
        suppress_enrichment: bool = False,
        suppress_explanation: Optional[bool] = None,
        **_,
    ):
        self.db_service = db_service

        # Core services
        self.translation_service = AWSTranslationService()
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
                model="gpt-4.1",
                temperature=0.0,
                theme="plotly_white",
                use_ai=True,
            ),
        )

        # Default off; route can enable
        self.enable_auto_visualization = False

        # ***** No suppressions *****
        # Always enable explanation; keep legacy attributes for compatibility.
        self.enable_explanation = True
        self.suppress_explanation = False
        self.suppress_enrichment = False  # currently unused in raw-only output

    @property
    def viz(self):
        return self.viz_service

    @property
    def person_table(self) -> str:
        return self.vector_search.person_table

    def vector_status(self) -> Dict[str, Any]:
        return self.vector_search.health_check()

    # ---------- language ----------
    @staticmethod
    def _normalize_lang(lang: Optional[str]) -> Literal["zh-tw", "en"]:
        if not lang:
            return "en"
        s = str(lang).strip().lower().replace("_", "-")
        if s.startswith("zh"):
            return "zh-tw"
        return "en"

    def _get_language_aware_explanation(
        self, query: str, lang: Literal["zh-tw", "en"], row_count: int, columns: List[str], aggregates: Dict, sample_text: str
    ) -> str:
        if lang == "zh-tw":
            return self.llm_service.generate_explanation_chinese(query, row_count, columns, aggregates, sample_text)
        return self.llm_service.generate_explanation(query, row_count, columns, aggregates, sample_text)

    # ---------- main entry ----------
    def process_complete_query(
        self,
        user_input: str,
        schema_name: Optional[str] = "dbo",
        rid: Optional[str] = None,
        include_visualization: bool = False,
        force_chart_type: Optional[str] = None,
        lang_override: Optional[Literal["zh-tw", "en"]] = None,
        # Per-call override; if None we still generate explanation (default ON)
        include_explanation: Optional[bool] = None,
    ) -> Dict[str, Any]:

        t0 = time.perf_counter()
        session_id = rid or "default"

        try:
            # 1) Language detect + normalize
            detected = detect_language_simple(user_input)
            lang = self._normalize_lang(lang_override or detected)
            logger.info("rid=%s query=%r lang=%s override=%s", rid, user_input, lang, bool(lang_override))

            # 2) Date rewrite
            query_with_dates = self.date_processor.rewrite_relative_dates(user_input, lang)

            # 3) Follow-up context rewrite
            grounded_query = self.context_rewriter.rewrite_followup_with_context(query_with_dates, lang, session_id)

            # 4) Intent-first planning
            plan = self.vector_search.plan_for(grounded_query, schema_filter=schema_name, rid=rid)
            rel_tables = plan.get("tables") or []
            join_hints = plan.get("join_hints") or "None"
            schema_ctx = plan.get("schema") or ""
            intent_ctx = plan.get("intent_context") or {}
            intent_ref = intent_ctx.get("template_ref")
            intent_slots = intent_ctx.get("slots") or {}

            # Prefer VAC snapshot when relevant
            if _prefers_vac_snapshot(grounded_query) and VAC_RESULT_TABLE not in rel_tables:
                rel_tables = [VAC_RESULT_TABLE] + rel_tables

            must_use = (
                list(dict.fromkeys([VAC_RESULT_TABLE] + intent_ctx.get("tables", [])))
                if _prefers_vac_snapshot(grounded_query) else
                intent_ctx.get("tables", rel_tables)
            )

            where_hard_hints = intent_ctx.get("where_hard_hints", [])
            where_hard_hints = list(dict.fromkeys(where_hard_hints + [
                f"Use {VAC_RESULT_TABLE} for balances (authoritative).",
                "Use GETDATE() (not @today) for validity checks.",
                "If asking for a specific year, filter VACAYEAR = <year>.",
                "If asking annual/特休, filter VACATIONTYPE = 1 (if applicable).",
                "If CANUSEDATE/DISABLEDDATE exist, require CAST(GETDATE() AS date) BETWEEN CAST(CANUSEDATE AS date) AND CAST(DISABLEDDATE AS date).",
                "Select REMAINDAYS (remaining days), plus person name via dbo.PSNACCOUNT and optional department via ORG table.",
            ]))

            vac_hint_block = (
                f"-- MUST USE {VAC_RESULT_TABLE} for remaining/balance queries.\n"
                f"-- Join: r.PERSONID → dbo.PSNACCOUNT.PERSONID; optional PSNACCOUNT.BRANCHID → {ORG_TABLE}.UNITID\n"
                f"-- Filters: r.REMAINDAYS > 0; year via r.VACAYEAR; annual via r.VACATIONTYPE = 1; validity via GETDATE()."
            )
            join_hints = f"{vac_hint_block}\n\n{join_hints}".strip()

            intent_ctx = {
                **intent_ctx,
                "tables": rel_tables,
                "must_use_tables": must_use,
                "forbid_tsql_variables": True,
                "where_hard_hints": where_hard_hints,
            }

            # Vector scores for UI/debug
            rel_with_scores = self.vector_search.find_relevant_tables_with_language(
                grounded_query, schema_filter=schema_name, language=lang, rid=rid
            )

            # 5) Memory lookup
            cached_sql, cached_conf = self.memory.check_memory_for_query(
                original_query=grounded_query,
                english_query=grounded_query,
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
                    final_sql = _sanitize_sql_variables(self.date_processor.rewrite_sql_dates(final_sql))
                    rows, columns = self.db_service.run_select(final_sql, max_rows=9999, query_timeout=10)
                    llm_attempts = 0
                    logger.info("CACHED_QUERY_EXECUTION: query='%s' cached_sql='%s'", grounded_query[:80], final_sql[:160])
                else:
                    if rel_tables:
                        kwargs = dict(
                            db_service=self.db_service,
                            user_question=grounded_query,
                            schema=schema_ctx,
                            join_hints=join_hints,
                            intent_context=intent_ctx,
                            params=None,
                            max_rows=9999,
                            query_timeout=10,
                            max_attempts=3,
                        )
                        try:
                            if plan.get("business_prompt"):
                                kwargs["business_prompt"] = plan["business_prompt"]
                            if plan.get("few_shots"):
                                kwargs["few_shots"] = plan["few_shots"]
                        except Exception:
                            pass

                        rows, columns, final_sql, llm_attempts = self.llm_service.run_query_with_llm_repair(**kwargs)  # type: ignore
                        final_sql = normalize_sql_columns(final_sql)
                        final_sql = _sanitize_sql_variables(self.date_processor.rewrite_sql_dates(final_sql))
                        if not rows and final_sql:
                            rows, columns = self.db_service.run_select(final_sql, max_rows=9999, query_timeout=10)
                    else:
                        logger.warning("NO_TABLES_FOUND: using fallback SQL")
                        english_for_template = self.translation_service.translate_to_english(grounded_query, lang)
                        alt = self.sql_template_service.get_fallback_sql(english_for_template or grounded_query)
                        final_sql = normalize_sql_columns(alt or "SELECT 1 WHERE 1=0")
                        final_sql = _sanitize_sql_variables(self.date_processor.rewrite_sql_dates(final_sql))
                        rows, columns = self.db_service.run_select(final_sql, max_rows=9999, query_timeout=10)
            except Exception as e:
                execution_error = str(e)
                logger.error("QUERY_EXECUTION_ERROR: %s", execution_error)

            exec_ms = _ms(exec_t0)

            # 6) Memory learn/record
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

            # 6.5) Online learning for vector DB
            try:
                if self.vector_search and getattr(self.vector_search, "vector", None):
                    touched = _extract_tables_from_sql(final_sql) or rel_tables
                    self.vector_search.vector.record_outcome(
                        query=grounded_query,
                        success=(execution_error is None) and bool(rows),
                        tables_used=touched,
                        recipe_id=intent_ref,
                        row_count=len(rows),
                        error=execution_error,
                        diagnostics={"attempts": llm_attempts},
                    )
            except Exception as e:
                logger.warning("VDB_RECORD_OUTCOME_FAIL: %s", e)

            # 7) Build raw-only outputs (no markdown/enriched display)
            if execution_error:
                explanation = ""
                visualization_payload: Optional[Dict[str, Any]] = None
                summary_text = ("查詢執行失敗：" + execution_error) if lang == "zh-tw" else ("Query execution failed: " + execution_error)
            else:
                # Basic summary from raw columns/rows
                def _idx(cols, names):
                    low = {c.lower(): i for i, c in enumerate(cols or [])}
                    for n in names:
                        i = low.get(n.lower())
                        if i is not None:
                            return i
                    return None

                n_rows = len(rows)
                i_year = _idx(columns, ["VACAYEAR", "year"])
                i_eid  = _idx(columns, ["EmployeeID", "EMPLOYEEID"])
                i_name = _idx(columns, ["Name", "TRUENAME", "姓名"])

                uniq_people = None
                if i_eid is not None:
                    uniq_people = len({str(r[i_eid]) for r in rows if i_eid < len(r) and r[i_eid] is not None})
                elif i_name is not None:
                    uniq_people = len({str(r[i_name]) for r in rows if i_name < len(r) and r[i_name] is not None})

                year_text = ""
                if i_year is not None and n_rows > 0:
                    vals = {str(r[i_year]) for r in rows if i_year < len(r)}
                    if len(vals) == 1:
                        y = list(vals)[0]
                        year_text = f"（{y}）" if lang == "zh-tw" else f" (Year {y})"

                if lang == "zh-tw":
                    base = f"共 {n_rows} 筆結果{year_text}。"
                    if uniq_people is not None:
                        base += f" 涉及 {uniq_people} 位員工。"
                    summary_text = base
                else:
                    base = f"Found {n_rows} rows{year_text}."
                    if uniq_people is not None:
                        base += f" Across {uniq_people} employees."
                    summary_text = base

                # --- ALWAYS generate explanation (LLM if available, else fallback) ---
                want_explanation = True if include_explanation is None else bool(include_explanation)
                explanation = ""
                if want_explanation:
                    try:
                        logger.info(
                            "EXPLAIN_START: rid=%s rows=%d cols=%d llm_enabled=%s",
                            rid, len(rows), len(columns), self.llm_service.llm_enabled
                        )
                        aggregates = self.data_analyzer.compute_aggregates(rows, columns)
                        sample_text = format_sample_data(rows, columns)
                        explanation = self._get_language_aware_explanation(
                            grounded_query, lang, len(rows), columns, aggregates, sample_text
                        )
                        if not explanation or not explanation.strip():
                            # Hard fallback to ensure non-empty
                            explanation = self.llm_service._fallback_explanation(aggregates, "zh-tw" if lang == "zh-tw" else "en")
                        logger.info("EXPLAIN_OK: rid=%s chars=%d", rid, len(explanation))
                    except Exception as ee:
                        logger.warning("EXPLAIN_FAIL: rid=%s err=%s", rid, ee)
                        # Last-ditch deterministic summary
                        try:
                            aggregates = self.data_analyzer.compute_aggregates(rows, columns)
                            explanation = self.llm_service._fallback_explanation(aggregates, "zh-tw" if lang == "zh-tw" else "en")
                        except Exception:
                            explanation = ""  # should rarely happen

                # Visualization
                visualization_payload = None
                if include_visualization and rows and columns and (_should_generate_chart(grounded_query, lang, columns) or force_chart_type):
                    try:
                        df = pd.DataFrame(rows, columns=columns)
                        for c in df.columns:
                            lc = str(c).lower()
                            if any(k in lc for k in ["date", "day", "time", "workdate", "startdate", "enddate", "canusedate", "disableddate"]):
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
                                sql_meta={"sql": final_sql, "tables": rel_tables, "lang": lang, "intent": intent_ref, "slots": intent_slots},
                                force_chart_type=None,
                                title=None,
                            )
                        if visualization_payload and lang == "zh-tw":
                            reason = visualization_payload.get("reasoning") or ""
                            visualization_payload["reasoning"] = f"圖表推薦：{reason}"
                    except Exception as viz_e:
                        logger.warning("VISUALIZATION_AGENT_FAILED: %s", viz_e)
                        visualization_payload = {"enabled": False, "reason": f"Visualization error: {viz_e}"}

            # 8) Build response
            stats = self.memory.get_memory_stats()
            chinese_chars = sum(1 for c in user_input if "\u4e00" <= c <= "\u9fff")
            total_chars = len([c for c in user_input if c.isalnum()])
            lang_confidence = min(1.0, (chinese_chars / max(total_chars, 1)) * 2) if lang == "zh-tw" else 0.9

            response = {
                "original_text": user_input,
                "detected_language": lang,
                "language_confidence": lang_confidence,
                "processed_query": grounded_query,
                "intent": intent_ref or "generic",
                "intent_slots": intent_slots,
                "schema": schema_name,
                "relevant_tables": [{"table": t, "score": round(s, 3)} for (t, s) in rel_with_scores],
                "generated_sql": final_sql or "SELECT 1 WHERE 1=0",
                "llm_attempts": llm_attempts,
                "execution_successful": execution_error is None,
                "execution_error": execution_error,

                "columns": columns,
                "results": [[jsonable_value(v) for v in r] for r in rows],
                "row_count": len(rows),

                # Always populated
                "summary": summary_text,
                # Always attempted; non-empty due to fallback if LLM disabled or empty
                "explanation": explanation,

                "success": execution_error is None,
                "language_native_processing": True,

                "visualization": visualization_payload if (execution_error is None and include_visualization) else None,
                "visualization_requested": bool(include_visualization),

                "memory": {
                    "session_id": session_id,
                    "used_cached_sql": bool(cached_sql),
                    "cached_confidence": float(cached_conf) if cached_sql else 0.0,
                    "cache_hit_rate": stats.get("cache_hit_rate"),
                    "language_aware": True,
                },

                "planner": plan,
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
                "explanation": "",
                "success": False,
                "language_native_processing": True,
                "visualization": None,
                "visualization_requested": False,
            }
