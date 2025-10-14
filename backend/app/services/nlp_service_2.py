# ================================================================================
# backend/app/services/nlp_service_2.py
"""
Pure orchestrator for NLP-to-SQL pipeline with comprehensive structured logging.
Delegates all business logic to specialized services.
"""
from __future__ import annotations

import os
import re
import json
import time
import logging
import pathlib
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Literal, Sequence, Iterable
from datetime import datetime

from app.services.db_service import SQLServerDatabaseService

# Core services
from app.services.llm.openai_service import UnifiedBilingualOpenAIService
from app.services.memory.simple_query_memory import SimpleQueryMemoryService

# Data processing
from .data_processing.data_analyzer import DataAnalyzer
from .data_processing.person_enrichment import PersonEnrichmentService

# Vector and retrieval (now owns all date/SQL/vector logic)
from .retrieval.vector_search_service import VectorSearchService
from .retrieval.vector_search_service import validate_generated_sql  # keep as-is

# Visualization
from .chart.chart_agent import ChartVisualizationAgent, ChartVisualizationAgentConfig
from .chart.visualization_service import VisualizationService

# Helpers
from .helpers.data_utils import jsonable_value, normalize_sql_columns, format_sample_data, normalize_column_labels



logger = logging.getLogger(__name__)
trace_logger = logging.getLogger("app.trace")  # JSON line logger


# ---------------------- Structured Trace Helpers ----------------------

APP_LOG_DIR = pathlib.Path(os.getenv("APP_LOG_DIR", "logs"))
ORCH_SQL_DIR = pathlib.Path(os.getenv("ORCH_SQL_DIR", str(APP_LOG_DIR / "sql_history")))
ORCH_SQL_SAVE = os.getenv("ORCH_SQL_SAVE", "1") != "0"
ORCH_SQL_DIR.mkdir(parents=True, exist_ok=True)


def _now_ms() -> int:
    return int(time.perf_counter() * 1000)


def _ms_since(t0_ms: int) -> int:
    return max(0, _now_ms() - t0_ms)


def _preview_text(s: Optional[str], limit: int = 240) -> str:
    if not s:
        return ""
    s = str(s).replace("\n", " ").replace("\r", " ")
    return (s[:limit] + "…") if len(s) > limit else s


def trace(rid: Optional[str], stage: str, **fields: Dict[str, Any]) -> None:
    """Emit a single JSON log line for reliable tracing."""
    try:
        payload = {"rid": rid or "default", "stage": stage, "ts": time.time()}
        payload.update(fields or {})
        trace_logger.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        # never crash the pipeline due to logging
        logger.debug("trace logging failed", exc_info=True)


def persist_sql(rid: Optional[str], sql: str, *, stage: str = "SQL_FINAL") -> Optional[str]:
    """Optionally persist the final SQL for re-run / DBA analysis."""
    if not ORCH_SQL_SAVE or not sql:
        return None
    try:
        fname = f"{(rid or 'default')}.sql"
        sql_path = ORCH_SQL_DIR / fname
        text = f"/* RID:{rid or 'default'} STAGE:{stage} */\nSET NOCOUNT ON;\n{sql.strip()}\n"
        sql_path.write_text(text, encoding="utf-8")
        trace(rid, stage, saved=str(sql_path))
        return str(sql_path)
    except Exception:
        logger.debug("persist_sql failed", exc_info=True)
        return None


# ---------------------- DataFrame Safety Utilities ----------------------

def _build_safe_dataframe(columns: Sequence[Any], rows: Iterable[Any]) -> pd.DataFrame:
    """
    Create a DataFrame even if row lengths != len(columns).
    - Truncates extra fields
    - Pads missing fields with None
    - Handles dict rows by selecting only provided columns
    - Drops fully empty columns
    """
    cols = [str(c) for c in (columns or [])]
    fixed_rows: List[Any] = []

    for r in rows or []:
        if isinstance(r, dict):
            fixed_rows.append({c: r.get(c) for c in cols})
        else:
            rr = list(r)
            if len(rr) >= len(cols):
                fixed_rows.append(rr[:len(cols)])
            else:
                fixed_rows.append(rr + [None] * (len(cols) - len(rr)))

    df = pd.DataFrame(fixed_rows, columns=cols)
    if not df.empty:
        df = df.loc[:, ~(df.isna().all())]
    return df


_DATE_COL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _maybe_melt_dates(df: pd.DataFrame, id_vars: Optional[List[str]] = None) -> pd.DataFrame:
    """
    If many columns look like YYYY-MM-DD, melt them into (date, value) long format.
    Helpful for trend/折線圖 queries.
    """
    if df is None or df.empty:
        return df
    date_like_cols = [c for c in df.columns if _DATE_COL_RE.match(str(c))]
    if len(date_like_cols) >= 5:
        id_vars = id_vars or [c for c in df.columns if c not in date_like_cols][:1]
        long_df = df.melt(id_vars=id_vars, value_vars=date_like_cols, var_name="date", value_name="value")
        try:
            long_df["date"] = pd.to_datetime(long_df["date"])
        except Exception:
            pass
        try:
            long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
        except Exception:
            pass
        return long_df
    return df


# ---------------------- Orchestrator ----------------------

class LanguageNativeNLPService:
    """
    Pure orchestrator: delegates to specialized services.
    Now with detailed structured tracing at each step.
    """

    def __init__(
        self, 
        db_service: SQLServerDatabaseService, 
        model_name: str = "gpt-4o-mini", 
        temperature: float = 0.1, 
        **_
    ):
        self.db_service = db_service

        # Core services
        self.llm_service = UnifiedBilingualOpenAIService(
            model_name=model_name, 
            temperature=temperature
        )
        self.data_analyzer = DataAnalyzer()
        self.person_enrichment = PersonEnrichmentService(db_service)
        self.memory = SimpleQueryMemoryService()

        # Vector search service (owns all vector/date/SQL logic)
        self.vector_search = VectorSearchService(db_service)

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

        # Log constructor config (once)
        trace(None, "ORCH_INIT", model=model_name, temperature=temperature)

    @property
    def viz(self):
        return self.viz_service

    @property
    def person_table(self) -> str:
        """Delegate to vector search"""
        return self.vector_search.person_table

    def vector_status(self) -> Dict[str, Any]:
        """Get vector database health status"""
        return self.vector_search.health_check()

    # ---------- UI Formatting Helpers ----------
    
    def _markdown_table(
        self, 
        columns: List[str], 
        rows: List[List[Any]], 
        limit: int = 20, 
        keep: Optional[List[str]] = None
    ) -> str:
        """Generate markdown table for preview"""
        if not rows or not columns:
            return ""

        # Project to specific columns if requested
        if keep:
            col_map = {c.lower(): i for i, c in enumerate(columns)}
            kept_cols = []
            kept_indices = []
            
            for k in keep:
                idx = col_map.get(k.lower())
                if idx is not None:
                    kept_cols.append(columns[idx])
                    kept_indices.append(idx)
            
            if kept_cols:
                columns = kept_cols
                rows = [
                    [row[i] if i < len(row) else "" for i in kept_indices]
                    for row in rows[:limit]
                ]
        else:
            rows = rows[:limit]

        # Build markdown
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        
        body_rows = []
        for row in rows:
            str_row = [str(v) if v is not None else "" for v in row]
            body_rows.append("| " + " | ".join(str_row) + " |")
        
        return "\n".join([header, separator] + body_rows)

    def _should_show_details(self, query: str, lang: Literal["zh-tw", "en"]) -> bool:
        """Determine if query wants detailed employee list"""
        if lang == "zh-tw":
            indicators = [
                "姓名", "員工", "員工編號", "列表", "顯示", 
                "樣本", "詳細", "明細", "誰", "哪些人", "具體", "清單"
            ]
        else:
            indicators = [
                "name", "names", "employee id", "employee ids", 
                "list", "show", "sample", "detail", "details", "who"
            ]
        
        q_lower = query.lower()
        return any(ind in q_lower for ind in indicators)

    def _should_generate_chart(
        self, 
        query: str, 
        lang: Literal["zh-tw", "en"], 
        columns: List[str]
    ) -> bool:
        """Determine if visualization is appropriate"""
        if lang == "zh-tw":
            keywords = [
                "圖", "圖表", "視覺化", "繪圖", "趨勢圖", 
                "長條圖", "柱狀圖", "折線圖", "圓餅圖"
            ]
        else:
            keywords = [
                "chart", "plot", "graph", "visualize", 
                "visualization", "trend", "bar", "line", "pie"
            ]
        
        has_keyword = any(kw in query for kw in keywords) if lang == "zh-tw" else \
                      any(kw in query.lower() for kw in keywords)
        
        return has_keyword or len(columns) >= 2

    def _infer_forced_chart_type(
        self, 
        query: str, 
        lang: Literal["zh-tw", "en"]
    ) -> Optional[str]:
        """Infer specific chart type from query"""
        if lang == "zh-tw":
            type_map = {
                "長條圖": "bar_chart", "柱狀圖": "bar_chart",
                "折線圖": "line_chart", "趨勢圖": "line_chart",
                "圓餅圖": "pie_chart", "散點圖": "scatter_plot",
                "箱型圖": "box_plot", "熱圖": "heatmap"
            }
            for keyword, chart_type in type_map.items():
                if keyword in query:
                    return chart_type
        else:
            q = query.lower()
            type_map = {
                "bar chart": "bar_chart", "bar": "bar_chart",
                "line chart": "line_chart", "line": "line_chart",
                "pie": "pie_chart", "scatter": "scatter_plot",
                "box": "box_plot", "heatmap": "heatmap"
            }
            for keyword, chart_type in type_map.items():
                if keyword in q:
                    return chart_type
        
        return None

    # UPDATED: delegate enrichment to PersonEnrichmentService
    def _enrich_rows_for_display(
        self, 
        columns: List[str], 
        rows: List[Tuple[Any, ...]]
    ) -> Tuple[List[str], List[List[Any]]]:
        if not columns or not rows:
            return columns or [], [list(r) for r in rows or []]
        try:
            new_cols, new_rows = self.person_enrichment.add_readable_columns(
                rows=rows,
                columns=columns,
                add_department_name=True,
                add_person_name=True,
                add_employee_id=True,
                add_leave_type_name=True
            )
            return new_cols, new_rows
        except Exception as e:
            logger.warning("DISPLAY_ENRICH_FAIL: %s", e)
            return columns, [list(r) for r in rows]

    # ---------- Main Processing Method ----------

    def process_complete_query(
        self,
        user_input: str,
        schema_name: Optional[str] = "dbo",
        rid: Optional[str] = None,
        include_visualization: bool = False,
        force_chart_type: Optional[str] = None,
        lang_override: Optional[Literal["zh-tw", "en"]] = None,
    ) -> Dict[str, Any]:
        """
        Main orchestration method with detailed tracing.
        Delegates all business logic to specialized services.
        """
        start_ms = _now_ms()
        session_id = rid or "default"

        trace(rid, "ORCH_START",
              schema=schema_name,
              include_viz=include_visualization,
              force_chart=force_chart_type,
              lang_override=lang_override,
              q_preview=_preview_text(user_input))

        try:
            current_year = datetime.now().year
            sql_warnings: List[str] = []

            # ========== STEP 1: Language Detection & Query Processing ==========
            t_ms = _now_ms()
            query_context = self.vector_search.process_query_with_context(
                user_input=user_input,
                lang_override=lang_override,
                session_id=session_id,
                current_year=current_year
            )
            lang = query_context.get("language")
            processed_query = query_context.get("processed_query", user_input)
            trace(rid, "LANG_DONE",
                  ms=_ms_since(t_ms),
                  language=lang,
                  processed_preview=_preview_text(processed_query),
                  context_keys=sorted(list(query_context.keys())))

            # ========== STEP 2: Vector Retrieval ==========
            t_ms = _now_ms()
            retrieval_result = self.vector_search.retrieve_schema_context(
                query=processed_query,
                schema_filter=schema_name,
                language=lang,
                current_year=current_year,
                rid=rid
            )
            relevant_tables = retrieval_result.get("tables", [])
            schema_context = retrieval_result.get("schema_context", {})
            join_hints = retrieval_result.get("join_hints", {})
            table_scores = retrieval_result.get("table_scores", [])
            trace(rid, "VEC_DONE",
                  ms=_ms_since(t_ms),
                  tables=[t for t, _ in table_scores],
                  table_scores=[round(s, 3) for _, s in table_scores],
                  join_hint_keys=sorted(list(join_hints.keys())) if isinstance(join_hints, dict) else None,
                  schema_ctx_size=len(json.dumps(schema_context, ensure_ascii=False)) if schema_context else 0)

            # ========== STEP 3: Memory Check ==========
            t_ms = _now_ms()
            cached_sql = None
            cached_confidence = 0.0
            if relevant_tables:
                cached_sql, cached_confidence = self.memory.check_memory_for_query(
                    query=processed_query,
                    relevant_tables=relevant_tables,
                    session_id=session_id
                )
            trace(rid, "MEM_CHECK",
                  ms=_ms_since(t_ms),
                  cache_hit=bool(cached_sql),
                  cached_confidence=round(float(cached_confidence or 0.0), 3),
                  cached_preview=_preview_text(cached_sql))

            # ========== STEP 4: SQL Generation & Execution ==========
            final_sql = ""
            llm_attempts = 0
            rows: List[Tuple[Any, ...]] = []
            columns: List[str] = []
            execution_error: Optional[str] = None

            def _exec_sql(sql_text: str, tag: str) -> Tuple[List[Tuple[Any, ...]], List[str], int]:
                exec_ms0 = _now_ms()
                try:
                    out_rows, out_cols = self.db_service.run_select(sql_text, max_rows=1000, query_timeout=10)
                    exec_ms = _ms_since(exec_ms0)
                    trace(rid, f"EXEC_OK", tag=tag, ms=exec_ms, rows=len(out_rows), cols=len(out_cols))
                    return out_rows, out_cols, exec_ms
                except Exception as e:
                    exec_ms = _ms_since(exec_ms0)
                    trace(rid, f"EXEC_ERR", tag=tag, ms=exec_ms, error=str(e))
                    raise

            # Branch A: cached SQL
            if cached_sql:
                t_ms = _now_ms()
                final_sql = normalize_sql_columns(cached_sql)
                final_sql = self.vector_search.anchor_sql_dates(final_sql, current_year)

                ok, corrected, warns = validate_generated_sql(final_sql, current_year, data_anchor_year=None)
                (sql_warnings or []).extend(warns or [])
                if corrected and corrected != final_sql:
                    final_sql = corrected
                    trace(rid, "SQL_VALIDATED", corrected=True, warn_count=len(sql_warnings or []))
                else:
                    trace(rid, "SQL_VALIDATED", corrected=False, warn_count=len(sql_warnings or []))

                persist_sql(rid, final_sql, stage="SQL_CACHED_FINAL")
                rows, columns, _ = _exec_sql(final_sql, tag="CACHED")
                trace(rid, "CACHED_SQL_DONE", ms=_ms_since(t_ms), row_count=len(rows), col_count=len(columns))

            # Branch B: LLM path
            elif relevant_tables:
                t_ms = _now_ms()
                try:
                    rows, columns, final_sql, llm_attempts = self.llm_service.run_query_with_llm_repair(
                        db_service=self.db_service,
                        user_question=processed_query,
                        schema=schema_context,
                        join_hints=join_hints,
                        params=None,
                        max_rows=1000,
                        query_timeout=10,
                        max_attempts=3,
                    )
                    trace(rid, "LLM_DONE",
                          ms=_ms_since(t_ms),
                          llm_attempts=int(llm_attempts or 0),
                          sql_preview=_preview_text(final_sql))

                    final_sql = normalize_sql_columns(final_sql or "")
                    final_sql = self.vector_search.anchor_sql_dates(final_sql, current_year)

                    ok, corrected, warns = validate_generated_sql(final_sql, current_year, data_anchor_year=None)
                    (sql_warnings or []).extend(warns or [])
                    if corrected and corrected != final_sql:
                        final_sql = corrected
                        persist_sql(rid, final_sql, stage="SQL_REPAIRED_FINAL")
                        rows, columns, _ = _exec_sql(final_sql, tag="LLM_REPAIR_RERUN")
                        trace(rid, "LLM_SQL_REPAIR_RERUN", warn_count=len(sql_warnings or []))
                    else:
                        persist_sql(rid, final_sql, stage="SQL_LLM_FINAL")
                        # If first run had rows/columns already returned by service, keep them.
                        # Safety: if empty, execute once ourselves
                        if not rows and final_sql:
                            rows, columns, _ = _exec_sql(final_sql, tag="LLM_SINGLE_RUN")

                except Exception as e:
                    execution_error = str(e)
                    trace(rid, "LLM_PATH_ERR", error=_preview_text(execution_error, 400))

            else:
                # No tables found → deterministic empty
                final_sql = "SELECT 1 WHERE 1=0"
                trace(rid, "NO_TABLES_FOUND", reason="vector retrieval returned no candidates")

            # ========== STEP 5: Memory Update ==========
            mem_ms0 = _now_ms()
            try:
                if execution_error is None:
                    self.memory.learn_from_query(
                        query=processed_query,
                        relevant_tables=relevant_tables,
                        generated_sql=final_sql,
                        success=True,
                        execution_time=_ms_since(t_ms) / 1000.0 if 't_ms' in locals() else 0.0,
                        session_id=session_id,
                    )
                    self.memory.record_success(
                        session_id=session_id,
                        query=processed_query,
                        generated_sql=final_sql,
                        columns=columns,
                        rows=rows,
                        relevant_tables=relevant_tables,
                        schema_ctx=schema_context,
                    )
                    trace(rid, "MEM_UPDATED", success=True)
                else:
                    self.memory.learn_from_query(
                        query=processed_query,
                        relevant_tables=relevant_tables,
                        generated_sql=final_sql or "",
                        success=False,
                        execution_time=_ms_since(t_ms) / 1000.0 if 't_ms' in locals() else 0.0,
                        session_id=session_id,
                    )
                    trace(rid, "MEM_UPDATED", success=False, error=_preview_text(execution_error))
            except Exception as me:
                trace(rid, "MEM_UPDATE_ERR", ms=_ms_since(mem_ms0), error=str(me))

            # ========== STEP 6: Display Enrichment & Explanation ==========
            if execution_error:
                msg = "查詢執行失敗：" + execution_error if lang == "zh-tw" else "Query execution failed: " + execution_error
                explanation = msg
                table_md = ""
                visualization_payload = None
                columns_enriched = normalize_column_labels(columns or [])
                rows_enriched = [tuple(r) for r in (rows or [])]
                trace(rid, "ENRICH_SKIP", error=_preview_text(execution_error))
            else:
                enr_ms0 = _now_ms()
                columns_enriched, rows_enriched = self._enrich_rows_for_display(columns, rows)
                columns_enriched = normalize_column_labels(columns_enriched or [])
                rows_enriched = [tuple(r) for r in (rows_enriched or [])]
                trace(rid, "ENRICH_DONE",
                      ms=_ms_since(enr_ms0),
                      rows=len(rows_enriched),
                      cols=len(columns_enriched))

                aggregates = self.data_analyzer.compute_aggregates(rows_enriched, columns_enriched)
                sample_text = format_sample_data(rows_enriched, columns_enriched)

                if (lang or "").lower() in ("zh-tw", "zh_tw", "zh", "zh-hant", "zh-hk", "zh-mo"):
                    explanation = self.llm_service.generate_explanation_chinese(
                        processed_query, len(rows_enriched), columns_enriched, aggregates, sample_text
                    )
                else:
                    explanation = self.llm_service.generate_explanation_english(
                        processed_query, len(rows_enriched), columns_enriched, aggregates, sample_text
                    )

                want_details = self._should_show_details(processed_query, lang)
                preferred_cols = [
                    "Department", "department_name",
                    "EmployeeID", "employeeid",
                    "Name", "truename", "name", "person_name",
                    "LeaveType", "classname", "attendancetype", "leavetype",
                    "HOURS", "TIMECLASSHOURS",
                    "StartDate", "WORKDATE", "EndDate"
                ]

                table_md = self._markdown_table(
                    columns_enriched,
                    rows_enriched,
                    limit=20,
                    keep=preferred_cols if want_details else None,
                )

                # ========== STEP 7: Visualization (Optional) ==========
                visualization_payload = None
                if (include_visualization and rows_enriched and columns_enriched and
                    (self._should_generate_chart(processed_query, lang, columns_enriched) or force_chart_type)):
                    try:
                        trace(rid, "VIZ_START",
                              rows=len(rows_enriched),
                              cols=len(columns_enriched),
                              forced=bool(force_chart_type))
                        df = _build_safe_dataframe(columns_enriched, rows_enriched)
                        df = _maybe_melt_dates(df)

                        for col in df.columns:
                            cl = str(col).lower()
                            if any(k in cl for k in ["date", "day", "time", "workdate", "startdate", "enddate"]):
                                try:
                                    df[col] = pd.to_datetime(df[col], errors="ignore")
                                except Exception:
                                    pass
                            if any(k in cl for k in ["hour", "hours", "timeclasshours", "duration"]):
                                try:
                                    df[col] = pd.to_numeric(df[col], errors="coerce")
                                except Exception:
                                    pass

                        forced_type = force_chart_type or self._infer_forced_chart_type(processed_query, lang)
                        if forced_type:
                            visualization_payload = self.viz_service.create_visualization(
                                df, user_query=processed_query, force_chart_type=forced_type, title=None
                            )
                        else:
                            visualization_payload = self.chart_agent.generate_chart(
                                df=df,
                                user_query=processed_query,
                                sql_meta={"sql": final_sql, "tables": relevant_tables, "lang": lang},
                                force_chart_type=None,
                                title=None
                            )

                        if visualization_payload and lang == "zh-tw":
                            reason = visualization_payload.get("reasoning", "")
                            visualization_payload["reasoning"] = f"圖表推薦：{reason}"
                        trace(rid, "VIZ_DONE", ok=True, chart_type=visualization_payload.get("type") if isinstance(visualization_payload, dict) else None)
                    except Exception as viz_e:
                        visualization_payload = {"enabled": False, "reason": f"Visualization error: {viz_e}"}
                        trace(rid, "VIZ_FAIL", ok=False, error=str(viz_e))

            # ========== STEP 8: Build Response ==========
            memory_stats = self.memory.get_memory_stats()

            response = {
                "original_text": user_input,
                "detected_language": lang,
                "language_confidence": query_context.get("language_confidence", 0.9),
                "processed_query": processed_query,
                "intent": "generic",
                "schema": schema_name,
                "relevant_tables": [
                    {"table": t, "score": round(s, 3)} 
                    for t, s in (table_scores or [])
                ],
                "generated_sql": final_sql or "SELECT 1 WHERE 1=0",
                "llm_attempts": int(llm_attempts or 0),
                "execution_successful": execution_error is None,
                "execution_error": execution_error,

                # Raw results (backwards compatibility)
                "columns": columns,
                "results": [[jsonable_value(v) for v in r] for r in rows],
                "row_count": len(rows),

                # Enriched results (for UI)
                "columns_enriched": columns_enriched if not execution_error else columns,
                "results_enriched": [[jsonable_value(v) for v in r] for r in (rows_enriched if not execution_error else rows)],
                "table_markdown": table_md if (not execution_error and table_md) else "",

                # Explanation
                "explanation": explanation if not execution_error else (execution_error or ""),
                "summary": explanation if not execution_error else (execution_error or ""),
                "success": execution_error is None,
                "language_native_processing": True,

                # Visualization
                "visualization": (visualization_payload if (execution_error is None and include_visualization) else None),
                "visualization_requested": bool(include_visualization),

                # Memory stats
                "memory": {
                    "session_id": session_id,
                    "used_cached_sql": bool(cached_sql),
                    "cached_confidence": float(cached_confidence or 0.0),
                    "cache_hit_rate": memory_stats.get("cache_hit_rate"),
                    "language_aware": True,
                },

                # SQL validator warnings
                "sql_warnings": sql_warnings,

                # Metadata
                "processing_time_ms": _ms_since(start_ms),
                "current_year": current_year,
            }

            trace(rid, "ORCH_DONE",
                  ms=_ms_since(start_ms),
                  lang=lang,
                  rows=len(rows),
                  cols=len(columns),
                  viz=bool(response.get("visualization")),
                  sql_saved=bool(persist_sql(rid, final_sql, stage="SQL_RESPONSE_FINAL")) if final_sql else False)

            logger.info("rid=%s pipeline_ok time=%dms lang=%s rows=%d", rid, _ms_since(start_ms), lang, len(rows))
            return response

        except Exception as e:
            trace(rid, "ORCH_FAIL", ms=_ms_since(start_ms), error=str(e))
            logger.error("rid=%s pipeline_failed time=%dms error=%s", rid, _ms_since(start_ms), e, exc_info=True)
            lang = "zh-tw" if any('\u4e00' <= c <= '\u9fff' for c in user_input) else "en"
            msg = "處理您的查詢時發生錯誤。" if lang == "zh-tw" else "An error occurred while processing your query."
            return {
                "original_text": user_input,
                "detected_language": lang,
                "language_confidence": 0.5,
                "execution_successful": False,
                "execution_error": str(e),
                "summary": msg,
                "explanation": msg,
                "success": False,
                "language_native_processing": True,
                "visualization": None,
                "visualization_requested": False,
                "processing_time_ms": _ms_since(start_ms),
            }
