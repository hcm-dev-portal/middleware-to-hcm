# backend/app/services/nlp/leave_nlp_service.py
"""
Leave NLP Pipeline Service - AWS Bedrock Edition
=================================================

Orchestrates the Leave AI Assistant with:
- AWS Bedrock (Claude) for LLM operations
- Vector search for intent routing
- Comprehensive debugging and SQL logging
"""
from __future__ import annotations

import logging
import uuid
import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Literal

from app.services.llm.llm_service import (
    LLMService,
    detect_query_language,
)
from app.services.vector_search_service import (
    VectorSearchService,
)

logger = logging.getLogger(__name__)

Language = Literal["zh-tw", "en"]

# ──────────────────────────────────────────────────────────────────────
# Debug logging configuration
# ──────────────────────────────────────────────────────────────────────
# Set to True to enable verbose SQL and result logging
DEBUG_SQL_LOGGING = True
DEBUG_RESULT_LOGGING = True
DEBUG_MAX_ROWS_TO_LOG = 10  # Limit rows logged to avoid huge outputs


def _json_safe(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-serializable forms.

    - datetime/date → ISO string
    - Enum → value
    - set → list
    - dict → dict with json-safe values
    - list/tuple → list with json-safe values
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]

    return str(obj)


def _format_sql_for_logging(sql: Optional[str], indent: int = 4) -> str:
    """Format SQL for readable logging output."""
    if not sql:
        return "(no SQL generated)"
    
    indent_str = " " * indent
    lines = sql.strip().split("\n")
    formatted = "\n".join(f"{indent_str}{line}" for line in lines)
    return f"\n{formatted}"


def _format_results_for_logging(
    rows: List[Any],
    columns: List[str],
    max_rows: int = DEBUG_MAX_ROWS_TO_LOG,
) -> str:
    """Format query results for readable logging output."""
    if not rows or not columns:
        return "(no results)"
    
    lines = []
    
    # Header
    header = " | ".join(f"{col[:20]:<20}" for col in columns)
    lines.append(f"    {header}")
    lines.append(f"    {'-' * len(header)}")
    
    # Data rows (limited)
    for i, row in enumerate(rows[:max_rows]):
        row_str = " | ".join(
            f"{str(v)[:20] if v is not None else 'NULL':<20}" 
            for v in row
        )
        lines.append(f"    {row_str}")
    
    if len(rows) > max_rows:
        lines.append(f"    ... ({len(rows) - max_rows} more rows)")
    
    return "\n".join(lines)


class LeaveNLPPipeline:
    """
    Orchestrator for the Leave AI Assistant.

    Now powered by AWS Bedrock (Claude) with enhanced debugging.

    Pipeline Flow:
        Controller
           ↓
        LeaveNLPPipeline.answer_question(...)
           1. Normalize + validate user_question
           2. Detect language (zh-tw / en)
           3. Build PLAN via VectorSearchService.plan_for(...)
           4. (Optional) Deep debug of vector index
           5. Call LLMService.answer_question(...) [Bedrock]
           6. Log SQL and results for debugging
           7. Attach all debug info to the final result

    Debug Features:
        - SQL generation logging with formatted output
        - Query results logging (configurable row limit)
        - Intent/slot extraction logging
        - Vector search debug information
        - Request timing metrics
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        vector_service: Optional[VectorSearchService] = None,
        debug_sql: bool = DEBUG_SQL_LOGGING,
        debug_results: bool = DEBUG_RESULT_LOGGING,
    ) -> None:
        """
        Initialize the NLP pipeline.
        
        Args:
            llm_service: Optional pre-configured LLMService (uses Bedrock by default)
            vector_service: Optional pre-configured VectorSearchService
            debug_sql: Enable SQL logging (default: True)
            debug_results: Enable results logging (default: True)
        """
        # Core LLM pipeline (SQL + DB + explanation) - Now uses Bedrock
        self.llm = llm_service or LLMService()
        
        # Debug flags
        self.debug_sql = debug_sql
        self.debug_results = debug_results

        # Vector-based planner / schema context
        self._vector_service: Optional[VectorSearchService] = vector_service
        self._vector_initialized: bool = vector_service is not None
        
        # Log initialization status
        logger.info(
            "[LeaveNLP] Initialized: llm_enabled=%s, debug_sql=%s, debug_results=%s",
            self.llm.llm_enabled,
            self.debug_sql,
            self.debug_results,
        )
        
        if not self.llm.llm_enabled:
            init_error = getattr(self.llm, 'bedrock', None)
            if init_error and hasattr(init_error, 'get_init_error'):
                logger.warning(
                    "[LeaveNLP] LLM not available: %s",
                    init_error.get_init_error(),
                )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def answer_question(
        self,
        db_service: Any,
        user_question: str,
        schema: str,
        join_hints: str,
        *,
        max_rows: int = 1000,
        query_timeout: int = 10,
        max_attempts: int = 3,
        rid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for the Leave AI assistant.

        Args:
            db_service: Database service for executing queries
            user_question: Natural language question from user
            schema: Fallback schema if planner doesn't provide one
            join_hints: Fallback join hints if planner doesn't provide them
            max_rows: Maximum rows to return
            query_timeout: Query timeout in seconds
            max_attempts: Max LLM repair attempts
            rid: Optional request ID for correlation

        Returns:
            Dict containing:
                - question, language_detected, sql, rows, columns
                - attempts, aggregates, explanation_zh
                - success, error, error_category
                - debug block with full transparency
        """
        import time
        start_time = time.perf_counter()
        
        # Per-request correlation ID for logs
        request_id = rid or self._new_request_id()

        user_question = (user_question or "").strip()
        
        # ══════════════════════════════════════════════════════════════
        # DEBUG: Log incoming request
        # ══════════════════════════════════════════════════════════════
        logger.info("=" * 70)
        logger.info("[LeaveNLP][%s] ═══ NEW REQUEST ═══", request_id)
        logger.info("[LeaveNLP][%s] Question: %r", request_id, user_question)
        logger.info("=" * 70)
        
        if not user_question:
            logger.warning("[LeaveNLP][%s] Empty question - rejecting", request_id)
            return self._empty_question_response(request_id)

        # 1) Detect language
        detected_lang: Language = detect_query_language(user_question)
        logger.info(
            "[LeaveNLP][%s] Language detected: %s",
            request_id,
            detected_lang,
        )

        # 2) Ensure vector service is initialized
        vector = self._ensure_vector_service(db_service=db_service)
        planner_plan: Optional[Dict[str, Any]] = None
        vector_debug: Optional[Dict[str, Any]] = None

        # 3) Build PLAN via VectorSearchService
        planner_plan = self._build_plan_safe(
            vector=vector,
            question=user_question,
            language=detected_lang,
            request_id=request_id,
        )
        
        # ══════════════════════════════════════════════════════════════
        # DEBUG: Log planner output
        # ══════════════════════════════════════════════════════════════
        if planner_plan:
            self._log_planner_output(request_id, planner_plan)

        # 4) Vector debug search
        vector_debug = self._debug_vector_safe(
            vector=vector,
            question=user_question,
            language=detected_lang,
            request_id=request_id,
        )

        # 5) Prepare schema + join_hints for LLMService
        if planner_plan:
            planner_schema = planner_plan.get("schema") or ""
            planner_join_hints = planner_plan.get("join_hints") or ""
            intent_context = planner_plan.get("intent_context") or {}
            tables_from_plan = planner_plan.get("tables") or []
        else:
            planner_schema = ""
            planner_join_hints = ""
            intent_context = {}
            tables_from_plan = []

        schema_for_llm = planner_schema or schema or ""
        join_hints_for_llm = planner_join_hints or join_hints or ""

        logger.info(
            "[LeaveNLP][%s] PLAN_READY: tables=%s, template_ref=%s",
            request_id,
            tables_from_plan,
            intent_context.get("template_ref") if intent_context else None,
        )

        # 6) Call LLMService (Bedrock)
        logger.info("[LeaveNLP][%s] Calling LLMService (Bedrock)...", request_id)
        
        llm_result = self.llm.answer_question(
            db_service=db_service,
            user_question=user_question,
            schema=schema_for_llm,
            join_hints=join_hints_for_llm,
            intent_context=intent_context,
            table_whitelist=None,
            max_rows=max_rows,
            query_timeout=query_timeout,
            max_attempts=max_attempts,
            allow_fallback=False,
        )

        # 7) Handle non-dict response (shouldn't happen)
        if not isinstance(llm_result, dict):
            logger.error(
                "[LeaveNLP][%s] LLMService returned non-dict: %r",
                request_id,
                type(llm_result),
            )
            return self._internal_error_response(
                request_id, detected_lang, planner_plan, vector_debug
            )

        result: Dict[str, Any] = dict(llm_result)

        # ══════════════════════════════════════════════════════════════
        # DEBUG: Log generated SQL
        # ══════════════════════════════════════════════════════════════
        if self.debug_sql:
            self._log_generated_sql(request_id, result)

        # ══════════════════════════════════════════════════════════════
        # DEBUG: Log query results
        # ══════════════════════════════════════════════════════════════
        if self.debug_results:
            self._log_query_results(request_id, result)

        # Ensure success flag exists
        if "success" not in result:
            result["success"] = False

        # Language detection metadata
        result.setdefault("language_detected", detected_lang)
        result["pipeline_language_detected"] = detected_lang

        # Attach intent context
        result["intent_context"] = intent_context

        # Build debug block
        llm_debug = {
            "attempts": result.get("attempts"),
            "final_sql": result.get("sql"),
            "llm_enabled": self.llm.llm_enabled,
            "model_id": getattr(self.llm, 'model_id', 'unknown'),
        }

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        
        result["debug"] = {
            "request_id": request_id,
            "pipeline_language_detected": detected_lang,
            "planner_plan": planner_plan,
            "vector_debug_search": vector_debug,
            "llm_debug": llm_debug,
            "elapsed_ms": elapsed_ms,
        }

        # ══════════════════════════════════════════════════════════════
        # DEBUG: Log final summary
        # ══════════════════════════════════════════════════════════════
        self._log_final_summary(request_id, result, elapsed_ms)

        # Ensure JSON-serializable
        safe_result = _json_safe(result)
        return safe_result

    # ------------------------------------------------------------------
    # Debug logging methods
    # ------------------------------------------------------------------
    def _log_planner_output(self, request_id: str, plan: Dict[str, Any]) -> None:
        """Log planner/intent routing output."""
        logger.info("[LeaveNLP][%s] ─── PLANNER OUTPUT ───", request_id)
        
        intent_ctx = plan.get("intent_context", {})
        logger.info(
            "[LeaveNLP][%s]   template_ref: %s",
            request_id,
            intent_ctx.get("template_ref"),
        )
        logger.info(
            "[LeaveNLP][%s]   slots: %s",
            request_id,
            intent_ctx.get("slots", {}),
        )
        logger.info(
            "[LeaveNLP][%s]   tables: %s",
            request_id,
            plan.get("tables", []),
        )
        
        # Log few-shot SQL if available
        few_shot = intent_ctx.get("few_shot_sql") or intent_ctx.get("example_sql")
        if few_shot:
            logger.info(
                "[LeaveNLP][%s]   few_shot_sql: %s",
                request_id,
                _format_sql_for_logging(few_shot),
            )

    def _log_generated_sql(self, request_id: str, result: Dict[str, Any]) -> None:
        """Log the generated SQL query."""
        sql = result.get("sql")
        attempts = result.get("attempts", 0)
        success = result.get("success", False)
        
        logger.info("[LeaveNLP][%s] ─── GENERATED SQL ───", request_id)
        logger.info(
            "[LeaveNLP][%s]   Attempts: %d | Success: %s",
            request_id,
            attempts,
            success,
        )
        
        if sql:
            logger.info(
                "[LeaveNLP][%s]   SQL:%s",
                request_id,
                _format_sql_for_logging(sql),
            )
        else:
            logger.warning(
                "[LeaveNLP][%s]   SQL: (none generated)",
                request_id,
            )
            if result.get("error"):
                logger.warning(
                    "[LeaveNLP][%s]   Error: %s",
                    request_id,
                    result.get("error"),
                )

    def _log_query_results(self, request_id: str, result: Dict[str, Any]) -> None:
        """Log query results summary and sample data."""
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        aggregates = result.get("aggregates", {})
        
        logger.info("[LeaveNLP][%s] ─── QUERY RESULTS ───", request_id)
        logger.info(
            "[LeaveNLP][%s]   Row count: %d",
            request_id,
            len(rows),
        )
        logger.info(
            "[LeaveNLP][%s]   Columns: %s",
            request_id,
            columns,
        )
        
        # Log aggregates
        if aggregates:
            logger.info(
                "[LeaveNLP][%s]   Aggregates: unique_people=%s, total_hours=%s, by_type=%s",
                request_id,
                aggregates.get("unique_people"),
                aggregates.get("total_hours"),
                aggregates.get("by_leave_type", {}),
            )
        
        # Log sample rows
        if rows and columns:
            logger.info(
                "[LeaveNLP][%s]   Sample data:\n%s",
                request_id,
                _format_results_for_logging(rows, columns),
            )

    def _log_final_summary(
        self, request_id: str, result: Dict[str, Any], elapsed_ms: int
    ) -> None:
        """Log final request summary."""
        logger.info("=" * 70)
        logger.info("[LeaveNLP][%s] ═══ REQUEST COMPLETE ═══", request_id)
        logger.info(
            "[LeaveNLP][%s]   Success: %s | Rows: %d | Attempts: %d | Time: %dms",
            request_id,
            result.get("success"),
            len(result.get("rows", [])),
            result.get("attempts", 0),
            elapsed_ms,
        )
        
        if not result.get("success"):
            logger.warning(
                "[LeaveNLP][%s]   Error: %s (%s)",
                request_id,
                result.get("error"),
                result.get("error_category"),
            )
        
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------
    def _empty_question_response(self, request_id: str) -> Dict[str, Any]:
        """Build response for empty question."""
        return {
            "question": "",
            "language_detected": "zh-tw",
            "sql": None,
            "rows": [],
            "columns": [],
            "attempts": 0,
            "aggregates": {
                "row_count": 0,
                "unique_people": None,
                "by_leave_type": {},
                "total_hours": None,
            },
            "explanation_zh": (
                "### 摘要\n"
                "• 問題不可為空白。\n\n"
                "### 資料品質說明\n"
                "• 請輸入有效的查詢問題。"
            ),
            "success": False,
            "error": "問題不可為空白。",
            "error_category": "validation_error",
            "debug": {
                "request_id": request_id,
                "pipeline_language_detected": "zh-tw",
                "planner_plan": None,
                "vector_debug_search": None,
                "llm_debug": None,
            },
        }

    def _internal_error_response(
        self,
        request_id: str,
        detected_lang: Language,
        planner_plan: Optional[Dict[str, Any]],
        vector_debug: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build response for internal errors."""
        return {
            "question": "",
            "language_detected": detected_lang,
            "sql": None,
            "rows": [],
            "columns": [],
            "attempts": 0,
            "aggregates": {
                "row_count": 0,
                "unique_people": None,
                "by_leave_type": {},
                "total_hours": None,
            },
            "explanation_zh": (
                "### 摘要\n"
                "• 系統內部錯誤，無法完成查詢。\n\n"
                "### 資料品質說明\n"
                "• 請稍後再試或聯絡系統管理員。"
            ),
            "success": False,
            "error": "LLM pipeline internal error",
            "error_category": "pipeline_internal_error",
            "debug": {
                "request_id": request_id,
                "pipeline_language_detected": detected_lang,
                "planner_plan": planner_plan,
                "vector_debug_search": vector_debug,
                "llm_debug": None,
            },
        }

    # ------------------------------------------------------------------
    # Vector service init
    # ------------------------------------------------------------------
    def _ensure_vector_service(self, db_service: Any) -> VectorSearchService:
        """
        Lazily construct VectorSearchService when db_service is first available.
        """
        if self._vector_initialized and self._vector_service is not None:
            return self._vector_service

        logger.info(
            "[LeaveNLP] Initializing VectorSearchService with db_service=%r",
            type(db_service).__name__,
        )
        self._vector_service = VectorSearchService(db_service=db_service)
        self._vector_initialized = True
        return self._vector_service

    # ------------------------------------------------------------------
    # Planner integration – safe wrapper
    # ------------------------------------------------------------------
    def _build_plan_safe(
        self,
        *,
        vector: VectorSearchService,
        question: str,
        language: Language,
        request_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Wrap VectorSearchService.plan_for with defensive error handling.
        """
        try:
            plan = vector.plan_for(
                question,
                schema_filter=None,
                rid=request_id,
            )
            logger.debug(
                "[LeaveNLP][%s] PLAN: lang=%s tables=%s template_ref=%s",
                request_id,
                plan.get("language"),
                plan.get("tables"),
                (plan.get("intent_context") or {}).get("template_ref"),
            )
            return plan
        except Exception as e:
            logger.error(
                "[LeaveNLP][%s] PLAN_FATAL: %s: %s (question=%r)",
                request_id,
                type(e).__name__,
                e,
                question,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Vector debug – what did the index actually search?
    # ------------------------------------------------------------------
    def _debug_vector_safe(
        self,
        *,
        vector: VectorSearchService,
        question: str,
        language: Language,
        request_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Use VectorSearchService.debug_search to introspect search behaviour.
        """
        try:
            dbg = vector.debug_search(question, language=language)
            logger.debug(
                "[LeaveNLP][%s] VECTOR_DEBUG: tried=%s results=%s",
                request_id,
                dbg.get("tried"),
                dbg.get("results_count"),
            )
            return dbg
        except Exception as e:
            logger.error(
                "[LeaveNLP][%s] VECTOR_DEBUG_FATAL: %s: %s (question=%r)",
                request_id,
                type(e).__name__,
                e,
                question,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Legacy slot extraction (kept for compatibility)
    # ------------------------------------------------------------------
    def _simple_slot_extraction(
        self,
        question: str,
        *,
        language: Language,
    ) -> Dict[str, Any]:
        """
        Legacy rule-based slot extraction.
        Not used in main pipeline but available for custom use.
        """
        import re
        from datetime import date
        
        q = question.strip()
        slots: Dict[str, Any] = {}
        
        current_year = date.today().year

        if "今年" in q:
            slots["year"] = current_year
        if "去年" in q:
            slots["year"] = current_year - 1

        if any(k in q for k in ["本月", "這個月"]):
            slots["month_scope"] = "current_month"
        if any(k in q for k in ["上個月", "上月"]):
            slots["month_scope"] = "previous_month"

        m = re.search(r"(?:員工|員編)\s*([0-9]{3,10})", q)
        if m:
            slots["emp_no"] = m.group(1)

        if "今天" in q or "目前" in q or "現在" in q:
            slots["today_scope"] = True

        return slots

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _new_request_id(self) -> str:
        """
        Short, log-friendly correlation ID.
        """
        return uuid.uuid4().hex[:8]


# ──────────────────────────────────────────────────────────────────────
# Convenience function for quick testing
# ──────────────────────────────────────────────────────────────────────
def create_pipeline(
    debug_sql: bool = True,
    debug_results: bool = True,
) -> LeaveNLPPipeline:
    """
    Factory function to create a configured pipeline.
    
    Usage:
        from app.services.nlp.leave_nlp_service import create_pipeline
        
        pipeline = create_pipeline(debug_sql=True, debug_results=True)
        result = pipeline.answer_question(db_service, question, schema, join_hints)
    """
    return LeaveNLPPipeline(
        debug_sql=debug_sql,
        debug_results=debug_results,
    )