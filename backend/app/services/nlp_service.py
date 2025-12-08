# backend/app/services/nlp/leave_nlp_service.py
from __future__ import annotations

import logging
import uuid
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


import datetime
from enum import Enum

def _json_safe(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-serializable forms.

    - datetime/date → ISO string
    - Enum → value
    - set → list
    - dict → dict with json-safe values
    - list/tuple → list with json-safe values
    """
    # primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # datetime-like
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    # enums
    if isinstance(obj, Enum):
        return obj.value

    # mappings
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    # sets / tuples / lists
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]

    # fallback: string representation
    return str(obj)

class LeaveNLPPipeline:
    """
    Orchestrator for the Leave AI Assistant (zh-tw / en).

    NEW PIPELINE (fully transparent):

        Controller
           ↓
        LeaveNLPPipeline.answer_question(...)
           1. Normalize + validate user_question
           2. Detect language (zh-tw / en)
           3. Build PLAN via VectorSearchService.plan_for(...)
                - intent_context (template_ref, slots, candidates, tables)
                - tables
                - join_hints
                - schema (enhanced: recipes + DB schema)
           4. (Optional) Deep debug of vector index via VectorSearchService.debug_search(...)
           5. Call LLMService.answer_question(...) using planner schema/join_hints
                - LLM generates SQL
                - LLMService executes via db_service.run_select(...)
                - LLMService computes aggregates + zh-tw explanation
           6. Attach **all debug info** to the final result:
                - request_id
                - pipeline language detection
                - planner plan
                - vector debug search results
                - LLM attempts + final SQL

    STRICT:
        - No fake "SELECT 1 WHERE 1=0".
        - No success=True if LLMService says failure.
        - Errors are surfaced with clear categories (from LLMService).
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        vector_service: Optional[VectorSearchService] = None,
    ) -> None:
        # Core LLM pipeline (SQL + DB + explanation)
        self.llm = llm_service or LLMService()

        # Vector-based planner / schema context
        # NOTE: VectorSearchService requires db_service, so we create it on first call
        #       when we actually have db_service available.
        self._vector_service: Optional[VectorSearchService] = vector_service
        self._vector_initialized: bool = vector_service is not None

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

        Controller can still pass `schema` and `join_hints`, but **planner output
        always takes precedence**. Old arguments are only used as last-resort fallback.

        Returns the normal LLM payload (rows, columns, sql, aggregates, explanation)
        PLUS a `debug` block with full transparency of:

          - request_id
          - pipeline_language_detected
          - planner_plan (tables, join_hints, schema, intent_context)
          - vector_debug_search
          - llm_debug (attempts, final sql)
        """
        # Per-request correlation ID for logs
        request_id = rid or self._new_request_id()

        user_question = (user_question or "").strip()
        if not user_question:
            logger.warning("[LeaveNLP][%s] Empty question.", request_id)
            return {
                "question": user_question,
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

        # 1) Detect language (pipeline-level; planner + LLM will re-detect as needed)
        detected_lang: Language = detect_query_language(user_question)
        logger.info(
            "[LeaveNLP][%s] START: lang=%s question=%r",
            request_id,
            detected_lang,
            user_question[:120],
        )

        # 2) Ensure vector service is initialised with db_service
        vector = self._ensure_vector_service(db_service=db_service)
        planner_plan: Optional[Dict[str, Any]] = None
        vector_debug: Optional[Dict[str, Any]] = None

        # 3) Build PLAN via VectorSearchService.plan_for(...)
        planner_plan = self._build_plan_safe(
            vector=vector,
            question=user_question,
            language=detected_lang,
            request_id=request_id,
        )

        # 4) Deep vector debug: what did the index actually search / return?
        vector_debug = self._debug_vector_safe(
            vector=vector,
            question=user_question,
            language=detected_lang,
            request_id=request_id,
        )

        # 5) Prepare schema + join_hints for LLMService
        #    Planner output always wins; controller-provided schema/join_hints
        #    are only used as last fallback.
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
            "[LeaveNLP][%s] PLAN_READY: tables=%s template_ref=%s",
            request_id,
            tables_from_plan,
            (intent_context.get("template_ref") if intent_context else None),
        )

        # 6) Delegate to LLMService.answer_question (which internally uses
        #    run_query_with_llm_repair for SQL + DB, then explanation).
        llm_result = self.llm.answer_question(
            db_service=db_service,
            user_question=user_question,
            schema=schema_for_llm,
            join_hints=join_hints_for_llm,
            intent_context=intent_context,
            table_whitelist=None,        # Vector recipes already steer tables via schema & join_hints
            max_rows=max_rows,
            query_timeout=query_timeout,
            max_attempts=max_attempts,
            allow_fallback=False,        # STRICT: no silent "fake" queries
        )

        # 7) Pipeline-level integrity and debug stitching
        if not isinstance(llm_result, dict):
            logger.error(
                "[LeaveNLP][%s] LLMService returned non-dict: %r",
                request_id,
                type(llm_result),
            )
            return {
                "question": user_question,
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

        result: Dict[str, Any] = dict(llm_result)

        # Do not override LLMService's success flag; only make sure it exists
        if "success" not in result:
            result["success"] = False

        # Expose pipeline-level language detection, but don't override any
        # "language_detected" field LLMService might have set.
        result.setdefault("language_detected", detected_lang)
        result["pipeline_language_detected"] = detected_lang

        # Attach final intent context from planner (even in error cases)
        result["intent_context"] = intent_context

        # Build LLM debug summary (attempts + final SQL)
        llm_debug = {
            "attempts": result.get("attempts"),
            "final_sql": result.get("sql"),
        }

        # Attach full debug block
        result["debug"] = {
            "request_id": request_id,
            "pipeline_language_detected": detected_lang,
            "planner_plan": planner_plan,
            "vector_debug_search": vector_debug,
            "llm_debug": llm_debug,
        }

        logger.info(
            "[LeaveNLP][%s] DONE: success=%s rows=%s attempts=%s",
            request_id,
            result.get("success"),
            len(result.get("rows") or []),
            result.get("attempts"),
        )

        # >>> NEW: ensure everything is JSON-serializable <<<
        safe_result = _json_safe(result)
        return safe_result

        return result

    # ------------------------------------------------------------------
    # Vector service init
    # ------------------------------------------------------------------
    def _ensure_vector_service(self, db_service: Any) -> VectorSearchService:
        """
        Lazily construct VectorSearchService when db_service is first available.
        """
        if self._vector_initialized and self._vector_service is not None:
            return self._vector_service

        logger.info("[LeaveNLP] Initialising VectorSearchService with db_service=%r", type(db_service).__name__)
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
        Returns the plan dict or None if planning fails.
        """
        try:
            plan = vector.plan_for(
                question,
                schema_filter=None,  # you can restrict (e.g., "dbo") later if needed
                rid=request_id,
            )
            # plan contains: language, intent_context, tables, join_hints, schema
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
        Returns a dict that can be directly returned to the frontend under debug.vector_debug_search.
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
    # Simple zh-tw oriented slot extraction (still here if you want to use it
    # separately; not used directly because VectorSearchService already extracts
    # slots into intent_context).
    # ------------------------------------------------------------------
    def _simple_slot_extraction(
        self,
        question: str,
        *,
        language: Language,
    ) -> Dict[str, Any]:
        """
        Legacy rule-based slot extraction. You can still call this if you want
        extra slots merged into intent_context before sending to LLMService.
        Currently not invoked by the main pipeline.
        """
        q = question.strip()
        slots: Dict[str, Any] = {}

        # TODO: derive these from datetime.today() in real code
        if "今年" in q:
            slots["year"] = 2025
        if "去年" in q:
            slots["year"] = 2024

        if any(k in q for k in ["本月", "這個月"]):
            slots["month_scope"] = "current_month"
        if any(k in q for k in ["上個月", "上月"]):
            slots["month_scope"] = "previous_month"

        import re

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
