# backend/app/services/retrieval/vector_search_service.py
"""
Vector Search Service - Optimized for AWS Bedrock Edition
==========================================================

Optimizations:
- Fast keyword-based intent routing (skips vector search for known patterns)
- Simplified zh-tw only processing (no language switching overhead)
- Direct recipe SQL lookup (single source of truth from LeaveVectorDB)
- No @variable DECLARE blocks (Bedrock LLM uses inline GETDATE())
- Efficient caching with better key structure
"""
from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from app.services.leave_vector import (
    LeaveVectorDB,
    ORG_TABLE,
    VAC_RESULT_TABLE,
    build_leave_index,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
ENABLE_FAST_ROUTING = True  # Use keyword matching before vector search
CACHE_MAX_SIZE = 256  # Reduced from 512 - most queries are unique
DEBUG_TIMING = True  # Log timing for performance analysis


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def _today() -> datetime:
    return datetime.now()


def _iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _week_window(dt: Optional[datetime] = None) -> Tuple[str, str]:
    """Return (monday, sunday) of current week."""
    dt = dt or _today()
    monday = dt - timedelta(days=dt.weekday())
    sunday = monday + timedelta(days=6)
    return _iso(monday), _iso(sunday)


def _parse_date_range(txt: str) -> Optional[Tuple[str, str]]:
    """
    Parse date ranges like:
    - 9/22-9/26, 09/22~09/26, 9/22到9/26
    - 2025/9/1-2025/9/30
    """
    # Try MM/DD-MM/DD format first
    m = re.search(r'(\d{1,2})\s*/\s*(\d{1,2})\s*[-~至到]\s*(\d{1,2})\s*/\s*(\d{1,2})', txt or "")
    if m:
        y = _today().year
        m1, d1, m2, d2 = map(int, m.groups())
        try:
            return _iso(datetime(y, m1, d1)), _iso(datetime(y, m2, d2))
        except ValueError:
            pass
    
    # Try YYYY/MM/DD format
    m = re.search(r'(\d{4})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})\s*[-~至到]\s*(\d{4})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})', txt or "")
    if m:
        y1, m1, d1, y2, m2, d2 = map(int, m.groups())
        try:
            return _iso(datetime(y1, m1, d1)), _iso(datetime(y2, m2, d2))
        except ValueError:
            pass
    
    return None


def _extract_year(txt: str) -> Optional[int]:
    """Extract year from query like '2024年' or '2024'."""
    m = re.search(r'(20\d{2})\s*年?', txt or "")
    if m:
        return int(m.group(1))
    return None


def _extract_emp_no(txt: str) -> Optional[str]:
    """Extract employee number from query."""
    m = re.search(r'(?:員編|員工編號|工號)\s*[：:]?\s*(\d{5,10})', txt or "")
    if m:
        return m.group(1)
    return None


def _extract_threshold_hours(txt: str) -> Optional[int]:
    """Extract threshold hours from query like '大於200小時'."""
    m = re.search(r'(?:大於|超過|>|>=|≥)\s*(\d+)\s*(?:小時|時)', txt or "")
    if m:
        return int(m.group(1))
    return None


def _should_prefer_vac_result(q: str) -> bool:
    """Check if query is about annual leave balance."""
    keywords = ["餘額", "剩餘", "還有", "可用", "未用"]
    leave_keywords = ["年假", "特休", "休假"]
    return any(k in q for k in keywords) and any(k in q for k in leave_keywords)


def _merge_hits(hit_lists: Iterable[List[Tuple[str, float]]], top_k: int = 5) -> List[Tuple[str, float]]:
    """Merge multiple hit lists, keeping highest score per table."""
    best: Dict[str, float] = {}
    for hits in hit_lists:
        for table, score in (hits or []):
            if score > best.get(table, 0):
                best[table] = score
    return sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]


class _LRUCache(OrderedDict):
    """Simple LRU cache with bounded size."""
    
    def __init__(self, maxsize: int = CACHE_MAX_SIZE):
        super().__init__()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self:
            self.move_to_end(key)
            self.hits += 1
            return self[key]
        self.misses += 1
        return None
    
    def set(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.maxsize:
            self.popitem(last=False)
    
    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self)}


# ────────────────────────────────────────────────────────────────────────────────
# Fast Intent Router (keyword-based, no vector search)
# ────────────────────────────────────────────────────────────────────────────────
class FastIntentRouter:
    """
    Keyword-based intent routing - faster than vector search for known patterns.
    
    Returns (template_ref, tables, slots) or None if no match.
    """
    
    # Intent patterns: (keywords, negative_keywords, template_ref, tables)
    # NOTE: Order matters! More specific patterns should come first.
    PATTERNS = [
        # Employee history by ID (most specific - has employee number)
        (
            ["員編", "工號", "員工編號"],
            [],
            "person_history_by_empno",
            ["dbo.ATDLEAVEDATA", "dbo.PSNACCOUNT", "dbo.ATDATTENDANCECLASS"],
        ),
        # Threshold queries - MUST come before general balance queries
        # (e.g., "大於200小時", "超過100時")
        (
            ["大於", "超過", ">=", "≥"],
            [],
            "balance_year_threshold_hours",
            [VAC_RESULT_TABLE, "dbo.PSNACCOUNT"],
        ),
        # Today's leave
        (
            ["今天", "今日", "目前"],
            ["歷史", "紀錄"],
            "today_who_on_leave",
            ["dbo.ATDLEAVEDATA", "dbo.PSNACCOUNT", ORG_TABLE],
        ),
        # This week
        (
            ["本週", "這週", "這周"],
            [],
            "weekly_count_people",
            ["dbo.ATDLEAVEDATA", "dbo.PSNACCOUNT", ORG_TABLE],
        ),
        # Date range queries (has "/" or date separators)
        (
            ["/"],
            [],
            "range_who_on_leave",
            ["dbo.ATDLEAVEDATA", "dbo.PSNACCOUNT", ORG_TABLE],
        ),
        # Balance/remaining annual leave (general balance query - comes last)
        (
            ["餘額", "剩餘", "還有", "可用"],
            ["大於", "超過"],  # Exclude threshold queries
            "annual_balance_by_person",
            [VAC_RESULT_TABLE, "dbo.PSNACCOUNT", ORG_TABLE],
        ),
    ]
    
    @classmethod
    def route(cls, query: str) -> Optional[Dict[str, Any]]:
        """
        Fast keyword-based routing.
        Returns dict with template_ref, tables, slots or None.
        """
        if not query:
            return None
        
        q = query.strip()
        
        for keywords, negatives, template_ref, tables in cls.PATTERNS:
            # Check positive keywords
            if not any(k in q for k in keywords):
                continue
            # Check negative keywords
            if any(k in q for k in negatives):
                continue
            
            # Extract slots
            slots = cls._extract_slots(q, template_ref)
            
            return {
                "template_ref": template_ref,
                "tables": tables,
                "slots": slots,
                "source": "fast_router",
            }
        
        return None
    
    @classmethod
    def _extract_slots(cls, query: str, template_ref: str) -> Dict[str, Any]:
        """Extract relevant slots based on template."""
        slots: Dict[str, Any] = {}
        
        # Year
        year = _extract_year(query)
        if year:
            slots["year"] = year
        
        # Date range
        date_range = _parse_date_range(query)
        if date_range:
            slots["start_date"], slots["end_date"] = date_range
        
        # Week scope
        if any(k in query for k in ["本週", "這週", "這周"]):
            ws, we = _week_window()
            slots["week_start"] = ws
            slots["week_end"] = we
        
        # Today scope
        if any(k in query for k in ["今天", "今日", "目前"]):
            slots["today_scope"] = True
            slots["today"] = _iso(_today())
        
        # Employee number
        emp_no = _extract_emp_no(query)
        if emp_no:
            slots["emp_no"] = emp_no
        
        # Threshold hours
        threshold = _extract_threshold_hours(query)
        if threshold:
            slots["threshold_hours"] = threshold
        
        # Vacation type (default to annual leave = 1)
        if any(k in query for k in ["年假", "特休"]):
            slots["vacationtype"] = 1
        
        return slots


# ────────────────────────────────────────────────────────────────────────────────
# Main Service
# ────────────────────────────────────────────────────────────────────────────────
class VectorSearchService:
    """
    Vector search and intent routing service (zh-tw focused).
    
    Optimizations:
    - Fast keyword routing before vector search
    - Single zh-tw language path (no translation overhead)
    - Direct recipe SQL lookup from LeaveVectorDB
    - Efficient caching
    """
    
    # Template alias map for normalization
    TEMPLATE_ALIAS_MAP: Dict[str, str] = {
        "remaining_balance_by_person": "annual_balance_by_person",
        "balance_by_person": "annual_balance_by_person",
        "who_on_leave_in_range": "range_who_on_leave",
        "current_on_leave_by_dept": "today_who_on_leave",
    }
    
    def __init__(self, db_service):
        self.db_service = db_service
        self.vector: Optional[LeaveVectorDB] = None
        self.person_table: str = "dbo.PSNACCOUNT"
        
        # Caches
        self._plan_cache = _LRUCache(maxsize=CACHE_MAX_SIZE)
        self._table_cache = _LRUCache(maxsize=CACHE_MAX_SIZE)
        
        # Initialize
        self._initialize_vector_index()
        self._determine_person_table()
    
    def _initialize_vector_index(self) -> None:
        """Initialize the vector index."""
        t0 = time.perf_counter()
        try:
            self.vector = build_leave_index()
            elapsed = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "[VectorSearch] Index initialized in %dms: tables=%d, ready=%s",
                elapsed,
                len(self.vector.tables) if self.vector else 0,
                self.vector.is_ready() if self.vector else False,
            )
        except Exception as e:
            self.vector = None
            logger.error("[VectorSearch] Index init failed: %s", e, exc_info=True)
    
    def _determine_person_table(self) -> None:
        """Determine the person table name from vector index."""
        if self.vector:
            pt = getattr(self.vector, "_person_table", None)
            if pt:
                self.person_table = pt
    
    def health_check(self) -> Dict[str, Any]:
        """Return health status of the service."""
        try:
            if not self.vector:
                return {"ready": False, "error": "no vector index"}
            
            health = self.vector.health_check()
            health["plan_cache"] = self._plan_cache.stats()
            health["table_cache"] = self._table_cache.stats()
            return health
        except Exception as e:
            return {"ready": False, "error": str(e)}
    
    # ────────────────────────────────────────────────────────────────────
    # Intent Routing (Fast + Vector fallback)
    # ────────────────────────────────────────────────────────────────────
    def get_intent_routing(self, query: str) -> Dict[str, Any]:
        """
        Get intent routing for a query.
        
        Uses fast keyword routing first, falls back to vector search.
        """
        # 1) Try fast keyword routing
        if ENABLE_FAST_ROUTING:
            fast_result = FastIntentRouter.route(query)
            if fast_result:
                logger.debug(
                    "[VectorSearch] Fast routing matched: template=%s",
                    fast_result.get("template_ref"),
                )
                return {
                    "language": "zh-tw",
                    "intent": fast_result["template_ref"],
                    "template_ref": fast_result["template_ref"],
                    "tables": fast_result["tables"],
                    "slots": fast_result["slots"],
                    "candidates": [fast_result],
                    "source": "fast_router",
                }
        
        # 2) Fall back to vector search
        if not self.vector:
            return {"language": "zh-tw", "slots": {}, "candidates": []}
        
        try:
            routing = self.vector.get_intent_routing(query)
            routing.setdefault("slots", {})
            routing.setdefault("candidates", [])
            routing["source"] = "vector_search"
            return routing
        except Exception as e:
            logger.error("[VectorSearch] Intent routing failed: %s", e)
            return {"language": "zh-tw", "slots": {}, "candidates": []}
    
    # ────────────────────────────────────────────────────────────────────
    # Table Search
    # ────────────────────────────────────────────────────────────────────
    def find_relevant_tables(
        self,
        query: str,
        schema_filter: Optional[str] = None,
        rid: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Find relevant tables for a query using vector search."""
        if not self.vector:
            return []
        
        # Check cache
        cache_key = (query.strip().lower(), schema_filter or "")
        cached = self._table_cache.get(cache_key)
        if cached is not None:
            return cached
        
        t0 = time.perf_counter()
        try:
            hits = self.vector.search_relevant_tables(query, top_k=5)
            
            # Apply schema filter if provided
            if schema_filter:
                hits = [
                    (t, s) for t, s in hits
                    if t.lower().startswith(schema_filter.lower() + ".")
                ]
            
            elapsed = int((time.perf_counter() - t0) * 1000)
            if DEBUG_TIMING:
                logger.debug(
                    "[VectorSearch] Table search: %dms, %d hits",
                    elapsed, len(hits),
                )
            
            self._table_cache.set(cache_key, hits)
            return hits
            
        except Exception as e:
            logger.error("[VectorSearch] Table search failed: %s", e)
            return []
    
    # ────────────────────────────────────────────────────────────────────
    # Join Hints
    # ────────────────────────────────────────────────────────────────────
    def get_join_hints(self, tables: List[str]) -> str:
        """Get join hints for a list of tables."""
        if not self.vector or not tables:
            return ""
        
        try:
            hints = self.vector.join_hints(tables)
            return "\n".join(hints) if hints else ""
        except Exception as e:
            logger.error("[VectorSearch] Join hints failed: %s", e)
            return ""
    
    # ────────────────────────────────────────────────────────────────────
    # Schema Context
    # ────────────────────────────────────────────────────────────────────
    def get_schema_context(
        self,
        tables: List[str],
        query: str = "",
        max_cols: int = 64,
    ) -> str:
        """
        Get schema context for tables.
        
        Combines:
        - Vector DB schema context (with business rules)
        - Database schema (actual columns)
        """
        if not tables:
            return "No relevant tables found"
        
        # Ensure person table is included
        selected = list(dict.fromkeys(
            tables[:3] + ([self.person_table] if self.person_table not in tables[:3] else [])
        ))
        
        parts = []
        
        # 1) Vector schema context (business rules, recipes)
        if self.vector and query:
            try:
                vector_ctx = self.vector.get_schema_context(query, include_examples=True)
                if vector_ctx:
                    parts.append(vector_ctx)
            except Exception as e:
                logger.warning("[VectorSearch] Vector schema context failed: %s", e)
        
        # 2) Database schema
        try:
            db_schema = self.db_service.get_compact_schema_for(
                selected, max_columns_per_table=max_cols
            )
            if db_schema:
                parts.append(f"=== DATABASE SCHEMA ===\n{db_schema}")
        except Exception as e:
            logger.error("[VectorSearch] DB schema failed: %s", e)
            parts.append(f"Schema unavailable for {selected}")
        
        return "\n\n".join(parts)
    
    # ────────────────────────────────────────────────────────────────────
    # Planning
    # ────────────────────────────────────────────────────────────────────
    def plan_for(
        self,
        query: str,
        *,
        schema_filter: Optional[str] = None,
        rid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a query plan with intent, tables, schema, and join hints.
        
        This is the main entry point for the NLP pipeline.
        """
        t0 = time.perf_counter()
        
        # Check cache
        cache_key = (query.strip().lower(), schema_filter or "", rid or "")
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            logger.debug("[VectorSearch] Plan cache hit")
            return cached
        
        # 1) Get intent routing (fast router or vector)
        routing = self.get_intent_routing(query)
        
        # 2) Get tables from intent
        intent_tables = routing.get("tables") or []
        
        # 3) Supplement with vector search if needed
        if len(intent_tables) < 2:
            vector_tables = [t for t, _ in self.find_relevant_tables(query, schema_filter, rid)]
            intent_tables = list(dict.fromkeys(intent_tables + vector_tables))
        
        # 4) Prefer VAC_RESULT_TABLE for balance queries
        if _should_prefer_vac_result(query) and VAC_RESULT_TABLE not in intent_tables:
            intent_tables.insert(0, VAC_RESULT_TABLE)
        
        # 5) Get join hints
        join_hints = self.get_join_hints(intent_tables)
        
        # Add VAC-specific hints if relevant
        if any("ATDCALCUVACATIONRESULT" in t.upper() for t in intent_tables):
            vac_hint = (
                f"-- {VAC_RESULT_TABLE} 為年假/特休餘額的權威資料來源\n"
                f"-- 建議條件：REMAINDAYS > 0, VACATIONTYPE = 1 (年假)\n"
                f"-- 有效期間：今天介於 CANUSEDATE 與 DISABLEDDATE 之間\n"
                f"-- Join: PERSONID → dbo.PSNACCOUNT.PERSONID"
            )
            join_hints = f"{vac_hint}\n\n{join_hints}".strip()
        
        # 6) Get schema context
        schema_ctx = self.get_schema_context(intent_tables, query)
        
        # 7) Normalize template ref
        template_ref = routing.get("template_ref") or ""
        if _should_prefer_vac_result(query):
            template_ref = "annual_balance_by_person"
        canonical = self.TEMPLATE_ALIAS_MAP.get(template_ref, template_ref)
        
        # 8) Get few-shot SQL from recipes
        few_shot_sql = self._get_recipe_sql(canonical)
        
        # 9) Build plan
        plan = {
            "language": "zh-tw",
            "intent_context": {
                "template_ref": canonical or None,
                "intent": routing.get("intent"),
                "slots": routing.get("slots", {}),
                "tables": intent_tables,
                "candidates": routing.get("candidates", []),
                "few_shot_sql": few_shot_sql,
                "source": routing.get("source", "unknown"),
            },
            "tables": intent_tables,
            "join_hints": join_hints,
            "schema": schema_ctx,
        }
        
        # Cache and log
        self._plan_cache.set(cache_key, plan)
        
        elapsed = int((time.perf_counter() - t0) * 1000)
        if DEBUG_TIMING:
            logger.info(
                "[VectorSearch] Plan built in %dms: template=%s, tables=%s, source=%s",
                elapsed,
                canonical,
                intent_tables[:3],
                routing.get("source"),
            )
        
        return plan
    
    def _get_recipe_sql(self, recipe_id: str) -> Optional[str]:
        """Get SQL template from LeaveVectorDB recipes."""
        if not self.vector or not recipe_id:
            return None
        
        try:
            recipes = getattr(self.vector, "_recipes", None)
            if not recipes:
                return None
            
            for recipe in recipes:
                if getattr(recipe, "recipe_id", None) == recipe_id:
                    return recipe.sql_template
        except Exception as e:
            logger.warning("[VectorSearch] Recipe lookup failed: %s", e)
        
        return None
    
    # ────────────────────────────────────────────────────────────────────
    # Execution (for direct SQL mode)
    # ────────────────────────────────────────────────────────────────────
    def run_with_openai(
        self,
        openai_service,  # Actually LLMService (Bedrock)
        query: str,
        *,
        schema_filter: Optional[str] = None,
        rid: Optional[str] = None,
        max_rows: int = 1000,
        query_timeout: int = 10,
        max_attempts: int = 3,
    ) -> Tuple[List[Tuple], List[str], str, int, Dict[str, Any]]:
        """
        Full pipeline: plan → LLM SQL generation → execution.
        
        Returns: (rows, columns, sql, attempts, plan)
        """
        t0 = time.perf_counter()
        
        # Build plan
        plan = self.plan_for(query, schema_filter=schema_filter, rid=rid)
        
        # Call LLM service
        rows, cols, sql, attempts = openai_service.run_query_with_llm_repair(
            db_service=self.db_service,
            user_question=query,
            schema=plan["schema"],
            join_hints=plan["join_hints"],
            intent_context=plan["intent_context"],
            max_rows=max_rows,
            query_timeout=query_timeout,
            max_attempts=max_attempts,
        )
        
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "[VectorSearch] run_with_openai complete: %dms, rows=%d, attempts=%d",
            elapsed, len(rows), attempts,
        )
        
        return rows, cols, sql, attempts, plan
    
    # ────────────────────────────────────────────────────────────────────
    # Debug
    # ────────────────────────────────────────────────────────────────────
    def debug_search(
        self,
        query: str,
        language: Optional[Literal["zh-tw", "en"]] = None,
    ) -> Dict[str, Any]:
        """
        Debug interface for inspecting search behavior.
        """
        try:
            if not self.vector:
                return {"error": "no vector index"}
            
            results = []
            
            # Get raw vector search results
            pairs = self.vector.search(query, top_k=10, min_score=0.1)
            for item, score in pairs:
                text_preview = getattr(item, "text_zh", "") or ""
                if len(text_preview) > 100:
                    text_preview = text_preview[:100] + "..."
                
                results.append({
                    "type": item.item_type.value,
                    "key": item.key,
                    "score": round(score, 4),
                    "priority": item.priority,
                    "text_preview": text_preview,
                })
            
            # Get intent routing
            routing = self.get_intent_routing(query)
            
            # Get fast router result
            fast_result = FastIntentRouter.route(query)
            
            return {
                "query": query,
                "language": "zh-tw",
                "vector_results_count": len(results),
                "vector_results": results,
                "intent_routing": routing,
                "fast_router_result": fast_result,
                "cache_stats": {
                    "plan": self._plan_cache.stats(),
                    "table": self._table_cache.stats(),
                },
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    # ────────────────────────────────────────────────────────────────────
    # Legacy compatibility
    # ────────────────────────────────────────────────────────────────────
    def find_relevant_tables_with_language(
        self,
        query: str,
        schema_filter: Optional[str] = None,
        language: Optional[Literal["zh-tw", "en"]] = None,
        rid: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Legacy method - redirects to find_relevant_tables."""
        return self.find_relevant_tables(query, schema_filter, rid)
    
    def get_schema_context_with_language(
        self,
        tables: List[str],
        query: str,
        language: Optional[Literal["zh-tw", "en"]] = None,
        max_cols: int = 64,
    ) -> str:
        """Legacy method - redirects to get_schema_context."""
        return self.get_schema_context(tables, query, max_cols)