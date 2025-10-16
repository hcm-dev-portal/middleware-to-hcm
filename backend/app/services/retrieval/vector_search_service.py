# backend/app/services/retrieval/vector_search_service.py
from __future__ import annotations

import logging
import functools
from typing import List, Tuple, Optional, Dict, Any, Literal, Iterable
from collections import defaultdict, OrderedDict

# Language-aware vector system
from app.services.leave_vector import LeaveVectorDB, build_leave_index, detect_language
# Bring in the DB-qualified VAC table name + ORG constant
from app.services.leave_vector import VAC_RESULT_TABLE, ORG_TABLE

# Translation fallback
from app.services.aws.translation_service import AWSTranslationService

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Traditional → Simplified conversion (best-effort, cached)
# ────────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _get_trad2simp_impl():
    try:
        import opencc  # type: ignore
        cc = opencc.OpenCC('t2s')
        return (True, lambda s: cc.convert(s))
    except Exception:
        pass
    try:
        from hanziconv import HanziConv  # type: ignore
        return (True, lambda s: HanziConv.toSimplified(s))
    except Exception:
        pass
    try:
        from zhconv import convert  # type: ignore
        return (True, lambda s: convert(s, 'zh-cn'))
    except Exception:
        pass
    _MINI_MAP = {
        "部門": "部门","單位":"单位","員工":"员工","工號":"工号",
        "請假":"请假","出勤":"出勤","假別":"假别","資料":"资料",
        "這":"这","個":"个","週":"周","當天":"当天","現":"现","狀":"状","態":"态",
        "數":"数","據":"据","離":"离","職":"职",
    }
    def _mini_trad2simp(s: str) -> str:
        return "".join(_MINI_MAP.get(ch, ch) for ch in s)
    return (False, _mini_trad2simp)

def _trad2simp(text: str) -> str:
    try:
        _ok, fn = _get_trad2simp_impl()
        out = fn(text)
        return out if isinstance(out, str) else text
    except Exception:
        return text

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def _merge_hits(hit_lists: Iterable[List[Tuple[str, float]]], top_k: int = 5) -> List[Tuple[str, float]]:
    best: Dict[str, float] = defaultdict(float)
    for hits in hit_lists:
        for t, s in (hits or []):
            if s > best[t]:
                best[t] = s
    merged = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return merged[:top_k]

# Simple keyword gate to prefer VAC snapshot for “remaining annual/特休”
def _should_prefer_vac_result(q: str) -> bool:
    ql = (q or "").lower()
    zh_hit = ("餘" in q or "餘額" in q or "剩" in q or "剩餘" in q or "還有" in q) and ("年假" in q or "特休" in q)
    en_hit = any(w in ql for w in ["remaining", "unused", "balance"]) and any(w in ql for w in ["annual", "pto", "vacation"])
    return zh_hit or en_hit

# Tiny bounded dict with eviction
class _BoundedCache(OrderedDict):
    def __init__(self, maxsize: int = 512):
        super().__init__()
        self.maxsize = maxsize
    def getset(self, key, compute_fn):
        if key in self:
            self.move_to_end(key)
            return self[key], True
        value = compute_fn()
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)
        return value, False

# ────────────────────────────────────────────────────────────────────────────────
# Service
# ────────────────────────────────────────────────────────────────────────────────
class VectorSearchService:
    """Enhanced language-aware vector-based retrieval, schema context & intent routing."""

    def __init__(self, db_service):
        self.db_service = db_service
        self.vector: Optional[LeaveVectorDB] = None
        self.person_table: str = "dbo.PSNACCOUNT_D"
        self.translator = AWSTranslationService()

        # Per-process tiny cache: key = (rid, lang, normalized_query, schema_filter or "")
        self._req_cache: _BoundedCache = _BoundedCache(maxsize=512)

        self._initialize_vector_index()
        self._determine_person_table()

    # ---- optional: health check used by native class ----
    def health_check(self) -> Dict[str, Any]:
        try:
            if not self.vector:
                return {"ready": False, "error": "no vector index"}
            return self.vector.health_check()  # type: ignore
        except Exception as e:
            return {"ready": False, "error": str(e)}

    def _initialize_vector_index(self):
        try:
            self.vector = build_leave_index()
            logger.info("Enhanced language-aware vector index ready.")
        except Exception as e:
            self.vector = None
            logger.warning("Language-aware vector unavailable: %s", e)

    def _determine_person_table(self):
        try:
            if self.vector:
                vt = getattr(self.vector, "_person_table", None)
                if isinstance(vt, str) and vt:
                    self.person_table = vt
        except Exception:
            pass

    # ---------------- INTENT ROUTING ----------------
    def get_intent_routing(self, query: str) -> Dict[str, Any]:
        if not self.vector:
            return {"lang": detect_language(query), "slots": {}, "candidates": []}
        try:
            routing = self.vector.get_intent_routing(query)
            routing.setdefault("slots", {})
            routing.setdefault("candidates", [])
            return routing
        except Exception as e:
            logger.error("INTENT_ROUTING_FAIL: %s", e, exc_info=True)
            return {"lang": detect_language(query), "slots": {}, "candidates": []}

    # ---------------- search (legacy wrapper) ----------------
    def find_relevant_tables(self, english_query: str, schema_filter: Optional[str] = None,
                             rid: Optional[str] = None) -> List[Tuple[str, float]]:
        lang = detect_language(english_query)
        logger.debug("VECTOR_SEARCH_LEGACY: query='%s' detected_lang=%s", english_query[:100], lang)
        return self.find_relevant_tables_with_language(
            english_query, schema_filter=schema_filter, language=lang, rid=rid
        )

    # ---------------- bilingual search (dedup + cache) ----------------
    def find_relevant_tables_with_language(
        self,
        query: str,
        schema_filter: Optional[str] = None,
        language: Optional[Literal["zh-tw", "en"]] = None,
        rid: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        try:
            if not self.vector:
                logger.warning("VECTOR_SEARCH: No vector index available")
                return []

            # Normalize language to only two rails: zh-tw or en
            if language is None:
                language = detect_language(query)
            language = "zh-tw" if (language or "").lower().startswith("zh") else "en"

            # -------- Per-request tiny cache to avoid duplicate passes --------
            norm_q = (query or "").strip()
            cache_key = (rid or "", language, norm_q, (schema_filter or "").lower())

            def _compute():
                logger.info("VECTOR_SEARCH_START: rid=%s lang=%s query='%s'", rid, language, norm_q[:100])

                hit_sets: List[List[Tuple[str, float]]] = []
                tried: List[Tuple[str, int]] = []

                def _search_and_filter(q: str, label: str) -> List[Tuple[str, float]]:
                    try:
                        hits = self.vector.search_relevant_tables(q, top_k=5)
                        if schema_filter:
                            hits = [(t, s) for (t, s) in hits if t.lower().startswith(schema_filter.lower() + ".")]
                        logger.info("VECTOR_HITS: rid=%s label=%s count=%d", rid, label, len(hits))
                        for t, s in hits:
                            logger.debug("VECTOR_HIT: rid=%s label=%s table=%s score=%.3f", rid, label, t, s)
                        return hits
                    except Exception as e:
                        logger.warning("VECTOR_SEARCH_FAIL: rid=%s label=%s err=%s", rid, label, e)
                        return []

                if language == "zh-tw":
                    # Only two passes: zh-tw + english translation (NO zh-cn pass)
                    hits_ztw = _search_and_filter(norm_q, "zh-tw")
                    hit_sets.append(hits_ztw); tried.append(("zh-tw", len(hits_ztw)))

                    q_en = self.translator.translate_to_english(norm_q, "zh-tw") or ""
                    if q_en:
                        hits_en = _search_and_filter(q_en, "en")
                        hit_sets.append(hits_en); tried.append(("en", len(hits_en)))
                else:
                    hits_en = _search_and_filter(norm_q, "en")
                    hit_sets.append(hits_en); tried.append(("en", len(hits_en)))

                logger.info("VECTOR_TRIED: rid=%s tried=%s", rid, tried)
                merged = _merge_hits(hit_sets, top_k=5)
                logger.info("VECTOR_MERGED: rid=%s merged=%s", rid, merged)
                return merged

            merged, was_cache = self._req_cache.getset(cache_key, _compute)
            if was_cache:
                logger.debug("VECTOR_CACHE_HIT: rid=%s lang=%s", rid, language)
            return merged

        except Exception as e:
            logger.error("VECTOR_SEARCH_ERROR: rid=%s query='%s' lang=%s error=%s",
                         rid, (query or "")[:100], language, str(e), exc_info=True)
            return []

    # ---------------- join hints ----------------
    def get_join_hints(self, tables: List[str]) -> str:
        try:
            if not self.vector:
                return "None"
            hints = self.vector.join_hints(tables)
            result = "\n".join(hints) if hints else "None"
            logger.debug("JOIN_HINTS: tables=%s hints_count=%d", tables, len(hints or []))
            return result
        except Exception as e:
            logger.error("JOIN_HINTS_ERROR: tables=%s error=%s", tables, str(e))
            return "None"

    # ---------------- schema context ----------------
    def get_schema_context(self, tables: List[str], max_cols: int = 64) -> str:
        if not tables:
            logger.debug("SCHEMA_CONTEXT: No tables provided")
            return "No relevant tables found"

        pick = list(dict.fromkeys(
            tables[:3] + ([self.person_table] if self.person_table not in tables[:3] else [])
        ))
        logger.debug("SCHEMA_CONTEXT: selected_tables=%s person_table=%s", pick, self.person_table)

        try:
            context = self.db_service.get_compact_schema_for(pick, max_columns_per_table=max_cols)
            logger.debug("SCHEMA_CONTEXT_SUCCESS: context_length=%d", len(context))
            return context
        except Exception as e:
            logger.error("SCHEMA_CONTEXT_ERROR: tables=%s error=%s", pick, str(e))
            return f"Schema context unavailable for {pick}"

    def get_schema_context_with_language(
        self,
        tables: List[str],
        query: str,
        language: Optional[Literal["zh-tw", "en"]] = None,
        max_cols: int = 64
    ) -> str:
        try:
            if not self.vector:
                return self.get_schema_context(tables, max_cols)

            if language is None:
                language = detect_language(query)
            language = "zh-tw" if (language or "").lower().startswith("zh") else "en"

            logger.debug("ENHANCED_SCHEMA_CONTEXT: lang=%s tables=%s query='%s'",
                         language, tables, (query or "")[:100])

            enhanced_context = self.vector.get_schema_context(query, include_examples=True)
            db_schema = self.get_schema_context(tables, max_cols)

            combined = f"{enhanced_context}\n\n=== DATABASE SCHEMA ===\n{db_schema}"
            logger.debug("ENHANCED_SCHEMA_CONTEXT_SUCCESS: total_length=%d", len(combined))
            return combined

        except Exception as e:
            logger.error("ENHANCED_SCHEMA_CONTEXT_ERROR: tables=%s error=%s", tables, str(e))
            return self.get_schema_context(tables, max_cols)

    # ---------------- planning helpers ----------------
    def plan_for(
        self,
        query: str,
        *,
        schema_filter: Optional[str] = None,
        rid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build plan (intent + schema + joins). Force VAC snapshot when question asks for remaining annual/特休.
        """
        language = detect_language(query)
        language = "zh-tw" if (language or "").lower().startswith("zh") else "en"

        routing = self.get_intent_routing(query)

        # Preferred tables from top intent
        top_cand = (routing.get("candidates") or [{}])[0]
        tables_from_intent = top_cand.get("tables") or []

        # Vector fallback (uses dedup + cache internally)
        vector_tables = [t for t, _ in self.find_relevant_tables_with_language(
            query, schema_filter=schema_filter, language=language, rid=rid
        )]

        # Merge
        tables = list(dict.fromkeys((tables_from_intent or []) + vector_tables))

        # ── PREFER VAC RESULT for “remaining annual/特休” ─────────────────────
        if _should_prefer_vac_result(query) and VAC_RESULT_TABLE not in tables:
            tables.insert(0, VAC_RESULT_TABLE)

        # Join hints from vector
        join_hints = self.get_join_hints(tables)

        # Strengthen hints when VAC is present
        if any(t.lower().endswith("atdcalcuvacationresult]") or "ATDCALCUVACATIONRESULT" in t.upper() for t in tables):
            vac_hint = (
                f"-- PREFER {VAC_RESULT_TABLE} as authoritative for balances.\n"
                f"-- Filters: r.REMAINDAYS > 0, r.VACAYEAR = @year (if provided), r.VACATIONTYPE = 1 (annual, if applicable),\n"
                f"--           validity window: @today BETWEEN CAST(r.CANUSEDATE AS date) AND CAST(r.DISABLEDDATE AS date) (when present).\n"
                f"-- Join: r.PERSONID → dbo.PSNACCOUNT.PERSONID; optional PSNACCOUNT.BRANCHID → {ORG_TABLE}.UNITID for department."
            )
            join_hints = f"{vac_hint}\n\n{join_hints or ''}".strip()

        schema_ctx = self.get_schema_context_with_language(tables, query, language)

        plan = {
            "language": language,
            "intent_context": {
                "template_ref": top_cand.get("template_ref"),
                "slots": routing.get("slots", {}),
                "tables": tables,
                "candidates": routing.get("candidates", []),
                # Surface recommended filters to the LLM
                "recommended_filters": ["REMAINDAYS > 0", "VACAYEAR = @year", "VACATIONTYPE = 1",
                                        "BETWEEN CANUSEDATE AND DISABLEDDATE"],
            },
            "tables": tables,
            "join_hints": join_hints,
            "schema": schema_ctx,
        }
        logger.info("PLAN_FOR: lang=%s tables=%s template_ref=%s", language, tables, plan["intent_context"]["template_ref"])
        return plan

    # ---------------- end-to-end runner ----------------
    def run_with_openai(
        self,
        openai_service,             # UnifiedBilingualOpenAIService
        query: str,
        *,
        schema_filter: Optional[str] = None,
        rid: Optional[str] = None,
        max_rows: int = 1000,
        query_timeout: int = 10,
        max_attempts: int = 3,
    ) -> Tuple[List[Tuple], List[str], str, int, Dict[str, Any]]:
        """
        Full pipeline:
        - Build plan (intent + schema + joins)
        - Call OpenAI service with intent_context
        - Execute with DB service, attempt LLM repair if needed
        Returns (rows, columns, sql, attempts, plan)
        """
        plan = self.plan_for(query, schema_filter=schema_filter, rid=rid)
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
        return rows, cols, sql, attempts, plan

    # ---------------- debug ----------------
    def debug_search(self, query: str, language: Optional[Literal["zh-tw", "en"]] = None) -> Dict[str, Any]:
        """Debug method to analyze search behavior and tried paths."""
        try:
            if not self.vector:
                return {"error": "no vector index"}

            if language is None:
                language = detect_language(query)
            language = "zh-tw" if (language or "").lower().startswith("zh") else "en"

            results_overall: List[Dict[str, Any]] = []
            tried = []

            def _collect(q: str, label: str):
                nonlocal results_overall, tried
                try:
                    pairs = self.vector.search(q, top_k=10, min_score=0.1)
                    tried.append((label, len(pairs)))
                    for item, score in pairs:
                        results_overall.append({
                            "label": label,
                            "type": item.item_type.value,
                            "key": item.key,
                            "score": round(score, 4),
                            "priority": item.priority,
                            "text_en_preview": (item.text_en[:100] + "...") if getattr(item, "text_en", "") and len(item.text_en) > 100 else getattr(item, "text_en", ""),
                            "text_zh_preview": (item.text_zh[:100] + "...") if getattr(item, "text_zh", "") and len(item.text_zh) > 100 else getattr(item, "text_zh", ""),
                        })
                except Exception as e:
                    tried.append((label, f"error: {e}"))

            if language == "zh-tw":
                _collect(query, "zh-tw")
                q_en = self.translator.translate_to_english(query, "zh-tw") or ""
                if q_en:
                    _collect(q_en, "en")
            else:
                _collect(query, "en")

            # Also surface current intent routing for transparency
            routing = self.get_intent_routing(query)

            return {
                "query": query,
                "detected_language": language,
                "tried": tried,
                "results_count": len(results_overall),
                "results": results_overall,
                "intent_routing": routing,
            }

        except Exception as e:
            return {"error": str(e)}
