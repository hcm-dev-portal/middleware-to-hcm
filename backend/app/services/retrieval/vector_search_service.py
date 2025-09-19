# ================================================================================
# backend/app/services/retrieval/vector_search_service.py
from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Dict, Any, Literal, Iterable
from collections import defaultdict

# Language-aware vector system
from app.services.leave_vector import LeaveVectorDB, build_leave_index, detect_language
# Translation fallback
from app.services.aws.translation_service import AWSTranslationService

logger = logging.getLogger(__name__)

# --- put near the top of vector_search_service.py, replace _try_opencc_trad2simp ---

# Translation fallback
from app.services.aws.translation_service import AWSTranslationService
import functools

@functools.lru_cache(maxsize=1)
def _get_trad2simp_impl():
    """
    Try several libraries for Traditional -> Simplified conversion, in order:
    1) opencc-python-reimplemented
    2) hanziconv
    3) zhconv
    If none are available, return (False, simple_map_fn).
    """
    # 1) opencc-python-reimplemented (pure Python)
    try:
        import opencc  # type: ignore
        # Its API mirrors OpenCC
        cc = opencc.OpenCC('t2s')
        return (True, lambda s: cc.convert(s))
    except Exception:
        pass

    # 2) hanziconv
    try:
        from hanziconv import HanziConv  # type: ignore
        return (True, lambda s: HanziConv.toSimplified(s))
    except Exception:
        pass

    # 3) zhconv
    try:
        from zhconv import convert  # type: ignore
        return (True, lambda s: convert(s, 'zh-cn'))
    except Exception:
        pass

    # 4) Minimal in-house char map for common HCM/HR words
    # NOTE: This is intentionally small; we only cover frequent tokens in your domain.
    _MINI_MAP = {
        "部門": "部门", "單位": "单位", "員工": "员工", "工號": "工号",
        "請假": "请假", "出勤": "出勤", "假別": "假别", "資料": "资料",
        "這": "这", "個": "个", "週": "周", "當天": "当天", "現": "现",
        "狀": "状", "態": "态", "數": "数", "據": "据", "離": "离", "職": "职",
    }
    def _mini_trad2simp(s: str) -> str:
        return "".join(_MINI_MAP.get(ch, ch) for ch in s)
    return (False, _mini_trad2simp)


def _trad2simp(text: str) -> str:
    """
    Convert zh-TW (Traditional) to zh-CN (Simplified) with best-effort fallbacks.
    Returns the input if no change occurs.
    """
    try:
        _ok, fn = _get_trad2simp_impl()
        out = fn(text)
        return out if isinstance(out, str) else text
    except Exception:
        return text



def _merge_hits(hit_lists: Iterable[List[Tuple[str, float]]], top_k: int = 5) -> List[Tuple[str, float]]:
    """Merge multiple (table, score) lists by taking the max score per table, then sort desc."""
    best: Dict[str, float] = defaultdict(float)
    for hits in hit_lists:
        for t, s in (hits or []):
            if s > best[t]:
                best[t] = s
    merged = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return merged[:top_k]


class VectorSearchService:
    """Enhanced language-aware vector-based table retrieval and schema operations."""

    def __init__(self, db_service):
        self.db_service = db_service
        self.vector: Optional[LeaveVectorDB] = None
        self.person_table: str = "dbo.PSNACCOUNT_D"
        self.translator = AWSTranslationService()  # English fallback when needed

        self._initialize_vector_index()
        self._determine_person_table()

    # ---------------- init ----------------
    def _initialize_vector_index(self):
        """Initialize the enhanced language-aware vector index."""
        try:
            self.vector = build_leave_index()
            logger.info("Enhanced language-aware vector index ready.")
        except Exception as e:
            self.vector = None
            logger.warning("Language-aware vector unavailable: %s", e)

    def _determine_person_table(self):
        """Determine the correct person table to use."""
        try:
            if self.vector:
                vt = getattr(self.vector, "_person_table", None)
                if isinstance(vt, str) and vt:
                    self.person_table = vt
        except Exception:
            pass

    # ---------------- search (legacy wrapper) ----------------
    def find_relevant_tables(self, english_query: str, schema_filter: Optional[str] = None,
                             rid: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Legacy entry point kept for compatibility: auto-detect language and route to bilingual search.
        """
        lang = detect_language(english_query)
        logger.debug("VECTOR_SEARCH_LEGACY: query='%s' detected_lang=%s", english_query[:100], lang)
        return self.find_relevant_tables_with_language(
            english_query, schema_filter=schema_filter, language=lang, rid=rid
        )

    # ---------------- bilingual search ----------------
    def find_relevant_tables_with_language(
        self,
        query: str,
        schema_filter: Optional[str] = None,
        language: Optional[Literal["zh-tw", "en"]] = None,
        rid: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Bilingual table search:
          zh-TW → direct
               → zh-CN (t2s) if available
               → EN (AWS translate)
          EN    → direct
        Merges results by max score per table; applies optional schema filter.
        """
        try:
            if not self.vector:
                logger.warning("VECTOR_SEARCH: No vector index available")
                return []

            if language is None:
                language = detect_language(query)

            logger.info("VECTOR_SEARCH_START: rid=%s lang=%s query='%s'", rid, language, query[:100])

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

            if (language or "").lower().startswith("zh"):
                # 1) zh-TW original
                hits_ztw = _search_and_filter(query, "zh-tw")
                hit_sets.append(hits_ztw); tried.append(("zh-tw", len(hits_ztw)))

                # 2) zh-CN simplified (optional)
                q_zcn = _trad2simp(query)
                if q_zcn and q_zcn != query:
                    hits_zcn = _search_and_filter(q_zcn, "zh-cn")
                    hit_sets.append(hits_zcn); tried.append(("zh-cn", len(hits_zcn)))

                # 3) English translation fallback
                q_en = self.translator.translate_to_english(query, "zh-tw") or ""
                if q_en:
                    hits_en = _search_and_filter(q_en, "en")
                    hit_sets.append(hits_en); tried.append(("en", len(hits_en)))
            else:
                # English-only path
                hits_en = _search_and_filter(query, language or "en")
                hit_sets.append(hits_en); tried.append((language or "en", len(hits_en)))

            logger.info("VECTOR_TRIED: rid=%s tried=%s", rid, tried)
            merged = _merge_hits(hit_sets, top_k=5)
            logger.info("VECTOR_MERGED: rid=%s merged=%s", rid, merged)
            return merged

        except Exception as e:
            logger.error("VECTOR_SEARCH_ERROR: rid=%s query='%s' lang=%s error=%s",
                         rid, query[:100], language, str(e), exc_info=True)
            return []

    # ---------------- join hints ----------------
    def get_join_hints(self, tables: List[str]) -> str:
        """Get join hints for the given tables."""
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
        """Get schema context for tables, ensuring person table is included."""
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
        """Language-aware schema context, combining vector examples with DB schema."""
        try:
            if not self.vector:
                return self.get_schema_context(tables, max_cols)

            if language is None:
                language = detect_language(query)

            logger.debug("ENHANCED_SCHEMA_CONTEXT: lang=%s tables=%s query='%s'",
                         language, tables, query[:100])

            # Vector-driven semantic schema/examples (bilingual inside the index)
            enhanced_context = self.vector.get_schema_context(query, include_examples=True)

            # Actual DB schema selection (ensure person table is included)
            db_schema = self.get_schema_context(tables, max_cols)

            combined = f"{enhanced_context}\n\n=== DATABASE SCHEMA ===\n{db_schema}"
            logger.debug("ENHANCED_SCHEMA_CONTEXT_SUCCESS: total_length=%d", len(combined))
            return combined

        except Exception as e:
            logger.error("ENHANCED_SCHEMA_CONTEXT_ERROR: tables=%s error=%s", tables, str(e))
            return self.get_schema_context(tables, max_cols)

    # ---------------- health ----------------
    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with language awareness details."""
        try:
            if not self.vector:
                return {"ready": False, "reason": "no index"}
            base = self.vector.health_check()
            enhanced = {
                **base,
                "service_version": "language_aware_v2",
                "person_table": self.person_table,
                "language_fallbacks": ["zh-tw", "zh-cn (optional)", "en"],
            }
            logger.debug("HEALTH_CHECK: %s", enhanced)
            return enhanced
        except Exception as e:
            logger.error("HEALTH_CHECK_ERROR: %s", str(e))
            return {"ready": False, "error": str(e)}

    # ---------------- business prompt ----------------
    def get_business_prompt(self, query: str, language: Optional[Literal["zh-tw", "en"]] = None) -> str:
        """Generate business-aware prompts for LLM with language context."""
        try:
            if not self.vector:
                return f"Business context unavailable. Query: {query}"
            if language is None:
                language = detect_language(query)
            logger.debug("BUSINESS_PROMPT: lang=%s query='%s'", language, query[:100])
            prompt = self.vector.get_business_prompt(query)
            logger.debug("BUSINESS_PROMPT_SUCCESS: prompt_length=%d", len(prompt))
            return prompt
        except Exception as e:
            logger.error("BUSINESS_PROMPT_ERROR: lang=%s error=%s", language, str(e))
            return f"Business context error. Query: {query}"

    # ---------------- debug ----------------
    def debug_search(self, query: str, language: Optional[Literal["zh-tw", "en"]] = None) -> Dict[str, Any]:
        """Debug method to analyze search behavior and tried paths."""
        try:
            if not self.vector:
                return {"error": "no vector index"}

            if language is None:
                language = detect_language(query)

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

            if (language or "").lower().startswith("zh"):
                _collect(query, "zh-tw")
                q_zcn = _trad2simp(query)
                if q_zcn and q_zcn != query:
                    _collect(q_zcn, "zh-cn")
                q_en = self.translator.translate_to_english(query, "zh-tw") or ""
                if q_en:
                    _collect(q_en, "en")
            else:
                _collect(query, language or "en")

            return {
                "query": query,
                "detected_language": language,
                "tried": tried,
                "results_count": len(results_overall),
                "results": results_overall,
            }

        except Exception as e:
            return {"error": str(e)}
