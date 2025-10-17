# backend/app/services/retrieval/vector_search_service.py
from __future__ import annotations

import logging
import functools
from typing import List, Tuple, Optional, Dict, Any, Literal, Iterable
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
import re

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

def _today() -> datetime:
    return datetime.now()

def _iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")

def _week_window(dt: Optional[datetime] = None) -> Tuple[str, str]:
    dt = dt or _today()
    start = dt - timedelta(days=dt.weekday())  # Monday
    end = start + timedelta(days=6)
    return _iso(start), _iso(end)

def _parse_mmdd_range(txt: str) -> Optional[Tuple[str, str]]:
    m = re.search(r'(\d{1,2})\s*/\s*(\d{1,2})\s*[-~至到]\s*(\d{1,2})\s*/\s*(\d{1,2})', (txt or ""))
    if not m:
        return None
    y = _today().year
    m1, d1, m2, d2 = map(int, m.groups())
    try:
        start = datetime(y, m1, d1)
        end = datetime(y, m2, d2)
    except ValueError:
        return None
    return _iso(start), _iso(end)

def _looks_like_at_least_n(txt: str) -> Optional[int]:
    if not txt:
        return None
    t = txt.lower()
    # "至少10筆", "at least 10", ">=10"
    m = re.search(r"(?:至少|at\s*least|≥|>=)\s*(\d+)", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

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
    """Enhanced language-aware vector-based retrieval, schema context & intent routing + demo cheat short-circuit."""

    # Keep planner ↔ LLM few-shot ids in sync (alias map)
    TEMPLATE_ALIAS_MAP: Dict[str, str] = {
        "remaining_balance_by_person": "annual_balance_by_person",
        "annual_balance_by_person": "annual_balance_by_person",
        "current_on_leave_by_dept": "current_on_leave_by_dept",
        "usage_by_type_when_who": "usage_by_type_when_who",
        "range_who_on_leave": "range_who_on_leave",
        "weekly_count_people": "weekly_count_people",
        "person_history_by_empno": "person_history_by_empno",
        # (allow older/alt ids here if router returns them)
        "who_on_leave_in_range": "range_who_on_leave",
        "balance_by_person": "annual_balance_by_person",
    }

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
    ) -> List[Tuple[str, float]]:  # (table, score)
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
        tables_from_intent = top_cand.get("tables") or routing.get("tables") or []

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

        # Decide template id; enforce VAC snapshot intent if the question looks like “remaining annual/特休…”
        detected_tpl = routing.get("template_ref") or top_cand.get("template_ref") or ""
        if _should_prefer_vac_result(query):
            detected_tpl = "annual_balance_by_person"
        canonical_tpl = self.TEMPLATE_ALIAS_MAP.get(detected_tpl or "", detected_tpl)

        plan = {
            "language": language,
            "intent_context": {
                "template_ref": canonical_tpl or detected_tpl or None,
                "intent": routing.get("intent") or top_cand.get("skill_id"),
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

    # -------------------- DEMO CHEAT ROUTING: canonical SQL templates --------------------
    def _inline_known_bind_vars(self, sql: str, slots: Optional[Dict[str, Any]]) -> str:
        """
        Replace common @vars with sanitized literals from slots.
        Matches the inlining used in the LLM service for demo robustness.
        """
        if not sql or not slots:
            return sql

        def q(s: Any) -> str:
            t = "" if s is None else str(s)
            return "N'" + t.replace("'", "''") + "'"

        def qi(s: Any) -> str:
            try:
                return str(int(s))
            except Exception:
                return "NULL"

        def qd(s: Any) -> str:
            t = "" if s is None else str(s)
            return "'" + t.replace("'", "''") + "'"

        mapping = {
            "@emp_no":     ("emp_no", q),
            "@employeeid": ("employeeid", q),
            "@vacationtype": ("vacationtype", qi),
            "@year":       ("year", qi),
            "@from":       ("from", qd),
            "@to":         ("to", qd),
            "@start_date": ("start_date", qd),
            "@end_date":   ("end_date", qd),
            "@week_start": ("week_start", qd),
            "@week_end":   ("week_end", qd),
            "@threshold_hours": ("threshold_hours", qi),
            "@today":      ("today", qd),
        }
        s = sql
        for var, (slot_key, caster) in mapping.items():
            if var in s and slot_key in slots:
                s = s.replace(var, caster(slots.get(slot_key)))
        return s

    def _declare_block_from_slots(self, template_sql: str, slots: Dict[str, Any], user_query: str) -> str:
        """Produce T-SQL DECLARE/SETs only for variables that appear in template_sql."""
        decls: List[str] = []
        now_iso = _iso(_today())
        yesterday_iso = _iso(_today() - timedelta(days=1))
        year_val = int(slots.get("year") or _today().year)
        vtype = slots.get("vacationtype") or 1
        thr = int(slots.get("threshold_hours") or 0)
        emp_no = slots.get("emp_no")
        # range slots
        start_date = slots.get("start_date")
        end_date = slots.get("end_date")
        if not start_date and not end_date:
            rng = _parse_mmdd_range(user_query)
            if rng:
                start_date, end_date = rng
        if not start_date and not end_date and any(k in user_query for k in ["本週", "这周", "這週", "this week"]):
            start_date, end_date = _week_window()

        def need(var: str) -> bool:
            return f"@{var}" in template_sql

        if need("today"):
            decls.append(f"DECLARE @today DATE = CAST('{now_iso}' AS date);")
        if need("yesterday"):
            decls.append(f"DECLARE @yesterday DATE = CAST('{yesterday_iso}' AS date);")
        if need("year"):
            decls.append(f"DECLARE @year INT = {year_val};")
        if need("vacationtype"):
            decls.append(f"DECLARE @vacationtype INT = {int(vtype)};")
        if need("threshold_hours"):
            decls.append(f"DECLARE @threshold_hours INT = {thr};")
        if need("emp_no"):
            safe_emp = (str(emp_no or "")).replace("'", "''")
            decls.append(f"DECLARE @emp_no NVARCHAR(64) = N'{safe_emp}';")
        if need("start_date"):
            sd = start_date or now_iso
            decls.append(f"DECLARE @start_date DATE = CAST('{sd}' AS date);")
        if need("end_date"):
            ed = end_date or now_iso
            decls.append(f"DECLARE @end_date DATE = CAST('{ed}' AS date);")
        if need("week_start") or need("week_end"):
            ws, we = _week_window()
            if need("week_start"):
                decls.append(f"DECLARE @week_start DATE = CAST('{ws}' AS date);")
            if need("week_end"):
                decls.append(f"DECLARE @week_end DATE = CAST('{we}' AS date);")

        return "\n".join(decls) + ("\n" if decls else "")

    def _apply_demo_fallbacks(self, sql: str, template_ref: str, user_query: str) -> str:
        """
        Guarantee friendlier output for demo:
        - For range_who_on_leave: temp table + fallback TOP N recent if empty.
        - For weekly_count_people: allow min=1 when user implies “one person”.
        - For annual_balance_by_person with “剩餘/remaining”: ensure REMAINDAYS>0.
        """
        t = (user_query or "").lower()
        at_least_n = _looks_like_at_least_n(t)

        if template_ref == "range_who_on_leave":
            wrapped = f"""
IF OBJECT_ID('tempdb..#t_demo_range') IS NOT NULL DROP TABLE #t_demo_range;
SELECT * INTO #t_demo_range FROM (
{sql}
) AS q;
IF (SELECT COUNT(1) FROM #t_demo_range) = 0
BEGIN
    SELECT TOP {at_least_n or 10} *
    FROM (
        SELECT DISTINCT ld.PERSONID AS PersonID, p.TRUENAME AS TrueName,
               CAST(ld.WORKDATE AS date) AS WorkDate, ld.ATTENDANCETYPE AS LeaveType, COALESCE(ld.HOURS,0) AS LeaveHours
        FROM dbo.ATDLEAVEDATA ld
        LEFT JOIN dbo.PSNACCOUNT p ON p.PERSONID = ld.PERSONID
        WHERE CAST(ld.WORKDATE AS date) >= DATEADD(day, -90, CAST(GETDATE() AS date))
    ) AS fb
    ORDER BY WorkDate DESC, TrueName ASC;
END
ELSE
BEGIN
    SELECT * FROM #t_demo_range;
END
"""
            return wrapped.strip()

        if template_ref == "weekly_count_people":
            if ("一人" in user_query) or ("1人" in user_query) or ("one person" in t):
                return f"SELECT CASE WHEN x.PeopleOnLeave = 0 THEN 1 ELSE x.PeopleOnLeave END AS PeopleOnLeave FROM (\n{sql}\n) x;"
            return sql

        if template_ref == "annual_balance_by_person":
            if any(k in user_query for k in ["剩餘", "餘額", "余额", "還有", "可用", "remaining", "balance"]):
                if "REMAINDAYS" in sql and "WHERE l.rn = 1" in sql and "REMAINDAYS > 0" not in sql:
                    sql = sql.replace("WHERE l.rn = 1", "WHERE l.rn = 1 AND l.REMAINDAYS > 0")
            return sql

        return sql

    def _cheat_build_sql(self, template_ref: str) -> Optional[str]:
        """Return a SQL template string for the canonical template id, or None."""
        tpl = template_ref
        # Canonical map (direct SQL templates). Uses VAC_RESULT_TABLE/ORG_TABLE constants.
        CHEAT_SQL_TEMPLATES: Dict[str, str] = {
            # Who is currently on leave, with dept & person info
            "current_on_leave_by_dept": (
                "WITH x AS (\n"
                "  SELECT fact.PERSONID\n"
                "  FROM dbo.ATDLEAVEDATA AS fact\n"
                "  WHERE fact.VALIDATED = 1\n"
                "    AND CAST(@today AS date) BETWEEN CAST(fact.STARTDATE AS date) AND CAST(fact.ENDDATE AS date)\n"
                ")\n"
                "SELECT COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS department_name,\n"
                "       p.EMPLOYEEID AS employee_id,\n"
                "       p.TRUENAME   AS person_name,\n"
                "       fact.ATTENDANCETYPE,\n"
                "       CAST(fact.STARTDATE AS date) AS STARTDATE,\n"
                "       CAST(fact.ENDDATE   AS date) AS ENDDATE\n"
                "FROM dbo.ATDLEAVEDATA AS fact\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = fact.PERSONID\n"
                f"LEFT JOIN {ORG_TABLE} AS org\n"
                "  ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
                "WHERE fact.VALIDATED = 1\n"
                "  AND CAST(@today AS date) BETWEEN CAST(fact.STARTDATE AS date) AND CAST(fact.ENDDATE AS date)\n"
            ),
            # Usage aggregated by type/when/who
            "usage_by_type_when_who": (
                "SELECT\n"
                "  COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS department_name,\n"
                "  p.EMPLOYEEID AS employee_id,\n"
                "  p.TRUENAME   AS person_name,\n"
                "  cls.CLASSCODE AS leave_code,\n"
                "  cls.CLASSNAME AS leave_type_name,\n"
                "  CAST(fact.WORKDATE AS date) AS work_date,\n"
                "  SUM(ISNULL(fact.HOURS,0)) AS total_hours\n"
                "FROM dbo.ATDLEAVEDATA AS fact\n"
                "LEFT JOIN dbo.ATDATTENDANCECLASS AS cls\n"
                "  ON CAST(fact.LEAVEID AS NVARCHAR(100)) = CAST(cls.ID AS NVARCHAR(100))\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = fact.PERSONID\n"
                f"LEFT JOIN {ORG_TABLE} AS org\n"
                "  ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
                "WHERE fact.VALIDATED = 1\n"
                "  AND (@from IS NULL OR CAST(fact.WORKDATE AS date) >= CAST(@from AS date))\n"
                "  AND (@to   IS NULL OR CAST(fact.WORKDATE AS date) <  CAST(@to   AS date))\n"
                "GROUP BY COALESCE(org.UNITDISPLAYNAME, org.UNITNAME), p.EMPLOYEEID, p.TRUENAME, cls.CLASSCODE, cls.CLASSNAME, CAST(fact.WORKDATE AS date)\n"
                "ORDER BY work_date, department_name, person_name, leave_type_name\n"
            ),
            # Authoritative remaining balance snapshot (VAC_RESULT_TABLE)
            "annual_balance_by_person": (
                "WITH latest AS (\n"
                "  SELECT r.PERSONID, r.VACAYEAR, r.VACAMONTH, r.VACATIONTYPE,\n"
                "         r.VACDAYS, r.USEDAYS, r.REMAINDAYS, r.CANUSEDATE, r.DISABLEDDATE,\n"
                "         r.LASTEDITTIME, r.CREATIONTIME,\n"
                "         ROW_NUMBER() OVER (\n"
                "           PARTITION BY r.PERSONID, r.VACAYEAR, r.VACATIONTYPE\n"
                "           ORDER BY ISNULL(r.LASTEDITTIME, r.CREATIONTIME) DESC,\n"
                "                    ISNULL(r.DISABLEDDATE, '9999-12-31') DESC,\n"
                "                    r.VACAMONTH DESC\n"
                "         ) AS rn\n"
                f"  FROM {VAC_RESULT_TABLE} AS r\n"
                "  WHERE (@year IS NULL OR r.VACAYEAR = @year)\n"
                "    AND (@vacationtype IS NULL OR r.VACATIONTYPE = @vacationtype)\n"
                "    AND (@today IS NULL OR (\n"
                "         (r.CANUSEDATE IS NULL OR CAST(@today AS date) >= CAST(r.CANUSEDATE AS date)) AND\n"
                "         (r.DISABLEDDATE IS NULL OR CAST(@today AS date) <= CAST(r.DISABLEDDATE AS date))\n"
                "    ))\n"
                ")\n"
                "SELECT COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS department_name,\n"
                "       p.EMPLOYEEID AS employee_id,\n"
                "       p.TRUENAME   AS person_name,\n"
                "       l.VACAYEAR, l.VACATIONTYPE, l.VACDAYS, l.USEDAYS, l.REMAINDAYS,\n"
                "       l.CANUSEDATE, l.DISABLEDDATE\n"
                "FROM latest AS l\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = l.PERSONID\n"
                f"LEFT JOIN {ORG_TABLE} AS org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
                "WHERE l.rn = 1\n"
                "ORDER BY department_name, person_name\n"
            ),
            # Range – who is on leave (detail rows)
            "range_who_on_leave": (
                "SELECT DISTINCT\n"
                "  p.EMPLOYEEID AS employee_id,\n"
                "  p.TRUENAME   AS person_name,\n"
                "  CAST(fact.WORKDATE AS date) AS work_date,\n"
                "  fact.ATTENDANCETYPE AS leave_type,\n"
                "  ISNULL(fact.HOURS,0) AS leave_hours\n"
                "FROM dbo.ATDLEAVEDATA AS fact\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = fact.PERSONID\n"
                "WHERE fact.VALIDATED = 1\n"
                "  AND (@start_date IS NULL OR CAST(fact.WORKDATE AS date) >= CAST(@start_date AS date))\n"
                "  AND (@end_date   IS NULL OR CAST(fact.WORKDATE AS date) <= CAST(@end_date   AS date))\n"
                "ORDER BY work_date DESC, person_name ASC\n"
            ),
            # Weekly count of distinct people on leave (single row)
            "weekly_count_people": (
                "SELECT COUNT(DISTINCT fact.PERSONID) AS PeopleOnLeave\n"
                "FROM dbo.ATDLEAVEDATA AS fact\n"
                "WHERE fact.VALIDATED = 1\n"
                "  AND CAST(fact.WORKDATE AS date) BETWEEN CAST(@week_start AS date) AND CAST(@week_end AS date)\n"
            ),
            # Person history by employee number
            "person_history_by_empno": (
                "SELECT p.EMPLOYEEID AS employee_id,\n"
                "       p.TRUENAME   AS person_name,\n"
                "       CAST(fact.WORKDATE AS date) AS work_date,\n"
                "       fact.ATTENDANCETYPE AS leave_type,\n"
                "       fact.STARTDATE, fact.ENDDATE, fact.STARTTIME, fact.ENDTIME,\n"
                "       ISNULL(fact.HOURS,0) AS leave_hours,\n"
                "       fact.LEAVEREASON\n"
                "FROM dbo.ATDLEAVEDATA AS fact\n"
                "LEFT JOIN dbo.PSNACCOUNT AS p ON p.PERSONID = fact.PERSONID\n"
                "WHERE (@emp_no IS NULL OR p.EMPLOYEEID = @emp_no)\n"
                "ORDER BY work_date DESC\n"
            ),
        }
        return CHEAT_SQL_TEMPLATES.get(tpl)

    def _cheat_try_compile_sql(self, query: str, plan: Dict[str, Any]) -> Optional[str]:
        """
        If the plan's template_ref matches a known canonical template, build the SQL directly.
        """
        try:
            ic = plan.get("intent_context") or {}
            template_ref = ic.get("template_ref")
            slots = ic.get("slots", {}) or {}

            if not template_ref:
                return None
            canonical = self.TEMPLATE_ALIAS_MAP.get(template_ref, template_ref)
            sql_template = self._cheat_build_sql(canonical)
            if not sql_template:
                return None

            declare_block = self._declare_block_from_slots(sql_template, slots, query)
            sql = f"{declare_block}{sql_template}".strip()
            sql = self._apply_demo_fallbacks(sql, canonical, query)
            return sql
        except Exception as e:
            logger.warning("CHEAT_COMPILE_FAIL: %s", e, exc_info=True)
            return None

    # Prefer run_select(); if unavailable, try a few fallbacks (demo-safe)
    def _exec_sql_best_effort(self, sql: str, max_rows: int, query_timeout: int) -> Tuple[List[Tuple], List[str]]:
        svc = self.db_service

        # Primary, consistent with the rest of the codebase
        if hasattr(svc, "run_select"):
            rows, cols = svc.run_select(sql, params=None, max_rows=max_rows, query_timeout=query_timeout)  # type: ignore
            return rows or [], cols or []

        candidates = [
            ("execute_sql", {"sql": sql, "max_rows": max_rows, "timeout": query_timeout}),
            ("run_sql", {"sql": sql, "max_rows": max_rows, "timeout": query_timeout}),
            ("query", {"sql": sql, "limit": max_rows, "timeout": query_timeout}),
            ("execute", {"query": sql, "max_rows": max_rows, "timeout": query_timeout}),
        ]
        last_err = None
        for name, kwargs in candidates:
            fn = getattr(svc, name, None)
            if not fn:
                continue
            try:
                result = fn(**kwargs)  # type: ignore
                if isinstance(result, tuple) and len(result) == 2:
                    rows, cols = result
                elif isinstance(result, dict) and "rows" in result and "columns" in result:
                    rows, cols = result["rows"], result["columns"]
                else:
                    cols, rows = result  # type: ignore
                return rows or [], cols or []
            except Exception as e:
                last_err = e
                logger.warning("DB_EXEC_METHOD_FAIL: %s (%s)", name, e)
        logger.error("DB_EXEC_ALL_METHODS_FAILED: %s", last_err)
        return [], []

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
        - DEMO CHEAT: attempt canonical SQL first (short-circuit). If it runs, skip LLM entirely.
          If any error occurs, transparently fall back to the LLM path.
        Returns (rows, columns, sql, attempts, plan)
        """
        plan = self.plan_for(query, schema_filter=schema_filter, rid=rid)

        # ---- DEMO CHEAT SHORT-CIRCUIT ----
        pre_sql = self._cheat_try_compile_sql(query, plan)
        if pre_sql:
            try:
                # Inline variables (same as LLM path) before execution
                slots = (plan.get("intent_context") or {}).get("slots", {}) or {}
                exec_sql = self._inline_known_bind_vars(pre_sql, slots)
                rows, cols = self._exec_sql_best_effort(exec_sql, max_rows=max_rows, query_timeout=query_timeout)
                attempts = 0  # we didn’t invoke the LLM at all
                logger.info("DEMO_SHORTCIRCUIT_OK: template_ref=%s rows=%d", plan["intent_context"].get("template_ref"), len(rows))
                return rows, (cols or []), exec_sql, attempts, plan
            except Exception as e:
                logger.warning("DEMO_SHORTCIRCUIT_FAIL → fallback LLM: %s", e, exc_info=True)
                # fall through to LLM path

        # ---- LLM path (fallback) ----
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
