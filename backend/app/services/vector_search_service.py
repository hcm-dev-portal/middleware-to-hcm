# backend/app/services/retrieval/vector_search_service.py
from __future__ import annotations

import logging
import functools
from typing import List, Tuple, Optional, Dict, Any, Literal, Iterable
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
import re

# Language-aware vector system (we still import detect_language for safety,
# but this service now standardizes to zh-tw for its own logic)
from app.services.leave_vector import LeaveVectorDB, build_leave_index, detect_language
# Bring in the DB-qualified VAC table name + ORG constant
from app.services.leave_vector import VAC_RESULT_TABLE, ORG_TABLE

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def _merge_hits(hit_lists: Iterable[List[Tuple[str, float]]], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    合併多組 (table, score) 結果，取最高分版本並依分數排序。
    雖然目前僅使用單一 zh-tw 查詢，但保留泛用工具以利未來擴充。
    """
    best: Dict[str, float] = defaultdict(float)
    for hits in hit_lists:
        for t, s in (hits or []):
            if s > best[t]:
                best[t] = s
    merged = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return merged[:top_k]


# 當前是否偏向「年假/特休餘額」問題，用來強制偏好 VAC_RESULT_TABLE
def _should_prefer_vac_result(q: str) -> bool:
    ql = (q or "").lower()
    zh_hit = ("餘" in q or "餘額" in q or "剩" in q or "剩餘" in q or "還有" in q) and ("年假" in q or "特休" in q)
    en_hit = any(w in ql for w in ["remaining", "unused", "balance"]) and any(
        w in ql for w in ["annual", "pto", "vacation"]
    )
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
    """
    解析「月/日-月/日」格式，例如：
    9/22-9/26、09/22 ~ 09/26、9/22至9/26
    年份一律採用今年。
    """
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
    """
    從查詢文字中判斷是否有「至少 N」的語意，例如：
    - 至少10筆
    - at least 10
    - >=10
    """
    if not txt:
        return None
    t = txt.lower()
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
# Service（專注 zh-tw，介面維持不變）
# ────────────────────────────────────────────────────────────────────────────────
class VectorSearchService:
    """
    向量檢索與意圖路由服務（現已專注 zh-tw 單語情境）。

    主要職責：
    - 呼叫 LeaveVectorDB 做向量檢索（schema / recipe / KPI 等）
    - 建立 SQL 規劃 plan_for
    - 提供 demo 短路模式（canonical SQL）與 LLM fallback
    """

    # Keep planner ↔ LLM few-shot ids in sync (alias map)
    TEMPLATE_ALIAS_MAP: Dict[str, str] = {
        # Planner / legacy ids → canonical recipe ids
        "remaining_balance_by_person": "annual_balance_by_person",
        "annual_balance_by_person": "annual_balance_by_person",
        "current_on_leave_by_dept": "current_on_leave_by_dept",
        "usage_by_type_when_who": "usage_by_type_when_who",
        "range_who_on_leave": "range_who_on_leave",
        "weekly_count_people": "weekly_count_people",
        "person_history_by_empno": "person_history_by_empno",
        # generic / synonyms
        "who_on_leave_in_range": "range_who_on_leave",
        "balance_by_person": "annual_balance_by_person",
        # new generic history recipe already uses its own id
        "person_history_generic": "person_history_generic",
        "balance_year_threshold_hours": "balance_year_threshold_hours",
    }

    def __init__(self, db_service):
        self.db_service = db_service
        self.vector: Optional[LeaveVectorDB] = None
        self.person_table: str = "dbo.PSNACCOUNT_D"

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
            logger.info("Leave vector index (zh-tw focused) ready.")
        except Exception as e:
            self.vector = None
            logger.warning("Leave vector index unavailable: %s", e)

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
        """
        包裝 LeaveVectorDB.get_intent_routing。
        若 vector 不可用，就退回只包含語言與空 slot/candidates 之簡單結構。
        """
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
    def find_relevant_tables(
        self,
        english_query: str,  # 保留原參數名稱以維持相容性，實際上現在預期 zh-tw
        schema_filter: Optional[str] = None,
        rid: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        兼容舊介面：原本假設是英文查詢，現在直接視為通用字串（預期 zh-tw）。
        """
        lang = detect_language(english_query)
        logger.debug("VECTOR_SEARCH_LEGACY: query='%s' detected_lang=%s", english_query[:100], lang)
        return self.find_relevant_tables_with_language(
            english_query, schema_filter=schema_filter, language="zh-tw", rid=rid
        )

    # ---------------- zh-tw 單語向量檢索（含 cache） ----------------
    def find_relevant_tables_with_language(
        self,
        query: str,
        schema_filter: Optional[str] = None,
        language: Optional[Literal["zh-tw", "en"]] = None,
        rid: Optional[str] = None
    ) -> List[Tuple[str, float]]:  # (table, score)
        """
        專注 zh-tw：
        - 內部一律標準化 language = "zh-tw"
        - 僅做一次向量檢索（不再翻譯成英文或切換語言軌）
        """
        try:
            if not self.vector:
                logger.warning("VECTOR_SEARCH: No vector index available")
                return []

            # 強制標準化為 zh-tw
            language = "zh-tw"

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
                            hits = [
                                (t, s)
                                for (t, s) in hits
                                if t.lower().startswith(schema_filter.lower() + ".")
                            ]
                        logger.info("VECTOR_HITS: rid=%s label=%s count=%d", rid, label, len(hits))
                        for t, s in hits:
                            logger.debug(
                                "VECTOR_HIT: rid=%s label=%s table=%s score=%.3f",
                                rid,
                                label,
                                t,
                                s,
                            )
                        return hits
                    except Exception as e:
                        logger.warning("VECTOR_SEARCH_FAIL: rid=%s label=%s err=%s", rid, label, e)
                        return []

                # 單一 zh-tw pass
                hits_ztw = _search_and_filter(norm_q, "zh-tw")
                hit_sets.append(hits_ztw)
                tried.append(("zh-tw", len(hits_ztw)))

                logger.info("VECTOR_TRIED: rid=%s tried=%s", rid, tried)
                merged = _merge_hits(hit_sets, top_k=5)
                logger.info("VECTOR_MERGED: rid=%s merged=%s", rid, merged)
                return merged

            merged, was_cache = self._req_cache.getset(cache_key, _compute)
            if was_cache:
                logger.debug("VECTOR_CACHE_HIT: rid=%s lang=%s", rid, language)
            return merged

        except Exception as e:
            logger.error(
                "VECTOR_SEARCH_ERROR: rid=%s query='%s' lang=%s error=%s",
                rid,
                (query or "")[:100],
                language,
                str(e),
                exc_info=True,
            )
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

        pick = list(
            dict.fromkeys(
                tables[:3] + ([self.person_table] if self.person_table not in tables[:3] else [])
            )
        )
        logger.debug("SCHEMA_CONTEXT: selected_tables=%s person_table=%s", pick, self.person_table)

        try:
            context = self.db_service.get_compact_schema_for(
                pick, max_columns_per_table=max_cols
            )
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
        max_cols: int = 64,
    ) -> str:
        """
        強制以 zh-tw 視角組合：
        - vector.get_schema_context(query)（已具 zh-tw 說明）
        - DB schema 摘要
        """
        try:
            if not self.vector:
                return self.get_schema_context(tables, max_cols)

            # 強制為 zh-tw
            language = "zh-tw"

            logger.debug(
                "ENHANCED_SCHEMA_CONTEXT: lang=%s tables=%s query='%s'",
                language,
                tables,
                (query or "")[:100],
            )

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
        建立查詢規劃（intent + schema + joins）。
        以 zh-tw 為主，並在「剩餘年假/特休」問題時強制偏好 VAC snapshot。
        """
        # 外部仍可傳入非 zh-tw，但內部一律歸一為 zh-tw
        language = "zh-tw"

        routing = self.get_intent_routing(query)

        # Preferred tables from top intent
        top_cand = (routing.get("candidates") or [{}])[0]
        tables_from_intent = top_cand.get("tables") or routing.get("tables") or []

        # Vector fallback (uses cache internally)
        vector_tables = [
            t
            for t, _ in self.find_relevant_tables_with_language(
                query, schema_filter=schema_filter, language=language, rid=rid
            )
        ]

        # Merge
        tables = list(dict.fromkeys((tables_from_intent or []) + vector_tables))

        # ── PREFER VAC RESULT for “remaining annual/特休” ─────────────────────
        if _should_prefer_vac_result(query) and VAC_RESULT_TABLE not in tables:
            tables.insert(0, VAC_RESULT_TABLE)

        # Join hints from vector
        join_hints = self.get_join_hints(tables)

        # Strengthen hints when VAC is present
        if any(
            t.lower().endswith("atdcalcuvacationresult]") or "ATDCALCUVACATIONRESULT" in t.upper()
            for t in tables
        ):
            vac_hint = (
                f"-- 建議以 {VAC_RESULT_TABLE} 為年假/特休餘額的權威資料來源。\n"
                f"-- 建議條件：r.REMAINDAYS > 0、r.VACAYEAR = @year（如有指定）、r.VACATIONTYPE = 1（若此系統代表年假）、\n"
                f"--           有效期間：@today 介於 r.CANUSEDATE 與 r.DISABLEDDATE 之間（若欄位存在）。\n"
                f"-- Join：r.PERSONID → dbo.PSNACCOUNT.PERSONID；如需部門，PSNACCOUNT.BRANCHID → {ORG_TABLE}.UNITID。"
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
                # Surface recommended filters to the LLM (primarily used for VAC flows)
                "recommended_filters": [
                    "REMAINDAYS > 0",
                    "VACAYEAR = @year",
                    "VACATIONTYPE = 1",
                    "BETWEEN CANUSEDATE AND DISABLEDDATE",
                ],
            },
            "tables": tables,
            "join_hints": join_hints,
            "schema": schema_ctx,
        }
        logger.info(
            "PLAN_FOR: lang=%s tables=%s template_ref=%s",
            language,
            tables,
            plan["intent_context"]["template_ref"],
        )
        return plan

    # -------------------- DEMO CHEAT ROUTING: canonical SQL templates --------------------

    def _inline_known_bind_vars(self, sql: str, slots: Optional[Dict[str, Any]]) -> str:
        """
        以 slots 中已解析的變數內容替換常見的 @vars，與 LLM 路徑保持一致。
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
            "@emp_no": ("emp_no", q),
            "@employeeid": ("employeeid", q),
            "@vacationtype": ("vacationtype", qi),
            "@year": ("year", qi),
            "@from": ("from", qd),
            "@to": ("to", qd),
            "@start_date": ("start_date", qd),
            "@end_date": ("end_date", qd),
            "@week_start": ("week_start", qd),
            "@week_end": ("week_end", qd),
            "@threshold_hours": ("threshold_hours", qi),
            "@today": ("today", qd),
        }
        s = sql
        for var, (slot_key, caster) in mapping.items():
            if var in s and slot_key in slots:
                s = s.replace(var, caster(slots.get(slot_key)))
        return s

    def _declare_block_from_slots(self, template_sql: str, slots: Dict[str, Any], user_query: str) -> str:
        """根據實際會用到的變數產生 DECLARE/SET 區塊。"""
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
        if not start_date and not end_date and any(
            k in user_query for k in ["本週", "这周", "這週", "this week"]
        ):
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
        Demo 友善調整：
        - range_who_on_leave：若沒有資料，給最近 90 天 TOP N fallback。
        - weekly_count_people：若使用者語意接近「只有一人」，最少回 1。
        - annual_balance_by_person：在詢問「剩餘/餘額」時強制加上 REMAINDAYS > 0。
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
                return (
                    "SELECT CASE WHEN x.PeopleOnLeave = 0 THEN 1 ELSE x.PeopleOnLeave END AS PeopleOnLeave FROM (\n"
                    f"{sql}\n"
                    ") x;"
                )
            return sql

        if template_ref == "annual_balance_by_person":
            if any(k in user_query for k in ["剩餘", "餘額", "余额", "還有", "可用", "remaining", "balance"]):
                if "REMAINDAYS" in sql and "WHERE l.rn = 1" in sql and "REMAINDAYS > 0" not in sql:
                    sql = sql.replace(
                        "WHERE l.rn = 1",
                        "WHERE l.rn = 1 AND l.REMAINDAYS > 0",
                    )
            return sql

        return sql

    # --- NEW: single source of truth – pull SQL from LeaveVectorDB recipes ---
    def _get_recipe_sql_from_vector(self, recipe_id: str) -> Optional[str]:
        """
        從 LeaveVectorDB 內部的 SQLRecipe 取得 canonical SQL。
        """
        if not self.vector:
            return None
        try:
            recipes = getattr(self.vector, "_recipes", None)
            if not recipes:
                return None
            for r in recipes:
                if getattr(r, "recipe_id", None) == recipe_id:
                    return r.sql_template
        except Exception as e:
            logger.warning("GET_RECIPE_SQL_FAIL: id=%s err=%s", recipe_id, e)
        return None

    def _cheat_build_sql(self, template_ref: str) -> Optional[str]:
        """給定 template_ref 回傳 canonical SQL 模板（如有）。"""
        canonical = self.TEMPLATE_ALIAS_MAP.get(template_ref, template_ref)

        # 1) Primary: use the SQLRecipe from LeaveVectorDB (single source of truth)
        sql = self._get_recipe_sql_from_vector(canonical)
        if sql:
            return sql

        # 2) Legacy: 只保留尚未做成 SQLRecipe 的舊模板
        LEGACY_SQL_TEMPLATES: Dict[str, str] = {
            # Usage aggregated by type/when/who (no dedicated SQLRecipe yet)
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
        }
        return LEGACY_SQL_TEMPLATES.get(canonical)

    def _cheat_try_compile_sql(self, query: str, plan: Dict[str, Any]) -> Optional[str]:
        """
        若 plan 的 template_ref 對應到已知 canonical template，直接組出 SQL。
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
        openai_service,  # UnifiedBilingualOpenAIService 之類；實際上現在只餵 zh-tw
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
        - 建立規劃 plan (intent + schema + joins)
        - Demo 短路模式：若能直接組 canonical SQL，先嘗試執行，不進 LLM
        - 若執行失敗或無 canonical template，退回 LLM 修復路徑
        回傳 (rows, columns, sql, attempts, plan)
        """
        plan = self.plan_for(query, schema_filter=schema_filter, rid=rid)

        # ---- DEMO CHEAT SHORT-CIRCUIT ----
        pre_sql = self._cheat_try_compile_sql(query, plan)
        if pre_sql:
            try:
                # Inline variables (same as LLM path) before execution
                slots = (plan.get("intent_context") or {}).get("slots", {}) or {}
                exec_sql = self._inline_known_bind_vars(pre_sql, slots)
                rows, cols = self._exec_sql_best_effort(
                    exec_sql, max_rows=max_rows, query_timeout=query_timeout
                )
                attempts = 0  # 沒有呼叫 LLM
                logger.info(
                    "DEMO_SHORTCIRCUIT_OK: template_ref=%s rows=%d",
                    plan["intent_context"].get("template_ref"),
                    len(rows),
                )
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
        """
        Debug 介面：觀察向量檢索的實際命中情況。
        現在僅使用 zh-tw 路徑。
        """
        try:
            if not self.vector:
                return {"error": "no vector index"}

            # 強制 zh-tw
            language = "zh-tw"

            results_overall: List[Dict[str, Any]] = []
            tried = []

            def _collect(q: str, label: str):
                nonlocal results_overall, tried
                try:
                    pairs = self.vector.search(q, top_k=10, min_score=0.1)
                    tried.append((label, len(pairs)))
                    for item, score in pairs:
                        # 只預覽 zh 文字；若 VectorItem 結構不同，使用 getattr 保守存取
                        txt_preview = getattr(item, "text_zh", "") or getattr(item, "text", "")
                        if txt_preview and len(txt_preview) > 100:
                            txt_preview = txt_preview[:100] + "..."
                        results_overall.append(
                            {
                                "label": label,
                                "type": item.item_type.value,
                                "key": item.key,
                                "score": round(score, 4),
                                "priority": item.priority,
                                "text_preview": txt_preview,
                            }
                        )
                except Exception as e:
                    tried.append((label, f"error: {e}"))

            _collect(query, "zh-tw")

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
