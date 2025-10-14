# ================================================================================
# backend/app/services/retrieval/vector_search_service.py
from __future__ import annotations

import re
import logging
import functools
from typing import List, Tuple, Optional, Dict, Any, Literal, Iterable
from collections import defaultdict
from datetime import datetime, date, timedelta

# Language-aware vector system
from app.services.leave_vector import LeaveVectorDB, detect_language, build_leave_index

# Prefer: try to load from disk, else build, else minimal
try:
    vdb = LeaveVectorDB.load_from_disk("leave_schema_vectors.db")
except Exception:
    vdb = build_leave_index()  # <-- now succeeds
    try:
        vdb.save_to_disk()
    except Exception:
        pass

# Translation fallback
from app.services.aws.translation_service import AWSTranslationService

logger = logging.getLogger(__name__)

# --- name normalization helpers -------------------------------------------------
def _normalize_table_name(t: str) -> str:
    """
    Normalize to 'schema.table' (lowercase), removing db prefix and brackets.
    Examples:
      [dbo].[ATDLEAVEDATA]                  -> dbo.atdleavedata
      [eHRAntung_DB].[dbo].[ATDLEAVEDATA]   -> dbo.atdleavedata
      dbo.ATDLEAVEDATA                      -> dbo.atdleavedata
      ATDLEAVEDATA                          -> atdleavedata (no schema)
    """
    if not t:
        return ""
    s = str(t).strip()
    s = s.replace("].[", ".").replace("]", "").replace("[", "").replace('"', "")
    parts = [p for p in s.split(".") if p]
    if len(parts) >= 2:
        schema, table = parts[-2], parts[-1]
        return f"{schema.lower()}.{table.lower()}"
    return parts[-1].lower() if parts else ""

def _schema_matches(t: str, wanted_schema: Optional[str]) -> bool:
    """True if normalized table schema == wanted (case-insensitive)."""
    if not wanted_schema:
        return True
    nt = _normalize_table_name(t)
    if "." not in nt:
        return False
    schema = nt.split(".", 1)[0]
    return schema == wanted_schema.lower()


# ========== Helper Functions ==========

@functools.lru_cache(maxsize=1)
def _get_trad2simp_impl():
    """
    Try several libraries for Traditional -> Simplified conversion, in order:
    1) opencc-python-reimplemented
    2) hanziconv
    3) zhconv
    If none are available, return (False, simple_map_fn).
    """
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
        "部門": "部门", "單位": "单位", "員工": "员工", "工號": "工号",
        "請假": "请假", "出勤": "出勤", "假別": "假别", "資料": "资料",
        "這": "这", "個": "个", "週": "周", "當天": "当天", "現": "现",
        "狀": "状", "態": "态", "數": "数", "據": "据", "離": "离", "職": "职",
    }
    def _mini_trad2simp(s: str) -> str:
        return "".join(_MINI_MAP.get(ch, ch) for ch in s)
    return (False, _mini_trad2simp)


def _trad2simp(text: str) -> str:
    """Convert zh-TW (Traditional) to zh-CN (Simplified) with best-effort fallbacks."""
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


def validate_generated_sql(
    sql: str,
    current_year: int,
    data_anchor_year: Optional[int] = None
) -> Tuple[bool, str, List[str]]:
    """
    Validate generated SQL for common issues.
    Returns: (is_valid, corrected_sql, warnings)
    """
    warnings = []
    corrected_sql = sql

    # Check 1: Outdated years (capture full YYYY)
    old_year_pattern = r"'((?:19|20)\d{2})-(\d{2})-(\d{2})'"
    matches = re.findall(old_year_pattern, sql)

    min_valid_year = data_anchor_year - 1 if data_anchor_year else current_year - 1

    for y, m, d in matches:
        year = int(y)
        if year < min_valid_year:
            old_date = f"'{y}-{m}-{d}'"
            new_date = f"'{current_year}-{m}-{d}'"
            warnings.append(f"WARNING: Found old year {year}, expected {current_year}")
            warnings.append(f"AUTO-CORRECTED: {old_date} → {new_date}")
            corrected_sql = corrected_sql.replace(old_date, new_date)

    # Check 2: Missing name resolution
    if 'PERSONID' in sql.upper() and 'TRUENAME' not in sql.upper():
        warnings.append("WARNING: Query includes PERSONID but may not resolve names")

    # Check 3: Missing VALIDATED filter for leave data
    if 'ATDLEAVEDATA' in sql.upper() and 'VALIDATED' not in sql.upper():
        warnings.append("WARNING: Querying ATDLEAVEDATA without VALIDATED filter")

    # Check 4: Large table without date filter
    if 'ATDLEAVEDATA' in sql.upper():
        has_date_filter = any(
            date_col in sql.upper()
            for date_col in ['WORKDATE', 'STARTDATE', 'ENDDATE']
        )
        if not has_date_filter:
            warnings.append("WARNING: Large table query without date range filter - may be slow")

    is_valid = len([w for w in warnings if w.startswith("WARNING")]) == 0
    return is_valid, corrected_sql, warnings


def extract_date_context_from_query(
    query: str,
    lang: Literal["zh-tw", "en"]
) -> Dict[str, Any]:
    """Extract date-related context from query."""
    context = {
        "has_explicit_year": False,
        "explicit_years": [],
        "has_relative_date": False,
        "relative_terms": [],
        "date_range_detected": False,
    }

    # FIX: capture full 4-digit years, not only '19' or '20'
    years = re.findall(r'\b(?:19|20)\d{2}\b', query)
    if years:
        context["has_explicit_year"] = True
        context["explicit_years"] = [int(y) for y in years]

    if lang == "zh-tw":
        relative_terms = [
            "今天", "昨天", "明天", "本月", "上月", "下月",
            "本週", "上週", "下週", "今年", "去年", "明年",
            "本季", "上季"
        ]
        found_terms = [t for t in relative_terms if t in query]
    else:
        relative_terms = [
            "today", "yesterday", "tomorrow", "this month", "last month", "next month",
            "this week", "last week", "next week", "this year", "last year", "next year",
            "this quarter", "last quarter"
        ]
        ql = query.lower()
        found_terms = [t for t in relative_terms if t in ql]

    if found_terms:
        context["has_relative_date"] = True
        context["relative_terms"] = found_terms

    # Simple date range like "9/22-9/26" or with Chinese connectors
    date_range_pattern = r'\d{1,2}/\d{1,2}\s*[-到至]\s*\d{1,2}/\d{1,2}'
    if re.search(date_range_pattern, query):
        context["date_range_detected"] = True

    return context


# ---------------------- Date math helpers (robust & reused) ----------------------

def _first_day_of_month(d: date) -> date:
    return d.replace(day=1)

def _first_day_next_month(d: date) -> date:
    return (_first_day_of_month(d).replace(day=28) + timedelta(days=4)).replace(day=1)

def _last_day_of_month(d: date) -> date:
    return _first_day_next_month(d) - timedelta(days=1)

def _week_bounds_iso(d: date) -> Tuple[date, date, date, date]:
    """
    Returns:
      this_week_start (Mon), this_week_end_incl (Sun),
      next_week_start (Mon), last_week_start (Mon)
    """
    this_mon = d - timedelta(days=d.weekday())
    this_sun = this_mon + timedelta(days=6)
    next_mon = this_mon + timedelta(days=7)
    last_mon = this_mon - timedelta(days=7)
    return this_mon, this_sun, next_mon, last_mon

def _quarter_bounds(d: date) -> Tuple[date, date, date, date]:
    """Return (this_q_start, this_q_end_incl, next_q_start, last_q_start)."""
    q = (d.month - 1) // 3  # 0..3
    this_q_start = date(d.year, q * 3 + 1, 1)
    next_q_start = date(d.year + (1 if q == 3 else 0), ((q + 1) % 4) * 3 + 1, 1)
    this_q_end = next_q_start - timedelta(days=1)
    last_q_start = date(d.year - (1 if q == 0 else 0), ((q - 1) % 4) * 3 + 1, 1)
    return this_q_start, this_q_end, next_q_start, last_q_start


# ========== Main Service Class ==========

class VectorSearchService:
    """Enhanced language-aware vector-based table retrieval and schema operations."""

    def __init__(self, db_service):
        self.db_service = db_service
        self.vector: Optional[LeaveVectorDB] = None
        self.person_table: str = "dbo.PSNACCOUNT"
        self.translator = AWSTranslationService()
        self.data_anchor: Optional[str] = None

        # track latest processed query for date snapping
        self._last_processed_query: Optional[str] = None
        self._last_date_context: Dict[str, Any] = {}

        # Date patterns for rewriting
        self._zh_date_patterns = {
            r"今天|今日": "today",
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

        self._initialize_vector_index()
        self._determine_person_table()
        self._initialize_data_anchor()

    # ========== Initialization ==========

    def _initialize_vector_index(self):
        """Initialize the enhanced language-aware vector index."""
        try:
            # Try lazy import of build function
            try:
                logger.debug("VECTOR_INIT: trying build_leave_index()")
                from app.services.leave_vector import build_leave_index  # type: ignore
                self.vector = build_leave_index()
                logger.info("Enhanced language-aware vector index ready (built in-memory).")
                return
            except Exception as e1:
                logger.warning("build_leave_index unavailable or failed: %s", e1)

            # Fallback 1: load saved index from disk (if previously persisted)
            try:
                logger.debug("VECTOR_INIT: trying LeaveVectorDB.load_from_disk()")
                self.vector = LeaveVectorDB.load_from_disk()
                logger.info("Enhanced language-aware vector index loaded from disk.")
                return
            except Exception as e2:
                logger.warning("LeaveVectorDB.load_from_disk failed: %s", e2)

            # Fallback 2: minimal instance (no embeddings) to keep service alive
            try:
                logger.debug("VECTOR_INIT: falling back to empty LeaveVectorDB(tables=[])")
                self.vector = LeaveVectorDB(tables=[])
                logger.info("Initialized minimal LeaveVectorDB (no tables).")
                return
            except Exception as e3:
                logger.warning("Minimal LeaveVectorDB init failed: %s", e3)

            # If we get here, nothing worked
            self.vector = None
            logger.warning("Language-aware vector unavailable after all fallbacks.")

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
                    logger.info("Using person table: %s", self.person_table)
        except Exception as e:
            logger.warning("Could not determine person table: %s", e)

    def _initialize_data_anchor(self):
        """Set data anchor to latest WORKDATE in database"""
        try:
            rows, _ = self.db_service.run_select(
                "SELECT CONVERT(varchar(10), MAX(CAST(WORKDATE AS date)), 23) "
                "FROM dbo.ATDLEAVEDATA"
            )
            if rows and rows[0][0]:
                self.data_anchor = str(rows[0][0])
                logger.info("Data anchor set to: %s (current date: %s)",
                            self.data_anchor, datetime.now().date())
        except Exception as e:
            logger.warning("Could not set data anchor: %s", e)
            self.data_anchor = None

    # ========== Query Processing ==========

    def process_query_with_context(
        self,
        user_input: str,
        lang_override: Optional[Literal["zh-tw", "en"]],
        session_id: str,
        current_year: int
    ) -> Dict[str, Any]:
        """
        Process user query: detect language, rewrite dates, handle context.
        Returns dict with processed query and metadata.
        """
        detected_lang = detect_language(user_input)
        lang = self._normalize_lang(lang_override or detected_lang)

        # Heuristic confidence
        chinese_chars = sum(1 for c in user_input if '\u4e00' <= c <= '\u9fff')
        alnum_chars = len([c for c in user_input if c.isalnum()])
        confidence = min(1.0, (chinese_chars / max(alnum_chars, 1)) * 2) if lang == "zh-tw" else 0.9

        processed_query = self._rewrite_relative_dates(user_input, lang, current_year)

        # store for later SQL snapping
        try:
            self._last_processed_query = processed_query
            self._last_date_context = extract_date_context_from_query(processed_query, lang)
        except Exception:
            self._last_processed_query = processed_query
            self._last_date_context = {}

        logger.info(
            "QUERY_PROCESSED: original=%r processed=%r lang=%s",
            user_input[:100], processed_query[:100], lang
        )

        return {
            "language": lang,
            "language_confidence": confidence,
            "original_query": user_input,
            "processed_query": processed_query,
            "session_id": session_id,
        }

    def retrieve_schema_context(
        self,
        query: str,
        schema_filter: Optional[str],
        language: Literal["zh-tw", "en"],
        current_year: int,
        rid: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant tables, schema context, join hints, and few-shot examples.
        All vector logic happens here.
        """
        table_scores = self.find_relevant_tables_with_language(
            query,
            schema_filter=schema_filter,
            language=language,
            rid=rid
        )
        tables = [t for t, _ in table_scores]

        if not tables and schema_filter:
            logger.warning("VECTOR_EMPTY_AFTER_FILTER: rid=%s schema_filter=%s -> retry without filter", rid, schema_filter)
            table_scores = self.find_relevant_tables_with_language(
                query,
                schema_filter=None,
                language=language,
                rid=rid
            )
            tables = [t for t, _ in table_scores]

        logger.info("VECTOR_TABLES_SELECTED: rid=%s tables=%s norm=%s",
                    rid, tables, [_normalize_table_name(t) for t, _ in table_scores])

        schema_context = self.get_schema_context_with_language(
            tables,
            query,
            language=language,
            current_year=current_year
        )

        join_hints = self.get_join_hints(tables)

        few_shot_examples = []
        if self.vector and hasattr(self.vector, 'search_few_shot_examples'):
            try:
                few_shot_examples = self.vector.search_few_shot_examples(
                    query,
                    top_k=3,
                    category_filter=None
                )
            except Exception as e:
                logger.warning("Could not retrieve few-shot examples: %s", e)

        return {
            "tables": tables,
            "table_scores": table_scores,
            "schema_context": schema_context,
            "join_hints": join_hints,
            "few_shot_examples": few_shot_examples,
            "language": language,
            "current_year": current_year,
        }

    # ========== Date & SQL Processing ==========

    def anchor_sql_dates(self, sql: str, current_year: int) -> str:
        """
        Replace year placeholders and validate/correct stale dates in SQL.
        Also SNAP incorrectly-chosen relative windows (e.g., "last month") to the correct bounds
        using the latest processed query context.
        """
        if not sql:
            return sql

        anchored = sql

        # Replace all common placeholders for current year
        placeholders = [
            '{{CURRENT_YEAR}}',
            '{{{{CURRENT_YEAR}}}}',  # sometimes double-braced via f-string
            '{CURRENT_YEAR}'
        ]
        for ph in placeholders:
            anchored = anchored.replace(ph, str(current_year))

        # Validate & correct stale year literals with optional data anchor
        data_anchor_year = None
        try:
            if self.data_anchor:
                data_anchor_year = int(str(self.data_anchor)[:4])
        except Exception:
            data_anchor_year = None

        ok, corrected, warnings = validate_generated_sql(
            anchored, current_year=current_year, data_anchor_year=data_anchor_year
        )
        if warnings:
            for w in warnings:
                logger.warning("SQL_VALIDATION: %s", w)

        if corrected != anchored:
            logger.info("SQL_DATES_CORRECTED: before=%r after=%r", anchored[:200], corrected[:200])

        # Snap month/week/year/quarter windows based on the latest processed query, if any
        try:
            corrected = self._snap_sql_windows_based_on_query(corrected)
        except Exception as e:
            logger.debug("SQL_DATE_SNAP_SKIPPED: %s", e)

        return corrected

    @staticmethod
    def _normalize_lang(lang: Optional[str]) -> Literal["zh-tw", "en"]:
        """
        Normalize language tags from UI or detectors.
        Accepts: 'zh', 'zh-tw', 'zh_TW', 'ZH-tw', 'zh-Hant', etc.
        """
        if not lang:
            return "en"
        s = str(lang).strip().lower().replace("_", "-")
        if s.startswith("zh"):
            return "zh-tw"
        return "en"

    @staticmethod
    def _month_add(d: date, months: int) -> date:
        """Add (or subtract) whole months while clamping day to month-end if needed."""
        y = d.year + (d.month - 1 + months) // 12
        m = (d.month - 1 + months) % 12 + 1
        from calendar import monthrange
        max_day = monthrange(y, m)[1]
        return d.replace(year=y, month=m, day=min(d.day, max_day))

    def _rewrite_last_n_ranges(self, text: str) -> str:
        """
        Support '過去N個月/近N個月/過去N天/近N天' → explicit 'from YYYY-MM-DD to YYYY-MM-DD'
        Works even after we translate zh to en tokens.
        """
        out = text
        today = datetime.now().date()

        # zh patterns → english tokens (keep both)
        patterns = [
            (r"過去(\d+)個?月", "last \\1 months"),
            (r"近(\d+)個?月",   "last \\1 months"),
            (r"過去(\d+)天",    "last \\1 days"),
            (r"近(\d+)天",      "last \\1 days"),
        ]
        for pat, rep in patterns:
            out = re.sub(pat, rep, out)

        # Now handle english tokens
        def _repl_months(m):
            n = int(m.group(1))
            end = today
            start = self._month_add(end.replace(day=1), -(n-1)) if n >= 1 else end
            return f"from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"

        def _repl_days(m):
            n = int(m.group(1))
            end = today
            start = end - timedelta(days=max(n-1, 0))
            return f"from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"

        out = re.sub(r"\blast\s+(\d+)\s+months\b", _repl_months, out, flags=re.I)
        out = re.sub(r"\blast\s+(\d+)\s+days\b", _repl_days, out, flags=re.I)
        return out

    def _rewrite_relative_dates(
        self,
        text: str,
        lang: Literal["zh-tw", "en"],
        current_year: int
    ) -> str:
        """
        Rewrite relative date expressions to absolute dates.
        Conservative: handles day/week/month/quarter/year + last N months/days.
        (Text rewrite only; SQL snapping happens later in anchor_sql_dates)
        """
        result = text
        today: date = datetime.now().date()

        # Translate zh date terms to English tokens first
        if lang == "zh-tw":
            for zh_pattern, en_replacement in self._zh_date_patterns.items():
                result = re.sub(zh_pattern, en_replacement, result)

        lower = result.lower()

        # today / yesterday / tomorrow
        repl = {
            r'\btoday\b': today.strftime('%Y-%m-%d'),
            r'\byesterday\b': (today - timedelta(days=1)).strftime('%Y-%m-%d'),
            r'\btomorrow\b': (today + timedelta(days=1)).strftime('%Y-%m-%d'),
        }
        for pat, val in repl.items():
            result = re.sub(pat, val, result, flags=re.IGNORECASE)

        # this week (Mon-Sun in ISO)
        if 'this week' in lower:
            this_mon, this_sun, _, _ = _week_bounds_iso(today)
            result = re.sub(r'\bthis week\b',
                            f'from {this_mon.strftime("%Y-%m-%d")} to {this_sun.strftime("%Y-%m-%d")}',
                            result, flags=re.IGNORECASE)

        # last week (Mon-Sun)
        if 'last week' in lower:
            _, _, _, last_mon = _week_bounds_iso(today)
            last_sun = last_mon + timedelta(days=6)
            result = re.sub(r'\blast week\b',
                            f'from {last_mon.strftime("%Y-%m-%d")} to {last_sun.strftime("%Y-%m-%d")}',
                            result, flags=re.IGNORECASE)

        # this month
        if 'this month' in lower:
            start = _first_day_of_month(today)
            end_incl = _last_day_of_month(today)
            result = re.sub(r'\bthis month\b',
                            f'from {start.strftime("%Y-%m-%d")} to {end_incl.strftime("%Y-%m-%d")}',
                            result, flags=re.IGNORECASE)

        # last month
        if 'last month' in lower:
            first_this = _first_day_of_month(today)
            last_month_end_incl = first_this - timedelta(days=1)
            last_month_start = _first_day_of_month(last_month_end_incl)
            result = re.sub(r'\blast month\b',
                            f'from {last_month_start.strftime("%Y-%m-%d")} to {last_month_end_incl.strftime("%Y-%m-%d")}',
                            result, flags=re.IGNORECASE)

        # this quarter
        if 'this quarter' in lower:
            q_start, q_end, _, _ = _quarter_bounds(today)
            result = re.sub(r'\bthis quarter\b',
                            f'from {q_start.strftime("%Y-%m-%d")} to {q_end.strftime("%Y-%m-%d")}',
                            result, flags=re.IGNORECASE)

        # last quarter
        if 'last quarter' in lower:
            _, _, _, last_q_start = _quarter_bounds(today)
            last_q_end = _first_day_next_month(last_q_start.replace(month=last_q_start.month+2 if last_q_start.month<=10 else 12, day=1)) - timedelta(days=1)
            # better: compute end as next_q_start - 1 day
            next_after_last_q = date(last_q_start.year + (1 if (last_q_start.month+2) > 12 else 0),
                                     ((last_q_start.month+2-1) % 12) + 1, 1)
            last_q_end = next_after_last_q - timedelta(days=1)
            result = re.sub(r'\blast quarter\b',
                            f'from {last_q_start.strftime("%Y-%m-%d")} to {last_q_end.strftime("%Y-%m-%d")}',
                            result, flags=re.IGNORECASE)

        # this year
        if 'this year' in lower:
            start = date(today.year, 1, 1)
            end_incl = date(today.year, 12, 31)
            result = re.sub(r'\bthis year\b',
                            f'from {start.strftime("%Y-%m-%d")} to {end_incl.strftime("%Y-%m-%d")}',
                            result, flags=re.IGNORECASE)

        # last year
        if 'last year' in lower:
            start = date(today.year - 1, 1, 1)
            end_incl = date(today.year - 1, 12, 31)
            result = re.sub(r'\blast year\b',
                            f'from {start.strftime("%Y-%m-%d")} to {end_incl.strftime("%Y-%m-%d")}',
                            result, flags=re.IGNORECASE)

        # finally support 過去N個月/近N個月, 過去N天/近N天
        return self._rewrite_last_n_ranges(result)

    # ---------- SQL window snapping (safe & surgical) ----------

    # patterns: target WORKDATE/STARTDATE/ENDDATE window shapes
    _RE_EXCL = re.compile(
        r"(?is)(?P<col>(?:\w+\.)?(?:WORKDATE|STARTDATE|ENDDATE))\s*>=\s*'(?P<s>\d{4}-\d{2}-\d{2})'\s*"
        r"(?:AND|and)\s*(?P=col)\s*<\s*'(?P<e>\d{4}-\d{2}-\d{2})'"
    )
    _RE_INCL = re.compile(
        r"(?is)(?P<col>(?:\w+\.)?(?:WORKDATE|STARTDATE|ENDDATE))\s*>=\s*'(?P<s>\d{4}-\d{2}-\d{2})'\s*"
        r"(?:AND|and)\s*(?P=col)\s*<=\s*'(?P<e>\d{4}-\d{2}-\d{2})'"
    )
    _RE_BETWEEN = re.compile(
        r"(?is)(?P<col>(?:\w+\.)?(?:WORKDATE|STARTDATE|ENDDATE))\s+BETWEEN\s+'(?P<s>\d{4}-\d{2}-\d{2})'\s+AND\s+'(?P<e>\d{4}-\d{2}-\d{2})'"
    )

    def _snap_sql_windows_based_on_query(self, sql: str) -> str:
        """
        If latest processed query indicates 'last month/this month/last week/this week/etc',
        and SQL contains a recognizable date window on WORKDATE/STARTDATE/ENDDATE,
        snap that window to the correct bounds while preserving style (exclusive/inclusive/between).
        """
        q = (self._last_processed_query or "")[:200].lower()
        ctx = self._last_date_context or {}
        if not q and not ctx:
            return sql

        today = datetime.now().date()

        # Desired windows (start_incl, end_incl, end_excl)
        want: Optional[Tuple[date, date, date]] = None

        def _mk_month_bounds_for(d: date) -> Tuple[date, date, date]:
            s = _first_day_of_month(d)
            e_incl = _last_day_of_month(d)
            e_excl = _first_day_next_month(d)
            return s, e_incl, e_excl

        def _mk_week_bounds_this(d: date) -> Tuple[date, date, date]:
            this_mon, this_sun, next_mon, _ = _week_bounds_iso(d)
            return this_mon, this_sun, next_mon

        def _mk_week_bounds_last(d: date) -> Tuple[date, date, date]:
            _, _, this_mon, last_mon = _week_bounds_iso(d)
            last_sun = last_mon + timedelta(days=6)
            return last_mon, last_sun, this_mon

        def _mk_year_bounds_this(d: date) -> Tuple[date, date, date]:
            s = date(d.year, 1, 1)
            e_incl = date(d.year, 12, 31)
            e_excl = date(d.year + 1, 1, 1)
            return s, e_incl, e_excl

        def _mk_year_bounds_last(d: date) -> Tuple[date, date, date]:
            s = date(d.year - 1, 1, 1)
            e_incl = date(d.year - 1, 12, 31)
            e_excl = date(d.year, 1, 1)
            return s, e_incl, e_excl

        def _mk_quarter_bounds_this(d: date) -> Tuple[date, date, date]:
            this_q_start, this_q_end, next_q_start, _ = _quarter_bounds(d)
            return this_q_start, this_q_end, next_q_start

        def _mk_quarter_bounds_last(d: date) -> Tuple[date, date, date]:
            _, _, _, last_q_start = _quarter_bounds(d)
            # compute last quarter end/excl
            next_after_last_q = date(last_q_start.year + (1 if (last_q_start.month+2) > 12 else 0),
                                     ((last_q_start.month+2-1) % 12) + 1, 1)
            last_q_end_incl = next_after_last_q - timedelta(days=1)
            return last_q_start, last_q_end_incl, next_after_last_q

        # Decide desired bounds from query
        if any(tok in q for tok in ("last month", "上月", "上個月", "上个月")):
            prev_month_end = _first_day_of_month(today) - timedelta(days=1)
            want = _mk_month_bounds_for(prev_month_end)
        elif any(tok in q for tok in ("this month", "本月", "這個月", "这个月")):
            want = _mk_month_bounds_for(today)
        elif any(tok in q for tok in ("last week", "上週", "上周")):
            want = _mk_week_bounds_last(today)
        elif any(tok in q for tok in ("this week", "本週", "這週", "这周")):
            want = _mk_week_bounds_this(today)
        elif any(tok in q for tok in ("last year", "去年")):
            want = _mk_year_bounds_last(today)
        elif any(tok in q for tok in ("this year", "今年")):
            want = _mk_year_bounds_this(today)
        elif any(tok in q for tok in ("last quarter", "上季")):
            want = _mk_quarter_bounds_last(today)
        elif any(tok in q for tok in ("this quarter", "本季", "這季")):
            want = _mk_quarter_bounds_this(today)

        if not want:
            return sql

        start_incl, end_incl, end_excl = want
        s_str = start_incl.strftime("%Y-%m-%d")
        e_incl_str = end_incl.strftime("%Y-%m-%d")
        e_excl_str = end_excl.strftime("%Y-%m-%d")

        def _same_shape_and_wrong(m_s: str, m_e: str, style: str) -> bool:
            try:
                s = datetime.strptime(m_s, "%Y-%m-%d").date()
                e = datetime.strptime(m_e, "%Y-%m-%d").date()
            except Exception:
                return False
            if style == "excl":
                # expect first-of-month/week and end_excl is 1st of next period
                # shape check: s is start-of-period, e > s
                return (e > s) and not (s == start_incl and e == end_excl)
            else:
                # inclusive shape: e >= s
                return (e >= s) and not (s == start_incl and e == end_incl)

        def _apply_rewrite(pattern: re.Pattern, text: str, style: str) -> Tuple[str, bool]:
            def repl(m: re.Match) -> str:
                ms, me = m.group("s"), m.group("e")
                if not _same_shape_and_wrong(ms, me, style):
                    return m.group(0)
                new_s = s_str
                new_e = e_excl_str if style == "excl" else e_incl_str
                before = f"{ms}→{me}"
                after = f"{new_s}→{new_e}"
                logger.info("SQL_DATE_SNAP: style=%s before=%s after=%s", style, before, after)
                seg = m.group(0)
                seg = seg.replace(ms, new_s).replace(me, new_e)
                return seg

            new_text, n = pattern.subn(repl, text, count=1)  # only snap first matched window
            return new_text, (n > 0)

        # Try exclusive (<) first, then inclusive (<=), then BETWEEN
        out, changed = _apply_rewrite(self._RE_EXCL, sql, "excl")
        if not changed:
            out, changed = _apply_rewrite(self._RE_INCL, out, "incl")
        if not changed:
            out, changed = _apply_rewrite(self._RE_BETWEEN, out, "incl")

        return out

    # ========== Vector Search ==========

    def find_relevant_tables(
        self,
        english_query: str,
        schema_filter: Optional[str] = None,
        rid: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Legacy entry point kept for compatibility: auto-detect language and route to bilingual search.
        """
        lang = detect_language(english_query)
        logger.debug("VECTOR_SEARCH_LEGACY: query=%r detected_lang=%s", english_query[:100], lang)
        return self.find_relevant_tables_with_language(
            english_query, schema_filter=schema_filter, language=lang, rid=rid
        )

    def find_relevant_tables_with_language(
        self,
        query: str,
        schema_filter: Optional[str] = None,
        language: Optional[Literal["zh-tw", "en"]] = None,
        rid: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Bilingual table search with multiple fallback paths.
        """
        try:
            if not self.vector:
                logger.warning("VECTOR_SEARCH: No vector index available")
                return []

            if language is None:
                language = detect_language(query)

            logger.info("VECTOR_SEARCH_START: rid=%s lang=%s query=%r", rid, language, query[:100])

            hit_sets: List[List[Tuple[str, float]]] = []
            tried: List[Tuple[str, int]] = []

            def _search_and_filter(q: str, label: str) -> List[Tuple[str, float]]:
                try:
                    raw_hits = self.vector.search_relevant_tables(q, top_k=10)
                    logger.debug("VECTOR_RAW_HITS: rid=%s label=%s raw=%s", rid, label, raw_hits)

                    # normalize + filter by schema if provided
                    hits = []
                    for (t, s) in (raw_hits or []):
                        keep = _schema_matches(t, schema_filter)
                        logger.debug(
                            "VECTOR_HIT_CHECK: rid=%s label=%s table_raw=%s norm=%s score=%.3f keep=%s",
                            rid, label, t, _normalize_table_name(t), s, keep
                        )
                        if keep:
                            hits.append((t, s))

                    # limit after filtering
                    hits = sorted(hits, key=lambda x: x[1], reverse=True)[:5]

                    logger.info("VECTOR_HITS: rid=%s label=%s count=%d (schema_filter=%s)",
                                rid, label, len(hits), schema_filter)
                    for t, s in hits:
                        logger.debug("VECTOR_HIT: rid=%s label=%s table=%s norm=%s score=%.3f",
                                     rid, label, t, _normalize_table_name(t), s)
                    return hits
                except Exception as e:
                    logger.warning("VECTOR_SEARCH_FAIL: rid=%s label=%s err=%s", rid, label, e)
                    return []

            if (language or "").lower().startswith("zh"):
                hits_ztw = _search_and_filter(query, "zh-tw")
                hit_sets.append(hits_ztw)
                tried.append(("zh-tw", len(hits_ztw)))

                q_zcn = _trad2simp(query)
                if q_zcn and q_zcn != query:
                    hits_zcn = _search_and_filter(q_zcn, "zh-cn")
                    hit_sets.append(hits_zcn)
                    tried.append(("zh-cn", len(hits_zcn)))

                q_en = self.translator.translate_to_english(query, "zh-tw") or ""
                if q_en:
                    hits_en = _search_and_filter(q_en, "en")
                    hit_sets.append(hits_en)
                    tried.append(("en", len(hits_en)))
            else:
                hits_en = _search_and_filter(query, language or "en")
                hit_sets.append(hits_en)
                tried.append((language or "en", len(hits_en)))

            logger.info("VECTOR_TRIED: rid=%s tried=%s", rid, tried)
            merged = _merge_hits(hit_sets, top_k=5)
            merged_norm = [(_normalize_table_name(t), s) for t, s in merged]
            logger.info("VECTOR_MERGED: rid=%s merged=%s merged_norm=%s", rid, merged, merged_norm)
            return merged

        except Exception as e:
            logger.error("VECTOR_SEARCH_ERROR: rid=%s query=%r lang=%s error=%s",
                         rid, query[:100], language, str(e), exc_info=True)
            return []

    # ========== Schema Context ==========

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
        current_year: Optional[int] = None,
        max_cols: int = 64
    ) -> str:
        """Language-aware schema context with few-shot examples + live DB schema."""
        try:
            if not self.vector:
                return self.get_schema_context(tables, max_cols)

            if language is None:
                language = detect_language(query)

            if current_year is None:
                current_year = datetime.now().year

            logger.debug("ENHANCED_SCHEMA_CONTEXT: lang=%s tables=%s query=%r current_year=%d",
                         language, tables, query[:100], current_year)

            # Guardrailed business context from vector (few-shots + rules)
            if hasattr(self.vector, 'get_business_prompt'):
                enhanced_context = self.vector.get_business_prompt(query, current_year)
            else:
                enhanced_context = self.vector.get_schema_context(query, include_examples=True)

            # Actual DB schema
            db_schema = self.get_schema_context(tables, max_cols)
            logger.debug("ENHANCED_SCHEMA_CONTEXT_PARTS: enhanced_len=%d db_len=%d",
                         len(enhanced_context or ""), len(db_schema or ""))

            combined = f"{enhanced_context}\n\n=== DATABASE SCHEMA ===\n{db_schema}"
            logger.debug("ENHANCED_SCHEMA_CONTEXT_SUCCESS: total_length=%d", len(combined))
            return combined

        except Exception as e:
            logger.error("ENHANCED_SCHEMA_CONTEXT_ERROR: tables=%s error=%s", tables, str(e), exc_info=True)
            return self.get_schema_context(tables, max_cols)

    def get_business_prompt(
        self,
        query: str,
        current_year: Optional[int] = None,
        language: Optional[Literal["zh-tw", "en"]] = None
    ) -> str:
        """Generate business-aware LLM prompt with guardrails/few-shots."""
        try:
            if not self.vector:
                return f"Business context unavailable. Query: {query}"

            if language is None:
                language = detect_language(query)

            if current_year is None:
                current_year = datetime.now().year

            logger.debug("BUSINESS_PROMPT: lang=%s query=%r current_year=%d",
                         language, query[:100], current_year)

            if hasattr(self.vector, 'get_business_prompt'):
                prompt = self.vector.get_business_prompt(query, current_year)
            else:
                prompt = self.vector.get_schema_context(query, include_examples=True)

            logger.debug("BUSINESS_PROMPT_SUCCESS: prompt_length=%d", len(prompt))
            return prompt
        except Exception as e:
            logger.error("BUSINESS_PROMPT_ERROR: lang=%s error=%s", language, str(e))
            return f"Business context error. Query: {query}"

    # ========== Health & Debug ==========

    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with language awareness details."""
        try:
            if not self.vector:
                return {"ready": False, "reason": "no index"}

            base = self.vector.health_check()
            enhanced = {
                **base,
                "service_version": "language_aware_v5_datesnap",
                "person_table": self.person_table,
                "data_anchor": self.data_anchor,
                "language_fallbacks": ["zh-tw", "zh-cn (optional)", "en"],
            }
            logger.debug("HEALTH_CHECK: %s", enhanced)
            return enhanced
        except Exception as e:
            logger.error("HEALTH_CHECK_ERROR: %s", str(e))
            return {"ready": False, "error": str(e)}

    def debug_search(
        self,
        query: str,
        language: Optional[Literal["zh-tw", "en"]] = None
    ) -> Dict[str, Any]:
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
                            "text_en_preview": (item.text_en[:100] + "...") if len(item.text_en) > 100 else item.text_en,
                            "text_zh_preview": (item.text_zh[:100] + "...") if len(item.text_zh) > 100 else item.text_zh,
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
