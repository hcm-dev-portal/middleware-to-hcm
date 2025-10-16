# ================================================================================
# backend/app/services/data_processing/date_processor.py
from __future__ import annotations

import re
import logging
from typing import Optional, Tuple, List
from datetime import datetime, date, timedelta
import calendar

logger = logging.getLogger(__name__)

# --------- tiny zh numerals (up to 99) ----------
_ZH_DIGIT = {"零":0,"〇":0,"一":1,"二":2,"兩":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
def _parse_zh_int(token: str) -> Optional[int]:
    if not token:
        return None
    m = re.search(r"\d{1,3}", token)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    token = token.strip()
    if token == "十":
        return 10
    if "十" in token:
        left, _, right = token.partition("十")
        tens = 1 if left == "" else _ZH_DIGIT.get(left)
        ones = 0 if right == "" else _ZH_DIGIT.get(right)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return _ZH_DIGIT.get(token)

# --------- date helpers ----------
def _parse_iso(d: str) -> Optional[date]:
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return None

def _fmt(d: date) -> str:
    return d.strftime("%Y-%m-%d")

def _month_bounds(d: date) -> Tuple[date, date]:
    first = d.replace(day=1)
    last_day = calendar.monthrange(d.year, d.month)[1]
    last = d.replace(day=last_day)
    return first, last

def _shift_months(d: date, delta: int) -> date:
    y = d.year + (d.month - 1 + delta) // 12
    m = (d.month - 1 + delta) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))

def _quarter_bounds(d: date) -> Tuple[date, date]:
    q = (d.month - 1) // 3  # 0..3
    start_m = q * 3 + 1
    end_m = start_m + 2
    start = date(d.year, start_m, 1)
    last_day = calendar.monthrange(d.year, end_m)[1]
    end = date(d.year, end_m, last_day)
    return start, end

def _year_bounds(d: date) -> Tuple[date, date]:
    start = date(d.year, 1, 1)
    end = date(d.year, 12, 31)
    return start, end


class DateProcessor:
    """
    Smart bilingual date processor.

    **Policy (updated):**
    - Relative words like 'today/this week/last 7 days/今天/本週/過去7天' always use the ACTUAL current date.
    - Only when the user explicitly asks for *latest data* (e.g., 'latest data', '最近的資料', '最近期資料'),
      do we anchor to the data_anchor.
    - SQL functions (GETDATE(), CURRENT_TIMESTAMP, etc.) are rewritten to the ACTUAL current date.
    """

    def __init__(self, data_anchor: Optional[str] = None):
        self.data_anchor = data_anchor            # Latest data date (YYYY-MM-DD)
        self.current_date = date.today()          # Actual current date

    # ------------ lifecycle ------------
    def set_data_anchor(self, anchor: str):
        self.data_anchor = anchor
        logger.info("Data anchor set to: %s (current date: %s)", anchor, self.current_date)

    def get_data_anchor(self) -> Optional[str]:
        return self.data_anchor

    def get_current_date(self) -> str:
        return _fmt(self.current_date)

    def has_data_anchor(self) -> bool:
        return self.data_anchor is not None

    # ------------ helpers ------------
    def _week_range(self, base: date) -> Tuple[date, date]:
        start = base - timedelta(days=base.weekday())  # Monday
        end = start + timedelta(days=6)                # Sunday
        return start, end

    # NEW: explicit anchor intent detection
    def _has_anchor_intent(self, text: str) -> bool:
        """
        Detect explicit 'latest data' intent; only then we use data_anchor.
        """
        t = (text or "").lower()
        patterns_en = [
            r"\blatest data\b",
            r"\bmost\s+recent\s+data\b",
            r"\bdata\s*as\s*of\b",      # e.g., "as of the latest data"
            r"\bas\s+of\s+latest\b",
        ]
        patterns_zh = [
            r"最新(的)?資料", r"最近的資料", r"最(近|新)期資料", r"資料錨點", r"数据锚点",
            r"數據錨點", r"截至最新資料", r"以最新資料為準",
        ]
        for p in patterns_en + patterns_zh:
            if re.search(p, t, flags=re.IGNORECASE):
                return True
        return False

    def _determine_base_date_for_range(self, query_context: str) -> date:
        """
        For ranges like 'last 7 days', default to CURRENT DATE.
        Only switch to data_anchor if the query explicitly asks for latest/anchored data.
        """
        if self._has_anchor_intent(query_context) and self.data_anchor:
            anchor_date = _parse_iso(self.data_anchor)
            if anchor_date:
                return anchor_date
        return self.current_date

    def _range_from_units(self, n: int, unit: str, direction: str, base: date) -> Tuple[date, date]:
        """
        Calculate date ranges for N units (days/weeks/months/quarters/years) relative to base.
        """
        unit = unit.lower()
        if unit.startswith("day"):
            if direction in ("past", "last", "recent"):
                return base - timedelta(days=n - 1), base
            return base, base + timedelta(days=n - 1)

        if unit.startswith("week"):
            if direction in ("past", "last", "recent"):
                start_week = base - timedelta(weeks=n - 1)
                start, _ = self._week_range(start_week)
                _, end = self._week_range(base)
                return start, end
            start, _ = self._week_range(base)
            end_week = base + timedelta(weeks=n - 1)
            _, end = self._week_range(end_week)
            return start, end

        if unit.startswith("month"):
            if direction in ("past", "last", "recent"):
                start_month = _shift_months(base, -(n - 1))
                start, _ = _month_bounds(start_month)
                _, end = _month_bounds(base)
                return start, end
            start, _ = _month_bounds(base)
            end_month = _shift_months(base, n - 1)
            _, end = _month_bounds(end_month)
            return start, end

        if unit.startswith("quarter"):
            months = n * 3
            if direction in ("past", "last", "recent"):
                start_m = _shift_months(base, -(months - 1))
                start, _ = _quarter_bounds(start_m)
                _, end = _quarter_bounds(base)
                return start, end
            start, _ = _quarter_bounds(base)
            end_m = _shift_months(base, months - 1)
            _, end = _quarter_bounds(end_m)
            return start, end

        if unit.startswith("year"):
            if direction in ("past", "last", "recent"):
                start = date(base.year - (n - 1), 1, 1)
                _, end = _year_bounds(base)
                return start, end
            start, _ = _year_bounds(base)
            end = date(base.year + (n - 1), 12, 31)
            return start, end

        return base, base

    def _pad_english_temporal_tokens(self, text: str) -> str:
        """
        Ensure English temporal tokens are separated by spaces when glued to CJK or alnum.
        Fixes cases like 'today有多少' so the English normalizer can match.
        """
        tokens = [
            "today", "yesterday", "tomorrow",
            "this week", "last week", "next week",
            "this month", "last month", "next month",
            "this quarter", "last quarter", "next quarter",
            "this year", "last year", "next year",
            "right now", "currently", "current", "most recent", "latest"
        ]
        tokens.sort(key=len, reverse=True)

        t = text
        for tok in tokens:
            t = re.sub(rf'(?P<pre>[\u4e00-\u9fffA-Za-z0-9])(?P<tok>{re.escape(tok)})',
                       r'\g<pre> \g<tok>', t, flags=re.IGNORECASE)
            t = re.sub(rf'(?P<tok>{re.escape(tok)})(?P<post>[\u4e00-\u9fffA-Za-z0-9])',
                       r'\g<tok> \g<post>', t, flags=re.IGNORECASE)
        return t

    # ------------ core normalization ------------
    def rewrite_relative_dates(self, query_text: str) -> str:
        """
        Convert relative date expressions into absolute dates with explicit policy:
        - Relative → current-date based
        - Explicit 'latest data' terms → data_anchor (if set)
        """
        original = query_text or ""
        text = original

        # 0) make EN tokens space-safe (for mixed zh/EN)
        text = self._pad_english_temporal_tokens(text)

        # 1) Immediate/current terms → current date
        current_day_patterns = [
            (r"\b(today)\b", 0), (r"\b(yesterday)\b", -1), (r"\b(tomorrow)\b", +1),
            (r"\b(right now)\b", 0), (r"\b(currently)\b", 0),
            (r"(?:今天|今日)", 0), (r"(?:昨天|昨日)", -1), (r"(?:明天|翌日)", +1),
            (r"(?:現在|现在|目前)", 0),
        ]
        for pat, delta in current_day_patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                target_date = self.current_date + timedelta(days=delta)
                text = re.sub(pat, _fmt(target_date), text, flags=re.IGNORECASE)

        # 2) Data-anchored terms → anchor (if explicitly asked)
        if self.data_anchor and self._has_anchor_intent(text):
            anchor = _parse_iso(self.data_anchor)
            if anchor:
                data_relative_patterns = [
                    (r"\b(latest)\b", 0),
                    (r"\b(most\s+recent)\b", 0),
                    (r"\b(current\s+data)\b", 0),
                    (r"(?:最新|最近的資料|最近的数据|最(近|新)期資料)", 0),
                    (r"(?:目前資料|目前数据)", 0),
                ]
                for pat, delta in data_relative_patterns:
                    if re.search(pat, text, flags=re.IGNORECASE):
                        text = re.sub(pat, _fmt(anchor + timedelta(days=delta)), text, flags=re.IGNORECASE)

        # 3) Week ranges (current-date basis)
        week_terms = [
            (r"\bthis\s+week\b", 0), (r"\bcurrent\s+week\b", 0),
            (r"\blast\s+week\b", -1), (r"\bnext\s+week\b", +1),
            (r"(?:本週|這週|这周|本周)", 0), (r"(?:上週|上周)", -1), (r"(?:下週|下周)", +1),
        ]
        for pat, wshift in week_terms:
            if re.search(pat, text, flags=re.IGNORECASE):
                base_date = self.current_date + timedelta(weeks=wshift)
                start, end = self._week_range(base_date)
                text = re.sub(pat, f"between {_fmt(start)} and {_fmt(end)}", text, flags=re.IGNORECASE)

        # 4) Month ranges (current-date basis)
        month_terms = [
            (r"\bthis\s+month\b", 0), (r"\bcurrent\s+month\b", 0),
            (r"\blast\s+month\b", -1), (r"\bnext\s+month\b", +1),
            (r"(?:本月|這個月|这个月)", 0), (r"(?:上月|上個月|上个月)", -1), (r"(?:下月|下個月|下个月)", +1),
        ]
        for pat, mshift in month_terms:
            if re.search(pat, text, flags=re.IGNORECASE):
                base_date = _shift_months(self.current_date, mshift)
                start, end = _month_bounds(base_date)
                text = re.sub(pat, f"between {_fmt(start)} and {_fmt(end)}", text, flags=re.IGNORECASE)

        # 4b) Quarter & Year ranges (current-date basis)
        qy_terms = [
            (r"\bthis\s+quarter\b", "quarter", 0),
            (r"\blast\s+quarter\b", "quarter", -1),
            (r"\bnext\s+quarter\b", "quarter", +1),
            (r"\bthis\s+year\b", "year", 0),
            (r"\blast\s+year\b", "year", -1),
            (r"\bnext\s+year\b", "year", +1),
            (r"(?:本季|本季度)", "quarter", 0),
            (r"(?:上季|上季度)", "quarter", -1),
            (r"(?:下季|下季度)", "quarter", +1),
            (r"(?:今年)", "year", 0),
            (r"(?:去年)", "year", -1),
            (r"(?:明年)", "year", +1),
        ]
        for pat, unit, shift in qy_terms:
            if re.search(pat, text, flags=re.IGNORECASE):
                if unit == "quarter":
                    base_m = _shift_months(self.current_date, shift * 3)
                    start, end = _quarter_bounds(base_m)
                else:
                    y = self.current_date.year + shift
                    start, end = date(y, 1, 1), date(y, 12, 31)
                text = re.sub(pat, f"between {_fmt(start)} and {_fmt(end)}", text, flags=re.IGNORECASE)

        # 5) EN: last/past/recent/next N days|weeks|months|quarters|years
        pattern_en = re.compile(
            r"\b(?P<dir>last|past|recent|next|upcoming)\s+"
            r"(?P<n>\d{1,3})\s*"
            r"(?P<u>day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
            flags=re.IGNORECASE
        )
        def _repl_en(m: re.Match) -> str:
            direction = m.group("dir").lower()
            n = int(m.group("n"))
            unit = m.group("u").lower()
            base_date = self._determine_base_date_for_range(text)  # now defaults to CURRENT
            start, end = self._range_from_units(n, unit, direction, base_date)
            return f"between {_fmt(start)} and {_fmt(end)}"
        text = pattern_en.sub(_repl_en, text)

        # 6) ZH: 過去/最近/近/未來 N 天/週/月/季/年  (support Chinese numerals)
        pattern_zh = re.compile(
            r"(?:(?P<past>過去|过去|最近|近)|(?P<next>未來|未来|接下來|接下来))\s*"
            r"(?P<n>[一二兩两三四五六七八九十\d]+)\s*"
            r"(?P<u>天|日|週|周|月|季|年)",
            flags=re.IGNORECASE
        )
        def _repl_zh(m: re.Match) -> str:
            direction = "next" if m.group("next") else "past"
            n_raw = m.group("n")
            unit_zh = m.group("u")
            n = _parse_zh_int(n_raw) or 1
            if unit_zh in ("天", "日"):
                unit = "days"
            elif unit_zh in ("週", "周"):
                unit = "weeks"
            elif unit_zh == "月":
                unit = "months"
            elif unit_zh == "季":
                unit = "quarters"
            else:
                unit = "years"
            base_date = self._determine_base_date_for_range(text)  # now defaults to CURRENT
            start, end = self._range_from_units(n, unit, direction, base_date)
            return f"between {_fmt(start)} and {_fmt(end)}"
        text = pattern_zh.sub(_repl_zh, text)

        if text != original:
            logger.info("Date rewrite completed:\n  before: %s\n  after:  %s", original, text)

        return text

    # ------------ SQL rewriting ------------
    def rewrite_sql_dates(self, sql: str) -> str:
        """
        Replace SQL date/time functions with **current date**.
        (Previously this sometimes used data_anchor implicitly.)
        """
        if not sql:
            return sql

        result = sql
        target_date = self.get_current_date()  # ALWAYS current date now

        replacements = [
            (r"CAST\s*\(\s*GETDATE\(\)\s*AS\s*date\s*\)", f"'{target_date}'"),
            (r"\bGETDATE\(\)", f"'{target_date} 00:00:00'"),
            (r"\bSYSDATETIME\(\)", f"'{target_date} 00:00:00'"),
            (r"\bCURRENT_TIMESTAMP\b", f"'{target_date} 00:00:00'"),
            (r"FORMAT\s*\(\s*GETDATE\(\)\s*,\s*'yyyy-MM-dd'\s*\)", f"'{target_date}'"),
            (r"CONVERT\s*\(\s*date\s*,\s*GETDATE\(\)\s*\)", f"'{target_date}'"),
        ]
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        if result != sql:
            logger.debug("SQL date rewrite (current-only): %s -> %s", sql[:120], result[:120])

        return result

    # ------------ query analysis ------------
    def extract_date_range_from_query(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract explicit date ranges from query text."""
        pat_iso = r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"
        pat_zh = r"(\d{4})年(\d{1,2})月(\d{1,2})日?"
        found: List[str] = []

        for y, m, d in re.findall(pat_iso, query or ""):
            try:
                dt = date(int(y), int(m), int(d))
                found.append(_fmt(dt))
            except ValueError:
                continue

        for y, m, d in re.findall(pat_zh, query or ""):
            try:
                dt = date(int(y), int(m), int(d))
                found.append(_fmt(dt))
            except ValueError:
                continue

        if len(found) >= 2:
            return min(found), max(found)
        elif len(found) == 1:
            return found[0], found[0]
        else:
            return None, None

    def is_relative_date_query(self, query: str) -> bool:
        """Check if query contains relative date expressions."""
        patterns = [
            r"\b(?:today|yesterday|tomorrow|now|current|recent|latest)\b",
            r"\b(?:this|last|next)\s+(?:day|week|month|quarter|year)\b",
            r"\b(?:past|last|recent|next|upcoming)\s+\d+\s*(?:day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
            r"(?:今天|今日|昨天|昨日|明天|翌日|現在|现在|目前|最近|最新)",
            r"(?:本週|這週|这周|本周|上週|上周|下週|下周)",
            r"(?:本月|這個月|这个月|上月|上個月|上个月|下月|下個月|下个月)",
            r"(?:本季|上季|下季|本季度|上季度|下季度|今年|去年|明年)",
            r"(?:過去|过去|最近|近|未來|未来|接下來|接下来)\s*[一二兩两三四五六七八九十\d]+\s*(?:天|日|週|周|月|季|年)",
        ]
        return any(re.search(p, query or "", flags=re.IGNORECASE) for p in patterns)

    def is_future_query(self, query: str) -> bool:
        """Check if query is asking about future dates/events."""
        future_indicators = [
            r"\b(?:future|upcoming|scheduled|planned|will be|going to|next)\b",
            r"\b(?:tomorrow|next week|next month|next quarter|next year)\b",
            r"(?:未來|未来|即將|即将|計劃|计划|預定|预定|排定|下週|下周|下月|下個月|下个月|下季|下季度|明年|明天)",
        ]
        return any(re.search(p, query or "", flags=re.IGNORECASE) for p in future_indicators)
