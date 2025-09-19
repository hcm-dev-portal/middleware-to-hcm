# backend/app/services/data_processing/date_processor.py
from __future__ import annotations

import re
import logging
from typing import Optional, Tuple, List
from datetime import datetime, date, timedelta
import calendar

logger = logging.getLogger(__name__)


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


class DateProcessor:
    """
    Smart bilingual date processor that handles both current dates and data-anchored queries.

    Principles:
    - 'today/今天' etc. -> actual current date (operational queries)
    - 'latest/最近(的資料)' -> data anchor (historical/analytical)
    - Ranges like 'last 7 days / 過去7天' choose base = current or data_anchor
      based on operational vs analytical cues.
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

    def _determine_base_date_for_range(self, query_context: str) -> date:
        """
        Decide whether to use current date or data anchor for range calculations.
        """
        # Operational/current context indicators
        operational_indicators = [
            "on leave", "currently", "active", "status", "who is", "working",
            "absent", "present", "available", "正在", "目前", "現在", "状态", "狀態"
        ]
        if any(indicator in query_context.lower() for indicator in operational_indicators):
            return self.current_date

        # Default to data anchor for analytical queries if available
        if self.data_anchor:
            anchor_date = _parse_iso(self.data_anchor)
            if anchor_date:
                return anchor_date

        return self.current_date

    def _range_from_units(self, n: int, unit: str, direction: str, base: date) -> Tuple[date, date]:
        """
        Calculate date ranges for N units (days/weeks/months) relative to base.
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
            "this quarter", "last quarter",
            "this year", "last year", "next year",
            "right now", "currently", "current", "most recent", "latest"
        ]
        tokens.sort(key=len, reverse=True)  # longest first to avoid partial overlaps

        t = text
        for tok in tokens:
            # pad left if CJK/alnum immediately before token
            t = re.sub(rf'(?P<pre>[\u4e00-\u9fffA-Za-z0-9])(?P<tok>{re.escape(tok)})', r'\g<pre> \g<tok>', t, flags=re.IGNORECASE)
            # pad right if CJK/alnum immediately after token
            t = re.sub(rf'(?P<tok>{re.escape(tok)})(?P<post>[\u4e00-\u9fffA-Za-z0-9])', r'\g<tok> \g<post>', t, flags=re.IGNORECASE)

        return t

    # ------------ core normalization ------------
    def rewrite_relative_dates(self, query_text: str) -> str:
        """
        Convert relative date expressions into absolute dates with smart context awareness.

        Strategy:
        - 'today/yesterday/tomorrow/今天/昨天/明天' -> actual current date ± days
        - 'this week/month' & zh equivalents -> explicit ranges
        - 'latest/最近(的資料)' -> data anchor (if available)
        - 'last/past/recent/next N days/weeks/months' & zh equivalents -> explicit ranges
        - Robust to mixed zh/EN with glued tokens (uses padding)
        """
        original = query_text
        text = query_text

        # 0) Make EN temporal tokens space-safe to support mixed zh/EN input
        text = self._pad_english_temporal_tokens(text)

        # 1) Immediate/Current Date Terms (Use Actual Current Date)
        current_day_patterns = [
            # English
            (r"\b(today)\b", 0),
            (r"\b(yesterday)\b", -1),
            (r"\b(tomorrow)\b", +1),
            (r"\b(right now)\b", 0),
            (r"\b(currently)\b", 0),
            # Chinese
            (r"(?:今天|今日)", 0),
            (r"(?:昨天|昨日)", -1),
            (r"(?:明天|翌日)", +1),
            (r"(?:現在|现在|目前)", 0),
        ]
        for pat, delta in current_day_patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                target_date = self.current_date + timedelta(days=delta)
                repl = _fmt(target_date)
                text = re.sub(pat, repl, text, flags=re.IGNORECASE)
                logger.debug("Current date conversion: %s -> %s", pat, repl)

        # 2) Data-Anchored Terms (Use Data Anchor if Available)
        if self.data_anchor:
            anchor = _parse_iso(self.data_anchor)
            if anchor:
                data_relative_patterns = [
                    # English
                    (r"\b(latest)\b", 0),
                    (r"\b(most\s+recent)\b", 0),
                    (r"\b(current\s+data)\b", 0),
                    # Chinese
                    (r"(?:最新|最近的資料|最近的数据)", 0),
                    (r"(?:目前資料|目前数据)", 0),
                ]
                for pat, delta in data_relative_patterns:
                    if re.search(pat, text, flags=re.IGNORECASE):
                        target_date = anchor + timedelta(days=delta)
                        repl = _fmt(target_date)
                        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
                        logger.debug("Data anchor conversion: %s -> %s", pat, repl)

        # 3) Week Ranges (Current Date basis)
        week_terms = [
            (r"\bthis\s+week\b", 0),
            (r"\bcurrent\s+week\b", 0),
            (r"\blast\s+week\b", -1),
            (r"\bnext\s+week\b", +1),
            (r"(?:本週|這週|这周|本周)", 0),
            (r"(?:上週|上周)", -1),
            (r"(?:下週|下周)", +1),
        ]
        for pat, wshift in week_terms:
            if re.search(pat, text, flags=re.IGNORECASE):
                base_date = self.current_date + timedelta(weeks=wshift)
                start, end = self._week_range(base_date)
                repl = f"between {_fmt(start)} and {_fmt(end)}"
                text = re.sub(pat, repl, text, flags=re.IGNORECASE)
                logger.debug("Week range conversion: %s -> %s", pat, repl)

        # 4) Month Ranges (Current Date basis)
        month_terms = [
            (r"\bthis\s+month\b", 0),
            (r"\bcurrent\s+month\b", 0),
            (r"\blast\s+month\b", -1),
            (r"\bnext\s+month\b", +1),
            (r"(?:本月|這個月|这个月)", 0),
            (r"(?:上月|上個月|上个月)", -1),
            (r"(?:下月|下個月|下个月)", +1),
        ]
        for pat, mshift in month_terms:
            if re.search(pat, text, flags=re.IGNORECASE):
                base_date = _shift_months(self.current_date, mshift)
                start, end = _month_bounds(base_date)
                repl = f"between {_fmt(start)} and {_fmt(end)}"
                text = re.sub(pat, repl, text, flags=re.IGNORECASE)
                logger.debug("Month range conversion: %s -> %s", pat, repl)

        # 5) Smart N-day/week/month Ranges (EN)
        pattern_en = re.compile(
            r"\b(?P<dir>last|past|recent|next|upcoming)\s+(?P<n>\d+)\s*(?P<u>day|days|week|weeks|month|months)\b",
            flags=re.IGNORECASE
        )

        def _repl_en(match: re.Match) -> str:
            direction = match.group("dir").lower()
            n = int(match.group("n"))
            unit = match.group("u").lower()
            base_date = self._determine_base_date_for_range(text)
            start, end = self._range_from_units(n, unit, direction, base_date)
            result = f"between {_fmt(start)} and {_fmt(end)}"
            logger.debug("EN range conversion: %s %d %s -> %s (base: %s)",
                         direction, n, unit, result, _fmt(base_date))
            return result

        text = pattern_en.sub(_repl_en, text)

        # 6) Smart N-day/week/month Ranges (ZH)
        pattern_zh = re.compile(
            r"(?:(?P<past>過去|过去|最近|近)|(?P<next>未來|未来|接下來|接下来))\s*(?P<n>\d+)\s*(?P<u>天|日|週|周|月)",
            flags=re.IGNORECASE
        )

        def _repl_zh(match: re.Match) -> str:
            direction = "next" if match.group("next") else "past"
            n = int(match.group("n"))
            unit_zh = match.group("u")
            unit = "days" if unit_zh in ("天", "日") else ("weeks" if unit_zh in ("週", "周") else "months")
            base_date = self._determine_base_date_for_range(text)
            start, end = self._range_from_units(n, unit, direction, base_date)
            result = f"between {_fmt(start)} and {_fmt(end)}"
            logger.debug("ZH range conversion: %s %d %s -> %s (base: %s)",
                         direction, n, unit_zh, result, _fmt(base_date))
            return result

        text = pattern_zh.sub(_repl_zh, text)

        # Final log if changed
        if text != original:
            logger.info("Date rewrite completed:\n  before: %s\n  after:  %s", original, text)

        return text

    # ------------ SQL rewriting ------------
    def rewrite_sql_dates(self, sql: str) -> str:
        """
        Replace SQL date functions with appropriate anchored dates.
        Uses current date for operational queries, data anchor for historical analysis.
        """
        if not sql:
            return sql

        result = sql

        # Heuristic: operational query markers
        is_operational_query = any(pattern in sql.upper() for pattern in [
            "VALIDATED = 1", "CURRENT", "ACTIVE", "STATUS"
        ])

        target_date = self.get_current_date() if is_operational_query else (self.data_anchor or self.get_current_date())

        replacements = [
            (r"CAST\s*\(\s*GETDATE\(\)\s*AS\s*date\s*\)", f"'{target_date}'"),
            (r"\bGETDATE\(\)", f"'{target_date}'"),
            (r"\bSYSDATETIME\(\)", f"'{target_date} 00:00:00'"),
            (r"\bCURRENT_TIMESTAMP\b", f"'{target_date} 00:00:00'"),
        ]

        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        if result != sql:
            logger.debug("SQL date rewrite: %s -> %s", sql[:100], result[:100])

        return result

    # ------------ query analysis ------------
    def extract_date_range_from_query(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract explicit date ranges from query text."""
        # ISO dates (YYYY-MM-DD or YYYY/MM/DD)
        pat_iso = r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"
        # Chinese dates (YYYY年M月D日)
        pat_zh = r"(\d{4})年(\d{1,2})月(\d{1,2})日?"

        found: List[str] = []

        for y, m, d in re.findall(pat_iso, query):
            try:
                dt = date(int(y), int(m), int(d))
                found.append(_fmt(dt))
            except ValueError:
                continue

        for y, m, d in re.findall(pat_zh, query):
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
            # English
            r"\b(?:today|yesterday|tomorrow|now|current|recent|latest)\b",
            r"\b(?:this|last|next)\s+(?:day|week|month|year)\b",
            r"\b(?:past|last|recent|next|upcoming)\s+\d+\s*(?:day|days|week|weeks|month|months)\b",
            # Chinese
            r"(?:今天|今日|昨天|昨日|明天|翌日|現在|现在|目前|最近|最新)",
            r"(?:本週|這週|这周|本周|上週|上周|下週|下周)",
            r"(?:本月|這個月|这个月|上月|上個月|上个月|下月|下個月|下个月)",
            r"(?:過去|过去|最近|近|未來|未来)\s*\d+\s*(?:天|日|週|周|月|年)",
        ]
        return any(re.search(p, query, flags=re.IGNORECASE) for p in patterns)

    def is_future_query(self, query: str) -> bool:
        """Check if query is asking about future dates/events."""
        future_indicators = [
            # English
            r"\b(?:future|upcoming|scheduled|planned|will be|going to|next)\b",
            r"\b(?:tomorrow|next week|next month)\b",
            # Chinese
            r"(?:未來|未来|即將|即将|計劃|计划|預定|预定|排定)",
            r"(?:明天|下週|下周|下月|下個月|下个月)",
        ]
        return any(re.search(p, query, flags=re.IGNORECASE) for p in future_indicators)
