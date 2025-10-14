# app/services/helpers/data_utils.py
from __future__ import annotations

from decimal import Decimal
from datetime import date, datetime, time, timedelta
from typing import Any, List, Tuple, Optional, Dict, Iterable, Set
import uuid as _uuid
import re
import os
import json

# ======================================================================
# JSON / primitive utilities
# ======================================================================

def jsonable_value(v: Any):
    """Convert various Python types to JSON-serializable values."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, Decimal):
        try:
            return int(v) if v == v.to_integral_value() else float(v)
        except Exception:
            return float(v)
    if isinstance(v, (datetime, date, time)):
        return v.isoformat()
    if isinstance(v, timedelta):
        return v.total_seconds()
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, _uuid.UUID):
        return str(v)
    return str(v)


def _decode_json_field(val: Any):
    """
    Accept JSON string or list; always returns a list (or []).
    Safe for SQL FOR JSON PATH payloads.
    """
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


# ======================================================================
# SQL helpers
# ======================================================================

def normalize_sql_columns(sql: str) -> str:
    """
    Fix a few common column-name variants the model may invent.
    Conservative to avoid breaking valid identifiers.
    """
    if not sql:
        return sql
    fixes = [
        (r"\bLEAVETYPE\b", "ATTENDANCETYPE"),
        (r"\bATTENDANCE[_\s]?TYPE\b", "ATTENDANCETYPE"),
        (r"\bEMPID\b", "EMPLOYEEID"),
        (r"\bTRUE[_\s]?NAME\b", "TRUENAME"),
        (r"\bBUSINESSUINTID\b", "BUSINESSUNITID"),  # frequent typo
        (r"\bEFFINIENTDATE\b", "EFFECTIVEDATE"),    # frequent typo
        (r"\bLEAVE[_\s]?REASON\b", "LEAVEREASON"),
    ]
    out = sql
    for pat, repl in fixes:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def get_today_sql_date() -> str:
    """
    Get SQL date expression for 'today', allowing override for demos.
    This should mirror the anchor date used in vector_search_service date rules.
    """
    override = os.getenv("NLP_TODAY_OVERRIDE")
    return f"'{override}'" if override else "CAST(GETDATE() AS date)"


# ======================================================================
# Date/time parsing (EN + zh-TW/CN)
# ======================================================================

# -- Week start policy (ISO Monday by default) ----------------------------------
_WEEK_START_ISO = int(os.getenv("WEEK_START_ISO", "1"))  # 1=Monday .. 7=Sunday


def _week_bounds(anchor: date, offset_weeks: int = 0) -> Tuple[date, date]:
    """Return start/end of ISO week containing anchor + offset, inclusive."""
    # Python's isoweekday: Mon=1..Sun=7
    anchor2 = anchor + timedelta(weeks=offset_weeks)
    delta = (anchor2.isoweekday() - _WEEK_START_ISO) % 7
    start = anchor2 - timedelta(days=delta)
    end = start + timedelta(days=6)
    return start, end


def _month_bounds(anchor: date, offset_months: int = 0) -> Tuple[date, date]:
    """Return first/last day of month offset relative to anchor."""
    y = anchor.year + ((anchor.month - 1 + offset_months) // 12)
    m = (anchor.month - 1 + offset_months) % 12 + 1
    start = date(y, m, 1)
    # next month first day minus one
    if m == 12:
        end = date(y + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(y, m + 1, 1) - timedelta(days=1)
    return start, end


def anchor_today() -> date:
    """
    Anchor date used to interpret relative or year-less ranges.
    Controlled by NLP_TODAY_OVERRIDE (YYYY-MM-DD).
    """
    override = os.getenv("NLP_TODAY_OVERRIDE")
    if override:
        try:
            return datetime.strptime(override, "%Y-%m-%d").date()
        except Exception:
            pass
    return date.today()


# -- Small zh numerals ----------------------------------------------------------
_ZH_DIGIT = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

def _parse_zh_int(token: str) -> Optional[int]:
    """Parse small Chinese numerals up to 99 (handles '十', '二十', '二十三', and Arabic digits)."""
    token = (token or "").strip()
    if not token:
        return None
    m = re.search(r"\d{1,3}", token)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    if token == "十":
        return 10
    if "十" in token:
        left, _, right = token.partition("十")
        tens = 1 if left == "" else _ZH_DIGIT.get(left)
        if tens is None:
            return None
        ones = 0 if right == "" else _ZH_DIGIT.get(right)
        if ones is None:
            return None
        return tens * 10 + ones
    return _ZH_DIGIT.get(token)


# -- Public days parser (kept) --------------------------------------------------
def parse_days_from_text(
    text: str,
    default_days: int = 14,
    min_days: int = 1,
    max_days: int = 90
) -> int:
    """
    Extract an interval (in days) from natural language text (EN + zh).
    Returns a bounded day count.
    """
    s = (text or "").lower()

    # English
    m = re.search(r"\b(\d{1,3})\s*(day|days|d|week|weeks|w|month|months|mo)\b", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith(("day", "d")):
            days = n
        elif unit.startswith(("week", "w")):
            days = n * 7
        else:
            days = n * 30
        return max(min_days, min(max_days, days))

    # Chinese (TW/CN)
    m2 = re.search(r"(過去|过去|最近|近|未來|未来|下)\s*([一二兩两三四五六七八九十\d]+)\s*(個|个)?\s*(天|日|週|周|月)", text or "")
    if m2:
        n = _parse_zh_int(m2.group(2)) or default_days
        unit_zh = m2.group(4)
        if unit_zh in ("天", "日"):
            days = n
        elif unit_zh in ("週", "周"):
            days = n * 7
        else:
            days = n * 30
        return max(min_days, min(max_days, days))

    m3 = re.search(r"([一二兩两三四五六七八九十\d]+)\s*(個|个)?\s*(天|日|週|周|月)", text or "")
    if m3:
        n = _parse_zh_int(m3.group(1)) or default_days
        unit_zh = m3.group(3)
        if unit_zh in ("天", "日"):
            days = n
        elif unit_zh in ("週", "周"):
            days = n * 7
        else:
            days = n * 30
        return max(min_days, min(max_days, days))

    return default_days


# -- New: date token parsers ----------------------------------------------------
_MONTH_NAME_EN = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def _mk_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except Exception:
        return None


def _infer_year_for_md(m: int, d: int, anchor_year: int) -> Optional[date]:
    """Build a date using (m,d) and anchor_year."""
    return _mk_date(anchor_year, m, d)


def _parse_md(token: str, anchor_year: int) -> Optional[date]:
    """Parse '9/22', '09-22', '9月22日' into a date using anchor_year."""
    t = (token or "").strip()
    t = t.replace("年", "-").replace("月", "-").replace("日", "").replace(".", "-").replace("/", "-")
    m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", t)
    if m:
        mm, dd = int(m.group(1)), int(m.group(2))
        return _infer_year_for_md(mm, dd, anchor_year)
    # e.g., '9-22-2025' handled elsewhere; here we only accept MD
    return None


def _parse_ymd(token: str) -> Optional[date]:
    """Parse YYYY-MM-DD or YYYY/MM/DD."""
    t = (token or "").strip().replace("/", "-").replace(".", "-")
    m = re.match(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$", t)
    if not m:
        return None
    y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return _mk_date(y, mm, dd)


def _parse_en_month_day(token: str, anchor_year: int) -> Optional[date]:
    """Parse 'Sep 22' or 'September 22nd' to date using anchor_year."""
    t = (token or "").lower()
    t = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", t)
    m = re.match(r"^\s*([a-z]+)\s+(\d{1,2})\s*$", t)
    if not m:
        return None
    name, dd = m.group(1), int(m.group(2))
    mm = _MONTH_NAME_EN.get(name)
    if not mm:
        return None
    return _infer_year_for_md(mm, dd, anchor_year)


def _clean_range_seps(text: str) -> str:
    """Normalize the common separators for ranges (~, -, to, 到, 至, —)."""
    s = (text or "")
    s = s.replace("～", "-").replace("–", "-").replace("—", "-").replace("~", "-")
    s = s.replace("至", "-").replace("到", "-").replace("—", "-")
    s = s.replace("－", "-")
    s = re.sub(r"\s+to\s+", "-", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*-\s*", "-", s)
    return s


_RELATIVE_PATTERNS = [
    # English
    (re.compile(r"\btoday\b", re.I), lambda a: (a, a)),
    (re.compile(r"\byesterday\b", re.I), lambda a: (a - timedelta(days=1), a - timedelta(days=1))),
    (re.compile(r"\btomorrow\b", re.I), lambda a: (a + timedelta(days=1), a + timedelta(days=1))),
    (re.compile(r"\bthis\s+week\b", re.I), lambda a: _week_bounds(a, 0)),
    (re.compile(r"\blast\s+week\b", re.I), lambda a: _week_bounds(a, -1)),
    (re.compile(r"\bnext\s+week\b", re.I), lambda a: _week_bounds(a, +1)),
    (re.compile(r"\bthis\s+month\b", re.I), lambda a: _month_bounds(a, 0)),
    (re.compile(r"\blast\s+month\b", re.I), lambda a: _month_bounds(a, -1)),
    (re.compile(r"\bnext\s+month\b", re.I), lambda a: _month_bounds(a, +1)),
    # Chinese
    (re.compile(r"(今日|今天)"), lambda a: (a, a)),
    (re.compile(r"(昨日|昨天)"), lambda a: (a - timedelta(days=1), a - timedelta(days=1))),
    (re.compile(r"(明日|明天)"), lambda a: (a + timedelta(days=1), a + timedelta(days=1))),
    (re.compile(r"(本週|這週|这周|本周)"), lambda a: _week_bounds(a, 0)),
    (re.compile(r"(上週|上周)"), lambda a: _week_bounds(a, -1)),
    (re.compile(r"(下週|下周)"), lambda a: _week_bounds(a, +1)),
    (re.compile(r"(本月|這個月|这个月)"), lambda a: _month_bounds(a, 0)),
    (re.compile(r"(上月|上個月|上个月)"), lambda a: _month_bounds(a, -1)),
    (re.compile(r"(下月|下個月|下个月)"), lambda a: _month_bounds(a, +1)),
]


def parse_date_range_from_text(
    text: str,
    anchor: Optional[date] = None
) -> Optional[Dict[str, Any]]:
    """
    Parse a date range from free text (EN + zh).
    Supports:
      - Relative: today/this week/last month, 今天/本週/上月, etc.
      - Explicit range: '9/22-9/26', '09-22 ~ 09-26' (assumes current year)
      - YMD: '2025-09-22 - 2025-09-26'
      - EN month names: 'Sep 22 - Sep 26'
      - zh: '9月22日-9月26日'
    Returns dict with: {'start': date, 'end': date, 'kind': 'relative|explicit', 'year_inferred': bool}
    """
    if not text:
        return None
    anchor = anchor or anchor_today()
    s = _clean_range_seps(text)

    # 1) Relative phrases
    for pat, fn in _RELATIVE_PATTERNS:
        if pat.search(s):
            start, end = fn(anchor)
            return {"start": start, "end": end, "kind": "relative", "year_inferred": False}

    # 2) Explicit ranges (two tokens)
    #    Try YMD first
    m = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})-(\d{4}[-/]\d{1,2}[-/]\d{1,2})", s)
    if m:
        d1, d2 = _parse_ymd(m.group(1)), _parse_ymd(m.group(2))
        if d1 and d2:
            if d1 > d2:
                d1, d2 = d2, d1
            return {"start": d1, "end": d2, "kind": "explicit", "year_inferred": False}

    # 3) Month/Day with inferred year
    #    e.g., '9/22-9/26', '9月22日-9月26日', 'Sep 22-Sep 26'
    #    Split around '-'
    if "-" in s:
        left, right = s.split("-", 1)
        left = left.strip()
        right = right.strip()

        # YMD + MD (mix) — try parse both ways
        d1 = _parse_ymd(left) or _parse_md(left, anchor.year) or _parse_en_month_day(left, anchor.year)
        d2 = _parse_ymd(right) or _parse_md(right, anchor.year) or _parse_en_month_day(right, anchor.year)
        if d1 and d2:
            if d1 > d2:
                d1, d2 = d2, d1
            # If either side came from MD, then year was inferred
            year_inferred = not (left.strip().startswith(str(d1.year)) and right.strip().startswith(str(d2.year)))
            return {"start": d1, "end": d2, "kind": "explicit", "year_inferred": year_inferred}

    # 4) Single YMD token (point date)
    m = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", s)
    if m:
        d = _parse_ymd(m.group(1))
        if d:
            return {"start": d, "end": d, "kind": "explicit", "year_inferred": False}

    # 5) Single EN month/day e.g., 'Sep 22'
    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2})(st|nd|rd|th)?\b", s)
    if m:
        d = _parse_en_month_day(m.group(0), anchor.year)
        if d:
            return {"start": d, "end": d, "kind": "explicit", "year_inferred": True}

    # 6) Single MD like '9/22' or '9月22日'
    m = re.search(r"(\d{1,2})\s*(/|\.|-|月)\s*(\d{1,2})\s*(日)?", s)
    if m:
        mm, dd = int(m.group(1)), int(m.group(3))
        d = _infer_year_for_md(mm, dd, anchor.year)
        if d:
            return {"start": d, "end": d, "kind": "explicit", "year_inferred": True}

    return None


# -- Time-of-day parsing --------------------------------------------------------
def parse_time_of_day(text: str) -> Optional[time]:
    """
    Parse time-of-day like '13:30', '1:30 pm', '下午1點半', '9', '0930'.
    Returns a time object or None if not understood.
    """
    if not text:
        return None
    s = (text or "").strip().lower()
    s = s.replace("：", ":").replace("點", ":").replace("時", ":")
    s = s.replace("半", ":30").replace("am", " am").replace("pm", " pm")
    s = re.sub(r"\s+", " ", s)

    # HH:MM with optional am/pm
    m = re.match(r"^\s*(\d{1,2})(?::(\d{1,2}))?\s*(am|pm)?\s*$", s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or "0")
        ap = m.group(3)
        if ap == "pm" and hh < 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return time(hour=hh, minute=mm)

    # Chinese 下午 / 上午 hints
    if "下午" in s or "晚上" in s or "pm" in s:
        m2 = re.search(r"(\d{1,2})(?::(\d{1,2}))?", s)
        if m2:
            hh = int(m2.group(1))
            mm = int(m2.group(2) or "0")
            if hh < 12:
                hh += 12
            return time(hour=hh, minute=mm)
    if "上午" in s or "早上" in s or "am" in s:
        m3 = re.search(r"(\d{1,2})(?::(\d{1,2}))?", s)
        if m3:
            hh = int(m3.group(1))
            mm = int(m3.group(2) or "0")
            if hh == 12:
                hh = 0
            return time(hour=hh, minute=mm)

    # 4-digit HHMM like '0930'
    m4 = re.match(r"^\s*(\d{3,4})\s*$", s)
    if m4:
        token = m4.group(1)
        if len(token) == 3:  # e.g., 930
            hh, mm = int(token[0]), int(token[1:])
        else:
            hh, mm = int(token[:2]), int(token[2:])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return time(hour=hh, minute=mm)

    return None


# -- SQL predicate builders -----------------------------------------------------
def sql_literal_date(d: date) -> str:
    return f"'{d.isoformat()}'"


def build_sql_between_date(col_expr: str, start: date, end: date, cast_to_date: bool = True) -> str:
    """
    Build a BETWEEN predicate for point-in-time date columns (e.g., WORKDATE).
    Casts to DATE by default for robust comparisons.
    """
    c = f"CAST({col_expr} AS date)" if cast_to_date else col_expr
    return f"{c} BETWEEN {sql_literal_date(start)} AND {sql_literal_date(end)}"


def build_sql_overlap_by_start_end(
    start_col: str,
    end_col: str,
    start: date,
    end: date,
    cast_to_date: bool = True
) -> str:
    """
    Build an overlap predicate: [start_col, end_col] overlaps with [start, end].
      start <= ENDDATE AND end >= STARTDATE
    Casts columns to DATE by default.
    """
    sc = f"CAST({start_col} AS date)" if cast_to_date else start_col
    ec = f"CAST({end_col} AS date)" if cast_to_date else end_col
    return f"{sql_literal_date(start)} <= {ec} AND {sql_literal_date(end)} >= {sc}"


# ======================================================================
# Misc numeric/time utilities
# ======================================================================

def minutes_to_hours_heuristic(vals: List[float]) -> Tuple[float, bool]:
    """
    Heuristic: if typical values are >=60 and mostly multiples of 30,
    treat as minutes; return (total_hours, converted?).
    """
    if not vals:
        return 0.0, False
    sample = vals[:1000]
    ge60 = sum(1 for v in sample if v is not None and float(v) >= 60)
    mult30ish = sum(1 for v in sample if v is not None and abs(float(v) % 30) < 1e-6)
    is_minutes = (ge60 >= 0.6 * len(sample)) and (mult30ish >= 0.5 * len(sample))
    total = sum(float(v or 0) for v in vals)
    return (total / 60.0 if is_minutes else total), is_minutes


# ======================================================================
# Column helpers (robust to tuple/MultiIndex)
# ======================================================================

def normalize_column_labels(columns):
    """
    Ensure we have a list[str] for column labels.
    Accepts strings, tuples/lists like (name, type), or other objects.
    """
    out = []
    for c in (columns or []):
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, (list, tuple)) and len(c) > 0:
            out.append(str(c[0]))
        else:
            out.append(str(c))
    return out


def _normalize_columns(columns: Iterable[Any]) -> List[str]:
    """
    Convert arbitrary column labels (str, tuple, list, etc.) into strings.
    Tuples/lists are joined with '.'.
    """
    out: List[str] = []
    if not columns:
        return out
    for c in columns:
        if isinstance(c, (tuple, list)):
            parts = [str(p) for p in c if p is not None and str(p) != ""]
            out.append(".".join(parts) if parts else str(c))
        else:
            out.append("" if c is None else str(c))
    return out


def _canon(s: str) -> str:
    """Case/underscore/space-insensitive canonical form."""
    return s.lower().replace("_", "").replace(" ", "")


def find_column_index(columns: Iterable[Any], *candidates: str) -> Optional[int]:
    """
    Robustly find the index of the first candidate column name.

    Handles:
      - non-string labels (tuple/list) by normalizing to 'a.b.c'
      - case differences
      - underscores / whitespace differences
    """
    norm_cols = _normalize_columns(columns)
    if not norm_cols:
        return None

    exact = {c.lower(): i for i, c in enumerate(norm_cols)}
    for name in candidates:
        if not name:
            continue
        i = exact.get(str(name).lower())
        if i is not None:
            return i

    canon_map = {_canon(c): i for i, c in enumerate(norm_cols)}
    for name in candidates:
        if not name:
            continue
        i = canon_map.get(_canon(str(name)))
        if i is not None:
            return i
    return None


def format_sample_data(
    rows: List[Tuple], columns: Iterable[Any], max_rows: int = 20, max_chars: int = 1800
) -> str:
    """Format a readable sample of data for LLM context."""
    if not rows or not columns:
        return "No data sample."

    cols_norm = _normalize_columns(columns)

    preferred = [
        "TRUENAME", "Name", "EMPLOYEEID", "PERSONID",
        "ATTENDANCETYPE", "LEAVETYPE", "HOURS",
        "STARTDATE", "ENDDATE", "WORKDATE",
    ]
    keep_idx: List[int] = []
    seen = set()

    for p in preferred:
        i = find_column_index(cols_norm, p)
        if i is not None and i not in seen:
            keep_idx.append(i)
            seen.add(i)

    if not keep_idx:
        keep_idx = list(range(min(5, len(cols_norm))))

    hdr = [cols_norm[i] for i in keep_idx]
    lines = [" | ".join(hdr)]

    char_count = sum(len(x) for x in lines)
    for r in rows[:max_rows]:
        vals = []
        for i in keep_idx:
            try:
                v = r[i] if i < len(r) else None
                s = "" if v is None else str(v)
                vals.append(s)
            except Exception:
                vals.append("")
        line = " | ".join(vals)
        char_count += len(line)
        lines.append(line)
        if char_count > max_chars:
            lines.append("... (truncated)")
            break

    return "\n".join(lines)


# ======================================================================
# Dashboard enricher data (kept; person resolution)
# ======================================================================

from app.services.person_resolver import PersonResolver  # local import to avoid cycles at module load time

def _collect_person_ids(*arrays: List[Dict[str, Any]]) -> List[str]:
    """Collect unique PERSONID values from already-decoded arrays."""
    ids: Set[str] = set()
    for arr in arrays:
        for r in arr or []:
            pid = r.get("person_id") or r.get("PERSONID")
            if pid:
                s = str(pid).strip()
                if s:
                    ids.add(s)
    return list(ids)


def _patch_rows(rows: List[Dict[str, Any]], resolved_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fill person_name/employee_id/email/cardnum from resolver map; keep existing when present."""
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        d = dict(r)
        pid = str(d.get("person_id") or d.get("PERSONID") or "").strip()
        info = resolved_map.get(pid, {}) if pid else {}
        d["person_id"] = pid or None
        d["person_name"] = info.get("name") or d.get("TRUENAME") or d.get("ENGNAME") or (pid or None)
        d.setdefault("employee_id", info.get("employee_id"))
        d.setdefault("email", info.get("email"))
        d["cardnum"] = d.get("cardnum") or info.get("cardnum")
        out.append(d)
    return out


def enrich_leave_metrics_payload(metrics: Dict[str, Any], resolver: PersonResolver) -> Dict[str, Any]:
    """
    Expect metrics like:
      {
        "on_leave_details": "[...JSON array...]",
        "upcoming_leave":   "[...JSON array...]",
        ...
      }
    Decodes arrays, resolves people, and patches person fields.
    """
    details = _decode_json_field(metrics.get("on_leave_details"))
    upcoming = _decode_json_field(metrics.get("upcoming_leave"))

    pid_list = _collect_person_ids(details, upcoming)
    resolved = resolver.resolve_many(pid_list)

    metrics["on_leave_details"] = _patch_rows(details, resolved)
    metrics["upcoming_leave"]   = _patch_rows(upcoming, resolved)
    return metrics


def enrich_leave_trend_payload(trend_root: Dict[str, Any], resolver: PersonResolver) -> Dict[str, Any]:
    """
    Expect shape:
      {
        "success": 1,
        "trend": "[{'date':'2025-01-01','people_on_leave':'[...]'}, ...]"  # or already parsed
      }
    Resolves and expands names inside each day's 'people_on_leave'.
    """
    trend_list = _decode_json_field(trend_root.get("trend"))
    perday_arrays = [_decode_json_field(day.get("people_on_leave")) for day in trend_list]
    all_ids = _collect_person_ids(*perday_arrays)
    resolved = resolver.resolve_many(all_ids)

    for day in trend_list:
        ppl = _decode_json_field(day.get("people_on_leave"))
        day["people_on_leave"] = _patch_rows(ppl, resolved)

    trend_root["trend"] = trend_list
    return trend_root


# ======================================================================
# Legacy back-compat helpers (kept, tidied)
# ======================================================================

def _collect_ids_from_rows(*rows_lists: Iterable[Optional[Dict[str, Any]]]) -> tuple[List[str], List[str]]:
    """
    Legacy helper: collect BOTH person IDs and employee IDs from mixed rows.

    Recognized keys:
      - person ids: person_id, PERSONID, pid
      - employee ids: employee_id, EMPLOYEEID, empno
    """
    pid_keys = {"person_id", "PERSONID", "pid"}
    eid_keys = {"employee_id", "EMPLOYEEID", "empno"}

    pids: set[str] = set()
    eids: set[str] = set()

    for rows in rows_lists:
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            # PERSON IDs
            for k in pid_keys:
                v = r.get(k)
                if v:
                    s = str(v).strip()
                    if s:
                        pids.add(s)
                        break
            # EMPLOYEE IDs
            for k in eid_keys:
                v = r.get(k)
                if v:
                    s = str(v).strip()
                    if s:
                        eids.add(s)
                        break

    return list(pids), list(eids)


def _apply_resolved(rows: List[Dict[str, Any]], resolved: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Legacy helper: attach 'display_name' to each row using whichever ID key is present.
    Lookup priority: person_id -> PERSONID -> employee_id -> EMPLOYEEID -> empno
    """
    if not rows:
        return []

    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            out.append(r)
            continue
        d = dict(r)
        lk = None
        for key in ("person_id", "PERSONID", "employee_id", "EMPLOYEEID", "empno"):
            if d.get(key):
                lk = str(d[key]).strip()
                if lk:
                    break

        name = None
        if lk and lk in resolved:
            name = resolved[lk].get("name")

        d["display_name"] = name or d.get("TRUENAME") or d.get("ENGNAME") or lk
        out.append(d)

    return out
