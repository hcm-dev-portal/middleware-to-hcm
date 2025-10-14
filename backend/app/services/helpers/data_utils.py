from __future__ import annotations

from decimal import Decimal
from datetime import date, datetime, time, timedelta
import uuid as _uuid
from typing import Any, List, Tuple, Optional, Dict, Iterable, Set
import re
import os
import json

# ===========================
# JSON / primitive utilities
# ===========================

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
    # Fallback: string
    return str(v)


def _safe_json_loads(val: Any) -> Any:
    """Parse JSON string → python value; tolerate bad inputs by returning []."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return val if isinstance(val, list) else []


def _decode_json_field(val: Any):
    """
    Accepts JSON string or list; always returns a list (or []).
    Safe for SQL `FOR JSON PATH` payloads.
    """
    return _safe_json_loads(val) or []


def _collect_ids_from_rows(*rows_lists: Iterable[Optional[Dict[str, Any]]]) -> tuple[list[str], list[str]]:
    """
    Collect BOTH person IDs and employee IDs from mixed rows.

    Recognized keys:
      - person ids: person_id, PERSONID, pid
      - employee ids: employee_id, EMPLOYEEID, empno
    """
    pid_keys = {"person_id", "PERSONID", "pid"}
    eid_keys = {"employee_id", "EMPLOYEEID", "empno"}

    pids: Set[str] = set()
    eids: Set[str] = set()
    for rows in rows_lists:
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            for k in pid_keys:
                v = r.get(k)
                if v:
                    s = str(v).strip()
                    if s:
                        pids.add(s)
                        break
            for k in eid_keys:
                v = r.get(k)
                if v:
                    s = str(v).strip()
                    if s:
                        eids.add(s)
                        break
    return list(pids), list(eids)


def _apply_resolved(rows: list[dict], resolved: dict) -> list[dict]:
    """
    Attach 'display_name' to each row using whichever ID key is present.
    Lookup priority: person_id -> PERSONID -> employee_id -> EMPLOYEEID -> empno
    """
    if not rows:
        return []
    out: list[dict] = []
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


# ==============
# SQL helpers
# ==============

def normalize_sql_columns(sql: str) -> str:
    """
    Fix a few common column-name variants the model may invent.
    Keep this conservative to avoid breaking valid identifiers.
    """
    if not sql:
        return sql

    # Whole-word, case-insensitive replacements
    fixes = [
        (r"\bLEAVETYPE\b", "ATTENDANCETYPE"),
        (r"\bATTENDANCE_TYPE\b", "ATTENDANCETYPE"),
        (r"\bEMPID\b", "EMPLOYEEID"),
        (r"\bTRUE_NAME\b", "TRUENAME"),
        (r"\bBUSINESSUINTID\b", "BUSINESSUNITID"),   # frequent typo
        (r"\bEFFINIENTDATE\b", "EFFECTIVEDATE"),     # frequent typo
    ]
    out = sql
    for pat, repl in fixes:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)

    return out


def get_today_sql_date() -> str:
    """
    Get SQL date expression for 'today', allowing override for demos.
    Date anchoring will be applied upstream by DateProcessor.rewrite_sql_dates().
    """
    override = os.getenv("NLP_TODAY_OVERRIDE")
    if override:
        return f"'{override}'"
    return "CAST(GETDATE() AS date)"


# ===============================
# Units & date parsing (EN + zh)
# ===============================

_ZH_DIGIT = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

def _parse_zh_int(token: str) -> Optional[int]:
    """Parse small Chinese numerals up to 99 (handles '十', '二十', '二十三', and Arabic digits)."""
    token = token.strip()
    if not token:
        return None
    # Arabic digits inside zh text (e.g., "近7天")
    m = re.search(r"\d{1,3}", token)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    # Pure Chinese numerals up to 99
    if token == "十":
        return 10
    if "十" in token:
        left, _, right = token.partition("十")
        tens = 1 if left == "" else _ZH_DIGIT.get(left, None)
        if tens is None:
            return None
        ones = 0 if right == "" else _ZH_DIGIT.get(right, None)
        if ones is None:
            return None
        return tens * 10 + ones
    # Single digit
    return _ZH_DIGIT.get(token)


def parse_days_from_text(
    text: str,
    default_days: int = 14,
    min_days: int = 1,
    max_days: int = 90
) -> int:
    """
    Extract an interval (in days) from natural language text (EN + zh).
    Supports days/weeks/months and common Chinese numerals.

    Examples:
      EN: "last 7 days", "next 2 weeks", "past 1 month"
      ZH: "近7天", "過去14天", "未來2週", "最近三十天", "下兩週", "近一個月/近一个月"

    Returns a bounded day count.
    """
    s = (text or "").lower()

    # --- English path ---
    m = re.search(r"\b(\d{1,3})\s*(day|days|d|week|weeks|w|month|months|mo)\b", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith(("day", "d")):
            days = n
        elif unit.startswith(("week", "w")):
            days = n * 7
        else:  # month-ish
            days = n * 30  # coarse but practical
        return max(min_days, min(max_days, days))

    # --- Chinese path (tw/cn) ---
    # Patterns like: 過去|过去|最近|近|未來|未来|下  + N(天/日/週/周/月) ; allow optional 個/个 before 月
    m2 = re.search(r"(過去|过去|最近|近|未來|未来|下)\s*([一二兩两三四五六七八九十\d]+)\s*(個|个)?\s*(天|日|週|周|月)", text)
    if m2:
        n_raw = m2.group(2)
        unit_zh = m2.group(4)
        n = _parse_zh_int(n_raw)
        if n is None:
            n = default_days
        if unit_zh in ("天", "日"):
            days = n
        elif unit_zh in ("週", "周"):
            days = n * 7
        else:  # 月
            days = n * 30
        return max(min_days, min(max_days, days))

    # Fallback: look for standalone numeric with a zh unit (allow 個/个 before 月)
    m3 = re.search(r"([一二兩两三四五六七八九十\d]+)\s*(個|个)?\s*(天|日|週|周|月)", text)
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

    # No match → default
    return default_days


def minutes_to_hours_heuristic(vals: List[float]) -> Tuple[float, bool]:
    """
    Heuristic: if typical values are >=60 and mostly multiples of 30,
    treat the unit as minutes and return total_hours.
    Returns (total_hours, converted_from_minutes?)
    """
    if not vals:
        return 0.0, False

    sample = vals[:1000]
    ge60 = sum(1 for v in sample if v is not None and float(v) >= 60)
    mult30ish = sum(1 for v in sample if v is not None and abs(float(v) % 30) < 1e-6)

    is_minutes = (ge60 >= 0.6 * len(sample)) and (mult30ish >= 0.5 * len(sample))
    total = sum(float(v or 0) for v in vals)
    return (total / 60.0 if is_minutes else total), is_minutes


# ==========================
# Column/preview helpers
# ==========================

def find_column_index(columns: List[str], *candidates: str) -> Optional[int]:
    """Case-insensitive column finder."""
    if not columns:
        return None
    lookup = {c.lower(): i for i, c in enumerate(columns)}
    for name in candidates:
        i = lookup.get(name.lower())
        if i is not None:
            return i
    return None


def format_sample_data(
    rows: List[Tuple], columns: List[str], max_rows: int = 20, max_chars: int = 1800
) -> str:
    """
    Format a readable sample of data for LLM context.
    Prefer human columns; remain schema-agnostic.
    """
    if not rows or not columns:
        return "No data sample."

    preferred = [
        "TRUENAME", "Name", "EMPLOYEEID", "PERSONID",
        "ATTENDANCETYPE", "LEAVETYPE", "HOURS",
        "STARTDATE", "ENDDATE", "WORKDATE"
    ]
    keep_idx: List[int] = []
    seen = set()

    for p in preferred:
        i = find_column_index(columns, p)
        if i is not None and i not in seen:
            keep_idx.append(i)
            seen.add(i)

    if not keep_idx:
        keep_idx = list(range(min(5, len(columns))))

    hdr = [columns[i] for i in keep_idx]
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


# ==========================
# Dashboard enricher data
# ==========================

from app.services.person_resolver import PersonResolver

def _collect_person_ids(*arrays: List[Dict[str, Any]]) -> List[str]:
    """Collect unique PERSONID values from already-decoded arrays."""
    ids: Set[str] = set()
    for arr in arrays:
        for r in arr or []:
            pid = r.get("person_id") or r.get("PERSONID")
            if pid:
                ids.add(str(pid).strip())
    return list(ids)


def _patch_rows(rows: List[Dict[str, Any]], resolved_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fill person_name/employee_id/email/cardnum from resolver map; keep existing when present."""
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        d = dict(r)
        pid = str(d.get("person_id") or d.get("PERSONID") or "").strip()
        info = resolved_map.get(pid, {}) if pid else {}
        # canonical keys
        d["person_id"] = pid or None
        d["person_name"] = info.get("name") or d.get("TRUENAME") or d.get("ENGNAME") or pid or None
        if not d.get("employee_id"):
            d["employee_id"] = info.get("employee_id")
        if not d.get("email"):
            d["email"] = info.get("email")
        d["cardnum"] = info.get("cardnum")
        out.append(d)
    return out


def enrich_leave_metrics_payload(metrics: Dict[str, Any], resolver: PersonResolver) -> Dict[str, Any]:
    """
    Expect metrics like:
      {
        "on_leave_details": "[...FOR JSON PATH array...]",
        "upcoming_leave":   "[...FOR JSON PATH array...]",
        ...
      }
    Will decode arrays, resolve people, and patch person fields.
    """
    # Decode JSON arrays produced by SQL
    details = _decode_json_field(metrics.get("on_leave_details"))
    upcoming = _decode_json_field(metrics.get("upcoming_leave"))

    # Collect unique person_ids and resolve
    pid_list = _collect_person_ids(details, upcoming)
    resolved = resolver.resolve_many(pid_list)  # {pid: {person_id,name,employee_id,email,cardnum}}

    # Patch arrays with person_name/cardnum
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
    Will resolve and expand names inside each day's 'people_on_leave'.
    """
    trend_list = _decode_json_field(trend_root.get("trend"))
    # Collect all person_ids across all days
    perday_arrays = [_decode_json_field(day.get("people_on_leave")) for day in trend_list]
    all_ids = _collect_person_ids(*perday_arrays)
    resolved = resolver.resolve_many(all_ids)

    # Patch each day's people_on_leave
    for day in trend_list:
        ppl = _decode_json_field(day.get("people_on_leave"))
        day["people_on_leave"] = _patch_rows(ppl, resolved)

    trend_root["trend"] = trend_list
    return trend_root
