# backend/app/services/helpers/data_utils.py
from __future__ import annotations

from decimal import Decimal
from datetime import date, datetime, time, timedelta
import uuid as _uuid
from typing import Any, List, Tuple, Optional, Dict
import re
import os


def jsonable_value(v):
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


def normalize_sql_columns(sql: str) -> str:
    """Fix common column synonyms the model might invent."""
    if not sql:
        return sql
    # whole-word, case-insensitive replace: LEAVETYPE -> ATTENDANCETYPE
    return re.sub(r"\bLEAVETYPE\b", "ATTENDANCETYPE", sql, flags=re.IGNORECASE)


def parse_days_from_text(text: str, default_days: int = 14, min_days: int = 1, max_days: int = 90) -> int:
    """Extract number of days from natural language text."""
    m = re.search(r"\b(\d{1,3})\s*(day|days|d)\b", text.lower())
    if not m:
        return default_days
    try:
        n = int(m.group(1))
        return max(min_days, min(max_days, n))
    except Exception:
        return default_days


def get_today_sql_date() -> str:
    """Get SQL date string for 'today', allowing override for demos."""
    override = os.getenv("NLP_TODAY_OVERRIDE")
    if override:
        return f"'{override}'"
    return "CAST(GETDATE() AS date)"


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


def format_sample_data(rows: List[Tuple], columns: List[str], max_rows: int = 20, max_chars: int = 1800) -> str:
    """Format a readable sample of data for LLM context."""
    if not rows or not columns:
        return "No data sample."
    
    preferred = ["TRUENAME", "Name", "EMPLOYEEID", "PERSONID", "ATTENDANCETYPE", 
                "LEAVETYPE", "HOURS", "STARTDATE", "ENDDATE", "WORKDATE"]
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
    
    for r in rows[:max_rows]:
        vals = []
        for i in keep_idx:
            try:
                v = r[i] if i < len(r) else None
                s = "" if v is None else str(v)
                vals.append(s)
            except Exception:
                vals.append("")
        lines.append(" | ".join(vals))
        if sum(len(x) for x in lines) > max_chars:
            lines.append("... (truncated)")
            break
    
    return "\n".join(lines)
