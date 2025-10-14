# utils/date_fill.py (or inline in your leave service)
from datetime import datetime, timedelta, date

def _to_date(s):
    if not s: return None
    return datetime.fromisoformat(str(s)[:10]).date()

def fill_trend_to_today(trend: list, *, end_date: date, days: int = 7):
    """Pad trend to `end_date` (today) with zeros, keep last `days` days."""
    # Build map
    m = {str(_to_date(t['date'])): (t.get('count') or 0) for t in (trend or [])}
    out = []
    for i in range(days-1, -1, -1):
        d = end_date - timedelta(days=i)
        out.append({"date": d.isoformat(), "count": int(m.get(d.isoformat(), 0))})
    return out

def build_meta(trend_raw: list, data_window: dict):
    today = datetime.utcnow().date()
    max_db = _to_date((data_window or {}).get("max_date"))
    stale_days = max(0, (today - (max_db or today)).days)
    return today, stale_days
