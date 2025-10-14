# ================================================================================
# backend/app/services/data_processing/data_analyzer.py
from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, date

from ..helpers.data_utils import (
    find_column_index,
    minutes_to_hours_heuristic,
    format_sample_data,  # noqa: F401  (kept for external callers that import it here)
)

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Analyzes query results and computes aggregates (leave-centric).
    Now resolves leave types directly from dbo.ATDATTENDANCECLASS instead of JSON.
    """

    def __init__(self, db_service: Optional[Any] = None):
        """
        :param db_service: Optional SQLServerDatabaseService for live leave-type lookup.
                           If not provided, analyzer falls back to raw values.
        """
        self.db_service = db_service
        # Unified map: keys can be ID (as str), CLASSCODE (upper), or CLASSNAME (as-is) → CLASSNAME (label)
        self._leave_type_map: Dict[str, str] = {}
        self._leave_type_loaded: bool = False

    # ----------------------------
    # Leave-type resolution
    # ----------------------------
    def _ensure_leave_type_map(self) -> None:
        """Lazy-load leave-type map from dbo.ATDATTENDANCECLASS (ID, CLASSCODE, CLASSNAME)."""
        if self._leave_type_loaded:
            return
        self._leave_type_loaded = True  # set early to avoid re-entry loops

        if not self.db_service:
            logger.info("DataAnalyzer: no db_service; using raw leave type values.")
            return

        try:
            sql = (
                "SELECT "
                "  CAST(ID AS NVARCHAR(100)) AS id_norm, "
                "  CAST(CLASSCODE AS NVARCHAR(100)) AS class_code, "
                "  CAST(CLASSNAME AS NVARCHAR(400)) AS class_name "
                "FROM dbo.ATDATTENDANCECLASS"
            )
            rows, cols = self.db_service.run_select(sql, max_rows=10000, query_timeout=8)
            col_map = {c.upper(): i for i, c in enumerate(cols or [])}

            i_id = col_map.get("ID_NORM") or col_map.get("ID")
            i_code = col_map.get("CLASS_CODE") or col_map.get("CLASSCODE")
            i_name = col_map.get("CLASS_NAME") or col_map.get("CLASSNAME")

            m: Dict[str, str] = {}
            for r in rows or []:
                idv = str(r[i_id]).strip() if (i_id is not None and len(r) > i_id and r[i_id] is not None) else ""
                code = str(r[i_code]).strip() if (i_code is not None and len(r) > i_code and r[i_code] is not None) else ""
                name = str(r[i_name]).strip() if (i_name is not None and len(r) > i_name and r[i_name] is not None) else ""

                if not name:
                    continue

                # Map by ID
                if idv:
                    m[idv] = name

                # Map by CLASSCODE (use UPPER for stable lookup)
                if code:
                    m[code.upper()] = name

                # Map by CLASSNAME (identity)
                m[name] = name

            self._leave_type_map = m
            logger.info("DataAnalyzer: loaded leave type map from DB (%d entries).", len(self._leave_type_map))
        except Exception as e:
            logger.warning("DataAnalyzer: failed to load leave types from DB, using raw values. Error: %s", e)

    def _norm_key(self, raw: Any) -> str:
        """Conservative normalization for lookup keys."""
        if raw is None:
            return ""
        s = str(raw).strip()
        return s

    def label_leave_type(self, raw: Any) -> str:
        """
        Return a human-friendly label for ATTENDANCETYPE / LEAVETYPE / CLASSNAME / LEAVEID-like values.
        Resolution order (after normalization):
        1) Exact ID string (e.g., '12') in map → CLASSNAME
        2) Uppercased CLASSCODE in map → CLASSNAME
        3) Exact CLASSNAME in map → CLASSNAME
        4) Fallback: original raw text
        """
        self._ensure_leave_type_map()

        key = self._norm_key(raw)
        if not key:
            return "(unknown)"

        # Try ID
        if key in self._leave_type_map:
            return self._leave_type_map[key]

        # Try CLASSCODE (UPPER)
        key_u = key.upper()
        if key_u in self._leave_type_map:
            return self._leave_type_map[key_u]

        # Try CLASSNAME (identity)
        if key in self._leave_type_map:
            return self._leave_type_map[key]

        return key  # fallback to raw

    # ----------------------------
    # Aggregation entry point
    # ----------------------------
    def compute_aggregates(self, rows: List[Tuple], columns: List[str]) -> Dict[str, Any]:
        """Compute comprehensive statistics from query results."""
        if not rows:
            return {
                "row_count": 0,
                "unique_people": 0,
                "by_leave_type": None,
                "by_leave_type_hours": None,
                "total_hours": 0.0,
                "hours_source": "hours",
                "avg_hours_per_person": None,
                "names_sample": None,
                "date_analysis": None,
                "min_date": None,
                "max_date": None,
            }

        # ---- Column indices (with tolerant aliases) ----
        idx_person = find_column_index(columns, "PERSONID", "person_id")
        idx_empid = find_column_index(columns, "EMPLOYEEID", "employee_id")
        idx_name = find_column_index(columns, "TRUENAME", "Name", "person_name")

        # Leave type can appear under multiple columns depending on the query
        idx_type_name = find_column_index(
            columns,
            "ATTENDANCETYPE", "LEAVETYPE", "LEAVE_TYPE", "LEAVE_TYPE_NAME", "CLASSNAME"
        )
        # Some queries return the ID instead of the name
        idx_type_id = find_column_index(columns, "LEAVEID", "LEAVE_ID", "ATTENDANCECLASSID", "CLASS_ID")

        # Hours may be emitted as HOURS or TIMECLASSHOURS (minutes)
        idx_hours = find_column_index(columns, "HOURS", "TIMECLASSHOURS")

        # Dates
        idx_start = find_column_index(columns, "STARTDATE", "StartDate")
        idx_end = find_column_index(columns, "ENDDATE", "EndDate")
        idx_workdate = find_column_index(columns, "WORKDATE", "WorkDate")

        # ---- Unique people ----
        unique_people = self._count_unique_people(rows, idx_person, idx_empid)

        # ---- Types & hours ----
        type_counts_raw, hours_by_type_raw, hours_vals, hours_col_name = self._analyze_types_and_hours(
            rows, idx_type_name, idx_type_id, idx_hours, columns
        )

        # Decide whether values look like minutes → convert
        total_hours_num, converted = minutes_to_hours_heuristic(hours_vals)
        hours_by_type_hrs = self._convert_hours_by_type(hours_by_type_raw, converted)

        # Apply labels after conversion to collapse synonyms / ids
        labeled_counts = self._apply_labels_to_counts(type_counts_raw)
        labeled_hours = self._apply_labels_to_hours(hours_by_type_hrs)

        # Sample names (purely UX)
        names_sample = self._extract_names_sample(rows, idx_name)

        # ---- Enhanced date analysis ----
        date_analysis = self._analyze_dates_enhanced(rows, idx_workdate, idx_start, idx_end)

        result = {
            "row_count": len(rows),
            "unique_people": unique_people,
            "by_leave_type": labeled_counts or None,
            "by_leave_type_hours": labeled_hours or None,
            "total_hours": round(total_hours_num, 2) if hours_vals else 0.0,
            "hours_source": ("minutes->hours" if converted else "hours") if hours_vals else "n/a",
            "avg_hours_per_person": round((total_hours_num / unique_people), 2) if unique_people else None,
            "names_sample": names_sample,
            "date_analysis": date_analysis,
            # legacy mirrors
            "min_date": date_analysis.get("min_date") if date_analysis else None,
            "max_date": date_analysis.get("max_date") if date_analysis else None,
            # small debug hint for downstream tuning
            "hours_column_used": hours_col_name,
        }
        return result

    # ----------------------------
    # Helpers: people / types / hours
    # ----------------------------
    def _count_unique_people(
        self, rows: List[Tuple], idx_person: Optional[int], idx_empid: Optional[int]
    ) -> int:
        """Count unique people using PERSONID or EMPLOYEEID (prefer PERSONID)."""
        people: set = set()
        if idx_person is not None:
            for r in rows:
                if len(r) > idx_person and r[idx_person] is not None:
                    people.add(str(r[idx_person]).strip())
        elif idx_empid is not None:
            for r in rows:
                if len(r) > idx_empid and r[idx_empid] is not None:
                    people.add(str(r[idx_empid]).strip())
        return len(people)

    def _analyze_types_and_hours(
        self,
        rows: List[Tuple],
        idx_type_name: Optional[int],
        idx_type_id: Optional[int],
        idx_hours: Optional[int],
        columns: List[str],
    ) -> Tuple[Dict[str, int], Dict[str, float], List[float], str]:
        """
        Analyze leave types and hours from rows.
        Returns (type_counts_raw, hours_by_type_raw, hours_vals, hours_col_name_used)
        """
        type_counts_raw: Dict[str, int] = {}
        hours_vals: List[float] = []
        hours_by_type_raw: Dict[str, float] = {}

        # Which column name we used for hours (for debugging)
        hours_col_name = ""
        if idx_hours is not None and 0 <= idx_hours < len(columns):
            hours_col_name = columns[idx_hours]

        def _num(v) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        for r in rows:
            # Determine raw type key (name preferred; fallback to id)
            raw_key = None
            if idx_type_name is not None and len(r) > idx_type_name:
                raw_key = "(unknown)" if r[idx_type_name] is None else str(r[idx_type_name]).strip()
            elif idx_type_id is not None and len(r) > idx_type_id:
                # Keep as string; label resolver will map IDs too
                if r[idx_type_id] is not None:
                    raw_key = str(r[idx_type_id]).strip()

            # Count leave types
            if raw_key is not None:
                type_counts_raw[raw_key] = type_counts_raw.get(raw_key, 0) + 1

            # Collect hours values
            if idx_hours is not None and len(r) > idx_hours:
                hv = _num(r[idx_hours])
                if hv is not None:
                    hours_vals.append(hv)
                    if raw_key is not None:
                        hours_by_type_raw[raw_key] = hours_by_type_raw.get(raw_key, 0.0) + hv

        return type_counts_raw, hours_by_type_raw, hours_vals, hours_col_name

    def _convert_hours_by_type(self, hours_by_type_raw: Dict[str, float], converted: bool) -> Dict[str, float]:
        """Convert hours-by-type if minutes were detected."""
        if not hours_by_type_raw:
            return {}
        factor = (1.0 / 60.0) if converted else 1.0
        return {k: round(v * factor, 2) for k, v in hours_by_type_raw.items()}

    def _apply_labels_to_counts(self, type_counts_raw: Dict[str, int]) -> Dict[str, int]:
        labeled_counts: Dict[str, int] = {}
        for raw_key, cnt in (type_counts_raw or {}).items():
            lbl = self.label_leave_type(raw_key)
            labeled_counts[lbl] = labeled_counts.get(lbl, 0) + cnt
        return labeled_counts

    def _apply_labels_to_hours(self, hours_by_type_hrs: Dict[str, float]) -> Dict[str, float]:
        labeled_hours: Dict[str, float] = {}
        for raw_key, hrs in (hours_by_type_hrs or {}).items():
            lbl = self.label_leave_type(raw_key)
            labeled_hours[lbl] = round(labeled_hours.get(lbl, 0.0) + hrs, 2)
        return labeled_hours

    def _extract_names_sample(self, rows: List[Tuple], idx_name: Optional[int]) -> Optional[List[str]]:
        """Extract a small sample of names from results."""
        if idx_name is None:
            return None
        names_sample: List[str] = []
        for r in rows[:15]:
            if len(r) > idx_name and r[idx_name]:
                names_sample.append(str(r[idx_name]).strip())
        clean_names = [n for n in names_sample if n][:10]
        return clean_names if clean_names else None

    # ----------------------------
    # Date normalization & analysis
    # ----------------------------
    def _normalize_date_value(self, v: Any) -> Optional[str]:
        """Normalize various date formats to YYYY-MM-DD string."""
        if v is None:
            return None

        # datetime/date objects
        try:
            if hasattr(v, "date"):
                try:
                    return v.date().isoformat()
                except Exception:
                    pass
            if isinstance(v, date) and not isinstance(v, datetime):
                return v.isoformat()
        except Exception:
            pass

        # strings
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None

            candidates = [
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d",
                "%Y/%m/%d %H:%M",
                "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%f%z",
            ]
            base = s.rstrip("Zz")
            try:
                if "+" in base and "T" in base:
                    base = base.split("+", 1)[0]
            except Exception:
                pass

            for fmt in candidates:
                try:
                    dt = datetime.strptime(base, fmt)
                    return dt.date().isoformat()
                except Exception:
                    continue

            if len(s) >= 8 and s[:8].isdigit():
                try:
                    dt = datetime.strptime(s[:8], "%Y%m%d")
                    return dt.date().isoformat()
                except Exception:
                    pass

            import re as _re
            m = _re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", s)
            if m:
                try:
                    yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    return date(yyyy, mm, dd).isoformat()
                except Exception:
                    return None

        return None

    def _analyze_dates_enhanced(
        self,
        rows: List[Tuple],
        idx_workdate: Optional[int],
        idx_start: Optional[int],
        idx_end: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Enhanced date analysis with proper normalization and current date context."""
        all_dates: List[str] = []
        workdates: List[str] = []
        start_dates: List[str] = []
        end_dates: List[str] = []

        for r in rows:
            if idx_workdate is not None and len(r) > idx_workdate:
                wd = self._normalize_date_value(r[idx_workdate])
                if wd:
                    all_dates.append(wd)
                    workdates.append(wd)
            if idx_start is not None and len(r) > idx_start:
                sd = self._normalize_date_value(r[idx_start])
                if sd:
                    all_dates.append(sd)
                    start_dates.append(sd)
            if idx_end is not None and len(r) > idx_end:
                ed = self._normalize_date_value(r[idx_end])
                if ed:
                    all_dates.append(ed)
                    end_dates.append(ed)

        if not all_dates:
            return None

        today = date.today().isoformat()
        uniq_sorted = sorted(set(all_dates))
        min_date = uniq_sorted[0]
        max_date = uniq_sorted[-1]

        freq: Dict[str, int] = {}
        for d in all_dates:
            freq[d] = freq.get(d, 0) + 1
        most_common_date = max(freq.items(), key=lambda kv: kv[1])[0] if freq else None

        try:
            range_days = (datetime.fromisoformat(max_date) - datetime.fromisoformat(min_date)).days
        except Exception:
            range_days = 0

        return {
            "min_date": min_date,
            "max_date": max_date,
            "total_date_records": len(all_dates),
            "unique_dates": len(uniq_sorted),
            "most_common_date": most_common_date,
            "current_date": today,
            "is_today_query": (most_common_date == today) if most_common_date else False,
            "workdate_count": len(workdates),
            "start_date_count": len(start_dates),
            "end_date_count": len(end_dates),
            "date_range_days": range_days,
        }
