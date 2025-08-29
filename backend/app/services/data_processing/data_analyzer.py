# ================================================================================
# backend/app/services/data_processing/data_analyzer.py
from __future__ import annotations

import json
import os
import logging
from typing import Dict, Any, List, Tuple, Optional

from ..helpers.data_utils import (
    find_column_index, 
    minutes_to_hours_heuristic,
    format_sample_data
)

logger = logging.getLogger(__name__)

LEAVE_TYPE_LABELS_PATH = os.getenv("LEAVE_TYPE_LABELS", "./storage/leave_type_labels.json")


class DataAnalyzer:
    """Analyzes query results and computes aggregates."""
    
    def __init__(self):
        self.leave_type_labels: Dict[str, str] = {}
        self._load_leave_type_labels()
    
    def _load_leave_type_labels(self):
        """Load friendly labels for leave types from JSON file."""
        try:
            if os.path.exists(LEAVE_TYPE_LABELS_PATH):
                with open(LEAVE_TYPE_LABELS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.leave_type_labels = {str(k): str(v) for k, v in data.items()}
                    logger.info("Loaded leave type labels: %d entries", len(self.leave_type_labels))
        except Exception as e:
            logger.warning("Could not load leave type labels: %s", e)
    
    def label_leave_type(self, raw: Any) -> str:
        """Return a human-friendly label for ATTENDANCETYPE/LEAVETYPE."""
        key = None if raw is None else str(raw).strip()
        if not key:
            return "(unknown)"
        return self.leave_type_labels.get(key, key)
    
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
                "names_sample": None,
                "min_date": None,
                "max_date": None,
            }
        
        # Find column indices
        idx_person = find_column_index(columns, "PERSONID")
        idx_empid = find_column_index(columns, "EMPLOYEEID")
        idx_name = find_column_index(columns, "TRUENAME", "Name")
        idx_type = find_column_index(columns, "ATTENDANCETYPE", "LEAVETYPE")
        idx_hours = find_column_index(columns, "HOURS")
        idx_start = find_column_index(columns, "STARTDATE", "StartDate")
        idx_end = find_column_index(columns, "ENDDATE", "EndDate")
        idx_workdate = find_column_index(columns, "WORKDATE", "WorkDate")
        
        # Calculate unique people
        unique_people = self._count_unique_people(rows, idx_person, idx_empid)
        
        # Analyze leave types and hours
        type_counts_raw, hours_by_type_raw, hours_vals = self._analyze_types_and_hours(
            rows, idx_type, idx_hours
        )
        
        # Convert hours if needed
        total_hours_num, converted = minutes_to_hours_heuristic(hours_vals)
        hours_by_type_hrs = self._convert_hours_by_type(hours_by_type_raw, converted)
        
        # Apply labels to types
        labeled_counts = self._apply_labels_to_counts(type_counts_raw)
        labeled_hours = self._apply_labels_to_hours(hours_by_type_hrs)
        
        # Extract names sample
        names_sample = self._extract_names_sample(rows, idx_name)
        
        # Find date range
        min_date, max_date = self._find_date_range(rows, idx_workdate, idx_start, idx_end)
        
        return {
            "row_count": len(rows),
            "unique_people": unique_people,
            "by_leave_type": labeled_counts or None,
            "by_leave_type_hours": labeled_hours or None,
            "total_hours": round(total_hours_num, 2) if hours_vals else 0.0,
            "hours_source": "minutes->hours" if converted else "hours",
            "avg_hours_per_person": round((total_hours_num / unique_people), 2) if unique_people else None,
            "names_sample": names_sample,
            "min_date": min_date,
            "max_date": max_date,
        }
    
    def _count_unique_people(self, rows: List[Tuple], idx_person: Optional[int], idx_empid: Optional[int]) -> int:
        """Count unique people using PERSONID or EMPLOYEEID."""
        people: set = set()
        
        if idx_person is not None:
            for r in rows:
                if len(r) > idx_person and r[idx_person] is not None:
                    people.add(str(r[idx_person]))
        elif idx_empid is not None:
            for r in rows:
                if len(r) > idx_empid and r[idx_empid] is not None:
                    people.add(str(r[idx_empid]))
        
        return len(people)
    
    def _analyze_types_and_hours(self, rows: List[Tuple], idx_type: Optional[int], 
                               idx_hours: Optional[int]) -> Tuple[Dict[str, int], Dict[str, float], List[float]]:
        """Analyze leave types and hours from rows."""
        type_counts_raw: Dict[str, int] = {}
        hours_vals: List[float] = []
        hours_by_type_raw: Dict[str, float] = {}
        
        def _num(v) -> Optional[float]:
            try:
                return float(v) if v is not None else None
            except Exception:
                return None
        
        for r in rows:
            # Count leave types
            if idx_type is not None and len(r) > idx_type:
                raw_t = "(unknown)" if r[idx_type] is None else str(r[idx_type]).strip()
                type_counts_raw[raw_t] = type_counts_raw.get(raw_t, 0) + 1
            
            # Collect hours values
            if idx_hours is not None and len(r) > idx_hours:
                hv = _num(r[idx_hours])
                if hv is not None:
                    hours_vals.append(hv)
                    if idx_type is not None and len(r) > idx_type:
                        raw_t = "(unknown)" if r[idx_type] is None else str(r[idx_type]).strip()
                        hours_by_type_raw[raw_t] = hours_by_type_raw.get(raw_t, 0.0) + hv
        
        return type_counts_raw, hours_by_type_raw, hours_vals
    
    def _convert_hours_by_type(self, hours_by_type_raw: Dict[str, float], converted: bool) -> Dict[str, float]:
        """Convert hours by type if needed."""
        if not hours_by_type_raw:
            return {}
        
        factor = (1.0 / 60.0) if converted else 1.0
        return {k: round(v * factor, 2) for k, v in hours_by_type_raw.items()}
    
    def _apply_labels_to_counts(self, type_counts_raw: Dict[str, int]) -> Dict[str, int]:
        """Apply friendly labels to leave type counts."""
        labeled_counts: Dict[str, int] = {}
        for raw_key, cnt in (type_counts_raw or {}).items():
            lbl = self.label_leave_type(raw_key)
            labeled_counts[lbl] = labeled_counts.get(lbl, 0) + cnt
        return labeled_counts
    
    def _apply_labels_to_hours(self, hours_by_type_hrs: Dict[str, float]) -> Dict[str, float]:
        """Apply friendly labels to leave type hours."""
        labeled_hours: Dict[str, float] = {}
        for raw_key, hrs in (hours_by_type_hrs or {}).items():
            lbl = self.label_leave_type(raw_key)
            labeled_hours[lbl] = round(labeled_hours.get(lbl, 0.0) + hrs, 2)
        return labeled_hours
    
    def _extract_names_sample(self, rows: List[Tuple], idx_name: Optional[int]) -> Optional[List[str]]:
        """Extract a sample of names from results."""
        if idx_name is None:
            return None
        
        names_sample: List[str] = []
        for r in rows[:15]:
            if len(r) > idx_name and r[idx_name]:
                names_sample.append(str(r[idx_name]))
        
        clean_names = [n for n in names_sample if n][:10]
        return clean_names if clean_names else None
    
    def _find_date_range(self, rows: List[Tuple], idx_workdate: Optional[int], 
                        idx_start: Optional[int], idx_end: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
        """Find date range across available date columns."""
        def _as_date_str(v):
            try:
                if hasattr(v, "date"):
                    return str(v.date())
                return str(v)
            except Exception:
                return str(v)
        
        all_dates: List[str] = []
        for r in rows:
            if idx_workdate is not None and len(r) > idx_workdate and r[idx_workdate] is not None:
                all_dates.append(_as_date_str(r[idx_workdate]))
            if idx_start is not None and len(r) > idx_start and r[idx_start] is not None:
                all_dates.append(_as_date_str(r[idx_start]))
            if idx_end is not None and len(r) > idx_end and r[idx_end] is not None:
                all_dates.append(_as_date_str(r[idx_end]))
        
        if not all_dates:
            return None, None
        
        return min(all_dates), max(all_dates)

