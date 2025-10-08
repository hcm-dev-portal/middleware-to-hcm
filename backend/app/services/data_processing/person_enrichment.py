# ================================================================================
# backend/app/services/data_processing/person_enrichment.py
from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Optional

from app.services.person_resolver import PersonResolver
from ..helpers.data_utils import find_column_index

logger = logging.getLogger(__name__)


class PersonEnrichmentService:
    """Handles person data enrichment and resolution."""

    def __init__(self, db_service):
        self.person_resolver = PersonResolver(db_service=db_service)

    # --- column detection helpers ------------------------------------------------
    def detect_personid_columns(self, columns: List[str]) -> List[int]:
        """
        Detect columns that contain PERSONID values (lenient match).
        Kept for backward compatibility.
        """
        hits: List[int] = []
        for i, c in enumerate(columns or []):
            if "personid" in (c or "").lower().replace("_", ""):
                hits.append(i)
        return hits

    def _detect_employeeid_columns(self, columns: List[str]) -> List[int]:
        """
        Detect columns that contain EMPLOYEEID values (empid/emp_no supported).
        """
        hits: List[int] = []
        for i, c in enumerate(columns or []):
            name = (c or "").lower().replace("_", "")
            if name in ("employeeid", "empid", "empno"):
                hits.append(i)
        return hits

    # --- main enrichment ---------------------------------------------------------
    def enrich_people_data(
        self,
        rows: List[Tuple],
        columns: List[str]
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Enrich query results with resolved person information.

        Returns a mapping keyed by **the same identifiers found in your data**:
          - If a PERSONID column exists, keys will include those PERSONIDs.
          - If an EMPLOYEEID column exists, keys will include those EMPLOYEEIDs.
        Each value has the normalized fields from PersonResolver:
          { person_id, name, employee_id, email, cardnum, department_id, department_name, department_code }
        """
        if not rows or not columns:
            return {}

        pid_idxs = self.detect_personid_columns(columns)
        eid_idxs = self._detect_employeeid_columns(columns)

        if not pid_idxs and not eid_idxs:
            return {}

        person_ids: set[str] = set()
        employee_ids: set[str] = set()

        for r in rows:
            # PERSONID harvest
            for i in pid_idxs:
                if i < len(r) and r[i] is not None:
                    s = str(r[i]).strip()
                    if s:
                        person_ids.add(s)
            # EMPLOYEEID harvest
            for i in eid_idxs:
                if i < len(r) and r[i] is not None:
                    s = str(r[i]).strip()
                    if s:
                        employee_ids.add(s)

        if not person_ids and not employee_ids:
            return {}

        try:
            # Resolve by both dimensions in one go
            resolved = self.person_resolver.resolve_many(
                list(person_ids),
                employee_ids=list(employee_ids) if employee_ids else None
            )
            # resolved already keyed by the IDs we passed in (pid and/or eid)
            return resolved or {}
        except Exception as e:
            logger.warning("Person resolve failed: %s", e)
            return {}

    # --- optional helper: annotate rows (non-breaking addition) ------------------
    def annotate_rows_with_display_name(
        self,
        rows: List[Tuple],
        columns: List[str],
        *,
        out_col_name: str = "display_name"
    ) -> List[Dict[str, Optional[str]]]:
        """
        Convenience: returns a list of dict rows with an added 'display_name' column,
        using PERSONID → name, falling back to EMPLOYEEID → name, then to raw ID.

        This does NOT modify the original rows; it’s safe to ignore if unused.
        """
        result_map = self.enrich_people_data(rows, columns)
        # Build quick lookups for both id types
        pid_lookup = {k: v.get("name") for k, v in result_map.items() if len(k) > 0 and k.startswith("{") is False}  # generic
        # A bit more explicit:
        # If the key looks like an employee id (not guaranteed format), we'll still try both maps below.

        out_rows: List[Dict[str, Optional[str]]] = []
        for r in rows or []:
            row = {columns[i]: (r[i] if i < len(r) else None) for i in range(len(columns))}
            display: Optional[str] = None

            # Try PERSONID col first
            for i in self.detect_personid_columns(columns):
                if i < len(r) and r[i]:
                    key = str(r[i]).strip()
                    display = (result_map.get(key) or {}).get("name")
                    if display:
                        break
            # Then EMPLOYEEID
            if not display:
                for i in self._detect_employeeid_columns(columns):
                    if i < len(r) and r[i]:
                        key = str(r[i]).strip()
                        display = (result_map.get(key) or {}).get("name")
                        if display:
                            break

            # Fallbacks
            if not display:
                display = (
                    row.get("TRUENAME")
                    or row.get("ENGNAME")
                    or next(
                        (str(r[i]).strip() for i in (self.detect_personid_columns(columns) + self._detect_employeeid_columns(columns))
                         if i < len(r) and r[i]), None
                    )
                )

            row[out_col_name] = display
            out_rows.append(row)

        return out_rows
