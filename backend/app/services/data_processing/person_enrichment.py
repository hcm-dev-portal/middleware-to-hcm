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
    
    def detect_personid_columns(self, columns: List[str]) -> List[int]:
        """Detect which columns contain PERSONID values."""
        hits = []
        for i, c in enumerate(columns or []):
            if "personid" in (c or "").lower().replace("_", ""):
                hits.append(i)
        return hits
    
    def enrich_people_data(self, rows: List[Tuple], columns: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
        """Enrich query results with resolved person information."""
        idxs = self.detect_personid_columns(columns)
        if not rows or not idxs:
            return {}
        
        # Collect all unique person IDs
        ids = set()
        for r in rows:
            for i in idxs:
                if i < len(r) and r[i]:
                    ids.add(str(r[i]).strip())
        
        try:
            return self.person_resolver.resolve_many(list(ids))
        except Exception as e:
            logger.warning("Person resolve failed: %s", e)
            return {}