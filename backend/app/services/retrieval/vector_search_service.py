# ================================================================================
# backend/app/services/retrieval/vector_search_service.py
from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Dict, Any

from app.services.leave_vector import build_leave_index

logger = logging.getLogger(__name__)


class VectorSearchService:
    """Handles vector-based table retrieval and schema operations."""
    
    def __init__(self, db_service):
        self.db_service = db_service
        self.vector = None
        self.person_table: str = "dbo.PSNACCOUNT_D"
        
        self._initialize_vector_index()
        self._determine_person_table()
    
    def _initialize_vector_index(self):
        """Initialize the leave vector index."""
        try:
            self.vector = build_leave_index()
            logger.info("Leave vector index ready.")
        except Exception as e:
            self.vector = None
            logger.warning("Leave vector unavailable: %s", e)
    
    def _determine_person_table(self):
        """Determine the correct person table to use."""
        try:
            vt = getattr(self.vector, "_person_table", None)
            if isinstance(vt, str) and vt:
                self.person_table = vt
        except Exception:
            pass
    
    def find_relevant_tables(self, english_query: str, schema_filter: Optional[str] = None, 
                           rid: Optional[str] = None) -> List[Tuple[str, float]]:
        """Find tables relevant to the query using vector search."""
        try:
            hits = self.vector.search_relevant_tables(english_query, top_k=5) if self.vector else []
            
            if schema_filter:
                hits = [(t, s) for (t, s) in hits if t.lower().startswith(schema_filter.lower() + ".")]
            
            return hits
        except Exception as e:
            logger.warning("rid=%s vector search failed: %s", rid, e)
            return []
    
    def get_join_hints(self, tables: List[str]) -> str:
        """Get join hints for the given tables."""
        try:
            hints = self.vector.join_hints(tables) if self.vector else []
            return "\n".join(hints) if hints else "None"
        except Exception:
            return "None"
    
    def get_schema_context(self, tables: List[str], max_cols: int = 64) -> str:
        """Get schema context for tables, ensuring person table is included."""
        if not tables:
            return "No relevant tables found"
        
        # Ensure person table is present in the context
        pick = list(dict.fromkeys(
            tables[:3] + ([self.person_table] if self.person_table not in tables[:3] else [])
        ))
        
        return self.db_service.get_compact_schema_for(pick, max_columns_per_table=max_cols)
    
    def health_check(self) -> Dict[str, Any]:
        """Check vector search service health."""
        try:
            return self.vector.health_check() if self.vector else {"ready": False, "reason": "no index"}
        except Exception as e:
            return {"ready": False, "error": str(e)}
