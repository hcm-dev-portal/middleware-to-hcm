# backend/app/services/memory/simple_query_memory.py
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CachedQuery:
    """Simple cached query result."""
    query_hash: str
    original_query: str
    generated_sql: str
    relevant_tables: List[str]
    success: bool
    cached_at: datetime
    use_count: int


@dataclass
class SessionContext:
    """Simple session context for follow-ups."""
    session_id: str
    recent_queries: List[str]
    recent_results: List[Dict[str, Any]]
    last_activity: datetime


class SimpleQueryMemoryService:
    """
    Simple in-memory query cache for short-term memory only.
    No database persistence - just keeps recent successful queries
    for 2-3 follow-up questions.
    """

    def __init__(self, cache_ttl_minutes: int = 30, max_cache_size: int = 100):
        self.cache_ttl_minutes = cache_ttl_minutes
        self.max_cache_size = max_cache_size
        
        # Simple in-memory storage
        self.query_cache: Dict[str, CachedQuery] = {}
        self.session_cache: Dict[str, SessionContext] = {}
        
        # Simple stats
        self.cache_hits = 0
        self.cache_misses = 0

    def _create_query_hash(self, query: str, tables: List[str]) -> str:
        """Create a simple hash for query + tables."""
        normalized_query = query.lower().strip()
        # Remove common variations that don't change intent
        normalized_query = normalized_query.replace("?", "").replace("!", "")
        
        # Combine query and sorted tables for consistency
        combined = f"{normalized_query}|{sorted(tables)}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()[:12]

    def _cleanup_expired_cache(self):
        """Remove expired entries from cache."""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=self.cache_ttl_minutes)
        
        # Clean query cache
        expired_queries = [
            hash_key for hash_key, cached in self.query_cache.items()
            if cached.cached_at < cutoff
        ]
        for hash_key in expired_queries:
            self.query_cache.pop(hash_key, None)
        
        # Clean session cache
        expired_sessions = [
            session_id for session_id, ctx in self.session_cache.items()
            if ctx.last_activity < cutoff
        ]
        for session_id in expired_sessions:
            self.session_cache.pop(session_id, None)
        
        # If cache is still too large, remove oldest entries
        if len(self.query_cache) > self.max_cache_size:
            sorted_cache = sorted(
                self.query_cache.items(),
                key=lambda x: (x[1].cached_at, x[1].use_count)
            )
            to_remove = len(self.query_cache) - self.max_cache_size
            for i in range(to_remove):
                self.query_cache.pop(sorted_cache[i][0], None)

    def check_memory_for_query(self, query: str, relevant_tables: List[str], 
                             session_id: str = "default") -> Tuple[Optional[str], float]:
        """
        Check if we have a cached SQL for this query.
        Returns (cached_sql, confidence_score) or (None, 0)
        """
        self._cleanup_expired_cache()
        
        query_hash = self._create_query_hash(query, relevant_tables)
        cached = self.query_cache.get(query_hash)
        
        if cached and cached.success and cached.generated_sql.strip():
            # Update usage stats
            cached.use_count += 1
            self.cache_hits += 1
            
            logger.info("Memory HIT: hash=%s uses=%d query=%s", 
                       query_hash, cached.use_count, query[:50])
            
            # Simple confidence based on recency and success
            age_minutes = (datetime.utcnow() - cached.cached_at).total_seconds() / 60
            confidence = max(0.7, 0.95 - (age_minutes / self.cache_ttl_minutes) * 0.25)
            
            return cached.generated_sql, confidence
        
        self.cache_misses += 1
        return None, 0.0

    def learn_from_query(self, query: str, relevant_tables: List[str], 
                        generated_sql: str, success: bool, execution_time: float,
                        session_id: str = "default"):
        """Store successful query for potential reuse."""
        if not success or not generated_sql.strip():
            return
        
        self._cleanup_expired_cache()
        
        query_hash = self._create_query_hash(query, relevant_tables)
        
        cached = CachedQuery(
            query_hash=query_hash,
            original_query=query,
            generated_sql=generated_sql,
            relevant_tables=list(relevant_tables),
            success=success,
            cached_at=datetime.utcnow(),
            use_count=0
        )
        
        self.query_cache[query_hash] = cached
        logger.info("Memory LEARN: hash=%s success=%s query=%s", 
                   query_hash, success, query[:50])

    def record_success(self, session_id: str, query: str, generated_sql: str,
                      columns: List[str], rows: List[Tuple], 
                      relevant_tables: List[str], schema_ctx: str):
        """Record successful execution for session context."""
        self._cleanup_expired_cache()
        
        # Get or create session
        if session_id not in self.session_cache:
            self.session_cache[session_id] = SessionContext(
                session_id=session_id,
                recent_queries=[],
                recent_results=[],
                last_activity=datetime.utcnow()
            )
        
        ctx = self.session_cache[session_id]
        ctx.last_activity = datetime.utcnow()
        
        # Store recent queries (keep last 5)
        ctx.recent_queries.append(query)
        ctx.recent_queries = ctx.recent_queries[-5:]
        
        # Store compact result preview (first 3 rows only)
        if columns and rows:
            preview_rows = []
            for i, row in enumerate(rows[:3]):
                row_dict = {}
                for j, col in enumerate(columns):
                    if j < len(row):
                        row_dict[col] = row[j]
                preview_rows.append(row_dict)
            
            result_summary = {
                "query": query,
                "columns": list(columns),
                "preview": preview_rows,
                "total_rows": len(rows),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            ctx.recent_results.append(result_summary)
            ctx.recent_results = ctx.recent_results[-3:]  # Keep last 3 results

    def get_last_focus_value(self, session_id: str, column_patterns: List[str]) -> Optional[str]:
        """Get the most recent value for a column pattern from session context."""
        ctx = self.session_cache.get(session_id)
        if not ctx or not ctx.recent_results:
            return None
        
        # Search recent results for matching column
        for result in reversed(ctx.recent_results):
            columns = result.get("columns", [])
            preview = result.get("preview", [])
            
            if not preview or not columns:
                continue
            
            # Case-insensitive column matching
            column_map = {col.lower(): col for col in columns}
            
            for pattern in column_patterns:
                matched_col = column_map.get(pattern.lower())
                if matched_col and preview:
                    value = preview[0].get(matched_col)
                    if value is not None and str(value).strip():
                        return str(value)
        
        return None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get simple memory statistics."""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_queries * 100) if total_queries > 0 else 0
        
        return {
            "total_cached_queries": len(self.query_cache),
            "active_sessions": len(self.session_cache),
            "cache_hit_rate": round(hit_rate, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_ttl_minutes": self.cache_ttl_minutes
        }