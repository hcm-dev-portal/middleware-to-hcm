# backend/app/services/memory/simple_query_memory.py
from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =========================
# Helpers
# =========================

_STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "in", "on", "by", "with",
    "who", "what", "which", "when", "where", "how", "is", "are",
    "me", "my", "that", "this", "those", "these", "do", "did", "does"
}

# Column aliases for entity extraction (lower-cased comparison)
_DEPT_COLS = {"department", "dept", "deptname", "org", "organization", "businessunit", "business_unit", "bu", "site"}
_EMP_COLS  = {"truename", "name", "employee", "employee_name", "employeeid", "empid", "personid", "person_id"}
_TYPE_COLS = {"attendancetype", "leave_type", "leavetype", "type", "type_code"}
_DATE_COLS = {"date", "workdate", "startdate", "enddate", "as_of"}

# Lightweight “follow-up” phrases
_PRONOUN_DEPT = re.compile(r"\b(that|this|the)\s+(dept|department|business\s*unit|unit|org|team)\b", re.I)
_PRONOUN_PERSON = re.compile(r"\b(who|which person|which employee|them|that person|that employee)\b", re.I)
_TIME_LAST = re.compile(r"\b(last|previous)\s+(week|month|quarter|year)\b", re.I)
_TIME_THIS = re.compile(r"\b(this)\s+(week|month|quarter|year)\b", re.I)
_TIME_PAST_N_DAYS = re.compile(r"\bpast\s+(\d{1,3})\s+days\b", re.I)

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _compact(s: str, maxlen: int = 120) -> str:
    s = (s or "").strip().replace("\n", " ")
    return (s[:maxlen] + "…") if len(s) > maxlen else s

def _safe_tables(tbls: List[str]) -> List[str]:
    return sorted({(t or "").strip().lower() for t in tbls if t})

def _tokenize(text: str) -> List[str]:
    s = re.sub(r"[^A-Za-z0-9_]+", " ", (text or "").lower()).strip()
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return toks

def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

# =========================
# Data classes
# =========================

@dataclass
class CachedQuery:
    query_hash: str
    original_query: str
    generated_sql: str
    relevant_tables: List[str]
    success: bool
    cached_at: datetime
    use_count: int = 0
    # Optional hints to validate reuse quickly
    last_schema: Optional[str] = None
    # For time-aware queries (helps follow-ups)
    time_anchor: Optional[str] = None  # e.g. "as_of=2025-01-30"
    time_range: Optional[Tuple[str, str]] = None  # ("start","end") ISO dates

@dataclass
class ResultPreview:
    query: str
    columns: List[str]
    preview: List[Dict[str, Any]]
    total_rows: int
    timestamp: str

@dataclass
class EntityFocus:
    department: Optional[str] = None
    employee: Optional[str] = None
    leave_type: Optional[str] = None
    person_id: Optional[str] = None
    employee_id: Optional[str] = None

@dataclass
class TimeFocus:
    # ISO dates as strings to avoid tz pitfalls when serializing
    as_of: Optional[str] = None
    range_start: Optional[str] = None
    range_end: Optional[str] = None

@dataclass
class SessionContext:
    session_id: str
    recent_queries: List[str] = field(default_factory=list)
    recent_results: List[ResultPreview] = field(default_factory=list)
    last_activity: datetime = field(default_factory=_now_utc)
    focus: EntityFocus = field(default_factory=EntityFocus)
    time_focus: TimeFocus = field(default_factory=TimeFocus)
    last_tables: List[str] = field(default_factory=list)
    last_sql: Optional[str] = None
    last_columns: List[str] = field(default_factory=list)

# =========================
# Service
# =========================

class SimpleQueryMemoryService:
    """
    Smarter in-memory query memory (thread-safe).
    - Exact cache + semantic-similarity lookup (token/Jaccard) scoped by tables
    - Entity/time focus extraction to resolve follow-ups like “that department”
    - TTL cleanup and simple LRU-like trimming
    NOTE: Memory is ephemeral and process-local by design.
    """

    def __init__(self, cache_ttl_minutes: int = 30, max_cache_size: int = 200):
        self.cache_ttl_minutes = cache_ttl_minutes
        self.max_cache_size = max_cache_size

        self.query_cache: Dict[str, CachedQuery] = {}
        self.session_cache: Dict[str, SessionContext] = {}

        self.cache_hits = 0
        self.cache_misses = 0

        self._lock = threading.RLock()

    # --------------- Core hashing ---------------

    def _normalize_query(self, query: str) -> str:
        q = (query or "").strip().lower()
        q = q.replace("?", "").replace("!", "")
        q = re.sub(r"\s+", " ", q)
        # Normalize common synonyms to stabilize the key
        q = re.sub(r"\bbusiness\s*unit\b", "department", q)
        q = re.sub(r"\bdept\b", "department", q)
        q = re.sub(r"\bleave\s*type\b", "leavetype", q)
        return q

    def _create_query_hash(self, query: str, tables: List[str]) -> str:
        normalized_query = self._normalize_query(query)
        combined = f"{normalized_query}|{_safe_tables(tables)}"
        return hashlib.md5(combined.encode("utf-8")).hexdigest()[:12]

    # --------------- Cleanup / trimming ---------------

    def _cleanup_expired_cache(self):
        now = _now_utc()
        cutoff = now - timedelta(minutes=self.cache_ttl_minutes)

        # Remove expired queries
        expired = [k for k, v in self.query_cache.items() if v.cached_at < cutoff]
        for k in expired:
            self.query_cache.pop(k, None)

        # Remove inactive sessions
        expired_sessions = [sid for sid, ctx in self.session_cache.items() if ctx.last_activity < cutoff]
        for sid in expired_sessions:
            self.session_cache.pop(sid, None)

        # Trim if still too large (oldest then least used)
        if len(self.query_cache) > self.max_cache_size:
            items = sorted(self.query_cache.items(), key=lambda kv: (kv[1].cached_at, kv[1].use_count))
            for k, _ in items[: len(self.query_cache) - self.max_cache_size]:
                self.query_cache.pop(k, None)

    # --------------- Public caching API (backward compatible) ---------------

    def check_memory_for_query(
        self, query: str, relevant_tables: List[str], session_id: str = "default"
    ) -> Tuple[Optional[str], float]:
        """
        Returns (cached_sql, confidence_score) or (None, 0)
        - Exact hash hit preferred
        - Otherwise try semantic near-match for the same table set
        """
        with self._lock:
            self._cleanup_expired_cache()

            qhash = self._create_query_hash(query, relevant_tables)
            cached = self.query_cache.get(qhash)
            if cached and cached.success and cached.generated_sql.strip():
                cached.use_count += 1
                self.cache_hits += 1

                age_min = (_now_utc() - cached.cached_at).total_seconds() / 60
                confidence = max(0.70, 0.95 - (age_min / self.cache_ttl_minutes) * 0.25)

                logger.info("MEM_HIT exact hash=%s uses=%d q=%s", qhash, cached.use_count, _compact(query))
                return cached.generated_sql, round(confidence, 3)

            # Semantic fallback (same tables, high textual similarity)
            toks_q = _tokenize(query)
            tbl_key = tuple(_safe_tables(relevant_tables))
            best: Tuple[Optional[CachedQuery], float] = (None, 0.0)

            for cq in self.query_cache.values():
                if tuple(cq.relevant_tables) != tbl_key:
                    continue
                sim = _jaccard(toks_q, _tokenize(cq.original_query))
                if sim > best[1]:
                    best = (cq, sim)

            cq, sim = best
            if cq and sim >= 0.82 and cq.success and cq.generated_sql.strip():
                cq.use_count += 1
                self.cache_hits += 1
                logger.info("MEM_HIT semantic hash=%s sim=%.2f q=%s <- %s", cq.query_hash, sim, _compact(query), _compact(cq.original_query))
                # Semantic hits carry lower confidence
                return cq.generated_sql, round(0.60 + 0.30 * (sim - 0.82) / 0.18, 3)

            self.cache_misses += 1
            return None, 0.0

    def learn_from_query(
        self,
        query: str,
        relevant_tables: List[str],
        generated_sql: str,
        success: bool,
        execution_time: float,
        session_id: str = "default",
        *,
        schema_ctx: Optional[str] = None,
        time_anchor: Optional[str] = None,
        time_range: Optional[Tuple[str, str]] = None,
    ):
        """Store successful query for potential reuse."""
        if not success or not (generated_sql or "").strip():
            return

        with self._lock:
            self._cleanup_expired_cache()
            qhash = self._create_query_hash(query, relevant_tables)
            self.query_cache[qhash] = CachedQuery(
                query_hash=qhash,
                original_query=query,
                generated_sql=generated_sql,
                relevant_tables=_safe_tables(relevant_tables),
                success=True,
                cached_at=_now_utc(),
                use_count=0,
                last_schema=schema_ctx,
                time_anchor=time_anchor,
                time_range=time_range,
            )
            logger.info("MEM_LEARN hash=%s tables=%s q=%s", qhash, ",".join(_safe_tables(relevant_tables)), _compact(query))

    # --------------- Session context enrichment ---------------

    def record_success(
        self,
        session_id: str,
        query: str,
        generated_sql: str,
        columns: List[str],
        rows: List[Tuple],
        relevant_tables: List[str],
        schema_ctx: str,
        *,
        meta_time: Optional[Dict[str, Any]] = None,  # e.g., {"effective_as_of":"YYYY-MM-DD", "effective_range":{"start":..,"end":..}}
    ):
        """Record successful execution for follow-up context."""
        with self._lock:
            self._cleanup_expired_cache()
            ctx = self.session_cache.get(session_id)
            if not ctx:
                ctx = SessionContext(session_id=session_id)
                self.session_cache[session_id] = ctx

            ctx.last_activity = _now_utc()
            ctx.recent_queries = (ctx.recent_queries + [query])[-5:]
            ctx.last_sql = generated_sql
            ctx.last_tables = _safe_tables(relevant_tables)
            ctx.last_columns = [str(c) for c in (columns or [])]

            # Preview rows (first 3)
            preview_rows: List[Dict[str, Any]] = []
            if columns and rows:
                for row in rows[:3]:
                    row_dict = {}
                    for j, col in enumerate(columns):
                        if j < len(row):
                            row_dict[str(col)] = row[j]
                    preview_rows.append(row_dict)

            ctx.recent_results = (
                ctx.recent_results
                + [
                    ResultPreview(
                        query=query,
                        columns=list(map(str, columns or [])),
                        preview=preview_rows,
                        total_rows=len(rows or []),
                        timestamp=_now_utc().isoformat(),
                    )
                ]
            )[-3:]

            # Extract & update focus from result preview
            self._update_focus_from_preview(ctx)

            # Update time focus from meta (e.g., /api/leave_data extra_ctx)
            if meta_time:
                tf = ctx.time_focus
                tf.as_of = str(meta_time.get("effective_as_of") or tf.as_of)
                rng = meta_time.get("effective_range") or {}
                tf.range_start = str(rng.get("start") or tf.range_start)
                tf.range_end = str(rng.get("end") or tf.range_end)

    def _update_focus_from_preview(self, ctx: SessionContext) -> None:
        """Infer entity focus (dept / employee / leave type) from latest preview."""
        if not ctx.recent_results:
            return
        last = ctx.recent_results[-1]
        cols_l = [c.lower() for c in last.columns]
        focus = ctx.focus

        # Pick values from the first preview row when available
        row0 = (last.preview[0] if last.preview else {}) or {}

        # Department
        for k in cols_l:
            if k in _DEPT_COLS:
                val = row0.get(last.columns[cols_l.index(k)])
                if val:
                    focus.department = str(val)
                    break

        # Employee (name or ID)
        for k in cols_l:
            if k in _EMP_COLS:
                val = row0.get(last.columns[cols_l.index(k)])
                if val:
                    sval = str(val)
                    if "id" in k:
                        # prefer IDs into the id fields
                        if "employee" in k:
                            ctx.focus.employee_id = sval
                        else:
                            ctx.focus.person_id = sval
                    else:
                        focus.employee = sval
                    break

        # Leave type
        for k in cols_l:
            if k in _TYPE_COLS:
                val = row0.get(last.columns[cols_l.index(k)])
                if val:
                    focus.leave_type = str(val)
                    break

    # --------------- Follow-up helpers ---------------

    def get_last_focus_value(self, session_id: str, column_patterns: List[str]) -> Optional[str]:
        """(Back-compat) Return most recent value for any case-insensitive column name match."""
        with self._lock:
            ctx = self.session_cache.get(session_id)
            if not ctx or not ctx.recent_results:
                return None

            for result in reversed(ctx.recent_results):
                columns = result.columns
                preview = result.preview
                if not preview or not columns:
                    continue
                colmap = {c.lower(): c for c in columns}
                for pat in column_patterns:
                    key = colmap.get(pat.lower())
                    if key and preview:
                        val = preview[0].get(key)
                        if val is not None and str(val).strip():
                            return str(val)
            return None

    def rewrite_with_context(self, session_id: str, user_query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Resolve pronouns + time hints using session focus.
        Returns (possibly rewritten_query, applied_context_dict)
        """
        applied: Dict[str, Any] = {"used_department": None, "used_employee": None, "time_hint": None}
        q = user_query or ""
        with self._lock:
            ctx = self.session_cache.get(session_id)
            if not ctx:
                return q, applied

            # Department pronouns
            if _PRONOUN_DEPT.search(q) and ctx.focus.department:
                q = _PRONOUN_DEPT.sub(ctx.focus.department, q)
                applied["used_department"] = ctx.focus.department

            # Person pronouns (we map to name; IDs can also be appended)
            if _PRONOUN_PERSON.search(q) and (ctx.focus.employee or ctx.focus.employee_id or ctx.focus.person_id):
                # Prefer display name, else employee_id/person_id
                who = ctx.focus.employee or ctx.focus.employee_id or ctx.focus.person_id
                q = _PRONOUN_PERSON.sub(str(who), q)
                applied["used_employee"] = str(who)

            # Explicit time hints: "last week", "this month", "past 14 days"
            if _TIME_LAST.search(q) or _TIME_THIS.search(q) or _TIME_PAST_N_DAYS.search(q):
                # Keep as-is so the NL model can resolve it, but record hint for callers
                applied["time_hint"] = "explicit_in_query"
            else:
                # If no explicit time in the follow-up but we have a stable anchor, append as hint
                tf = ctx.time_focus
                if tf.as_of and tf.range_start and tf.range_end:
                    q += f" (as_of {tf.as_of}, range {tf.range_start}..{tf.range_end})"
                    applied["time_hint"] = "anchored_from_session"

            return q, applied

    def suggest_cached_candidates(
        self, query: str, relevant_tables: List[str], top_k: int = 3
    ) -> List[Tuple[str, float, str]]:
        """
        Return up to K candidate SQLs with similarity scores for the same tables.
        Each item: (generated_sql, score, original_query)
        """
        with self._lock:
            toks = _tokenize(query)
            tbl_key = tuple(_safe_tables(relevant_tables))
            ranked: List[Tuple[str, float, str]] = []
            for cq in self.query_cache.values():
                if tuple(cq.relevant_tables) != tbl_key:
                    continue
                score = _jaccard(toks, _tokenize(cq.original_query))
                if cq.success and cq.generated_sql.strip():
                    ranked.append((cq.generated_sql, score, cq.original_query))
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked[:top_k]

    # --------------- Stats / admin ---------------

    def get_memory_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total * 100.0) if total > 0 else 0.0
            return {
                "total_cached_queries": len(self.query_cache),
                "active_sessions": len(self.session_cache),
                "cache_hit_rate": round(hit_rate, 2),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_ttl_minutes": self.cache_ttl_minutes,
            }

    # --------------- Convenience APIs for time anchoring ---------------

    def update_time_focus(self, session_id: str, *, as_of: Optional[str], range_start: Optional[str], range_end: Optional[str]) -> None:
        with self._lock:
            ctx = self.session_cache.get(session_id)
            if not ctx:
                ctx = SessionContext(session_id=session_id)
                self.session_cache[session_id] = ctx
            ctx.last_activity = _now_utc()
            if as_of:
                ctx.time_focus.as_of = as_of
            if range_start:
                ctx.time_focus.range_start = range_start
            if range_end:
                ctx.time_focus.range_end = range_end

    def get_time_focus(self, session_id: str) -> Optional[TimeFocus]:
        with self._lock:
            ctx = self.session_cache.get(session_id)
            return ctx.time_focus if ctx else None
