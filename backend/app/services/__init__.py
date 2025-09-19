# backend/app/services/__init__.py
"""
Lightweight utilities used across services.

This module intentionally has **no side effects on import**:
- No FastAPI app creation
- No lifespan handlers
- No database or vector bootstrap logic

It only exposes tiny helpers for schema/table discovery and intent parsing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# tokens for fuzzy table search
_WORD = re.compile(r"[A-Za-z0-9_]+")

# validate SQL identifiers we may interpolate into metadata queries
# (schema, table). Keep this strict: letters, digits, underscore only.
_ID_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# detect a single "schema.table" reference anywhere in a string
_TABLE_REF = re.compile(r"\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\b")


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _tok(s: str) -> List[str]:
    """Tokenize input for simple fuzzy matches (lowercased alnum+underscore chunks)."""
    if not s:
        return []
    return [w.lower() for w in _WORD.findall(s)]


def _safe_ident(x: Optional[str], default: str = "dbo") -> str:
    """
    Return a safe SQL identifier (letters/digits/underscore) or `default`.
    We *only* use these in INFORMATION_SCHEMA metadata queries.
    """
    x = (x or "").strip()
    if _ID_SAFE.match(x):
        return x
    logger.debug("Unsafe identifier %r replaced with default=%r", x, default)
    return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_tables(
    index: Dict[str, Any],
    query: str,
    schema_filter: Optional[str] = None,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Lightweight in-memory search over a prebuilt index:
      index = {"by_token": {"emp": ["dbo.Employee", ...], "att": [...], ...}}

    Returns a list of (table_fullname, score) ordered by decreasing score.

    Score is the token overlap count. This is intentionally simple & fast.
    """
    if not index or not isinstance(index, dict) or top_k <= 0:
        return []

    toks = _tok(query)
    if not toks:
        return []

    by_token = index.get("by_token", {}) or {}
    hits: Dict[str, int] = {}
    schema_prefix = (schema_filter or "").lower().strip()
    if schema_prefix and not schema_prefix.endswith("."):
        schema_prefix += "."

    for t in toks:
        for tbl in by_token.get(t, []) or []:
            # Filter by schema if requested
            if schema_prefix and not tbl.lower().startswith(schema_prefix):
                continue
            hits[tbl] = hits.get(tbl, 0) + 1

    ranked = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(tbl, float(score)) for tbl, score in ranked[:top_k]]


def intent_from_question(q: str) -> str:
    """
    Heuristic intent detection for simple console/admin use-cases.
    This is NOT an NLP classifier and should remain predictable & safe.
    """
    ql = (q or "").strip().lower()
    if not ql:
        return "freeform"

    # Raw SQL (guard rails elsewhere must still validate!)
    if ql.startswith(("select", "with")):
        return "raw_sql"

    # Show/list tables
    if "show tables" in ql or "list tables" in ql:
        return "show_tables"

    # Describe table
    if ql.startswith(("describe ", "desc ")) or "describe table" in ql:
        return "describe_table"

    return "freeform"


def parse_table_ref(q: str) -> Optional[Tuple[str, str]]:
    """
    Extract the first 'schema.table' occurrence in `q`.

    NOTE (backward compatibility):
    - Returns (table, schema) to preserve your original function’s order.
      If you prefer conventional order, use `parse_table_ref_parts`.
    """
    m = _TABLE_REF.search(q or "")
    if not m:
        return None
    return (m.group(2), m.group(1))  # (table, schema) — legacy order


def parse_table_ref_parts(q: str) -> Optional[Tuple[str, str]]:
    """
    Extract 'schema.table' and return (schema, table) in the conventional order.
    """
    m = _TABLE_REF.search(q or "")
    if not m:
        return None
    return (m.group(1), m.group(2))  # (schema, table)


def make_sql_for_intent(intent: str, schema: Optional[str], table: Optional[str]) -> Optional[str]:
    """
    Produce safe metadata SQL for **read-only** introspection on SQL Server.
    Uses INFORMATION_SCHEMA.* and strict identifier validation.

    We cannot parameterize identifiers, so we:
      - validate with `_ID_SAFE`
      - fall back to defaults (schema='dbo')
      - rely on INFORMATION_SCHEMA only (read-only)

    Returns a SQL string or None if the intent isn't supported here.
    """
    schema_safe = _safe_ident(schema, default="dbo")

    if intent == "show_tables":
        return (
            "SELECT TABLE_SCHEMA, TABLE_NAME\n"
            "FROM INFORMATION_SCHEMA.TABLES\n"
            f"WHERE TABLE_SCHEMA = '{schema_safe}' AND TABLE_TYPE = 'BASE TABLE'\n"
            "ORDER BY TABLE_NAME"
        )

    if intent == "describe_table" and table:
        table_safe = _safe_ident(table, default="")
        if not table_safe:
            logger.debug("Refusing to generate DESCRIBE SQL: invalid table name %r", table)
            return None
        return (
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH\n"
            "FROM INFORMATION_SCHEMA.COLUMNS\n"
            f"WHERE TABLE_SCHEMA = '{schema_safe}' AND TABLE_NAME = '{table_safe}'\n"
            "ORDER BY ORDINAL_POSITION"
        )

    # Nothing to do for raw_sql/freeform here
    return None


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "_tok",
    "search_tables",
    "intent_from_question",
    "parse_table_ref",          # legacy (table, schema)
    "parse_table_ref_parts",    # conventional (schema, table)
    "make_sql_for_intent",
]
