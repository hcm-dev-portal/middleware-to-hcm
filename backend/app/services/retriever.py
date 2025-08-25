# backend/app/services/retriever.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple, Optional

WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tok(s: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(s or "")]


def intent_from_question(q: str) -> str:
    s = q.strip().lower()
    if "show" in s and "table" in s and "column" not in s:
        return "show_tables"
    if ("describe" in s or "columns" in s or "schema of" in s or "structure" in s) and "table" in s:
        return "describe_table"
    return "search"


def parse_table_ref(q: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Try to extract table reference:
      - dbo.ATDOVERTIME
      - table ATDOVERTIME
    Returns (table_name, schema_or_none)
    """
    m = re.search(r"\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\b", q)  # schema.table
    if m:
        return m.group(2), m.group(1)
    m2 = re.search(r"\btable\s+([A-Za-z0-9_]+)\b", q, flags=re.I)
    if m2:
        return m2.group(1), None
    return None


def search_tables(index: Dict[str, Any], query: str, schema_filter: Optional[str] = None, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Search for relevant tables using both exact and partial token matching.
    """
    query_tokens = _tok(query)
    scores: Dict[str, float] = {}
    by_token = index.get("by_token", {})
    
    # First pass: exact token matches
    for query_token in query_tokens:
        if query_token in by_token:
            for full_table, weight in by_token[query_token]["tables"].items():
                if schema_filter and not full_table.lower().startswith(schema_filter.lower() + "."):
                    continue
                scores[full_table] = scores.get(full_table, 0.0) + float(weight)
    
    # Second pass: partial token matches (substring matching)
    for query_token in query_tokens:
        for index_token in by_token.keys():
            # Check if query token is a substring of index token, or vice versa
            if (query_token in index_token or index_token in query_token) and len(query_token) >= 3:
                # Partial match gets lower score
                partial_weight = 0.5
                for full_table, weight in by_token[index_token]["tables"].items():
                    if schema_filter and not full_table.lower().startswith(schema_filter.lower() + "."):
                        continue
                    # Only add partial score if we don't already have an exact match
                    if full_table not in scores:
                        scores[full_table] = scores.get(full_table, 0.0) + (float(weight) * partial_weight)
    
    # Third pass: semantic matching for common HR terms
    semantic_matches = {
        "employee": ["psn", "emp", "person", "staff"],
        "department": ["dept", "org", "unit"],
        "overtime": ["ovt", "overtime", "atd"],
        "salary": ["pay", "sal", "wage"],
        "leave": ["leave", "vacation", "absence"]
    }
    
    for query_token in query_tokens:
        if query_token in semantic_matches:
            for semantic_token in semantic_matches[query_token]:
                for index_token in by_token.keys():
                    if semantic_token in index_token:
                        semantic_weight = 0.3
                        for full_table, weight in by_token[index_token]["tables"].items():
                            if schema_filter and not full_table.lower().startswith(schema_filter.lower() + "."):
                                continue
                            scores[full_table] = scores.get(full_table, 0.0) + (float(weight) * semantic_weight)

    # Boost for exact table name matches
    for full_table in index.get("tables", {}).keys():
        table_name = full_table.split(".", 1)[1].lower()
        if table_name in [t.lower() for t in query_tokens]:
            scores[full_table] = scores.get(full_table, 0.0) + 5.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def make_sql_for_intent(intent: str, schema: str, table: Optional[str] = None) -> str:
    if intent == "show_tables":
        return (
            "SELECT TABLE_SCHEMA, TABLE_NAME\n"
            "FROM INFORMATION_SCHEMA.TABLES\n"
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE='BASE TABLE'\n"
            "ORDER BY TABLE_NAME"
        )
    if intent == "describe_table" and table:
        return (
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE\n"
            "FROM INFORMATION_SCHEMA.COLUMNS\n"
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'\n"
            "ORDER BY ORDINAL_POSITION"
        )
    return ""  # for generic search we don't auto-SQL