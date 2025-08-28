from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple, Optional

_WORD = re.compile(r"[A-Za-z0-9_]+")

def _tok(s: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(s or "")]

def search_tables(index: Dict[str, Any], query: str, schema_filter: Optional[str] = None, top_k: int = 10) -> List[Tuple[str, float]]:
    if not index: 
        return []
    toks = _tok(query)
    hits: Dict[str, int] = {}
    for t in toks:
        for tbl in index.get("by_token", {}).get(t, []):
            if schema_filter and not tbl.lower().startswith(schema_filter.lower() + "."):
                continue
            hits[tbl] = hits.get(tbl, 0) + 1
    ranked = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))
    # score = token overlap count
    return [(tbl, float(score)) for tbl, score in ranked[:top_k]]

def intent_from_question(q: str) -> str:
    ql = (q or "").lower()
    if ql.startswith(("select", "with")): return "raw_sql"
    if "show tables" in ql or "list tables" in ql: return "show_tables"
    if ql.startswith(("describe ", "desc ")) or "describe table" in ql: return "describe_table"
    return "freeform"

_TABLE_REF = re.compile(r"\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\b")

def parse_table_ref(q: str) -> Optional[Tuple[str, str]]:
    m = _TABLE_REF.search(q or "")
    if not m: return None
    return (m.group(2), m.group(1))  # (table, schema)

def make_sql_for_intent(intent: str, schema: str, table: Optional[str]) -> Optional[str]:
    schema = schema or "dbo"
    if intent == "show_tables":
        return f"""
        SELECT TABLE_SCHEMA, TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE='BASE TABLE'
        ORDER BY TABLE_NAME
        """
    if intent == "describe_table" and table:
        return f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'
        ORDER BY ORDINAL_POSITION
        """
    return None
