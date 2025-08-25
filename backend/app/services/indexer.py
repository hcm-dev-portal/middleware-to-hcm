# backend/app/services/indexer.py
from __future__ import annotations
import json, re, logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from app.core.paths import METADATA_PATH, INDEX_PATH

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[A-Za-z0-9_]+")  # keep underscores to match column/table tokens


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


def build_index(metadata: Dict[str, Any], boost_table: int = 3, boost_column: int = 1) -> Dict[str, Any]:
    """
    Build a simple inverted index:
    {
      "database": "...",
      "by_token": { token: { "tables": { "schema.table": score } } },
      "tables":   { "schema.table": { "columns":[...], "schema":"dbo", "table":"ATDOVERTIME" } }
    }
    """
    index: Dict[str, Any] = {
        "database": metadata.get("database"),
        "by_token": {},
        "tables": {}
    }

    schemas = metadata.get("schemas", {})
    for schema_name, tables in schemas.items():
        for table_name, info in tables.items():
            full = f"{schema_name}.{table_name}"
            cols = [c["name"] for c in info.get("columns", [])]
            index["tables"][full] = {
                "schema": schema_name,
                "table": table_name,
                "columns": cols
            }

            # tokens from table + columns
            table_tokens = _tokenize(table_name)
            col_tokens = []
            for c in cols:
                col_tokens.extend(_tokenize(c))

            # score table-name tokens higher
            token_weights: Dict[str, int] = {}
            for tk in table_tokens:
                token_weights[tk] = token_weights.get(tk, 0) + boost_table
            for tk in col_tokens:
                token_weights[tk] = token_weights.get(tk, 0) + boost_column

            # fill inverted
            for tk, w in token_weights.items():
                entry = index["by_token"].setdefault(tk, {"tables": {}})
                entry["tables"][full] = entry["tables"].get(full, 0) + w

    return index


def save_index(index: Dict[str, Any], path: Optional[Path] = None) -> str:
    p = path or INDEX_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    return str(p.resolve())


def load_index(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    p = path or INDEX_PATH
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
