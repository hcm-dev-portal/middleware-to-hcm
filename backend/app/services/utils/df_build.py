# app/services/utils/df_build.py
from __future__ import annotations
from typing import Any, Iterable, List, Sequence, Dict
import pandas as pd

def build_safe_dataframe(columns: Sequence[Any], rows: Iterable[Any]) -> pd.DataFrame:
    """
    Create a DataFrame even if row lengths != len(columns).
    - Truncates extra fields
    - Pads missing fields with None
    - Handles dict rows by selecting only provided columns
    """
    cols = [str(c) for c in columns]
    fixed_rows: List[Any] = []

    for r in rows:
        if isinstance(r, dict):
            fixed_rows.append({c: r.get(c) for c in cols})
        else:
            # assume list/tuple/np array-like
            r = list(r)
            if len(r) >= len(cols):
                fixed_rows.append(r[:len(cols)])
            else:
                fixed_rows.append(r + [None] * (len(cols) - len(r)))

    df = pd.DataFrame(fixed_rows, columns=cols)
    # Optional hygiene: drop fully empty columns
    df = df.loc[:, ~(df.isna().all())]
    return df
