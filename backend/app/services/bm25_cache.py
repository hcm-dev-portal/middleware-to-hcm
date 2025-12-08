# backend/app/services/retrieval/bm25_cache.py
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
from typing import List, Tuple, Optional, Dict, Any


class BM25SQLiteCache:
    """
    Lightweight BM25 + SQLite cache for vector search results.

    - Keyed by (language | normalized_query | schema_filter).
    - Payload: JSON-serialized list of (table, score) pairs.
    - BM25 is applied on top of vector hits (table name text), then
      the final re-ranked list is cached.
    """

    def __init__(
        self,
        db_path: str,
        ttl_seconds: int = 7 * 24 * 3600,  # 7 days
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.db_path = db_path
        self.ttl = ttl_seconds
        self.k1 = k1
        self.b = b

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Some pragmatic SQLite pragmas for low-write / read-heavy usage
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")

        self._init_db()

    # ------------------------------------------------------------------ #
    # SQLite primitives
    # ------------------------------------------------------------------ #
    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key   TEXT PRIMARY KEY,
                    created_at  INTEGER NOT NULL,
                    payload     TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def get(self, key: str) -> Optional[List[Tuple[str, float]]]:
        """
        Return cached value if present and not expired.
        """
        now = int(time.time())
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT created_at, payload FROM query_cache WHERE cache_key = ?",
                (key,),
            )
            row = cur.fetchone()
            if not row:
                return None

            created_at, payload = row
            if self.ttl and (now - created_at) > self.ttl:
                # expire
                try:
                    self._conn.execute(
                        "DELETE FROM query_cache WHERE cache_key = ?", (key,)
                    )
                except Exception:
                    pass
                return None

            try:
                data = json.loads(payload)
                # Normalize types: [(table:str, score:float), ...]
                return [(str(t), float(s)) for t, s in data]
            except Exception:
                return None

    def set(self, key: str, value: List[Tuple[str, float]]) -> None:
        """
        Persist list[(table, score)] to SQLite.
        """
        now = int(time.time())
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO query_cache(cache_key, created_at, payload)
                VALUES (?, ?, ?)
                """,
                (key, now, payload),
            )

    # ------------------------------------------------------------------ #
    # BM25 core
    # ------------------------------------------------------------------ #
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _build_bm25_model(self, docs_tokens: List[List[str]]) -> Dict[str, Any]:
        """
        Build IDF + avgdl for a small doc set (candidate tables).
        """
        N = len(docs_tokens) or 1
        df: Dict[str, int] = {}

        for tokens in docs_tokens:
            seen = set()
            for t in tokens:
                if t in seen:
                    continue
                seen.add(t)
                df[t] = df.get(t, 0) + 1

        idf: Dict[str, float] = {}
        for t, df_t in df.items():
            # standard BM25-ish IDF
            idf[t] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)

        avgdl = sum(len(d) for d in docs_tokens) / float(N)
        return {"idf": idf, "avgdl": avgdl}

    def _bm25_scores(
        self,
        query: str,
        docs_text: List[str],
    ) -> List[float]:
        """
        Compute BM25 scores of docs_text against query.
        """
        if not query or not docs_text:
            return [0.0 for _ in docs_text]

        docs_tokens = [self._tokenize(t) for t in docs_text]
        model = self._build_bm25_model(docs_tokens)
        idf = model["idf"]
        avgdl = model["avgdl"] or 1.0

        q_tokens = self._tokenize(query)
        scores: List[float] = []

        for tokens in docs_tokens:
            dl = len(tokens) or 1
            score = 0.0
            for term in q_tokens:
                if term not in idf:
                    continue
                tf = tokens.count(term)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                score += idf[term] * (tf * (self.k1 + 1.0)) / denom
            scores.append(score)

        return scores

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Combine vector score with BM25 lexical score on table "documents".
        """
        if not candidates or not query:
            return candidates

        doc_texts = [str(table or "") for table, _ in candidates]
        bm_scores = self._bm25_scores(query, doc_texts)
        max_bm = max(bm_scores) if bm_scores else 0.0

        reranked: List[Tuple[str, float]] = []
        for (table, vec_score), bm in zip(candidates, bm_scores):
            combined = vec_score
            if max_bm > 0:
                # 70% vector, 30% lexical by default
                combined = 0.7 * float(vec_score) + 0.3 * float(bm / max_bm)
            reranked.append((table, combined))

        reranked.sort(key=lambda kv: kv[1], reverse=True)
        return reranked

    # ------------------------------------------------------------------ #
    # Public: lookup + compute
    # ------------------------------------------------------------------ #
    def get_or_compute(
        self,
        cache_key: str,
        query: str,
        base_hits: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        1) Try SQLite cache.
        2) If miss: run BM25 rerank on base_hits, store, return.
        """
        cached = self.get(cache_key)
        if cached is not None:
            return cached

        reranked = self.rerank(query, base_hits)
        try:
            self.set(cache_key, reranked)
        except Exception:
            # Cache failures should never break the main pipeline.
            pass
        return reranked
