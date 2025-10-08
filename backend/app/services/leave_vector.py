# backend/app/services/leave_vector.py
from __future__ import annotations

import os
import re
import json
import pickle
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Iterable, Set, Optional, Any, Literal
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Optional deps: sentence-transformers + faiss
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# Language detection
try:
    from langdetect import detect as _langdetect_detect  # type: ignore
    from langdetect.lang_detect_exception import LangDetectException  # type: ignore
except Exception:
    _langdetect_detect = None
    LangDetectException = Exception  # type: ignore

# Use ORG_TABLE to override fully-qualified org table name if needed
ORG_TABLE = os.getenv("ORG_TABLE", "[eHRAntung_DB].[dbo].[ORGStdStruct]")

_ZH_TO_EN_SYNONYMS: List[Tuple[re.Pattern, str]] = [
    # time
    (re.compile(r"今天|今日"), "today"),
    (re.compile(r"昨天|昨日"), "yesterday"),
    (re.compile(r"明天|翌日"), "tomorrow"),
    (re.compile(r"本週|這週|这周|本周"), "this week"),
    (re.compile(r"上週|上周"), "last week"),
    (re.compile(r"下週|下周"), "next week"),
    (re.compile(r"本月|這個月|这个月"), "this month"),
    (re.compile(r"上月|上個月|上个月"), "last month"),
    (re.compile(r"下月|下個月|下个月"), "next month"),
    (re.compile(r"歷史|历史|趨勢|趋势|過去|过去"), "history trend past"),
    (re.compile(r"未來|未来|即將|即将"), "future upcoming"),

    # domain nouns/verbs
    (re.compile(r"請假|休假"), "leave"),
    (re.compile(r"員工|人員|人力|同仁"), "employee person"),
    (re.compile(r"部門|單位|分行|分部"), "department branch unit"),
    (re.compile(r"部門名稱|單位名稱|分行名稱"), "department name branch name"),
    (re.compile(r"假別|假種|假期類型"), "leave type vacation type"),
    (re.compile(r"工號|員工編號|員編|人員代碼"), "employee id employeeid personid"),
    (re.compile(r"事業部|公司別|BU"), "business unit"),
    (re.compile(r"已核准|已批准|已驗證|已验证"), "validated approved"),
    (re.compile(r"餘額|余额"), "balance"),
    (re.compile(r"取消"), "cancel cancelation"),
]

def _expand_zh_synonyms(q: str) -> str:
    out = q
    for pat, en in _ZH_TO_EN_SYNONYMS:
        if pat.search(out):
            out += f" {en}"
    return out

def detect_language(text: str) -> Literal["zh-tw", "en"]:
    if not text or not text.strip():
        return "en"
    cnt_zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if cnt_zh >= 1:
        return "zh-tw"
    if _langdetect_detect is not None:
        try:
            d = _langdetect_detect(text)
            if d in ('zh-cn', 'zh-tw', 'zh'):
                return "zh-tw"
        except (LangDetectException, Exception):
            pass
    return "en"

class JoinType(Enum):
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

class Cardinality(Enum):
    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:M"
    MANY_TO_ONE = "M:1"
    MANY_TO_MANY = "M:M"

@dataclass
class TableJoin:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: JoinType = JoinType.LEFT
    cardinality: Cardinality = Cardinality.ONE_TO_MANY
    is_required: bool = True
    condition: Optional[str] = None

    description: str = ""
    description_zh: str = ""
    purpose: str = ""
    tags: List[str] = field(default_factory=list)

    def on_clause(self) -> str:
        cond = ""
        if self.condition:
            cond = " AND " + self.condition.format(left=self.left_table, right=self.right_table)
        return (f"{self.join_type.value} JOIN {self.right_table} "
                f"ON {self.left_table}.{self.left_column} = {self.right_table}.{self.right_column}{cond} "
                f"-- {self.cardinality.value}")

@dataclass
class QueryPattern:
    pattern: str
    description: str
    primary_tables: List[str]
    suggested_joins: List[str]
    required_filters: List[str] = field(default_factory=list)
    performance_notes: List[str] = field(default_factory=list)
    description_zh: str = ""
    examples: List[str] = field(default_factory=list)
    examples_zh: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class TableSchema:
    full: str
    columns: List[str]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    indexed_columns: List[str] = field(default_factory=list)
    row_estimate: Optional[int] = None
    is_historical: bool = False
    is_deleted_data: bool = False
    temporal_columns: List[str] = field(default_factory=list)

    description_zh: str = ""
    business_context: str = ""
    business_context_zh: str = ""
    common_queries: List[str] = field(default_factory=list)
    common_queries_zh: List[str] = field(default_factory=list)
    key_columns: Dict[str, str] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    row_count_estimate: str = ""
    priority: int = 1
    last_updated: Optional[str] = None
    kpi_relevance: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

@dataclass
class KPIDef:
    name: str
    description: str
    description_zh: str = ""
    formula_sql_hint: str = ""
    tables: List[str] = field(default_factory=list)
    grain: str = ""
    interpretation: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class SQLRecipe:
    recipe_id: str
    title: str
    description: str
    description_zh: str = ""
    sql_template: str = ""
    variables: Dict[str, str] = field(default_factory=dict)
    tables: List[str] = field(default_factory=list)
    expected_columns: List[str] = field(default_factory=list)
    caution_notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

class VectorItemType(Enum):
    TABLE = "TABLE"
    JOIN = "JOIN"
    PATTERN = "PATTERN"
    KPI = "KPI"
    RECIPE = "RECIPE"

@dataclass
class VectorItem:
    key: str
    item_type: VectorItemType
    text_en: str
    text_zh: str
    priority: int = 2
    payload: Dict[str, Any] = field(default_factory=dict)

    def get_text_for_lang(self, lang: Literal["zh-tw", "en"]) -> str:
        return self.text_zh if lang == "zh-tw" else self.text_en

COLUMN_ALIASES: Dict[str, Set[str]] = {
    "BUSINESSUNITID": {"BUSINESSUINTID"},
    "EFFECTIVEDATE": {"EFFINIENTDATE", "EFFICIENTDATE", "EFFDATE"},
    "TIMECARDDATE": {"CARDDATE"},
}

def _has_col(table: "TableSchema", name: str) -> bool:
    cols = {c.upper() for c in table.columns}
    nameU = name.upper()
    if nameU in cols:
        return True
    for canonical, aliases in COLUMN_ALIASES.items():
        if nameU == canonical or nameU in {a.upper() for a in aliases}:
            if canonical in cols or any(a.upper() in cols for a in aliases):
                return True
    return False

class LeaveVectorDB:
    def __init__(
        self,
        tables: List[TableSchema],
        db_path: str = "leave_schema_vectors.db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        kpis: Optional[List[KPIDef]] = None,
        recipes: Optional[List[SQLRecipe]] = None,
        patterns: Optional[List[QueryPattern]] = None,
        joins: Optional[List[TableJoin]] = None,
    ):
        self.tables = tables
        self._by_name: Dict[str, TableSchema] = {t.full.lower(): t for t in tables}
        self._person_table = self._resolve_person_table()

        self.db_path = db_path
        self.model_name = model_name

        self._joins = joins if joins is not None else self._build_comprehensive_joins()
        self._query_patterns = patterns if patterns is not None else self._build_query_patterns()
        self._kpis = kpis if kpis is not None else self._build_kpis()
        self._recipes = recipes if recipes is not None else self._build_recipes()

        self.model = None
        self.index_en = None
        self.index_zh = None
        self._vector_items: List[VectorItem] = []
        self._id2item: Dict[int, VectorItem] = {}
        self.embeddings_en: Optional[np.ndarray] = None
        self.embeddings_zh: Optional[np.ndarray] = None

        self._load_model()
        self._build_vector_items()
        self._build_indexes()

    def _resolve_person_table(self) -> Optional[str]:
        candidates = [
            "dbo.PSNACCOUNT",
            "dbo.PSNACCOUNT_D",
        ]
        for name in candidates:
            if name.lower() in self._by_name:
                return name
        return None

    def _exists(self, full: str) -> bool:
        return full.lower() in self._by_name

    def _load_model(self) -> None:
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not installed; using hashing-based fallback embeddings.")
            self.model = None
            return
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model: %s", self.model_name)
        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_name, e)
            self.model = None

    def _combine_text_by_language(self, obj, obj_type: str) -> Tuple[str, str]:
        if obj_type == "table":
            t: TableSchema = obj
            en_parts = [
                t.description, t.business_context, " ".join(t.tags),
                " ".join(t.common_queries), " ".join(t.relationships),
                " ".join(t.kpi_relevance), t.full, " ".join(t.columns[:40])
            ]
            zh_parts = [
                t.description_zh or t.description, t.business_context_zh or t.business_context, " ".join(t.tags),
                " ".join(t.common_queries_zh or t.common_queries), " ".join(t.relationships),
                " ".join(t.kpi_relevance), t.full, " ".join(t.columns[:40])
            ]

        elif obj_type == "join":
            j: TableJoin = obj
            en_parts = [
                j.description, j.purpose, " ".join(j.tags),
                j.left_table, j.right_table,
                f"{j.left_table}.{j.left_column}={j.right_table}.{j.right_column}",
                j.join_type.value, j.cardinality.value
            ]
            zh_parts = [
                j.description_zh or j.description, j.purpose, " ".join(j.tags),
                j.left_table, j.right_table,
                f"{j.left_table}.{j.left_column}={j.right_table}.{j.right_column}",
                j.join_type.value, j.cardinality.value
            ]

        elif obj_type == "pattern":
            p: QueryPattern = obj
            en_parts = [p.pattern, p.description, " ".join(p.tags), " ".join(p.primary_tables), " ".join(p.examples)]
            zh_parts = [p.pattern, p.description_zh or p.description, " ".join(p.tags),
                        " ".join(p.primary_tables), " ".join(p.examples_zh or p.examples)]

        elif obj_type == "kpi":
            k: KPIDef = obj
            en_parts = [k.name, k.description, " ".join(k.tags), " ".join(k.tables),
                        k.formula_sql_hint, k.grain, k.interpretation]
            zh_parts = [k.name, k.description_zh or k.description, " ".join(k.tags), " ".join(k.tables),
                        k.formula_sql_hint, k.grain, k.interpretation]

        elif obj_type == "recipe":
            r: SQLRecipe = obj
            en_parts = [r.title, r.description, " ".join(r.tags), " ".join(r.tables), " ".join(r.expected_columns)]
            zh_parts = [r.title, r.description_zh or r.description, " ".join(r.tags),
                        " ".join(r.tables), " ".join(r.expected_columns)]
        else:
            en_parts = zh_parts = [""]

        en_text = " ".join([p for p in en_parts if p])
        zh_text = " ".join([p for p in zh_parts if p])

        if not zh_text.strip() or len(zh_text.strip()) < 10:
            zh_text = en_text
        return en_text, zh_text

    def _build_vector_items(self) -> None:
        self._vector_items = []

        for t in self.tables:
            en_text, zh_text = self._combine_text_by_language(t, "table")
            self._vector_items.append(VectorItem(
                key=t.full, item_type=VectorItemType.TABLE,
                text_en=en_text, text_zh=zh_text, priority=t.priority,
                payload={"table": t}
            ))

        for j in self._joins:
            en_text, zh_text = self._combine_text_by_language(j, "join")
            self._vector_items.append(VectorItem(
                key=f"JOIN::{j.left_table}::{j.right_table}::{j.left_column}::{j.right_column}",
                item_type=VectorItemType.JOIN,
                text_en=en_text, text_zh=zh_text, priority=2, payload={"join": j}
            ))

        for p in self._query_patterns:
            en_text, zh_text = self._combine_text_by_language(p, "pattern")
            self._vector_items.append(VectorItem(
                key=f"PATTERN::{p.pattern}",
                item_type=VectorItemType.PATTERN,
                text_en=en_text, text_zh=zh_text, priority=1, payload={"pattern": p}
            ))

        for k in self._kpis:
            en_text, zh_text = self._combine_text_by_language(k, "kpi")
            self._vector_items.append(VectorItem(
                key=f"KPI::{k.name}",
                item_type=VectorItemType.KPI,
                text_en=en_text, text_zh=zh_text, priority=1, payload={"kpi": k}
            ))

        for r in self._recipes:
            en_text, zh_text = self._combine_text_by_language(r, "recipe")
            self._vector_items.append(VectorItem(
                key=f"RECIPE::{r.recipe_id}",
                item_type=VectorItemType.RECIPE,
                text_en=en_text, text_zh=zh_text, priority=1, payload={"recipe": r}
            ))

        logger.info("VECTOR_ITEMS: built=%d", len(self._vector_items))

    @staticmethod
    def _hashing_embed(texts: List[str], dim: int = 2048) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                idx = hash(tok) % dim
                out[i, idx] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out

    def _build_indexes(self) -> None:
        if not self._vector_items:
            logger.warning("No vector items to index.")
            return

        texts_en = [vi.text_en for vi in self._vector_items]
        texts_zh = [vi.text_zh for vi in self._vector_items]

        if self.model is not None:
            try:
                self.embeddings_en = self.model.encode(texts_en, normalize_embeddings=True, show_progress_bar=False).astype("float32")
                self.embeddings_zh = self.model.encode(texts_zh, normalize_embeddings=True, show_progress_bar=False).astype("float32")
                logger.info("EMBEDDINGS: EN=%s ZH=%s", self.embeddings_en.shape, self.embeddings_zh.shape)
            except Exception as e:
                logger.error("Embedding error; falling back to hashing. %s", e)
                self.embeddings_en = self._hashing_embed(texts_en)
                self.embeddings_zh = self._hashing_embed(texts_zh)
        else:
            self.embeddings_en = self._hashing_embed(texts_en)
            self.embeddings_zh = self._hashing_embed(texts_zh)

        if faiss is not None:
            try:
                dim = int(self.embeddings_en.shape[1])
                self.index_en = faiss.IndexFlatIP(dim)  # type: ignore
                self.index_zh = faiss.IndexFlatIP(dim)  # type: ignore
                self.index_en.add(self.embeddings_en)  # type: ignore
                self.index_zh.add(self.embeddings_zh)  # type: ignore
                logger.info("FAISS: built EN/ZH indexes, dim=%d, items=%d", dim, len(self._vector_items))
            except Exception as e:
                logger.error("FAISS error; using numpy fallback. %s", e)
                self.index_en = None
                self.index_zh = None
        else:
            self.index_en = None
            self.index_zh = None

        self._id2item = {i: vi for i, vi in enumerate(self._vector_items)}

    def _numpy_search(self, query_vec: np.ndarray, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        sims = embeddings @ query_vec.T
        sims = sims.squeeze()
        idxs = np.argsort(-sims)[:k]
        return sims[idxs], idxs

    def _encode_query(self, query: str) -> np.ndarray:
        if self.model is not None:
            try:
                q = self.model.encode([query], normalize_embeddings=True)
                return q.astype("float32")
            except Exception as e:
                logger.error("Query embedding error; falling back to hashing. %s", e)
        return self._hashing_embed([query])

    def _boost_score(self, vi: VectorItem, sim: float, query: str, lang: Literal["zh-tw", "en"]) -> float:
        score = float(sim)
        score *= (1.0 + (4 - vi.priority) * 0.10)

        ql = query.lower()

        if vi.item_type == VectorItemType.TABLE:
            t: TableSchema = vi.payload["table"]
            for kname in t.kpi_relevance:
                if kname and kname.lower() in ql:
                    score *= 1.10
            for col in t.columns[:20]:
                c = col.lower()
                if c in ql or any(w in ql for w in c.split("_")):
                    score *= 1.05
                    break

        elif vi.item_type == VectorItemType.KPI:
            kpi: KPIDef = vi.payload["kpi"]
            if kpi.name.lower() in ql:
                score *= 1.15
            if any(w in (kpi.description_zh or kpi.description).lower() for w in ql.split()):
                score *= 1.05

        elif vi.item_type == VectorItemType.PATTERN:
            pat: QueryPattern = vi.payload["pattern"]
            examples = pat.examples_zh if lang == "zh-tw" and pat.examples_zh else pat.examples
            if any(any(w in ex.lower() for w in ql.split()) for ex in (examples or [])):
                score *= 1.05

        return score

    def _do_search_once(self, query: str, lang: Literal["zh-tw", "en"], top_k: int) -> List[Tuple[VectorItem, float]]:
        if lang == "zh-tw":
            index, embeddings = self.index_zh, self.embeddings_zh
        else:
            index, embeddings = self.index_en, self.embeddings_en

        if embeddings is None:
            return []

        qvec = self._encode_query(query).astype("float32")
        k = min(top_k * 3, len(self._vector_items))

        if index is not None and faiss is not None:
            try:
                distances, indices = index.search(qvec, k)  # type: ignore
                sims, idxs = distances[0], indices[0]
            except Exception as e:
                logger.error("FAISS search error; falling back to numpy. %s", e)
                sims, idxs = self._numpy_search(qvec, embeddings, k)  # type: ignore
        else:
            sims, idxs = self._numpy_search(qvec, embeddings, k)  # type: ignore

        results: List[Tuple[VectorItem, float]] = []
        for sim, idx in zip(sims, idxs):
            if int(idx) >= len(self._vector_items):
                continue
            vi = self._id2item[int(idx)]
            weighted = self._boost_score(vi, float(sim), query, lang)
            results.append((vi, weighted))
        return results

    def search(self, query: str, top_k: int = 8, min_score: float = 0.25) -> List[Tuple[VectorItem, float]]:
        if not self._vector_items or (self.embeddings_en is None and self.embeddings_zh is None):
            return []

        base_lang = detect_language(query)
        q_expanded = _expand_zh_synonyms(query) if base_lang == "zh-tw" else query

        logger.info("VDB_SEARCH: lang=%s base_query='%s' expanded='%s'",
                    base_lang, query, q_expanded if q_expanded != query else "(none)")

        results = self._do_search_once(q_expanded, base_lang, top_k)

        strong = [(vi, s) for (vi, s) in results if s >= min_score]
        if len(strong) < max(2, top_k // 3):
            other_lang: Literal["zh-tw", "en"] = "en" if base_lang == "zh-tw" else "zh-tw"
            logger.debug("VDB_SEARCH: weak results (%d). Trying other_lang=%s", len(strong), other_lang)
            results += self._do_search_once(q_expanded, other_lang, top_k)

        dedup: Dict[str, Tuple[VectorItem, float]] = {}
        for vi, s in results:
            if s < min_score:
                continue
            if vi.key not in dedup or s > dedup[vi.key][1]:
                dedup[vi.key] = (vi, s)

        out = sorted(dedup.values(), key=lambda x: x[1], reverse=True)[:top_k]
        logger.info("VDB_SEARCH: final_hits=%d", len(out))
        for vi, s in out[:6]:
            logger.debug("VDB_HIT: type=%s key=%s score=%.3f", vi.item_type.value, vi.key, s)
        return out

    def search_relevant_tables(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        mixed = self.search(query, top_k=top_k * 2)
        tables: List[Tuple[str, float]] = []
        for vi, s in mixed:
            if vi.item_type == VectorItemType.TABLE:
                tables.append((vi.payload["table"].full, s))
            if len(tables) >= top_k:
                break

        if not tables:
            ql = (query or "").lower()
            likely = []
            if any(k in ql for k in ["leave", "請假", "休假", "today", "今天", "current", "當前"]):
                for guess in ["dbo.ATDLEAVEDATA"]:
                    if self._exists(guess):
                        likely.append((guess, 0.24))
            if any(k in ql for k in ["employee", "員工", "person", "personid", "姓名"]):
                if self._person_table:
                    likely.append((self._person_table, 0.23))
            tables = likely[:top_k]
            if tables:
                logger.warning("VDB_SEARCH: returning heuristic tables due to empty vector hits: %s", tables)

        return tables

    def join_hints(self, tables: Iterable[str]) -> List[str]:
        table_set = {(t or "").lower() for t in tables if t}
        hints: List[str] = []

        if self._person_table:
            hints.append(f"-- FACT.PERSONID → {self._person_table}.PERSONID (LEFT JOIN)")
            hints.append(f"-- e.g. LEFT JOIN {self._person_table} p ON p.PERSONID = l.PERSONID")

            if self._exists(ORG_TABLE):
                hints.append(f"-- {self._person_table}.BRANCHID → {ORG_TABLE}.UNITID (LEFT JOIN)")
                hints.append(f"-- e.g. LEFT JOIN {ORG_TABLE} org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))")
            elif self._exists('dbo.orgstdstruct'):
                hints.append(f"-- {self._person_table}.BRANCHID → dbo.ORGStdStruct.UNITID (LEFT JOIN)")
                hints.append(f"-- e.g. LEFT JOIN dbo.ORGStdStruct org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))")

        for j in self._joins:
            lt, rt = j.left_table.lower(), j.right_table.lower()
            if lt in table_set and rt in table_set:
                hints.append(j.on_clause())

        for tname in table_set:
            t = self._by_name.get(tname)
            if t and t.row_estimate and t.row_estimate > 100_000:
                hints.append(f"-- Performance: filter {t.full} by date range when possible")

        if any(n.startswith("dbo.atd") for n in table_set):
            hints.append("-- Performance: ATD* facts are large; add STARTDATE/ENDDATE (or WORKDATE) range predicates")

        return list(dict.fromkeys(hints))

    def get_schema_context(self, query: str, include_examples: bool = True) -> str:
        lang = detect_language(query)
        ranked = self.search(query, top_k=10)

        top_tables: List[TableSchema] = []
        top_patterns: List[QueryPattern] = []
        top_kpis: List[KPIDef] = []
        top_recipes: List[SQLRecipe] = []

        for vi, _score in ranked:
            if vi.item_type == VectorItemType.TABLE and len(top_tables) < 4:
                top_tables.append(vi.payload["table"])
            elif vi.item_type == VectorItemType.PATTERN and len(top_patterns) < 2:
                top_patterns.append(vi.payload["pattern"])
            elif vi.item_type == VectorItemType.KPI and len(top_kpis) < 3:
                top_kpis.append(vi.payload["kpi"])
            elif vi.item_type == VectorItemType.RECIPE and len(top_recipes) < 2:
                top_recipes.append(vi.payload["recipe"])

        join_strs: List[str] = []
        if top_tables:
            join_strs = self.join_hints([t.full for t in top_tables])

        lines: List[str] = []
        if lang == "zh-tw":
            lines.append("=== 相關資料庫上下文 ===")
            lines.append(f"查詢: '{query}'\n")
        else:
            lines.append("=== RELEVANT DATABASE CONTEXT ===")
            lines.append(f"Query: '{query}'\n")

        for i, t in enumerate(top_tables, 1):
            if lang == "zh-tw":
                desc = t.description_zh or t.description
                business = t.business_context_zh or t.business_context
                examples = t.common_queries_zh if t.common_queries_zh else t.common_queries
                lines.extend([
                    f"[{i}] 資料表: {t.full}",
                    f"  描述: {desc}",
                    f"  業務背景: {business}",
                    f"  資料量: {t.row_count_estimate or ('大型' if (t.row_estimate or 0) > 200000 else '中型' if t.row_estimate else '')}",
                ])
            else:
                lines.extend([
                    f"[{i}] TABLE: {t.full}",
                    f"  Description: {t.description}",
                    f"  Business Context: {t.business_context}",
                    f"  Data Volume: {t.row_count_estimate or (('large' if (t.row_estimate or 0) > 200000 else 'medium') if t.row_estimate else '')}",
                ])
                examples = t.common_queries

            if t.key_columns:
                header = "  關鍵欄位:" if lang == "zh-tw" else "  Key Columns:"
                lines.append(header)
                for c, d in list(t.key_columns.items())[:8]:
                    lines.append(f"    • {c}: {d}")

            if t.kpi_relevance:
                lines.append(f"  KPIs: {', '.join(t.kpi_relevance)}")

            if examples and include_examples:
                header = "  範例:" if lang == "zh-tw" else "  Examples:"
                lines.append(header)
                for ex in examples[:2]:
                    lines.append(f"    - {ex}")
            lines.append("")

        if join_strs:
            header = "=== 建議關聯 ===" if lang == "zh-tw" else "=== Suggested Joins ==="
            lines.append(header)
            for j in join_strs[:6]:
                lines.append(j)
            lines.append("")

        if top_patterns:
            header = "=== 相關查詢模式 ===" if lang == "zh-tw" else "=== Relevant Patterns ==="
            lines.append(header)
            for p in top_patterns:
                desc = p.description_zh if lang == "zh-tw" and p.description_zh else p.description
                lines.append(f"- {p.pattern}: {desc}")
            lines.append("")

        if top_kpis:
            header = "=== KPI 提示 ===" if lang == "zh-tw" else "=== KPI Hints ==="
            lines.append(header)
            for k in top_kpis:
                desc = k.description_zh if lang == "zh-tw" and k.description_zh else k.description
                grain_label = "資料粒度" if lang == "zh-tw" else "grain"
                lines.append(f"- {k.name}: {desc} ({grain_label}: {k.grain})")
            lines.append("")

        if top_recipes:
            header = "=== 經典SQL模版 ===" if lang == "zh-tw" else "=== Canonical SQL Recipes ==="
            lines.append(header)
            for r in top_recipes:
                desc = r.description_zh if lang == "zh-tw" and r.description_zh else r.description
                lines.append(f"- {r.title}: {desc}")
                if r.variables:
                    var_label = "變數" if lang == "zh-tw" else "Variables"
                    lines.append(f"  {var_label}: {', '.join(r.variables.keys())}")
            lines.append("")

        if lang == "zh-tw":
            lines.extend([
                "=== 查詢建構建議 ===",
                "• 歷史大表務必加日期範圍過濾",
                "• 包含 VALIDATED=1 以僅統計已批准假期",
                "• 僅在需要時加人員維度以避免不必要 JOIN",
                "• 考慮按 PERSONID/DEPARTMENTID 分組做彙總",
            ])
        else:
            lines.extend([
                "=== Query Construction Tips ===",
                "• Filter historical tables by date range",
                "• Include VALIDATED=1 for approved leave",
                "• Only join the person dimension when needed",
                "• Consider grouping by PERSONID/DEPARTMENTID",
            ])

        return "\n".join(lines)

    @staticmethod
    def preview_projection_sql(lang: Literal["zh-tw","en"] = "en") -> str:
        dept_expr = "COALESCE(org.UNITDISPLAYNAME, org.UNITNAME)"
        if lang == "zh-tw":
            return (
                f"{dept_expr} AS 部門, "
                "p.EMPLOYEEID AS 員編, "
                "p.TRUENAME AS 姓名"
            )
        else:
            return (
                f"{dept_expr} AS department_name, "
                "p.EMPLOYEEID AS employee_id, "
                "p.TRUENAME AS person_name"
            )

    def get_business_prompt(self, query: str) -> str:
        lang = detect_language(query)
        context = self.get_schema_context(query, include_examples=True)
        pv = self.preview_projection_sql("zh-tw" if lang == "zh-tw" else "en")
        if lang == "zh-tw":
            prompt = f"""
    您是一位請假/考勤領域的專業分析工程師。請提供SQL和推理，生成業務就緒的答案，而不僅僅是原始資料。

    使用者查詢：
    {query}

    {context}

    必須遵守：
    1) 產出正確的SQL，並包含以下 JOIN（若需要人員/部門顯示）：
    • FACT.PERSONID → dbo.PSNACCOUNT.PERSONID
    • dbo.PSNACCOUNT.BRANCHID → {ORG_TABLE}.UNITID
    2) 預覽欄位務必包含：{pv}
    3) 查詢暗示時間時，加入合理的日期視窗；涉及核准資料時加入 VALIDATED = 1
    4) 提供聚合/KPI時，說明粒度（人員/部門/日/月）與效能建議（索引、範圍條件）
    5) 可能為空時，建議切換歷史表（如 ATDHIS*）

    回傳格式：
    - 簡短推理要點
    - SQL
    - 預期欄位與資料粒度
    - 後續驗證建議（可選）
    """
        else:
            prompt = f"""
    You are an expert analytics engineer for the leave/attendance domain. Provide SQL and reasoning that deliver
    business-ready answers.

    USER QUERY:
    {query}

    {context}

    MUST DO:
    1) When person/department context is needed, include JOINs:
    • FACT.PERSONID → dbo.PSNACCOUNT.PERSONID
    • dbo.PSNACCOUNT.BRANCHID → {ORG_TABLE}.UNITID
    2) The preview SELECT must include: {pv}
    3) Add a sensible date window if time is implied; use VALIDATED = 1 for approved data
    4) For aggregations/KPIs, state the grain (person/department/day/month) and add performance tips
    5) If current tables may be empty, suggest switching to historical (ATDHIS*) tables

    Return:
    - Short reasoning bullets
    - SQL
    - Expected columns and grain
    - Optional follow-up checks
    """
        return prompt.strip()

    def _build_comprehensive_joins(self) -> List[TableJoin]:
        joins: List[TableJoin] = []

        leave_core = [
            "dbo.ATDLEAVEDATA",
            "dbo.ATDLEAVECANCELDATA",
        ]

        if self._person_table:
            for lt in leave_core:
                if self._exists(lt) and _has_col(self._by_name[lt.lower()], "PERSONID"):
                    joins.append(TableJoin(
                        left_table=lt, left_column="PERSONID",
                        right_table=self._person_table, right_column="PERSONID",
                        join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                        description="Resolve PERSONID to person attributes (TRUENAME, EMPLOYEEID, BRANCHID).",
                        description_zh="將 PERSONID 關聯到人員屬性（姓名、員編、部門BRANCHID）。",
                        purpose="Preview requires 姓名/員編 and further join to department.",
                        tags=["person","dimension","name","employee_id","branch"]
                    ))

        if self._exists(self._person_table or "") and self._exists(ORG_TABLE):
            joins.append(TableJoin(
                left_table=self._person_table, left_column="BRANCHID",
                right_table=ORG_TABLE, right_column="UNITID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                description="Resolve BRANCHID to department/branch (UNITID → UNITDISPLAYNAME/UNITNAME/UNITCODE).",
                description_zh="以 BRANCHID 關聯部門（UNITID → UNITDISPLAYNAME/UNITNAME/UNITCODE）。",
                purpose="Show 部門名稱 in previews and groupings.",
                tags=["department","org","branch","unit"]
            ))

        return joins

    def _build_query_patterns(self) -> List[QueryPattern]:
        return [
            QueryPattern(
                pattern="current_leave_status",
                description="Who is currently on leave (approved)",
                description_zh="當前正在休假的人員（已批准）",
                primary_tables=["dbo.ATDLEAVEDATA"],
                suggested_joins=["PERSONID"],
                required_filters=[
                    "VALIDATED = 1",
                    "CAST(GETDATE() AS date) BETWEEN CAST(STARTDATE AS date) AND CAST(ENDDATE AS date)"
                ],
                performance_notes=[
                    "Index/Filter on STARTDATE, ENDDATE",
                    "Group by PERSONID to collapse multiple days"
                ],
                examples=["Which employees are on leave today by department?"],
                examples_zh=["今天各部門有哪些員工在休假？"],
                tags=["current", "today", "validated"]
            ),
            QueryPattern(
                pattern="leave_cancellations",
                description="Cancelled leave requests and reasons",
                description_zh="已取消的請假申請及原因",
                primary_tables=["dbo.ATDLEAVECANCELDATA"],
                suggested_joins=["PERSONID"],
                required_filters=["WORKDATE range filter"],
                performance_notes=["Filter VALIDATED if needed; use REASON/LEAVEREASON"],
                examples=["Cancellation count and reasons last quarter"],
                examples_zh=["上季度取消請假的數量與原因"],
                tags=["cancel", "reason"]
            ),
        ]

    def _build_kpis(self) -> List[KPIDef]:
        return [
            KPIDef(
                name="total_leave_hours",
                description="Total approved leave hours in a period",
                description_zh="某時間段內已批准請假總小時數",
                formula_sql_hint="SUM(HOURS) WHERE VALIDATED=1 AND date filter",
                tables=["dbo.ATDLEAVEDATA"],
                grain="department-day / person-day",
                interpretation="Higher values may indicate seasonal effects or issues",
                tags=["volume", "hours"]
            ),
        ]

    def _build_recipes(self) -> List[SQLRecipe]:
        return [
           SQLRecipe(
                recipe_id="current_on_leave_by_dept",
                title="Current on-leave employees by department",
                description="Lists employees currently on validated leave grouped by department.",
                description_zh="按部門列出當前已批准且在休假的員工（含部門＋員編＋姓名）",
                tables=["dbo.ATDLEAVEDATA","dbo.PSNACCOUNT", ORG_TABLE],
                expected_columns=["department_name","employee_id","person_name","ATTENDANCETYPE","STARTDATE","ENDDATE"],
                sql_template=(
                    "SELECT "
                    "  COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS department_name,\n"
                    "  p.EMPLOYEEID AS employee_id,\n"
                    "  p.TRUENAME   AS person_name,\n"
                    "  l.ATTENDANCETYPE,\n"
                    "  CAST(l.STARTDATE AS date) AS STARTDATE,\n"
                    "  CAST(l.ENDDATE   AS date) AS ENDDATE\n"
                    "FROM dbo.ATDLEAVEDATA l\n"
                    "LEFT JOIN dbo.PSNACCOUNT p\n"
                    "  ON p.PERSONID = l.PERSONID\n"
                    f"LEFT JOIN {ORG_TABLE} org\n"
                    "  ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))\n"
                    "WHERE l.VALIDATED = 1\n"
                    "  AND CAST(GETDATE() AS date) BETWEEN CAST(l.STARTDATE AS date) AND CAST(l.ENDDATE AS date)\n"
                    "ORDER BY department_name, person_name;"
                ),
                caution_notes=["Ensure BRANCHID data type matches ORG.UNITID; CAST to NVARCHAR when needed."],
                tags=["current","validated","preview"]
            ),
        ]

    def is_ready(self) -> bool:
        return bool(self.tables) and (self.embeddings_en is not None or self.embeddings_zh is not None)

    def health_check(self) -> Dict[str, object]:
        faiss_used = False
        try:
            if faiss is not None and self.index_en is not None and self.index_zh is not None:
                faiss_used = True
        except Exception:
            faiss_used = False

        return {
            "ready": self.is_ready(),
            "tables_indexed": len(self.tables),
            "vector_items": len(self._vector_items),
            "join_relationships": len(self._joins),
            "query_patterns": len(self._query_patterns),
            "kpis": len(self._kpis),
            "recipes": len(self._recipes),
            "person_table": self._person_table,
            "model_loaded": self.model is not None,
            "faiss_used": faiss_used,
            "embeddings_en_shape": tuple(self.embeddings_en.shape) if self.embeddings_en is not None else None,
            "embeddings_zh_shape": tuple(self.embeddings_zh.shape) if self.embeddings_zh is not None else None,
            "model_name": self.model_name,
            "db_path": self.db_path,
            "language_detection_available": _langdetect_detect is not None
        }

    def save_to_disk(self) -> None:
        data = {
            "schemas": [asdict(s) for s in self.tables],
            "joins": [asdict(j) for j in self._joins],
            "patterns": [asdict(p) for p in self._query_patterns],
            "kpis": [asdict(k) for k in self._kpis],
            "recipes": [asdict(r) for r in self._recipes],
            "model_name": self.model_name,
            "created_at": datetime.now().isoformat(),
            "version": "4.0",
        }
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Language-aware Vector DB saved to %s", self.db_path)

    @classmethod
    def load_from_disk(cls, db_path: str = "leave_schema_vectors.db") -> "LeaveVectorDB":
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"No vector DB at {db_path}")
        with open(db_path, "rb") as f:
            data = pickle.load(f)

        tables = [TableSchema(**d) for d in data.get("schemas", [])]
        joins = [TableJoin(**d) for d in data.get("joins", [])]
        patterns = [QueryPattern(**d) for d in data.get("patterns", [])]
        kpis = [KPIDef(**d) for d in data.get("kpis", [])]
        recipes = [SQLRecipe(**d) for d in data.get("recipes", [])]

        inst = cls(
            tables=tables,
            db_path=db_path,
            model_name=data.get("model_name", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
            joins=joins,
            patterns=patterns,
            kpis=kpis,
            recipes=recipes,
        )
        return inst

    def get_query_pattern(self, query: str) -> Optional[QueryPattern]:
        q = query.lower()
        if any(x in q for x in ["當前", "今天", "現在", "今日"]) or ("leave" in q and any(x in q for x in ["current", "today", "now"])):
            return next((p for p in self._query_patterns if p.pattern == "current_leave_status"), None)
        if any(x in q for x in ["取消", "已取消"]) or any(x in q for x in ["cancel", "cancelled", "cancellation"]):
            return next((p for p in self._query_patterns if p.pattern == "leave_cancellations"), None)
        return None

    def relationships_sanity_check(self) -> Dict[str, List[str]]:
        errors, warnings = [], []
        for j in self._joins:
            lt = self._by_name.get(j.left_table.lower())
            rt = self._by_name.get(j.right_table.lower())
            if not lt or not rt:
                warnings.append(f"Table absent: {j.left_table} or {j.right_table}")
                continue
            if not _has_col(lt, j.left_column):
                errors.append(f"Missing column {j.left_table}.{j.left_column}")
            if not _has_col(rt, j.right_column):
                errors.append(f"Missing column {j.right_table}.{j.right_column}")
            if j.condition and ("{left}" not in j.condition or "{right}" not in j.condition):
                warnings.append(f"Condition placeholders missing in join {j.left_table} → {j.right_table}")
        return {"errors": errors, "warnings": warnings}

def build_leave_index() -> LeaveVectorDB:
    def T(full: str, cols: List[str], desc: str = "", tags: List[str] = None,  # type: ignore
          pks: List[str] = None, indexed: List[str] = None, rows: int = None,  # type: ignore
          is_hist: bool = False, is_del: bool = False, temporal: List[str] = None,  # type: ignore
          business_context: str = "", common_queries: List[str] = None, key_cols: Dict[str, str] = None,  # type: ignore
          relationships: List[str] = None, kpis: List[str] = None, priority: int = 1,  # type: ignore
          description_zh: str = "", business_context_zh: str = "", common_queries_zh: List[str] = None,  # type: ignore
          row_count_estimate: Optional[str] = None) -> TableSchema:

        rce = row_count_estimate or (("large" if (rows or 0) > 200000 else "medium") if rows else "")
        return TableSchema(
            full=full, columns=cols, description=desc, tags=tags or [],
            primary_keys=pks or [], indexed_columns=indexed or [], row_estimate=rows,
            is_historical=is_hist, is_deleted_data=is_del, temporal_columns=temporal or [],
            description_zh=description_zh, business_context=business_context,
            business_context_zh=business_context_zh, common_queries=common_queries or [],
            common_queries_zh=common_queries_zh or [], key_columns=key_cols or {},
            relationships=relationships or [], row_count_estimate=rce,
            priority=priority, kpi_relevance=kpis or []
        )

    tables: List[TableSchema] = [
        T("dbo.ATDLEAVEDATA",
          ["ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","STARTDATE","ENDDATE"],
          desc="Current validated leave data",
          description_zh="當前已驗證的請假資料",
          tags=["leave","current","validated"],
          priority=1,
          business_context="Operational day-level leave records",
          business_context_zh="日粒度的作業性請假紀錄",
          common_queries=[
              "Who is on leave today?",
              "Count validated leaves by department"
          ],
          common_queries_zh=[
              "今天有哪些員工在休假？",
              "各部門已核准請假數量"
          ],
          key_cols={"PERSONID": "Person surrogate key", "DEPARTMENTID": "Department surrogate key"},
        ),
        T("dbo.ATDLEAVECANCELDATA",
          ["OID","ATTENDANCETYPE","PERSONID","WORKDATE","STARTDATE","STARTTIME","ENDDATE","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","REASON","LEAVEREASON","CREATEDATE","LASTEDITTIME"],
          desc="Leave cancellation records",
          description_zh="請假取消資料",
          tags=["leave","cancel","reason"],
          priority=2,
          temporal=["WORKDATE","CREATEDATE","LASTEDITTIME"],
        ),
        T("dbo.PSNACCOUNT",
          ["CARDNUM","TRUENAME","PERSONID","EMPLOYEEID","COMPANYEMAIL","BRANCHID","BUSINESSUNITID","FIRSTNAME","MIDDLENAME","LASTNAME","ENGNAME"],
          desc="Person dimension (authoritative; includes BRANCHID for department)",
          description_zh="人員維度（權威來源；含BRANCHID以解析部門）",
          tags=["person","dimension","branch"],
          key_cols={
              "PERSONID": "人員鍵 (join to facts)",
              "EMPLOYEEID": "員編",
              "BRANCHID": "部門/單位鍵 → ORGStdStruct.UNITID"
          },
          priority=1
        ),
        T(ORG_TABLE,
          ["UNITID","UNITCODE","UNITNAME","UNITDISPLAYNAME","ISDELETE"],
          desc="Organization structure / branch dimension (UNITID as key)",
          description_zh="組織/單位維度（以 UNITID 為鍵）",
          tags=["org","department","branch"],
          key_cols={
              "UNITID": "部門鍵 ← PSNACCOUNT.BRANCHID",
              "UNITNAME": "部門名稱",
              "UNITDISPLAYNAME": "部門顯示名稱",
              "UNITCODE": "部門代碼"
          },
          priority=1
        ),
        T("dbo.ATDATTENDANCECLASS",
          ["CLASSCODE","ID","CLASSNAME","CLASSTYPE"],
          desc="Attendance/leave class dictionary (maps LEAVEID → readable type)",
          description_zh="出勤/請假類別字典（LEAVEID → 假別名稱）",
          tags=["dictionary","leave-type"],
          key_cols={"ID": "LEAVEID ↔ ATDLEAVEDATA.LEAVEID"},
          priority=1
        ),
    ]

    return LeaveVectorDB(tables)

def _contains_all(haystack: str, needles: Iterable[str]) -> bool:
    hs = (haystack or "").lower()
    return all(n.lower() in hs for n in needles)

def run_zh_tw_smoke_tests():
    print("=== zh-TW Vector→SQL Smoke Tests ===")
    db = build_leave_index()
    health = db.health_check()
    print("Health:", health)

    zh_queries = [
        "今天有誰在休假？",
        "查看本月請假統計",
        "請列出各部門今天在休假的人員（顯示部門＋員編＋姓名）",
    ]

    required_join_hints = [
        f"-- FACT.PERSONID → {db._person_table}.PERSONID (LEFT JOIN)".lower() if db._person_table else "",
        f"-- {db._person_table}.BRANCHID → {ORG_TABLE}.UNITID (LEFT JOIN)".lower() if db._person_table else "",
    ]
    required_join_hints = [h for h in required_join_hints if h]

    pv_tokens = ["部門", "員編", "姓名"]

    failures = 0
    for q in zh_queries:
        print(f"\n--- Query: {q} ---")
        lang = detect_language(q)
        print("Detected lang:", lang, "(expected: zh-tw)")
        if lang != "zh-tw":
            print("❌ lang detection mismatch")
            failures += 1

        hits = db.search(q, top_k=5)
        for item, score in hits:
            print(f"  HIT {item.item_type.value:<7} {item.key:<55} score={score:.3f}")

        prompt = db.get_business_prompt(q)
        print("\nPrompt (first 20 lines):")
        for i, line in enumerate(prompt.splitlines()[:20], 1):
            print(f"{i:02d}: {line}")

        if not all(tok in prompt for tok in pv_tokens):
            print("❌ preview tokens missing (需要『部門／員編／姓名』)")
            failures += 1

        low = prompt.lower()
        if required_join_hints and not _contains_all(low, required_join_hints):
            print("❌ join hints missing (PERSONID→PSNACCOUNT, BRANCHID→ORG)")
            failures += 1

        expanded = _expand_zh_synonyms(q)
        if expanded == q:
            print("⚠️  no synonym expansion appended (may be fine depending on wording)")
        else:
            print("Synonym expansion:", expanded)

    print("\n=== Summary ===")
    if failures == 0:
        print("✅ All zh-TW smoke tests passed.")
    else:
        print(f"❌ zh-TW smoke tests found {failures} issue(s).")

if __name__ == "__main__":
    def test_language_detection():
        tests = [
            ("今天有誰在休假？", "zh-tw"),
            ("Who is on leave today?", "en"),
            ("查看本月請假統計", "zh-tw"),
            ("Show monthly leave statistics", "en"),
            ("查看 leave 資料", "zh-tw"),
            ("Check the 請假 data", "zh-tw"),
        ]
        for q, expect in tests:
            got = detect_language(q)
            print(f"Query: '{q}' -> {got} (expected: {expect})")

    test_language_detection()
    db = build_leave_index()
    print(f"Health check: {db.health_check()}")
    for query in ["今天有誰在休假？", "Who is on leave today?"]:
        print(f"\n--- Query: {query} ---")
        for item, score in db.search(query, top_k=3):
            print(f"  {item.item_type.value}: {item.key} (score: {score:.3f})")

    print("\n\n==============================")
    run_zh_tw_smoke_tests()
