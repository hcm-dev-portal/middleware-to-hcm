# ================================================================================
# backend/app/services/leave_vector.py
# Language-aware schema vector index with zh-TW optimized few-shots and join guardrails
# Public API preserved:
#   - detect_language(text) -> "zh-tw" | "en"
#   - build_leave_index() -> LeaveVectorDB
#   - class LeaveVectorDB: load_from_disk(...), save_to_disk(...),
#       search_relevant_tables(...), search(...),
#       join_hints(tables), search_few_shot_examples(...),
#       get_business_prompt(query, current_year), health_check()
# ================================================================================
from __future__ import annotations

import os
import re
import pickle
import logging
import unicodedata
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Iterable, Set, Literal
from datetime import datetime, date

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------
# FQ normalizer (kept)
# ---------------------------
def _fq(table: str, schema: str = "dbo") -> str:
    """
    Normalize a table identifier to 'schema.table' in lowercase.
    Accepts: 'ATDLEAVEDATA', 'dbo.ATDLEAVEDATA', '[dbo].[ATDLEAVEDATA]',
             '[db].[dbo].[ATDLEAVEDATA]' → returns 'dbo.atdleavedata'.
    If no schema present, prefixes with the given `schema` (default 'dbo').
    """
    t = (table or "").strip()
    if not t:
        return t
    s = t.replace("].[", ".").replace("[", "").replace("]", "").replace('"', "")
    parts = [p for p in s.split(".") if p]
    if len(parts) >= 2:
        schema_name, table_name = parts[-2], parts[-1]
        return f"{schema_name.lower()}.{table_name.lower()}"
    return f"{schema}.{parts[0].lower()}" if parts else ""


# Optional deps: sentence-transformers + faiss
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# Language detection (fast heuristic + optional langdetect fallback)
try:
    from langdetect import detect as _langdetect_detect  # type: ignore
    from langdetect.lang_detect_exception import LangDetectException  # type: ignore
except Exception:
    _langdetect_detect = None
    LangDetectException = Exception  # type: ignore

# Allow overriding org table (retains your current env usage)
ORG_TABLE = os.getenv("ORG_TABLE", "[eHRAntung_DB].[dbo].[ORGStdStruct]")

# ---------------------------
# Chinese → English signal expansion (kept/extended)
# ---------------------------
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
    (re.compile(r"過去|过去|近"), "past recent last"),
    # domain nouns/verbs
    (re.compile(r"請假|休假"), "leave vacation"),
    (re.compile(r"員工|人員|人力|同仁"), "employee person staff"),
    (re.compile(r"部門|單位|分行|分部"), "department branch unit"),
    (re.compile(r"部門名稱|單位名稱|分行名稱"), "department name branch name"),
    (re.compile(r"假別|假種|假期類型"), "leave type vacation type"),
    (re.compile(r"工號|員工編號|員編|人員代碼"), "employee id employeeid personid"),
    (re.compile(r"事業部|公司別|BU"), "business unit"),
    (re.compile(r"已核准|已批准|已驗證|已验证"), "validated approved"),
    (re.compile(r"取消|作廢"), "cancel cancellation"),
    (re.compile(r"統計|計算|彙總|汇总"), "statistics calculate aggregate sum"),
    (re.compile(r"時數|小時|钟头"), "hours time duration"),
    (re.compile(r"天數|日數"), "days count"),
    (re.compile(r"前10|前十|Top\s*10|TOP\s*10"), "top 10 rank"),
]

# ---------------------------
# Light canonicalization utils (kept)
# ---------------------------
_META_PATTERNS_ZH = [
    r'[，,]?\s*至少\s*\d+\s*(筆|条)',
    r'[，,]?\s*但(是)?(查詢|查询)?(沒|无|没有)資料',
    r'[，,]?\s*(如果)?沒有(資料)?就(算了|算|忽略)',
]
_META_PATTERNS_EN = [
    r'[,\s]*at\s*least\s*\d+\s*rows?',
    r'[,\s]*(no|zero)\s*data\s*(found)?',
    r'[,\s]*if\s*(none|no\s*data)\s*(then\s*ignore|skip)?',
]

def _to_halfwidth(s: str) -> str:
    return unicodedata.normalize('NFKC', s or "")

def _normalize_punct(s: str) -> str:
    s = (s or "")
    s = s.replace('，', ',').replace('：', ':').replace('／', '/').replace('－', '-').replace('～', '-')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def detect_language(text: str) -> Literal["zh-tw", "en"]:
    """Detect if text is Chinese or English"""
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

def _strip_meta(text: str, lang: Literal["zh-tw", "en"]) -> str:
    pats = _META_PATTERNS_ZH if lang == "zh-tw" else _META_PATTERNS_EN
    out = text
    for p in pats:
        out = re.sub(p, '', out, flags=re.IGNORECASE)
    return _normalize_punct(out)

def _canonicalize_query(original: str) -> Tuple[str, Literal["zh-tw", "en"]]:
    """Light canonicalization for retrieval stability (no semantic rewrites)."""
    s0 = _to_halfwidth(original or "")
    lang = detect_language(s0)
    s1 = _normalize_punct(s0)
    s2 = _strip_meta(s1, lang)
    return s2, lang

def _expand_zh_synonyms(q: str) -> str:
    """Expand Chinese query with English synonyms for better bilingual matching (deterministic, no dups)."""
    out = q
    added: Set[str] = set()
    for pat, en in _ZH_TO_EN_SYNONYMS:
        if pat.search(out) and en not in added:
            out += f" {en}"
            added.add(en)
    return out


# ---------- Vector structures ----------
class JoinType(Enum):
    LEFT = "LEFT"
    INNER = "INNER"
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
class FewShotExample:
    """Few-shot SQL example with bilingual support"""
    example_id: str
    category: str  # "date_handling", "name_resolution", "aggregation", "complex", "cancellation"
    user_query_zh: str
    user_query_en: str
    sql_template: str  # Supports {{CURRENT_YEAR}} / {{ANCHOR_DATE}}
    tables_used: List[str]
    join_pattern: str
    key_concepts: List[str]
    notes: str
    output_columns: List[str]
    expected_result_type: str  # "employee_list", "summary", "time_series"
    priority: int = 2

    def get_sql_for_year(self, year: int) -> str:
        return self.sql_template.replace('{{CURRENT_YEAR}}', str(year))

class VectorItemType(Enum):
    TABLE = "TABLE"
    JOIN = "JOIN"
    PATTERN = "PATTERN"
    KPI = "KPI"
    RECIPE = "RECIPE"
    FEWSHOT = "FEWSHOT"

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


# ---------- LeaveVectorDB ----------
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
        few_shot_examples: Optional[List[FewShotExample]] = None,
    ):
        # Normalize table names up-front
        self.tables = tables
        for t in self.tables:
            t.full = _fq(t.full)

        self._by_name = {t.full.lower(): t for t in self.tables}
        self._person_table = self._resolve_person_table()
        self._person_table = _fq(self._person_table) if self._person_table else None

        self.db_path = db_path
        self.model_name = model_name

        self._joins = joins if joins is not None else self._build_comprehensive_joins()
        self._query_patterns = patterns if patterns is not None else self._build_query_patterns()
        self._kpis = kpis if kpis is not None else self._build_kpis()
        self._recipes = recipes if recipes is not None else self._build_recipes()
        self._few_shot_examples = few_shot_examples if few_shot_examples is not None else self._build_few_shot_examples()

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

    # ----- helpers -----
    def _resolve_person_table(self) -> Optional[str]:
        for name in ("dbo.PSNACCOUNT", "dbo.PSNACCOUNT_D"):
            if _fq(name).lower() in {t.full.lower() for t in self.tables}:
                return _fq(name)
        return None

    def _exists(self, full: str) -> bool:
        return _fq(full).lower() in self._by_name

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

    # ----- vector items -----
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
        elif obj_type == "fewshot":
            f: FewShotExample = obj
            en_parts = [
                f.user_query_en, f.category, f.join_pattern, f.notes,
                " ".join(f.key_concepts), " ".join(f.tables_used), " ".join(f.output_columns)
            ]
            zh_parts = [
                f.user_query_zh, f.category, f.join_pattern, f.notes,
                " ".join(f.key_concepts), " ".join(f.tables_used), " ".join(f.output_columns)
            ]
        else:
            en_parts = zh_parts = [""]
        en_text = " ".join([p for p in en_parts if p])
        zh_text = " ".join([p for p in zh_parts if p]) or en_text
        return en_text, zh_text

    def _build_vector_items(self) -> None:
        self._vector_items = []

        for t in self.tables:
            t.full = _fq(t.full)
            en_text, zh_text = self._combine_text_by_language(t, "table")
            self._vector_items.append(VectorItem(
                key=t.full,
                item_type=VectorItemType.TABLE,
                text_en=en_text, text_zh=zh_text, priority=t.priority,
                payload={"table": t}
            ))

        for j in self._joins:
            j.left_table = _fq(j.left_table)
            j.right_table = _fq(j.right_table)
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

        for f in self._few_shot_examples:
            en_text, zh_text = self._combine_text_by_language(f, "fewshot")
            self._vector_items.append(VectorItem(
                key=f"FEWSHOT::{f.example_id}",
                item_type=VectorItemType.FEWSHOT,
                text_en=en_text, text_zh=zh_text, priority=f.priority, payload={"fewshot": f}
            ))

        logger.info(
            "VECTOR_ITEMS: built=%d (tables=%d joins=%d patterns=%d kpis=%d recipes=%d fewshot=%d)",
            len(self._vector_items), len(self.tables), len(self._joins),
            len(self._query_patterns), len(self._kpis), len(self._recipes), len(self._few_shot_examples)
        )

    # ----- embeddings / index -----
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
        score *= (1.0 + (4 - vi.priority) * 0.10)  # lower priority number = stronger

        ql = query.lower()
        if vi.item_type == VectorItemType.FEWSHOT:
            fs: FewShotExample = vi.payload["fewshot"]
            for concept in fs.key_concepts:
                if concept.lower() in ql:
                    score *= 1.25
            if fs.category in ["date_handling", "date"] and any(w in ql for w in ["日期","date","月","month","週","week","天","day"]):
                score *= 1.20
            if fs.category in ["aggregation"] and any(w in ql for w in ["sum","count","top 10","前10","前十","rank","排行"]):
                score *= 1.15
        elif vi.item_type == VectorItemType.TABLE:
            t: TableSchema = vi.payload["table"]
            for kname in t.kpi_relevance:
                if kname and kname.lower() in ql:
                    score *= 1.10
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

    # ────────────────────────────────────────────────────────────────────────────
    # Search APIs
    # ────────────────────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 8, min_score: float = 0.25) -> List[Tuple[VectorItem, float]]:
        if not self._vector_items or (self.embeddings_en is None and self.embeddings_zh is None):
            return []
        canon, base_lang = _canonicalize_query(query)
        q_expanded = _expand_zh_synonyms(canon) if base_lang == "zh-tw" else canon

        logger.info("VDB_SEARCH: lang=%s base_query='%s' expanded='%s'",
                    base_lang, query, (q_expanded if q_expanded != canon else "(none)"))

        results = self._do_search_once(q_expanded, base_lang, top_k)
        strong = [(vi, s) for (vi, s) in results if s >= min_score]
        if len(strong) < max(2, top_k // 3):
            other_lang: Literal["zh-tw", "en"] = "en" if base_lang == "zh-tw" else "zh-tw"
            results += self._do_search_once(q_expanded, other_lang, top_k)

        # dedupe
        dedup: Dict[str, Tuple[VectorItem, float]] = {}
        for vi, s in results:
            if s < min_score:
                continue
            if vi.key not in dedup or s > dedup[vi.key][1]:
                dedup[vi.key] = (vi, s)
        out = sorted(dedup.values(), key=lambda x: x[1], reverse=True)[:top_k]
        logger.info("VDB_SEARCH: final_hits=%d", len(out))
        return out

    def search_few_shot_examples(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> List[FewShotExample]:
        mixed = self.search(query, top_k=top_k * 3)
        examples: List[Tuple[FewShotExample, float]] = []
        for vi, s in mixed:
            if vi.item_type == VectorItemType.FEWSHOT:
                example: FewShotExample = vi.payload["fewshot"]
                if category_filter is None or example.category == category_filter:
                    examples.append((example, s))
            if len(examples) >= top_k:
                break
        return [ex for ex, _ in sorted(examples, key=lambda x: x[1], reverse=True)]

    def search_relevant_tables(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        mixed = self.search(query, top_k=top_k * 2)
        tables: List[Tuple[str, float]] = []
        seen: set[str] = set()
        for vi, s in mixed:
            if vi.item_type == VectorItemType.TABLE:
                full = vi.payload["table"].full
                norm = _fq(full)
                if norm.lower() not in seen:
                    tables.append((norm, s))
                    seen.add(norm.lower())
            if len(tables) >= top_k:
                break

        # Heuristic fallback
        if not tables:
            ql = (query or "").lower()
            likely: List[Tuple[str, float]] = []
            if any(k in ql for k in ["leave", "請假", "休假", "today", "今天", "current", "當前"]):
                guess = _fq("dbo.ATDLEAVEDATA")
                if self._exists(guess):
                    likely.append((guess, 0.24))
            if any(k in ql for k in ["employee", "員工", "person", "personid", "姓名", "名字", "name"]):
                if self._person_table:
                    likely.append((_fq(self._person_table), 0.23))
            tables = likely[:top_k]
            if tables:
                logger.warning("VDB_SEARCH: returning heuristic tables due to empty vector hits: %s", tables)
        return tables

    # ────────────────────────────────────────────────────────────────────────────
    # Join hints & schema context
    # ────────────────────────────────────────────────────────────────────────────
    def join_hints(self, tables: Iterable[str]) -> List[str]:
        table_set = {_fq((t or "")).lower() for t in tables if t}
        hints: List[str] = []

        # Person dim join (names are essential)
        if self._person_table:
            hints.append(f"-- FACT.PERSONID → {self._person_table}.PERSONID (LEFT JOIN)")
            hints.append(f"-- e.g. LEFT JOIN {self._person_table} p ON CAST(l.PERSONID AS NVARCHAR(100)) = CAST(p.PERSONID AS NVARCHAR(100))")

            # Person → Org via BRANCHID
            if self._exists(ORG_TABLE):
                hints.append(f"-- {self._person_table}.BRANCHID → {ORG_TABLE}.UNITID (LEFT JOIN)")
                hints.append(f"-- e.g. LEFT JOIN {ORG_TABLE} org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))")
            elif self._exists('dbo.ORGStdStruct'):
                hints.append(f"-- {self._person_table}.BRANCHID → dbo.ORGStdStruct.UNITID (LEFT JOIN)")
                hints.append(f"-- e.g. LEFT JOIN dbo.ORGStdStruct org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))")

        # Fact → Org via DEPARTMENTID
        if self._exists(ORG_TABLE):
            for fact in ("dbo.ATDLEAVEDATA", "dbo.ATDLEAVECANCELDATA"):
                fqn = _fq(fact)
                t = self._by_name.get(fqn.lower())
                if t and _has_col(t, "DEPARTMENTID"):
                    hints.append(f"-- {fqn}.DEPARTMENTID → {ORG_TABLE}.UNITID (LEFT JOIN)")
                    hints.append(f"-- e.g. LEFT JOIN {ORG_TABLE} org ON CAST(l.DEPARTMENTID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))")

        # Fact → Class for readable leave type
        cls_fqn = _fq("dbo.ATDATTENDANCECLASS")
        if self._exists(cls_fqn):
            cls_tbl = self._by_name.get(cls_fqn.lower())
            for fact in ("dbo.ATDLEAVEDATA", "dbo.ATDLEAVECANCELDATA"):
                fqn = _fq(fact)
                t = self._by_name.get(fqn.lower())
                if t and _has_col(t, "LEAVEID") and cls_tbl and _has_col(cls_tbl, "ID"):
                    hints.append(f"-- {fqn}.LEAVEID → {cls_fqn}.ID (LEFT JOIN)")
                    hints.append(f"-- e.g. LEFT JOIN {cls_fqn} c ON CAST(l.LEAVEID AS NVARCHAR(100)) = CAST(c.ID AS NVARCHAR(100))")

        # Add explicit anti-join/match guidance for cancellations
        if {"dbo.atdleavedata", "dbo.atdleavecanceldata"} <= table_set:
            hints.append("-- To compute NET hours: match cancellations to leave by FORM_NO or RECORD_ID and subtract cd.HOURS")
            hints.append("-- e.g. join on: CAST(cd.FORM_NO AS NVARCHAR(100)) = CAST(ld.FORM_NO AS NVARCHAR(100)) OR")
            hints.append("--      CAST(cd.RECORD_ID AS NVARCHAR(100)) = CAST(ld.RECORD_ID AS NVARCHAR(100))")

        # Include explicit join clauses that match the selected tables
        for j in self._joins:
            lt, rt = _fq(j.left_table).lower(), _fq(j.right_table).lower()
            if lt in table_set and rt in table_set:
                hints.append(j.on_clause())

        # Performance nudges
        for tname in table_set:
            t = self._by_name.get(tname)
            if t and t.row_estimate and t.row_estimate > 100_000:
                hints.append(f"-- Performance: filter {t.full} by date range when possible")
        if any(n.startswith("dbo.atd") for n in table_set):
            hints.append("-- Performance: ATD* facts are large; add STARTDATE/ENDDATE (or WORKDATE) range predicates")

        return list(dict.fromkeys(hints))

    def get_schema_context(self, query: str, include_examples: bool = True) -> str:
        lang = detect_language(query)
        ranked = self.search(query, top_k=12)
        top_tables: List[TableSchema] = []
        top_patterns: List[QueryPattern] = []
        top_kpis: List[KPIDef] = []
        top_recipes: List[SQLRecipe] = []
        top_fewshots: List[FewShotExample] = []

        for vi, _score in ranked:
            if vi.item_type == VectorItemType.TABLE and len(top_tables) < 4:
                top_tables.append(vi.payload["table"])
            elif vi.item_type == VectorItemType.PATTERN and len(top_patterns) < 2:
                top_patterns.append(vi.payload["pattern"])
            elif vi.item_type == VectorItemType.KPI and len(top_kpis) < 3:
                top_kpis.append(vi.payload["kpi"])
            elif vi.item_type == VectorItemType.RECIPE and len(top_recipes) < 2:
                top_recipes.append(vi.payload["recipe"])
            elif vi.item_type == VectorItemType.FEWSHOT and len(top_fewshots) < 3:
                top_fewshots.append(vi.payload["fewshot"])

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
                examples = t.common_queries
                lines.extend([
                    f"[{i}] TABLE: {t.full}",
                    f"  Description: {t.description}",
                    f"  Business Context: {t.business_context}",
                    f"  Data Volume: {t.row_count_estimate or (('large' if (t.row_estimate or 0) > 200000 else 'medium') if t.row_estimate else '')}",
                ])

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
            for j in join_strs[:10]:
                lines.append(j)
            lines.append("")

        if top_fewshots and include_examples:
            header = "=== 相關SQL範例 ===" if lang == "zh-tw" else "=== Relevant SQL Examples ==="
            lines.append(header)
            for fs in top_fewshots:
                query_text = fs.user_query_zh if lang == "zh-tw" else fs.user_query_en
                lines.append(f"\n範例: {query_text}" if lang == "zh-tw" else f"\nExample: {query_text}")
                lines.append(f"Tables: {', '.join(fs.tables_used)}")
                lines.append(f"Join Pattern: {fs.join_pattern}")
                lines.append(f"Notes: {fs.notes}")
                if include_examples:
                    lines.append("SQL Template:")
                    sql_lines = fs.sql_template.split('\n')
                    for sql_line in sql_lines[:18]:
                        lines.append(f"  {sql_line}")
                    if len(sql_lines) > 18:
                        lines.append("  ...")
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
                "• 歷史大表務必加日期範圍過濾（以 WORKDATE 為主；必要時 STARTDATE/ENDDATE）",
                "• 需要『已核准』時一律加上 VALIDATED = 1",
                "• 顯示姓名/部門時，必須 JOIN PSNACCOUNT 與 ORG 表",
                "• 『淨』時數需以 FORM_NO/RECORD_ID 比對取消單再扣除",
                "• 當查詢未指定年份時，使用當前年份 {{CURRENT_YEAR}}",
            ])
        else:
            lines.extend([
                "=== Query Construction Tips ===",
                "• Filter large facts by date window (prefer WORKDATE; START/END when needed)",
                "• Use VALIDATED = 1 for approved-only logic",
                "• Always join PSNACCOUNT for names; join ORG for department labels",
                "• NET hours: match cancellations by FORM_NO/RECORD_ID before subtracting",
                "• Use {{CURRENT_YEAR}} when year is omitted",
            ])
        return "\n".join(lines)

    @staticmethod
    def preview_projection_sql(lang: Literal["zh-tw","en"] = "en") -> str:
        dept_expr = "COALESCE(org.UNITDISPLAYNAME, org.UNITNAME)"
        if lang == "zh-tw":
            return f"{dept_expr} AS 部門, p.EMPLOYEEID AS 員編, p.TRUENAME AS 姓名"
        else:
            return f"{dept_expr} AS department_name, p.EMPLOYEEID AS employee_id, p.TRUENAME AS person_name"

    def get_business_prompt(self, query: str, current_year: int, anchor_date: date | None = None) -> str:
        lang = detect_language(query)
        context = self.get_schema_context(query, include_examples=True)
        pv = self.preview_projection_sql("zh-tw" if lang == "zh-tw" else "en")
        anchor_date = anchor_date or date.today()

        if lang == "zh-tw":
            prompt = f"""
您是一位請假/考勤領域的專業分析工程師。請輸出**業務就緒**的結果：先給出精簡推理要點，再給完整 SQL（使用 CTE），最後列出預期欄位與資料粒度。

使用者查詢（原文）：
{query}

**關鍵日期規則**
- 當前年份：{current_year}
- 若查詢含日期但未指明年份（例如「9/22-9/26」），一律視為 {current_year}
- 除非使用者明確指定，切勿使用過去年份
- 一律以 ISO 日期（YYYY-MM-DD）比較，並以 CAST(col AS date) 做日期範圍過濾

{context}

**必須遵守的 SQL 規範**
1) CTE（WITH）結構、明確欄位（避免 SELECT *）。
2) 顯示姓名/部門：
   • FACT.PERSONID → dbo.PSNACCOUNT.PERSONID（**必須**）
   • dbo.PSNACCOUNT.BRANCHID → {ORG_TABLE}.UNITID（需要部門名稱時）
3) 涉及「已核准」時，一律加上 `ld.VALIDATED = 1`（或別名對應）。
4) 預覽欄位必須包含：{pv}
5) 『淨』時數：僅在可透過 FORM_NO/RECORD_ID 精準比對時，才扣除取消單（ATDLEAVECANCELDATA）。
6) 統一以 WORKDATE 作時間視角；必要時使用 STARTDATE/ENDDATE 覆蓋區間。
7) 排序要固定（例如：部門、姓名、日期 ASC）；無資料時回傳空集合，不得改動條件。

**回傳格式**
- 推理要點（最多 5 點）
- SQL（CTE 模式，含固定排序）
- 預期欄位與資料粒度
""".strip()
        else:
            prompt = f"""
You are an expert analytics engineer for the leave/attendance domain. Provide business-ready output:
short reasoning bullets, then complete SQL (WITH/CTE), then expected columns + grain.

USER QUERY:
{query}

**Critical date context**
- Current Year: {current_year}
- If dates are given without a year (e.g., 9/22–9/26), assume {current_year}
- Prefer WORKDATE for windows; use START/END only when appropriate

{context}

**Must follow**
1) Use CTEs and explicit columns.
2) Names/departments require joins:
   • FACT.PERSONID → dbo.PSNACCOUNT.PERSONID (required)
   • dbo.PSNACCOUNT.BRANCHID → {ORG_TABLE}.UNITID (for department labels)
3) When the intent is “approved”, add `ld.VALIDATED = 1`.
4) Preview projection must include: {pv}
5) NET hours: subtract cancellations **only** when you can match by FORM_NO/RECORD_ID (ATDLEAVECANCELDATA).
6) Stable ordering; return empty sets rather than loosening filters.
""".strip()
        return prompt

    # ────────────────────────────────────────────────────────────────────────────
    # Join graph, patterns, KPIs, recipes, few-shots
    # ────────────────────────────────────────────────────────────────────────────
    def _build_comprehensive_joins(self) -> List[TableJoin]:
        joins: List[TableJoin] = []

        fact_leave = _fq("dbo.ATDLEAVEDATA")
        fact_cancel = _fq("dbo.ATDLEAVECANCELDATA")
        dim_person = _fq(self._person_table) if self._person_table else None
        dim_org = _fq(ORG_TABLE)
        dim_class = _fq("dbo.ATDATTENDANCECLASS")

        def _ok(table_full: str, col: str) -> bool:
            tbl = self._by_name.get(table_full.lower())
            return bool(tbl and _has_col(tbl, col))

        # FACT → PERSON
        if dim_person and self._exists(dim_person) and _ok(dim_person, "PERSONID"):
            for lt in (fact_leave, fact_cancel):
                if self._exists(lt) and _ok(lt, "PERSONID"):
                    joins.append(TableJoin(
                        left_table=lt, left_column="PERSONID",
                        right_table=dim_person, right_column="PERSONID",
                        join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                        description="Resolve PERSONID to person attributes (TRUENAME, EMPLOYEEID, BRANCHID).",
                        description_zh="以 PERSONID 關聯人員維度，取得姓名、員編與 BRANCHID。",
                        purpose="Get names/employee_id and enable downstream org joins.",
                        tags=["person", "dimension", "name", "employee_id", "branch"]
                    ))

        # FACT → CLASS
        if self._exists(dim_class) and _ok(dim_class, "ID"):
            for lt in (fact_leave, fact_cancel):
                if self._exists(lt) and _ok(lt, "LEAVEID"):
                    joins.append(TableJoin(
                        left_table=lt, left_column="LEAVEID",
                        right_table=dim_class, right_column="ID",
                        join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                        description="Map LEAVEID to readable leave type/class.",
                        description_zh="利用 LEAVEID 連至 ATDATTENDANCECLASS 取得假別名稱。",
                        purpose="Expose leave type names for grouping.",
                        tags=["dictionary", "class", "leave_type"]
                    ))

        # FACT → ORG via DEPARTMENTID
        if self._exists(dim_org) and _ok(dim_org, "UNITID"):
            for lt in (fact_leave, fact_cancel):
                if self._exists(lt) and _ok(lt, "DEPARTMENTID"):
                    joins.append(TableJoin(
                        left_table=lt, left_column="DEPARTMENTID",
                        right_table=dim_org, right_column="UNITID",
                        join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                        description="Resolve FACT.DEPARTMENTID to organization unit.",
                        description_zh="以 FACT.DEPARTMENTID 連至組織維度（UNITID）。",
                        purpose="Label fact rows with department.",
                        tags=["department", "org", "unit"]
                    ))

        # PERSON → ORG via BRANCHID
        if dim_person and self._exists(dim_person) and _ok(dim_person, "BRANCHID") and self._exists(dim_org) and _ok(dim_org, "UNITID"):
            joins.append(TableJoin(
                left_table=dim_person, left_column="BRANCHID",
                right_table=dim_org, right_column="UNITID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                description="Resolve BRANCHID to department/branch (UNITDISPLAYNAME/UNITNAME).",
                description_zh="以 BRANCHID 連至部門（UNITDISPLAYNAME/UNITNAME）。",
                purpose="Show 部門名稱; support org-level aggregation.",
                tags=["department", "org", "branch"]
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
                performance_notes=["Index/Filter on STARTDATE, ENDDATE"],
                examples=["Which employees are on leave today by department?"],
                examples_zh=["今天各部門有哪些員工在休假？"],
                tags=["current", "today", "validated"]
            ),
            QueryPattern(
                pattern="last_month_dept_top10",
                description="Last month TOP 10 departments by approved leave hours",
                description_zh="上月各部門請假時數 Top 10（僅已核准）",
                primary_tables=["dbo.ATDLEAVEDATA"],
                suggested_joins=["DEPARTMENTID→ORGStdStruct", "LEAVEID→ATDATTENDANCECLASS (optional for type)"],
                required_filters=[
                    "VALIDATED = 1",
                    "WORKDATE in last-month window"
                ],
                performance_notes=["Aggregate first then top 10"],
                examples=["Top 10 departments by SUM(HOURS) last month"],
                examples_zh=["上月請假時數排名前十的部門"],
                tags=["aggregation","rank","top10","last month"]
            ),
            QueryPattern(
                pattern="leave_cancellations",
                description="Cancelled leave requests and reasons",
                description_zh="已取消的請假申請及原因",
                primary_tables=["dbo.ATDLEAVECANCELDATA"],
                suggested_joins=["PERSONID"],
                required_filters=["CREATEDATE within range (when filtering by cancel time)"],
                performance_notes=["Filter by CREATEDATE; join to PSNACCOUNT for names"],
                examples=["Cancellation count and reasons last quarter"],
                examples_zh=["上季度取消請假的數量與原因"],
                tags=["cancel", "reason"]
            ),
        ]

    def _build_kpis(self) -> List[KPIDef]:
        return [
            KPIDef(
                name="total_leave_hours",
                description="Total approved leave hours in a period (VALIDATED=1)",
                description_zh="某時間段內已批准請假總小時數（VALIDATED=1）",
                formula_sql_hint="SUM(HOURS) WHERE VALIDATED=1 AND date filter",
                tables=["dbo.ATDLEAVEDATA"],
                grain="department-day / person-day",
                interpretation="Higher values may indicate seasonal effects or issues",
                tags=["volume", "hours"]
            ),
            KPIDef(
                name="net_leave_hours",
                description="Net approved leave hours = Leaves - Matched Cancellations",
                description_zh="『淨』請假小時 = 請假時數 − 成功比對之取消時數",
                formula_sql_hint="SUM(ld.HOURS) - SUM(cd.HOURS matched by FORM_NO/RECORD_ID)",
                tables=["dbo.ATDLEAVEDATA","dbo.ATDLEAVECANCELDATA"],
                grain="department-month / person-month",
                interpretation="Use only when reliable matching keys exist",
                tags=["net","cancellation","hours"]
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
                    "SELECT \n"
                    "  COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS department_name,\n"
                    "  p.EMPLOYEEID AS employee_id,\n"
                    "  p.TRUENAME   AS person_name,\n"
                    "  l.ATTENDANCETYPE,\n"
                    "  CAST(l.STARTDATE AS date) AS STARTDATE,\n"
                    "  CAST(l.ENDDATE   AS date) AS ENDDATE\n"
                    "FROM dbo.ATDLEAVEDATA l\n"
                    "LEFT JOIN dbo.PSNACCOUNT p\n"
                    "  ON CAST(p.PERSONID AS NVARCHAR(100)) = CAST(l.PERSONID AS NVARCHAR(100))\n"
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

    def _build_few_shot_examples(self) -> List[FewShotExample]:
        """Few-shots aligned with date rules and validated-only semantics."""
        return [
            # DATE HANDLING
            FewShotExample(
                example_id="date_001",
                category="date_handling",
                user_query_zh="查9/22-9/26休假的人（已核准）",
                user_query_en="Find people on approved leave from 9/22-9/26",
                sql_template="""WITH L AS (
    SELECT 
        CAST(ld.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        CAST(ld.LEAVEID AS NVARCHAR(100)) AS leave_id_norm,
        ld.ATTENDANCETYPE,
        ld.WORKDATE,
        ld.STARTDATE,
        ld.ENDDATE,
        ld.HOURS
    FROM eHRAntung_DB.dbo.ATDLEAVEDATA ld
    WHERE ld.VALIDATED = 1
),
P AS (
    SELECT 
        CAST(p.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        p.EMPLOYEEID AS employee_id,
        p.TRUENAME AS person_name
    FROM eHRAntung_DB.dbo.PSNACCOUNT p
),
C AS (
    SELECT 
        CAST(c.ID AS NVARCHAR(100)) AS leave_id_norm,
        c.CLASSNAME AS leave_type_name
    FROM eHRAntung_DB.dbo.ATDATTENDANCECLASS c
)
SELECT
    P.employee_id,
    P.person_name,
    C.leave_type_name,
    L.WORKDATE,
    L.HOURS
FROM L
LEFT JOIN P ON L.person_id_norm = P.person_id_norm
LEFT JOIN C ON L.leave_id_norm = C.leave_id_norm
WHERE L.WORKDATE >= '{{CURRENT_YEAR}}-09-22'
  AND L.WORKDATE <= '{{CURRENT_YEAR}}-09-26'
ORDER BY P.person_name, L.WORKDATE;""",
                tables_used=["dbo.ATDLEAVEDATA", "dbo.PSNACCOUNT", "dbo.ATDATTENDANCECLASS"],
                join_pattern="ATDLEAVEDATA.PERSONID → PSNACCOUNT.PERSONID; ATDLEAVEDATA.LEAVEID → ATDATTENDANCECLASS.ID",
                key_concepts=["date_range_no_year","approved","leave_records","employee_names"],
                notes="When dates omit a year, use {{CURRENT_YEAR}}. Approved = VALIDATED=1.",
                output_columns=["employee_id","person_name","leave_type_name","WORKDATE","HOURS"],
                expected_result_type="employee_list",
                priority=2
            ),
            FewShotExample(
                example_id="date_002",
                category="date_handling",
                user_query_zh="本月已核准請假紀錄",
                user_query_en="Approved leave records this month",
                sql_template="""WITH L AS (
    SELECT 
        CAST(ld.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        ld.WORKDATE,
        ld.HOURS
    FROM eHRAntung_DB.dbo.ATDLEAVEDATA ld
    WHERE ld.VALIDATED = 1
),
P AS (
    SELECT 
        CAST(p.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        p.EMPLOYEEID AS employee_id,
        p.TRUENAME AS person_name
    FROM eHRAntung_DB.dbo.PSNACCOUNT p
)
SELECT
    P.employee_id,
    P.person_name,
    L.WORKDATE,
    L.HOURS
FROM L
LEFT JOIN P ON L.person_id_norm = P.person_id_norm
WHERE L.WORKDATE >= DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()), 0)
  AND L.WORKDATE <  DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()) + 1, 0)
ORDER BY L.WORKDATE DESC, P.person_name;""",
                tables_used=["dbo.ATDLEAVEDATA","dbo.PSNACCOUNT"],
                join_pattern="ATDLEAVEDATA.PERSONID → PSNACCOUNT.PERSONID",
                key_concepts=["this_month","approved","names"],
                notes="Use WORKDATE window and VALIDATED=1.",
                output_columns=["employee_id","person_name","WORKDATE","HOURS"],
                expected_result_type="employee_list",
                priority=2
            ),
            # AGGREGATION & RANK
            FewShotExample(
                example_id="fs_last_month_dept_top10",
                category="aggregation",
                user_query_zh="上月各部門請假時數 Top 10（僅已核准）",
                user_query_en="Last month TOP 10 departments by approved leave hours",
                sql_template=f"""SELECT TOP 10 
    COALESCE(o.UNITDISPLAYNAME, o.UNITNAME) AS dept_name,
    SUM(COALESCE(ld.HOURS, 0)) AS total_hours
FROM eHRAntung_DB.dbo.ATDLEAVEDATA ld
LEFT JOIN {ORG_TABLE} o 
  ON CAST(ld.DEPARTMENTID AS NVARCHAR(100)) = CAST(o.UNITID AS NVARCHAR(100))
WHERE ld.VALIDATED = 1
  AND ld.WORKDATE >= DATEFROMPARTS(YEAR(DATEADD(MONTH, -1, GETDATE())), MONTH(DATEADD(MONTH, -1, GETDATE())), 1)
  AND ld.WORKDATE <  DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)
GROUP BY COALESCE(o.UNITDISPLAYNAME, o.UNITNAME)
ORDER BY total_hours DESC;""",
                tables_used=["dbo.ATDLEAVEDATA", ORG_TABLE],
                join_pattern=f"ATDLEAVEDATA.DEPARTMENTID → {ORG_TABLE}.UNITID",
                key_concepts=["last month","top 10","approved","department ranking"],
                notes="Uses last-month window via DATEFROMPARTS; VALIDATED=1 enforced.",
                output_columns=["dept_name","total_hours"],
                expected_result_type="summary",
                priority=3
            ),
            FewShotExample(
                example_id="agg_001",
                category="aggregation",
                user_query_zh="統計每個人的請假天數（已核准）",
                user_query_en="Count approved leave days for each person",
                sql_template="""WITH L AS (
    SELECT 
        CAST(ld.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        COUNT(DISTINCT CAST(ld.WORKDATE AS date)) AS leave_days,
        SUM(COALESCE(ld.HOURS, 0)) / 8.0 AS leave_days_calculated
    FROM eHRAntung_DB.dbo.ATDLEAVEDATA ld
    WHERE ld.VALIDATED = 1
      AND ld.WORKDATE >= '{{CURRENT_YEAR}}-01-01'
      AND ld.WORKDATE <= '{{CURRENT_YEAR}}-12-31'
    GROUP BY ld.PERSONID
),
P AS (
    SELECT 
        CAST(p.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        p.EMPLOYEEID AS employee_id,
        p.TRUENAME AS person_name
    FROM eHRAntung_DB.dbo.PSNACCOUNT p
)
SELECT
    P.employee_id,
    P.person_name,
    L.leave_days,
    L.leave_days_calculated
FROM L
LEFT JOIN P ON L.person_id_norm = P.person_id_norm
ORDER BY L.leave_days DESC, P.person_name;""",
                tables_used=["dbo.ATDLEAVEDATA","dbo.PSNACCOUNT"],
                join_pattern="ATDLEAVEDATA.PERSONID → PSNACCOUNT.PERSONID",
                key_concepts=["count_days","approved","aggregation"],
                notes="Count distinct WORKDATE or use HOURS/8. Always include names.",
                output_columns=["employee_id","person_name","leave_days","leave_days_calculated"],
                expected_result_type="summary",
                priority=2
            ),
            # NAME RESOLUTION
            FewShotExample(
                example_id="name_001",
                category="name_resolution",
                user_query_zh="剩餘休假時數大於200小時的人（以已核准記錄統計）",
                user_query_en="People with total approved leave hours > 200 this year",
                sql_template="""WITH L AS (
    SELECT 
        CAST(ld.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        SUM(COALESCE(ld.HOURS, 0)) AS total_leave_hours
    FROM eHRAntung_DB.dbo.ATDLEAVEDATA ld
    WHERE ld.VALIDATED = 1
      AND ld.WORKDATE >= '{{CURRENT_YEAR}}-01-01'
      AND ld.WORKDATE <= '{{CURRENT_YEAR}}-12-31'
    GROUP BY ld.PERSONID
),
P AS (
    SELECT 
        CAST(p.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        p.EMPLOYEEID AS employee_id,
        p.TRUENAME AS person_name
    FROM eHRAntung_DB.dbo.PSNACCOUNT p
)
SELECT
    P.employee_id,
    P.person_name,
    L.total_leave_hours
FROM L
INNER JOIN P ON L.person_id_norm = P.person_id_norm
WHERE L.total_leave_hours > 200
ORDER BY L.total_leave_hours DESC, P.person_name;""",
                tables_used=["dbo.ATDLEAVEDATA","dbo.PSNACCOUNT"],
                join_pattern="ATDLEAVEDATA.PERSONID → PSNACCOUNT.PERSONID",
                key_concepts=["threshold","approved"],
                notes="Interpretation aligns with approved leave usage, not entitlement balance.",
                output_columns=["employee_id","person_name","total_leave_hours"],
                expected_result_type="employee_list",
                priority=1
            ),
            # COMPLEX MULTI-TABLE
            FewShotExample(
                example_id="complex_001",
                category="complex",
                user_query_zh="查詢售服零件課的員工在9月的請假明細（已核准）",
                user_query_en="Leave details for Sales Parts Dept in September (approved)",
                sql_template=f"""WITH L AS (
    SELECT 
        CAST(ld.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        CAST(ld.LEAVEID AS NVARCHAR(100)) AS leave_id_norm,
        ld.WORKDATE,
        ld.HOURS,
        ld.DEPARTMENTID
    FROM eHRAntung_DB.dbo.ATDLEAVEDATA ld
    WHERE ld.VALIDATED = 1
),
P AS (
    SELECT 
        CAST(p.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        p.EMPLOYEEID AS employee_id,
        p.TRUENAME AS person_name
    FROM eHRAntung_DB.dbo.PSNACCOUNT p
),
C AS (
    SELECT 
        CAST(c.ID AS NVARCHAR(100)) AS leave_id_norm,
        c.CLASSNAME AS leave_type_name
    FROM eHRAntung_DB.dbo.ATDATTENDANCECLASS c
),
O AS (
    SELECT 
        CAST(o.UNITID AS NVARCHAR(100)) AS unit_id_norm,
        COALESCE(o.UNITDISPLAYNAME, o.UNITNAME) AS dept_name,
        o.UNITCODE AS dept_code
    FROM {ORG_TABLE} o
    WHERE ISNULL(o.ISDELETE, 0) = 0
)
SELECT
    P.employee_id,
    P.person_name,
    O.dept_name,
    O.dept_code,
    C.leave_type_name,
    L.WORKDATE,
    L.HOURS
FROM L
LEFT JOIN P ON L.person_id_norm = P.person_id_norm
LEFT JOIN C ON L.leave_id_norm = C.leave_id_norm
LEFT JOIN O ON CAST(L.DEPARTMENTID AS NVARCHAR(100)) = O.unit_id_norm
WHERE O.dept_name LIKE N'%售服零件課%'
  AND L.WORKDATE >= '{{{{CURRENT_YEAR}}}}-09-01'
  AND L.WORKDATE <  '{{{{CURRENT_YEAR}}}}-10-01'
ORDER BY L.WORKDATE, P.person_name;""",
                tables_used=["dbo.ATDLEAVEDATA", "dbo.PSNACCOUNT", "dbo.ATDATTENDANCECLASS", ORG_TABLE],
                join_pattern=f"ATDLEAVEDATA → PSNACCOUNT → ATDATTENDANCECLASS → {ORG_TABLE}",
                key_concepts=["department_filter","monthly_data","approved"],
                notes=f"Filter by department name via {ORG_TABLE}; use WORKDATE window and VALIDATED=1.",
                output_columns=["employee_id","person_name","dept_name","dept_code","leave_type_name","WORKDATE","HOURS"],
                expected_result_type="employee_list",
                priority=2
            ),
            # 6-MONTH TREND (approved)
            FewShotExample(
                example_id="trend_6m_approved_by_type",
                category="aggregation",
                user_query_zh="過去六個月請假趨勢（僅已核准），依假別分色",
                user_query_en="Past six months leave trend by leave type (approved only)",
                sql_template="""WITH L AS (
    SELECT
        CAST(ld.LEAVEID AS NVARCHAR(100)) AS leave_id_norm,
        CAST(ld.WORKDATE AS date) AS work_date,
        IIF(ld.VALIDATED = 1, 1, 0) AS is_validated
    FROM eHRAntung_DB.dbo.ATDLEAVEDATA ld
),
C AS (
    SELECT
        CAST(c.ID AS NVARCHAR(100)) AS leave_id_norm,
        c.CLASSNAME AS leave_type_name
    FROM eHRAntung_DB.dbo.ATDATTENDANCECLASS c
),
RANGE AS (
    SELECT
        CAST('{{ANCHOR_DATE}}' AS date) AS anchor_date,
        DATEADD(MONTH, DATEDIFF(MONTH, 0, CAST('{{ANCHOR_DATE}}' AS date)) - 5, 0) AS start_month,
        EOMONTH(CAST('{{ANCHOR_DATE}}' AS date)) AS end_month
)
SELECT
    DATEFROMPARTS(YEAR(L.work_date), MONTH(L.work_date), 1) AS month_start,
    C.leave_type_name,
    SUM(CASE WHEN L.is_validated = 1 THEN 1 ELSE 0 END) AS records
FROM L
CROSS JOIN RANGE r
LEFT JOIN C ON C.leave_id_norm = L.leave_id_norm
WHERE L.work_date >= r.start_month
  AND L.work_date <= r.end_month
GROUP BY DATEFROMPARTS(YEAR(L.work_date), MONTH(L.work_date), 1),
         C.leave_type_name
ORDER BY month_start, leave_type_name;""",
                tables_used=["dbo.ATDLEAVEDATA","dbo.ATDATTENDANCECLASS"],
                join_pattern="ATDLEAVEDATA.LEAVEID → ATDATTENDANCECLASS.ID",
                key_concepts=["six_month_trend","validated_only","group_by_month","leave_type"],
                notes="Compute window from the anchor month (inclusive) going back five full months; only approved records counted.",
                output_columns=["month_start","leave_type_name","records"],
                expected_result_type="time_series",
                priority=1
            ),
            # CANCELLATIONS
            FewShotExample(
                example_id="cancel_001",
                category="cancellation",
                user_query_zh="查詢本月的請假取消紀錄（顯示姓名）",
                user_query_en="Cancellation records this month (with names)",
                sql_template="""WITH cancel AS (
    SELECT 
        CAST(c.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        CAST(c.FORM_NO AS NVARCHAR(100)) AS form_no_norm,
        CAST(c.RECORD_ID AS NVARCHAR(100)) AS record_id_norm,
        c.OID,
        c.REASON AS cancel_reason,
        c.CREATEDATE AS cancel_createdate,
        CAST(c.WORKDATE AS date) AS WORKDATE
    FROM eHRAntung_DB.dbo.ATDLEAVECANCELDATA c
),
P AS (
    SELECT 
        CAST(p.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        p.EMPLOYEEID AS employee_id,
        p.TRUENAME AS person_name
    FROM eHRAntung_DB.dbo.PSNACCOUNT p
)
SELECT
    P.employee_id,
    P.person_name,
    c.form_no_norm AS form_no,
    c.record_id_norm AS record_id,
    c.cancel_reason,
    c.cancel_createdate,
    c.WORKDATE
FROM cancel c
LEFT JOIN P ON c.person_id_norm = P.person_id_norm
WHERE c.cancel_createdate >= DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()), 0)
  AND c.cancel_createdate <  DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()) + 1, 0)
ORDER BY c.cancel_createdate DESC, P.person_name;""",
                tables_used=["dbo.ATDLEAVECANCELDATA","dbo.PSNACCOUNT"],
                join_pattern="ATDLEAVECANCELDATA.PERSONID → PSNACCOUNT.PERSONID",
                key_concepts=["cancellation","reason","createdate"],
                notes="Filter by cancellation CREATEDATE. VALIDATED may not apply; list cancels regardless.",
                output_columns=["employee_id","person_name","form_no","record_id","cancel_reason","cancel_createdate","WORKDATE"],
                expected_result_type="employee_list",
                priority=2
            ),
        ]

    # ────────────────────────────────────────────────────────────────────────────
    # Health / persistence
    # ────────────────────────────────────────────────────────────────────────────
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
            "few_shot_examples": len(self._few_shot_examples),
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
            "few_shot_examples": [asdict(f) for f in self._few_shot_examples],
            "model_name": self.model_name,
            "created_at": datetime.now().isoformat(),
            "version": "6.1",
        }
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Enhanced Language-aware Vector DB saved to %s", self.db_path)

    @classmethod
    def load_from_disk(cls, db_path: str = "leave_schema_vectors.db") -> "LeaveVectorDB":
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"No vector DB at {db_path}")
        with open(db_path, "rb") as f:
            data = pickle.load(f)

        tables = [TableSchema(**d) for d in data.get("schemas", [])]
        for t in tables:
            t.full = _fq(t.full)

        joins = [TableJoin(**d) for d in data.get("joins", [])]
        patterns = [QueryPattern(**d) for d in data.get("patterns", [])]
        kpis = [KPIDef(**d) for d in data.get("kpis", [])]
        recipes = [SQLRecipe(**d) for d in data.get("recipes", [])]
        few_shot_examples = [FewShotExample(**d) for d in data.get("few_shot_examples", [])]

        return cls(
            tables=tables,
            db_path=db_path,
            model_name=data.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            joins=joins,
            patterns=patterns,
            kpis=kpis,
            recipes=recipes,
            few_shot_examples=few_shot_examples,
        )

    def get_query_pattern(self, query: str) -> Optional[QueryPattern]:
        q = (query or "").lower()
        if any(x in q for x in ["當前", "今天", "現在", "今日"]) or ("leave" in q and any(x in q for x in ["current", "today", "now"])):
            return next((p for p in self._query_patterns if p.pattern == "current_leave_status"), None)
        if any(x in q for x in ["取消", "已取消", "作廢"]) or any(x in q for x in ["cancel", "cancelled", "cancellation"]):
            return next((p for p in self._query_patterns if p.pattern == "leave_cancellations"), None)
        if "top 10" in q or "top10" in q or "前十" in q or "前10" in q:
            return next((p for p in self._query_patterns if p.pattern == "last_month_dept_top10"), None)
        return None

    def relationships_sanity_check(self) -> Dict[str, List[str]]:
        errors, warnings = [], []
        for j in self._joins:
            lt = self._by_name.get(_fq(j.left_table).lower())
            rt = self._by_name.get(_fq(j.right_table).lower())
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


# ---- PUBLIC BUILDER -----------------------------------------------------------
def build_leave_index() -> LeaveVectorDB:
    """Build the leave vector DB focusing on five core tables."""
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
          ["ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS","LEAVEID",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","STARTDATE","ENDDATE","FORM_NO","RECORD_ID"],
          desc="Primary leave fact table (use VALIDATED=1 for approved records).",
          description_zh="主要請假事實表（已核准請加 VALIDATED=1）。",
          tags=["leave","approved","validated","fact"],
          priority=1,
          business_context="Operational day-level leave records.",
          business_context_zh="日粒度之作業性請假紀錄。",
          common_queries=["Who is on leave today?","Top 10 departments by leave hours last month"],
          common_queries_zh=["今天有哪些員工在休假？","上月各部門請假時數 Top 10"],
          key_cols={
              "PERSONID": "Join to PSNACCOUNT for names",
              "LEAVEID": "Join to ATDATTENDANCECLASS for readable type",
              "DEPARTMENTID": "Join to ORGStdStruct",
              "WORKDATE": "Default date column for windows",
              "VALIDATED": "1 = approved, 0 = pending"
          },
          kpis=["total_leave_hours","net_leave_hours"]),
        T("dbo.ATDLEAVECANCELDATA",
          ["OID","ATTENDANCETYPE","PERSONID","WORKDATE","STARTDATE","STARTTIME","ENDDATE","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","REASON","LEAVEREASON","CREATEDATE","LASTEDITTIME","FORM_NO","RECORD_ID","FROM_SOURCE"],
          desc="Leave cancellation records (use to compute NET when matched).",
          description_zh="請假取消紀錄（與請假單以 FORM_NO/RECORD_ID 比對後方可扣除，計算『淨』時數）。",
          tags=["leave","cancel","reason"],
          priority=2,
          temporal=["WORKDATE","CREATEDATE","LASTEDITTIME"],
          kpis=["net_leave_hours"]),
        T("dbo.PSNACCOUNT",
          ["CARDNUM","TRUENAME","PERSONID","EMPLOYEEID","COMPANYEMAIL","BRANCHID","BUSINESSUNITID",
           "FIRSTNAME","MIDDLENAME","LASTNAME","ENGNAME"],
          desc="Person dimension (authoritative; includes BRANCHID for department).",
          description_zh="人員維度（權威來源；含 BRANCHID 以解析部門）。",
          tags=["person","dimension","branch"],
          key_cols={
              "PERSONID": "Join to facts",
              "EMPLOYEEID": "Employee visible id",
              "TRUENAME": "Person full name",
              "BRANCHID": f"Org key → {ORG_TABLE}.UNITID"
          },
          priority=1),
        T(ORG_TABLE,
          ["UNITID","UNITCODE","UNITNAME","UNITDISPLAYNAME","ISDELETE"],
          desc="Organization units (UNITID as key).",
          description_zh="組織單位（UNITID 為鍵）。",
          tags=["org","department","branch"],
          key_cols={
              "UNITID": f"Department key ← PSNACCOUNT.BRANCHID / FACT.DEPARTMENTID",
              "UNITNAME": "Department name",
              "UNITDISPLAYNAME": "Display name",
              "UNITCODE": "Department code"
          },
          priority=1),
        T("dbo.ATDATTENDANCECLASS",
          ["CLASSCODE","ID","CLASSNAME","CLASSTYPE"],
          desc="Attendance/leave class dictionary (maps LEAVEID → CLASSNAME).",
          description_zh="出勤/請假類別字典（LEAVEID → CLASSNAME）。",
          tags=["dictionary","leave-type"],
          key_cols={"ID": "LEAVEID ↔ ATDLEAVEDATA.LEAVEID"},
          priority=1),
    ]

    return LeaveVectorDB(tables)


__all__ = ["LeaveVectorDB", "TableSchema", "build_leave_index", "detect_language"]
