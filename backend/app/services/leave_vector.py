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
from typing import List, Dict, Tuple, Iterable, Set, Optional, Any

from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Optional deps: sentence-transformers + faiss
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # graceful fallback

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # graceful fallback

# ───────────────────────────────────────────────
# Enums & Core Models
# ───────────────────────────────────────────────

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
    condition: Optional[str] = None  # Use placeholders {left} and {right} for table names

    # Enriched metadata
    description: str = ""   # EN description of why this join exists
    description_zh: str = ""  # CN description
    purpose: str = ""       # human explanation (analytics intent)
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
    # Enriched metadata
    description_zh: str = ""
    examples: List[str] = field(default_factory=list)
    examples_zh: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class TableSchema:
    # Existing fields
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

    # Enriched metadata
    description_zh: str = ""
    business_context: str = ""
    business_context_zh: str = ""
    common_queries: List[str] = field(default_factory=list)
    common_queries_zh: List[str] = field(default_factory=list)
    key_columns: Dict[str, str] = field(default_factory=dict)  # col -> description
    relationships: List[str] = field(default_factory=list)
    row_count_estimate: str = ""  # "small", "medium", "large"
    priority: int = 1  # 1=high, 2=medium, 3=low
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
    formula_sql_hint: str = ""  # human-readable or templated SQL fragments
    tables: List[str] = field(default_factory=list)  # required tables
    grain: str = ""  # person-day, department-month, etc.
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
    key: str                 # unique identifier (e.g., table name or recipe_id)
    item_type: VectorItemType
    text: str                # combined text for embedding
    priority: int = 2        # 1=highest
    payload: Dict[str, Any] = field(default_factory=dict)  # pointer to the actual object


# ───────────────────────────────────────────────
# Helper Aliases
# ───────────────────────────────────────────────

COLUMN_ALIASES: Dict[str, Set[str]] = {
    "BUSINESSUNITID": {"BUSINESSUINTID"},
    "EFFECTIVEDATE": {"EFFINIENTDATE", "EFFICIENTDATE", "EFFDATE"},
    "TIMECARDDATE": {"CARDDATE"},
}

def _has_col(table: "TableSchema", name: str) -> bool:
    """True if table has the column or an alias."""
    cols = {c.upper() for c in table.columns}
    nameU = name.upper()
    if nameU in cols:
        return True
    for canonical, aliases in COLUMN_ALIASES.items():
        if nameU == canonical or nameU in {a.upper() for a in aliases}:
            if canonical in cols or any(a.upper() in cols for a in aliases):
                return True
    return False


# ───────────────────────────────────────────────
# Smart Leave Vector DB (multilingual, FAISS-backed)
# ───────────────────────────────────────────────

class LeaveVectorDB:
    """
    Multilingual, vector-backed knowledge store for leave/attendance schema.
    Indexes tables, joins, query patterns, KPIs, and SQL recipes.
    """

    def __init__(
        self,
        tables: List[TableSchema],
        db_path: str = "leave_schema_vectors.db",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        kpis: Optional[List[KPIDef]] = None,
        recipes: Optional[List[SQLRecipe]] = None,
        patterns: Optional[List[QueryPattern]] = None,
        joins: Optional[List[TableJoin]] = None,
    ):
        # Data
        self.tables = tables
        self._by_name: Dict[str, TableSchema] = {t.full.lower(): t for t in tables}
        self._person_table = self._resolve_person_table()

        # Config
        self.db_path = db_path
        self.model_name = model_name

        # Knowledge
        self._joins = joins if joins is not None else self._build_comprehensive_joins()
        self._query_patterns = patterns if patterns is not None else self._build_query_patterns()
        self._kpis = kpis if kpis is not None else self._build_kpis()
        self._recipes = recipes if recipes is not None else self._build_recipes()

        # Embedding infra
        self.model = None
        self.index = None
        self._vector_items: List[VectorItem] = []
        self._id2item: Dict[int, VectorItem] = {}
        self.embeddings: Optional[np.ndarray] = None

        # Build lookup tables and vector index
        self._load_model()
        self._build_vector_items()
        self._build_index()

    # ---------- Resolution helpers ----------

    def _resolve_person_table(self) -> Optional[str]:
        candidates = [
            "dbo.PSNACCOUNT_D",
            "dbo.BIPSNACCOUNTSP",
            "BIPSNACCOUNTSP",
        ]
        for name in candidates:
            if name.lower() in self._by_name:
                return name
        return None

    def _exists(self, full: str) -> bool:
        return full.lower() in self._by_name

    # ---------- Model & Index ----------

    def _load_model(self) -> None:
        """Load multilingual sentence transformer if available; else enable hashing-fallback."""
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not installed; using hashing-based fallback embeddings.")
            self.model = None
            return
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model: %s", self.model_name)
        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_name, e)
            self.model = None  # fallback to hashing

    def _combine_table_text(self, t: TableSchema) -> str:
        parts = [
            t.description, t.description_zh, t.business_context, t.business_context_zh,
            " ".join(t.tags),
            " ".join(t.common_queries),
            " ".join(t.common_queries_zh),
            " ".join([f"{k}:{v}" for k, v in t.key_columns.items()]),
            " ".join(t.relationships),
            " ".join(t.kpi_relevance),
            t.full,
            " ".join(t.columns)
        ]
        return " ".join([p for p in parts if p])

    def _combine_join_text(self, j: TableJoin) -> str:
        parts = [
            j.description, j.description_zh, j.purpose, " ".join(j.tags),
            j.left_table, j.right_table,
            f"{j.left_table}.{j.left_column}={j.right_table}.{j.right_column}",
            j.condition or "", j.join_type.value, j.cardinality.value
        ]
        return " ".join([p for p in parts if p])

    def _combine_pattern_text(self, p: QueryPattern) -> str:
        parts = [
            p.pattern, p.description, p.description_zh, " ".join(p.tags),
            " ".join(p.primary_tables), " ".join(p.suggested_joins),
            " ".join(p.required_filters), " ".join(p.performance_notes),
            " ".join(p.examples), " ".join(p.examples_zh)
        ]
        return " ".join([x for x in parts if x])

    def _combine_kpi_text(self, k: KPIDef) -> str:
        parts = [
            k.name, k.description, k.description_zh, " ".join(k.tags),
            " ".join(k.tables), k.formula_sql_hint, k.grain, k.interpretation
        ]
        return " ".join([p for p in parts if p])

    def _combine_recipe_text(self, r: SQLRecipe) -> str:
        parts = [
            r.title, r.description, r.description_zh, " ".join(r.tags),
            " ".join(r.tables), " ".join(r.expected_columns), r.sql_template
        ]
        return " ".join([p for p in parts if p])

    def _build_vector_items(self) -> None:
        self._vector_items = []
        self._id2item = {}

        # Tables
        for t in self.tables:
            text = self._combine_table_text(t)
            self._vector_items.append(VectorItem(
                key=t.full,
                item_type=VectorItemType.TABLE,
                text=text,
                priority=t.priority,
                payload={"table": t}
            ))

        # Joins
        for j in self._joins:
            text = self._combine_join_text(j)
            self._vector_items.append(VectorItem(
                key=f"JOIN::{j.left_table}::{j.right_table}::{j.left_column}::{j.right_column}",
                item_type=VectorItemType.JOIN,
                text=text,
                priority=2,
                payload={"join": j}
            ))

        # Patterns
        for p in self._query_patterns:
            text = self._combine_pattern_text(p)
            self._vector_items.append(VectorItem(
                key=f"PATTERN::{p.pattern}",
                item_type=VectorItemType.PATTERN,
                text=text,
                priority=1,
                payload={"pattern": p}
            ))

        # KPIs
        for k in self._kpis:
            text = self._combine_kpi_text(k)
            self._vector_items.append(VectorItem(
                key=f"KPI::{k.name}",
                item_type=VectorItemType.KPI,
                text=text,
                priority=1,
                payload={"kpi": k}
            ))

        # Recipes
        for r in self._recipes:
            text = self._combine_recipe_text(r)
            self._vector_items.append(VectorItem(
                key=f"RECIPE::{r.recipe_id}",
                item_type=VectorItemType.RECIPE,
                text=text,
                priority=1,
                payload={"recipe": r}
            ))

        logger.info("Vector items built: %d", len(self._vector_items))

    # --------- Embedding fallbacks ---------

    @staticmethod
    def _hashing_embed(texts: List[str], dim: int = 2048) -> np.ndarray:
        """
        Very light-weight hashing-based bag-of-words embedding.
        Provides deterministic, language-agnostic vectors as a fallback.
        """
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                idx = hash(tok) % dim
                out[i, idx] += 1.0
            # L2 normalize
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out

    def _build_index(self) -> None:
        texts = [vi.text for vi in self._vector_items]
        if not texts:
            logger.warning("No texts to index.")
            return

        # Create embeddings
        if self.model is not None:
            try:
                emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                self.embeddings = emb.astype("float32")
                logger.info("Built model embeddings: shape=%s", self.embeddings.shape)
            except Exception as e:
                logger.error("Embedding error; falling back to hashing. %s", e)
                self.embeddings = self._hashing_embed(texts)
        else:
            self.embeddings = self._hashing_embed(texts)

        # Build FAISS / numpy index
        if faiss is not None:
            try:
                # Use Inner Product for cosine with normalized embeddings
                dim = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(self.embeddings)
                logger.info("Built FAISS index: %d items, dim=%d", len(self._vector_items), dim)
            except Exception as e:
                logger.error("FAISS error; using numpy index. %s", e)
                self.index = None
        else:
            self.index = None  # numpy fallback

        # id -> item mapping
        self._id2item = {i: vi for i, vi in enumerate(self._vector_items)}

    # ───────────────────────────────────────────────
    # Knowledge construction (joins/patterns/kpis/recipes)
    # ───────────────────────────────────────────────

    def _build_comprehensive_joins(self) -> List[TableJoin]:
        joins: List[TableJoin] = []

        leave_core = [
            "dbo.ATDLEAVEDATA",
            "dbo.ATDHISLEAVEDATA",
            "dbo.ATDLEAVEDATA_D",
            "dbo.ATDHISLEAVEDATA_D",
            "dbo.ATDLEAVEDATA_T",
            "dbo.ATDLEAVEDATAEX",
            "dbo.ATDLEAVEDATAEX_D",
            "dbo.ATDLEAVECANCELDATA",
            "dbo.ATDNONCALCULATEDVACATION",
            "dbo.ATDHISNONCALCULATEDVACATION",
            "dbo.ATDNONCALCULATEDVACATION_D",
            "dbo.EDFATDLEAVEDATA",
        ]

        # Person joins (M:1) for any table that has PERSONID
        if self._person_table:
            for lt in leave_core + [
                "dbo.ATDHISLATEEARLY",
                "dbo.ATDHISTIMECARDDATA",
                "dbo.ATDHISNOTIMECARD",
                "dbo.ATDRESULTDATAIMPORT",
            ]:
                if self._exists(lt) and _has_col(self._by_name[lt.lower()], "PERSONID"):
                    joins.append(TableJoin(
                        left_table=lt, left_column="PERSONID",
                        right_table=self._person_table, right_column="PERSONID",
                        join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                        description="Resolve PERSONID to person dimension for names/attributes",
                        description_zh="将 PERSONID 关联到人员维度以获取姓名/属性",
                        purpose="Show employee names, departments, and attributes alongside leave rows",
                        tags=["person", "dimension", "lookup"]
                    ))

        # LeaveEX ↔ Vacation balance
        if self._exists("dbo.ATDLEAVEDATAEX") and self._exists("dbo.ATDNONCALCULATEDVACATION"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVEDATAEX", left_column="VACATIONID",
                right_table="dbo.ATDNONCALCULATEDVACATION", right_column="OID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                description="Link leave accounting with current vacation balance",
                description_zh="关联请假核算与当前假期余额",
                purpose="Reconcile used/remaining balances per person and vacation type",
                tags=["balance", "reconciliation"]
            ))

        # Leave ↔ LeaveEX
        if self._exists("dbo.ATDLEAVEDATA") and self._exists("dbo.ATDLEAVEDATAEX"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVEDATA", left_column="LEAVEID",
                right_table="dbo.ATDLEAVEDATAEX", right_column="LEAVEID",
                join_type=JoinType.LEFT, cardinality=Cardinality.ONE_TO_ONE,
                description="Enrich live leave rows with extended accounting fields",
                description_zh="使用扩展核算字段丰富当前请假记录",
                purpose="Add MINUSDAYS / VACATIONID context to validated leave",
                tags=["enrichment"]
            ))

        # Historical/Deleted ↔ Original by LEAVEID (audit trace)
        if self._exists("dbo.ATDHISLEAVEDATA_D") and self._exists("dbo.ATDHISLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.ATDHISLEAVEDATA_D", left_column="LEAVEID",
                right_table="dbo.ATDHISLEAVEDATA", right_column="LEAVEID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                description="Trace deleted historical leave to original historical rows",
                description_zh="将已删除的历史请假记录追溯到原始历史记录",
                purpose="Audit reporting",
                tags=["audit", "history"]
            ))
        if self._exists("dbo.ATDLEAVEDATA_D") and self._exists("dbo.ATDLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVEDATA_D", left_column="LEAVEID",
                right_table="dbo.ATDLEAVEDATA", right_column="LEAVEID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                description="Trace deleted current leave to original",
                description_zh="将已删除的当前请假记录追溯到原始记录",
                purpose="Audit reporting",
                tags=["audit"]
            ))

        # Leave ↔ Cancel (composite ON)
        if self._exists("dbo.ATDLEAVECANCELDATA") and self._exists("dbo.ATDLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVECANCELDATA", left_column="PERSONID",
                right_table="dbo.ATDLEAVEDATA", right_column="PERSONID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                condition=(
                    "({left}.ATTENDANCETYPE = {right}.ATTENDANCETYPE "
                    "AND {left}.STARTDATE = {right}.STARTDATE "
                    "AND {left}.ENDDATE = {right}.ENDDATE)"
                ),
                description="Map cancellations back to original requests by person/type/date window",
                description_zh="按人员/考勤类型/起止日期将取消记录映射到原始申请",
                purpose="Analyze cancellation reasons and rates",
                tags=["cancel"]
            ))

        # Leave ↔ Legal calendar
        if self._exists("dbo.ATDLEAVEDATA") and self._exists("dbo.ATDLEGALCALENDAR"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVEDATA", left_column="WORKDATE",
                right_table="dbo.ATDLEGALCALENDAR", right_column="CALENDARDATE",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                description="Mark leave occurrences on legal holidays",
                description_zh="识别请假日期是否法定节假日",
                purpose="Policy validation / reporting",
                tags=["calendar", "holiday"]
            ))

        # Dept-day operation state ↔ Leave
        if self._exists("dbo.ATDDEPTOPERSTATE") and self._exists("dbo.ATDLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVEDATA", left_column="DEPARTMENTID",
                right_table="dbo.ATDDEPTOPERSTATE", right_column="DEPARTMENTID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                condition="({left}.WORKDATE = {right}.WORKDATE)",
                description="Align leave with dept-day calculation/validation checkpoints",
                description_zh="将请假与部门日计算/校验状态对齐",
                purpose="Troubleshoot miscalculations and processing windows",
                tags=["ops", "validation"]
            ))

        # EDF external leave ↔ Leave
        if self._exists("dbo.EDFATDLEAVEDATA") and self._exists("dbo.ATDLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.EDFATDLEAVEDATA", left_column="PERSONID",
                right_table="dbo.ATDLEAVEDATA", right_column="PERSONID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                condition="( {right}.WORKDATE BETWEEN {left}.WORKDATES AND {left}.WORKDATEE )",
                description="Overlap EDF-imported leave windows with actual leave by person",
                description_zh="按人员将 EDF 导入的请假时间窗与实际请假重叠匹配",
                purpose="Reconciliation of external vs internal sources",
                tags=["edf", "integration"]
            ))

        # Monthly import results ↔ Leave by PERSONID + ATTENDANCETYPE
        if self._exists("dbo.ATDRESULTDATAIMPORT") and self._exists("dbo.ATDLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.ATDRESULTDATAIMPORT", left_column="PERSONID",
                right_table="dbo.ATDLEAVEDATA", right_column="PERSONID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                condition="({left}.ATTENDANCETYPE = {right}.ATTENDANCETYPE)",
                description="Relate aggregated monthly results to detailed leave rows",
                description_zh="将月度汇总结果与明细请假记录关联",
                purpose="Drill-through from monthly KPIs to row-level evidence",
                tags=["rollup", "drillthrough"]
            ))

        return joins

    def _build_query_patterns(self) -> List[QueryPattern]:
        return [
            QueryPattern(
                pattern="current_leave_status",
                description="Who is currently on leave (approved)",
                description_zh="当前正在休假的人员（已批准）",
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
                examples=[
                    "Which employees are on leave today by department?",
                    "Count current sick leaves by business unit"
                ],
                examples_zh=[
                    "今天各部门有哪些员工在休假？",
                    "按事业部统计当前病假人数"
                ],
                tags=["current", "today", "validated"]
            ),
            QueryPattern(
                pattern="upcoming_leave",
                description="Who will be on leave in a future window",
                description_zh="未来一段时间将要休假的人员",
                primary_tables=["dbo.ATDLEAVEDATA"],
                suggested_joins=["PERSONID"],
                required_filters=["STARTDATE >= CAST(GETDATE() AS date)", "VALIDATED = 1"],
                performance_notes=["Index on STARTDATE, ENDDATE"],
                examples=["Show next week's planned leave by type"],
                examples_zh=["展示下周按类型的请假计划"],
                tags=["upcoming", "future"]
            ),
            QueryPattern(
                pattern="historical_leave_analysis",
                description="Analyze leave patterns over time periods",
                description_zh="按时间段分析请假模式",
                primary_tables=["dbo.ATDHISLEAVEDATA"],
                suggested_joins=["PERSONID"],
                required_filters=["Date range filter required (WORKDATE/STARTDATE/ENDDATE)"],
                performance_notes=["Historical tables are large — always filter by date range"],
                examples=["Trend of annual leave hours by month for the past year"],
                examples_zh=["过去一年每月年假小时趋势"],
                tags=["history", "trend"]
            ),
            QueryPattern(
                pattern="leave_balance_reconciliation",
                description="Reconcile used/remaining leave with vacation balance",
                description_zh="对账已用/剩余请假与假期余额",
                primary_tables=["dbo.ATDLEAVEDATAEX", "dbo.ATDNONCALCULATEDVACATION"],
                suggested_joins=["PERSONID", "VACATIONID"],
                required_filters=["VACATIONTYPE filter recommended",
                                  "WORKDATE or STARTDATE/ENDDATE window"],
                performance_notes=["Join VACATIONID ↔ OID; group by PERSONID, VACATIONTYPE"],
                examples=["Remaining annual leave days per person"],
                examples_zh=["每人剩余年假天数"],
                tags=["balance", "reconciliation"]
            ),
            QueryPattern(
                pattern="leave_cancellations",
                description="Cancelled leave requests and reasons",
                description_zh="已取消的请假申请及原因",
                primary_tables=["dbo.ATDLEAVECANCELDATA"],
                suggested_joins=["PERSONID"],
                required_filters=["WORKDATE range filter"],
                performance_notes=["Filter VALIDATED if needed; use REASON/LEAVEREASON"],
                examples=["Cancellation count and reasons last quarter"],
                examples_zh=["上季度取消请假的数量与原因"],
                tags=["cancel", "reason"]
            ),
        ]

    def _build_kpis(self) -> List[KPIDef]:
        return [
            KPIDef(
                name="total_leave_hours",
                description="Total approved leave hours in a period",
                description_zh="某时间段内已批准请假总小时数",
                formula_sql_hint="SUM(HOURS) WHERE VALIDATED=1 AND date filter",
                tables=["dbo.ATDLEAVEDATA"],
                grain="department-day / person-day",
                interpretation="Higher values may indicate seasonal effects or issues",
                tags=["volume", "hours"]
            ),
            KPIDef(
                name="absence_rate",
                description="Share of employees on leave on a given day",
                description_zh="某日请假率（在休假员工占比）",
                formula_sql_hint="COUNT(distinct PERSONID on leave) / headcount",
                tables=["dbo.ATDLEAVEDATA", "dbo.BIPSNACCOUNTSP"],
                grain="businessunit-day",
                interpretation="Use for capacity planning",
                tags=["rate", "capacity"]
            ),
            KPIDef(
                name="balance_utilization",
                description="Used leave vs allocated balance by type",
                description_zh="已用假期与分配余额的对比（按类型）",
                formula_sql_hint="SUM(MINUSDAYS) vs REMAINDAYS by VACATIONTYPE",
                tables=["dbo.ATDLEAVEDATAEX", "dbo.ATDNONCALCULATEDVACATION"],
                grain="person-vacationtype",
                interpretation="Identify overdraw risk or underuse",
                tags=["balance"]
            ),
        ]

    def _build_recipes(self) -> List[SQLRecipe]:
        return [
            SQLRecipe(
                recipe_id="current_on_leave_by_dept",
                title="Current on-leave employees by department",
                description="Lists employees currently on validated leave grouped by department.",
                description_zh="按部门列出当前已批准且在休假的员工。",
                tables=["dbo.ATDLEAVEDATA"],
                expected_columns=["DEPARTMENTID", "PERSONID", "ATTENDANCETYPE", "STARTDATE", "ENDDATE"],
                sql_template=(
                    "SELECT DEPARTMENTID, PERSONID, ATTENDANCETYPE, STARTDATE, ENDDATE, HOURS "
                    "FROM dbo.ATDLEAVEDATA "
                    "WHERE VALIDATED = 1 "
                    "AND CAST(GETDATE() AS date) BETWEEN CAST(STARTDATE AS date) AND CAST(ENDDATE AS date);"
                ),
                caution_notes=["Consider group by PERSONID to collapse multiple rows"],
                tags=["current", "validated"]
            ),
            SQLRecipe(
                recipe_id="leave_hours_trend",
                title="Monthly leave hours trend (historical)",
                description="Sums leave hours by month over a given range.",
                description_zh="在给定时间范围内，按月汇总请假小时数。",
                tables=["dbo.ATDHISLEAVEDATA"],
                expected_columns=["BELONG_MONTH", "TOTAL_HOURS"],
                sql_template=(
                    "SELECT FORMAT(WORKDATE,'yyyy-MM') AS BELONG_MONTH, SUM(HOURS) AS TOTAL_HOURS "
                    "FROM dbo.ATDHISLEAVEDATA "
                    "WHERE WORKDATE BETWEEN @start_date AND @end_date "
                    "AND VALIDATED = 1 "
                    "GROUP BY FORMAT(WORKDATE,'yyyy-MM') "
                    "ORDER BY BELONG_MONTH;"
                ),
                variables={"@start_date": "YYYY-MM-01", "@end_date": "YYYY-MM-31"},
                tags=["trend", "history"]
            ),
            SQLRecipe(
                recipe_id="balance_reconciliation",
                title="Balance reconciliation by person/type",
                description="Reconciles ATDLEAVEDATAEX to current balances.",
                description_zh="将 ATDLEAVEDATAEX 与当前假期余额进行对账。",
                tables=["dbo.ATDLEAVEDATAEX", "dbo.ATDNONCALCULATEDVACATION"],
                expected_columns=["PERSONID", "VACATIONTYPE", "USED_DAYS", "REMAINDAYS"],
                sql_template=(
                    "SELECT ex.PERSONID, vac.VACATIONTYPE, SUM(ex.MINUSDAYS) AS USED_DAYS, MAX(vac.REMAINDAYS) AS REMAINDAYS "
                    "FROM dbo.ATDLEAVEDATAEX ex "
                    "LEFT JOIN dbo.ATDNONCALCULATEDVACATION vac ON ex.VACATIONID = vac.OID "
                    "WHERE ex.WORKDATE BETWEEN @start_date AND @end_date "
                    "GROUP BY ex.PERSONID, vac.VACATIONTYPE;"
                ),
                variables={"@start_date": "YYYY-MM-01", "@end_date": "YYYY-MM-31"},
                tags=["balance", "reconciliation"]
            ),
        ]

    # ───────────────────────────────────────────────
    # Health & Persistence
    # ───────────────────────────────────────────────

    def is_ready(self) -> bool:
        return bool(self.tables) and self.embeddings is not None

    def health_check(self) -> Dict[str, object]:
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
            "faiss_used": faiss is not None and isinstance(self.index, (faiss.IndexFlat, faiss.IndexFlatIP)),
            "embeddings_shape": tuple(self.embeddings.shape) if self.embeddings is not None else None,
            "model_name": self.model_name,
            "db_path": self.db_path
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
            "version": "3.0",
        }
        # Note: embeddings are rebuilt on load to avoid cross-version mismatch
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Vector DB saved to %s", self.db_path)

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
            model_name=data.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2"),
            joins=joins,
            patterns=patterns,
            kpis=kpis,
            recipes=recipes,
        )
        return inst

    # ───────────────────────────────────────────────
    # Search
    # ───────────────────────────────────────────────

    def _numpy_search(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # Cosine similarity assuming embeddings are normalized
        sims = self.embeddings @ query_vec.T  # shape (N, 1)
        idxs = np.argsort(-sims.squeeze())[:k]
        return sims[idxs], idxs

    def _encode_query(self, query: str) -> np.ndarray:
        if self.model is not None:
            try:
                q = self.model.encode([query], normalize_embeddings=True)
                return q.astype("float32")
            except Exception as e:
                logger.error("Query embedding error; falling back to hashing. %s", e)
        return self._hashing_embed([query])

    def _boost_score(self, vi: VectorItem, sim: float, query: str) -> float:
        # Base score
        score = float(sim)

        # Priority boost (1 is highest priority)
        score *= (1.0 + (4 - vi.priority) * 0.1)

        # KPI keyword boost
        ql = query.lower()
        if vi.item_type == VectorItemType.TABLE:
            t: TableSchema = vi.payload["table"]
            for k in t.kpi_relevance:
                if k.lower() in ql:
                    score *= 1.15

        if vi.item_type == VectorItemType.KPI:
            k: KPIDef = vi.payload["kpi"]
            if k.name.lower() in ql:
                score *= 1.2
            for tag in k.tags:
                if tag in ql:
                    score *= 1.05

        return score

    def search(self, query: str, top_k: int = 8, min_score: float = 0.25) -> List[Tuple[VectorItem, float]]:
        if self.embeddings is None or not self._vector_items:
            return []

        qvec = self._encode_query(query).astype("float32")
        k = min(top_k * 3, len(self._vector_items))

        if self.index is not None and faiss is not None:
            # FAISS IP search
            distances, indices = self.index.search(qvec, k)  # distances are inner product sims
            sims = distances[0]
            idxs = indices[0]
        else:
            sims, idxs = self._numpy_search(qvec, k)

        results: List[Tuple[VectorItem, float]] = []
        for sim, idx in zip(sims, idxs):
            vi = self._id2item[int(idx)]
            weighted = self._boost_score(vi, float(sim), query)
            if weighted >= min_score:
                results.append((vi, weighted))

        # Dedup by key, keep best score
        dedup: Dict[str, Tuple[VectorItem, float]] = {}
        for vi, s in results:
            if vi.key not in dedup or s > dedup[vi.key][1]:
                dedup[vi.key] = (vi, s)

        out = sorted(dedup.values(), key=lambda x: x[1], reverse=True)
        return out[:top_k]

    # Convenience: table-only search (compatible with your previous usage)
    def search_relevant_tables(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        mixed = self.search(query, top_k=top_k * 2)
        tables: List[Tuple[str, float]] = []
        for vi, s in mixed:
            if vi.item_type == VectorItemType.TABLE:
                tables.append((vi.payload["table"].full, s))
            if len(tables) >= top_k:
                break
        return tables

    # ───────────────────────────────────────────────
    # Context building & Hints
    # ───────────────────────────────────────────────

    def _select_joins_for_tables(self, tables: Iterable[str]) -> List[TableJoin]:
        tset = {t.lower() for t in tables}
        selected = []
        for j in self._joins:
            if j.left_table.lower() in tset and j.right_table.lower() in tset:
                selected.append(j)
        return selected

    def join_hints(self, tables: Iterable[str]) -> List[str]:
        table_set = {t.lower() for t in tables}
        hints: List[str] = []

        # Filter applicable joins
        for j in self._joins:
            if j.left_table.lower() in table_set and j.right_table.lower() in table_set:
                hints.append(j.on_clause())

        # Perf notes
        for tname in table_set:
            t = self._by_name.get(tname)
            if t and t.row_estimate and t.row_estimate > 100_000:
                hints.append(f"-- Performance: filter {t.full} by date range when possible")

        # Deduplicate preserve order
        return list(dict.fromkeys(hints))

    def get_query_pattern(self, query: str) -> Optional[QueryPattern]:
        # Retain light keyword fallback; primary selection is vector-based elsewhere
        q = query.lower()
        if "leave" in q and any(x in q for x in ["current", "today", "now"]):
            return next((p for p in self._query_patterns if p.pattern == "current_leave_status"), None)
        if "leave" in q and any(x in q for x in ["future", "upcoming", "next"]):
            return next((p for p in self._query_patterns if p.pattern == "upcoming_leave"), None)
        if "leave" in q and any(x in q for x in ["history", "historical", "trend", "past"]):
            return next((p for p in self._query_patterns if p.pattern == "historical_leave_analysis"), None)
        if any(x in q for x in ["balance", "vacation", "remaining"]):
            return next((p for p in self._query_patterns if p.pattern == "leave_balance_reconciliation"), None)
        if any(x in q for x in ["cancel", "cancelled", "cancellation"]):
            return next((p for p in self._query_patterns if p.pattern == "leave_cancellations"), None)
        return None

    def get_schema_context(self, query: str, include_examples: bool = True) -> str:
        """
        Returns a compact, bilingual context for an LLM including:
        - top tables
        - applicable joins
        - relevant query patterns
        - KPI hints
        - few-shot SQL recipes
        """
        ranked = self.search(query, top_k=10)

        # Partition by type
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
        lines.append("=== RELEVANT DATABASE CONTEXT / 相关数据库上下文 ===")
        lines.append(f"Query / 查询: '{query}'\n")

        for i, t in enumerate(top_tables, 1):
            lines.extend([
                f"[{i}] TABLE 表: {t.full}",
                f"  Desc/描述: {t.description or ''}",
                f"  业务/Business: {t.business_context or ''}",
                f"  Data Volume/数据量: {t.row_count_estimate or (('large' if (t.row_estimate or 0) > 200000 else 'medium') if t.row_estimate else '')}",
            ])
            if t.key_columns:
                lines.append("  Key Columns/关键字段:")
                for c, d in list(t.key_columns.items())[:8]:
                    lines.append(f"    • {c}: {d}")
            if t.kpi_relevance:
                lines.append(f"  KPIs: {', '.join(t.kpi_relevance)}")
            if t.relationships:
                lines.append(f"  Related/关联: {', '.join(t.relationships)}")
            if t.common_queries and include_examples:
                lines.append("  Examples/示例:")
                for ex in t.common_queries[:2]:
                    lines.append(f"    - {ex}")
            lines.append("")

        if join_strs:
            lines.append("=== Suggested Joins / 推荐关联 ===")
            for j in join_strs[:6]:
                lines.append(j)
            lines.append("")

        if top_patterns:
            lines.append("=== Relevant Patterns / 相关查询模式 ===")
            for p in top_patterns:
                lines.append(f"- {p.pattern}: {p.description}")
            lines.append("")

        if top_kpis:
            lines.append("=== KPI Hints / KPI 提示 ===")
            for k in top_kpis:
                lines.append(f"- {k.name}: {k.description} (grain: {k.grain})")
            lines.append("")

        if top_recipes:
            lines.append("=== Canonical SQL Recipes / 经典SQL模版 ===")
            for r in top_recipes:
                lines.append(f"- {r.title}: {r.description}")
                if r.variables:
                    lines.append(f"  Variables: {', '.join(r.variables.keys())}")
            lines.append("")

        lines.extend([
            "=== Query Construction Tips / 构建建议 ===",
            "• Filter historical tables by date range (历史大表务必加日期过滤)",
            "• Include VALIDATED=1 for approved leave (仅统计已批准假期)",
            "• Use PERSON dimension only when needed to avoid unnecessary joins",
            "• Consider grouping by PERSONID/DEPARTMENTID for rollups",
        ])

        return "\n".join(lines)

    def get_business_prompt(self, query: str) -> str:
        context = self.get_schema_context(query, include_examples=True)
        prompt = f"""
You are an expert analytics engineer for a leave/attendance domain. Provide SQL and reasoning that
produce business-ready answers, not just raw data.

USER QUERY:
{query}

{context}

REQUIREMENTS:
1) Generate correct SQL with proper JOINs and filters (VALIDATED=1 where appropriate).
2) Add a sensible date window if the query implies time context.
3) Include aggregations and derived metrics for KPIs when relevant.
4) Mention performance tips (indexes, range predicates) as comments.
5) If the result may be empty on current tables, suggest switching to historical tables.

Return:
- Short reasoning bullets
- SQL
- Expected columns and grain
- Optional follow-up/validation checks
"""
        return prompt.strip()

    # ───────────────────────────────────────────────
    # Validation
    # ───────────────────────────────────────────────

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


# ───────────────────────────────────────────────
# Index Builder (your concrete leave schema)
# ───────────────────────────────────────────────

def build_leave_index() -> LeaveVectorDB:
    """
    Builds the enriched index with your actual tables.
    """

    def T(full: str, cols: List[str], desc: str = "", tags: List[str] = None,
          pks: List[str] = None, indexed: List[str] = None, rows: int = None,
          is_hist: bool = False, is_del: bool = False, temporal: List[str] = None,
          business_context: str = "", common_queries: List[str] = None, key_cols: Dict[str, str] = None,
          relationships: List[str] = None, kpis: List[str] = None, priority: int = 1,
          description_zh: str = "", business_context_zh: str = "", common_queries_zh: List[str] = None,
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
        # Leave core
        T("dbo.ATDLEAVEDATA",
          ["ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","STARTDATE","ENDDATE","AutoRevise",
           "TIMECLASSID","TIMECLASSHOURS","CREATIONTIME","CREATEDBY","LASTUPDATETIME",
           "LASTUPDATEDBY","LEAVEID","FORMKIND","FROM_SOURCE","FORM_NO","RECORD_ID",
           "FLEAVEBYDAYTYPE","TLEAVEBYDAYTYPE","LEAVEREASON"],
          desc="Current validated leave data",
          description_zh="当前已验证的请假数据",
          tags=["leave","current","validated"],
          pks=["LEAVEID"], indexed=["PERSONID","STARTDATE","ENDDATE"], rows=50_000,
          temporal=["WORKDATE","STARTDATE","ENDDATE"],
          business_context="Operational view of approved leave used for day-of reporting and capacity checks.",
          business_context_zh="用于当日报表和产能检查的已批准请假数据。",
          common_queries=[
              "Who is on leave today by department?",
              "Total approved leave hours this week by type"
          ],
          common_queries_zh=["今天各部门谁在休假？","本周按类型统计已批准的请假小时数"],
          key_cols={"VALIDATED":"1=approved","HOURS":"Leave hours","ATTENDANCETYPE":"Leave type"},
          relationships=["dbo.BIPSNACCOUNTSP","dbo.ATDLEGALCALENDAR","dbo.ATDDEPTOPERSTATE","dbo.ATDLEAVEDATAEX"],
          kpis=["total_leave_hours","absence_rate"],
          priority=1
        ),

        T("dbo.ATDHISLEAVEDATA",
          ["LEAVEID","ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","STARTDATE","ENDDATE","AutoRevise",
           "TIMECLASSID","TIMECLASSHOURS","CREATIONTIME","CREATEDBY","LASTUPDATETIME",
           "LASTUPDATEDBY","FORMKIND","FROM_SOURCE","FORM_NO","RECORD_ID","RECORD_USERID",
           "RECORD_DATE","FLEAVEBYDAYTYPE","TLEAVEBYDAYTYPE","LEAVEREASON"],
          desc="Historical leave data",
          description_zh="历史请假数据",
          tags=["leave","history"], pks=["LEAVEID"], indexed=["PERSONID","WORKDATE"],
          rows=500_000, is_hist=True, temporal=["WORKDATE","STARTDATE","ENDDATE"],
          business_context="Trend analysis and historical KPI computation.",
          business_context_zh="趋势分析和历史 KPI 计算。",
          common_queries=["Monthly leave hours trend", "Year-over-year leave by type"],
          common_queries_zh=["按月趋势的请假小时数","按类型同比比较"],
          key_cols={"WORKDATE":"Event date","VALIDATED":"Approval flag"},
          relationships=["dbo.BIPSNACCOUNTSP"],
          kpis=["total_leave_hours"],
          priority=1
        ),

        T("dbo.ATDLEAVEDATA_D",
          ["DELETEID","ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","STARTDATE","ENDDATE","AUTOREVISE",
           "TIMECLASSID","TIMECLASSHOURS","CREATIONTIME","CREATEDBY","LASTUPDATETIME",
           "LASTUPDATEDBY","LEAVEID","DELETE_USER_ID","DELETE_TIME","FORMKIND","FROM_SOURCE",
           "FORM_NO","RECORD_ID","FLEAVEBYDAYTYPE","TLEAVEBYDAYTYPE","LEAVEREASON"],
          desc="Deleted leave data (audit)",
          description_zh="已删除的请假数据（审计）",
          tags=["leave","deleted","audit"], pks=["DELETEID"], indexed=["PERSONID","DELETE_TIME"],
          rows=25_000, is_del=True, temporal=["WORKDATE","DELETE_TIME"],
          business_context="Audit and compliance reporting.",
          business_context_zh="审计与合规报表。",
          relationships=["dbo.ATDLEAVEDATA"],
          priority=3
        ),

        T("dbo.ATDHISLEAVEDATA_D",
          ["DELETEID","ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","STARTDATE","ENDDATE","AUTOREVISE",
           "TIMECLASSID","TIMECLASSHOURS","CREATIONTIME","CREATEDBY","LASTUPDATETIME",
           "LASTUPDATEDBY","LEAVEID","DELETE_USER_ID","DELETE_TIME","FORMKIND","FROM_SOURCE",
           "FORM_NO","RECORD_ID","RECORD_USERID","RECORD_DATE","FLEAVEBYDAYTYPE",
           "TLEAVEBYDAYTYPE","LEAVEREASON"],
          desc="Deleted historical leave (audit)",
          description_zh="已删除的历史请假（审计）",
          tags=["leave","deleted","audit"], pks=["DELETEID"], indexed=["PERSONID","DELETE_TIME"],
          rows=25_000, is_del=True, temporal=["WORKDATE","DELETE_TIME"],
          business_context="Historical audit trails.",
          business_context_zh="历史审计追踪。",
          relationships=["dbo.ATDHISLEAVEDATA"],
          priority=3
        ),

        T("dbo.ATDLEAVEDATA_T",
          ["ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "BUSINESSUNITID","STARTDATE","ENDDATE","LEAVEID","TRANSFERUNITID","LEAVEREASON"],
          desc="Transferred leave across units",
          description_zh="跨单位转移的请假记录",
          tags=["leave","transferred"], pks=["LEAVEID"], indexed=["PERSONID","TRANSFERUNITID"],
          rows=5_000, temporal=["WORKDATE"],
          business_context="Inter-unit transfer tracking.",
          business_context_zh="跨单位转移跟踪。",
          relationships=["dbo.ATDLEAVEDATA"],
          priority=2
        ),

        T("dbo.ATDLEAVEDATAEX",
          ["ATTENDANCETYPE","PERSONID","WORKDATE","HOURS","BUSINESSUNITID","VACATIONID",
           "MINUSDAYS","LEAVETYPE","STARTTIME","ENDTIME","STARTDATE","ENDDATE",
           "TIMECLASSID","TIMECLASSHOURS","LEAVEID"],
          desc="Extended leave accounting / vacation linkage",
          description_zh="扩展请假核算与假期余额关联",
          tags=["leave","extended","balance"], pks=["LEAVEID"], indexed=["PERSONID","VACATIONID"],
          rows=50_000, temporal=["WORKDATE"],
          business_context="Reconciliation between leave usage and balances.",
          business_context_zh="请假使用与余额之间的对账。",
          relationships=["dbo.ATDNONCALCULATEDVACATION","dbo.ATDLEAVEDATA"],
          kpis=["balance_utilization"],
          priority=1
        ),

        T("dbo.ATDLEAVEDATAEX_D",
          ["DELETEID","ATTENDANCETYPE","PERSONID","WORKDATE","HOURS","BUSINESSUNITID",
           "VACATIONID","MINUSDAYS","LEAVETYPE","STARTTIME","ENDTIME","STARTDATE","ENDDATE",
           "TIMECLASSID","TIMECLASSHOURS","LEAVEID","DELETE_USER_ID","DELETE_TIME"],
          desc="Deleted extended leave (audit)",
          description_zh="已删除的扩展请假（审计）",
          tags=["leave","extended","deleted","audit"], pks=["DELETEID"], indexed=["PERSONID","DELETE_TIME"],
          rows=10_000, is_del=True, temporal=["WORKDATE","DELETE_TIME"],
          business_context="Audit for extended accounting records.",
          business_context_zh="扩展核算记录的审计。",
          relationships=["dbo.ATDLEAVEDATAEX"],
          priority=3
        ),

        T("dbo.ATDLEAVECANCELDATA",
          ["OID","ATTENDANCETYPE","PERSONID","WORKDATE","STARTDATE","STARTTIME","ENDDATE",
           "ENDTIME","HOURS","DEPARTMENTID","VALIDATED","BUSINESSUNITID","ISHISDATA",
           "AutoRevise","CREATEDATE","CREATEUSERID","LASTEDITUSERID","LASTEDITTIME","REASON",
           "FLEAVEBYDAYTYPE","TLEAVEBYDAYTYPE","LEAVEREASON","FROM_SOURCE","FORM_NO",
           "RECORD_ID","FORMKIND"],
          desc="Cancelled leave requests",
          description_zh="已取消的请假申请",
          tags=["leave","cancelled"], pks=["OID"], indexed=["PERSONID","WORKDATE"],
          rows=10_000, temporal=["WORKDATE","STARTDATE","ENDDATE"],
          business_context="Cancellation analytics and reasons.",
          business_context_zh="取消分析与原因统计。",
          relationships=["dbo.ATDLEAVEDATA"],
          priority=2
        ),

        # Balance / holiday / ops
        T("dbo.ATDNONCALCULATEDVACATION",
          ["OID","PERSONID","BUSINESSUINTID","VACATIONDAYS","USEDDAYS","REMAINDAYS","VACATIONTYPE",
           "LASTUPDATETIME","EFFINIENTDATE","INVALIDATIONDATE","DESCRIPTION","CREATIONTIME",
           "CREATEDBY","LASTUPDATEDBY","FLAG"],
          desc="Current vacation balances",
          description_zh="当前假期余额",
          tags=["vacation","balance","current"], pks=["OID"], indexed=["PERSONID","VACATIONTYPE"],
          rows=25_000, temporal=["EFFINIENTDATE","INVALIDATIONDATE"],
          business_context="Live balances for policy and employee self-service.",
          business_context_zh="用于政策及员工自助的实时余额。",
          key_cols={"VACATIONDAYS":"Allocated","USEDDAYS":"Used","REMAINDAYS":"Remaining"},
          relationships=["dbo.ATDLEAVEDATAEX"],
          kpis=["balance_utilization"],
          priority=1
        ),

        T("dbo.ATDHISNONCALCULATEDVACATION",
          ["OID","PERSONID","BUSINESSUINTID","VACATIONDAYS","USEDDAYS","REMAINDAYS","VACATIONTYPE",
           "LASTUPDATETIME","EFFINIENTDATE","INVALIDATIONDATE","DESCRIPTION","CREATIONTIME",
           "CREATEDBY","LASTUPDATEDBY"],
          desc="Historical vacation balances",
          description_zh="历史假期余额",
          tags=["vacation","balance","history"], pks=["OID"], indexed=["PERSONID"],
          rows=100_000, is_hist=True, temporal=["EFFINIENTDATE","INVALIDATIONDATE"],
          business_context="Back-testing and balance history analysis.",
          business_context_zh="回溯及余额历史分析。",
          relationships=["dbo.ATDNONCALCULATEDVACATION"],
          priority=2
        ),

        T("dbo.ATDLEGALCALENDAR",
          ["CALENDARDATE","LEGALID","CALENDARTYPE"],
          desc="Legal holidays / calendar",
          description_zh="法定节假日/日历",
          tags=["calendar","holidays"], pks=["CALENDARDATE","LEGALID"], indexed=["CALENDARDATE"],
          rows=1_000, temporal=["CALENDARDATE"],
          business_context="Holiday detection and policy compliance.",
          business_context_zh="节假日识别与政策合规。",
          relationships=["dbo.ATDLEAVEDATA"],
          priority=2
        ),

        T("dbo.ATDDEPTOPERSTATE",
          ["DEPARTMENTID","WORKDATE","STATEROLLCALL","STATEOVERTIME","STATEALLOWANCE",
           "CALCULATOR","CALCULATETIME","SUBMITER","SUBMITTIME","CHECKER","CHECKTIME",
           "VALIDATER","VALIDATETIME","BUSINESSUNITID"],
          desc="Department-day operation status (calc/validate checkpoints)",
          description_zh="部门-日运作状态（计算/验证检查点）",
          tags=["dept","ops","status"], pks=["DEPARTMENTID","WORKDATE"], indexed=["WORKDATE"],
          rows=50_000, temporal=["WORKDATE"],
          business_context="Process status alignment for troubleshooting.",
          business_context_zh="故障排查的流程状态对齐。",
          relationships=["dbo.ATDLEAVEDATA"],
          priority=2
        ),

        # Aux attendance/timecard
        T("dbo.ATDHISLATEEARLY",
          ["LATEEARLYID","ATTENDDATE","PERSONID","SHOULDTIME","DEPARTMENTID","TIMECLASS",
           "CARDTIME","MINUTES","FUNCFLAG","PROCSTATE","ATDTYPE","ALLEGEREASON",
           "BUSINESSUNITID","SHOULDDATE","CREATIONTIME","CREATEDBY","LASTUPDATETIME",
           "LASTUPDATEDBY","PROCSTATEBEFORETODEAL","RECORD_USERID","RECORD_DATE",
           "TOPAYROLLDATE","PACKAGEID","CARDDATE","EVENTRULECODE"],
          desc="Historical late/early events",
          description_zh="历史迟到/早退事件",
          tags=["attendance","late","early","history"], pks=["LATEEARLYID"], indexed=["PERSONID","ATTENDDATE"],
          rows=750_000, is_hist=True, temporal=["ATTENDDATE","CARDDATE"],
          business_context="Exception analytics and policy adherence.",
          business_context_zh="异常分析与政策遵循。",
          priority=2
        ),

        T("dbo.ATDHISTIMECARDDATA",
          ["RECORDID","DATAID","TIMECARDDATE","TIMECARDTIME","RECEIVEDATE","MACHINEID",
           "DEPARTMENTID","PERSONID","REGIONID","DATAFROM","BUSINESSUNITID","CREATIONTIME",
           "CREATEDBY","LASTUPDATETIME","LASTUPDATEDBY","RECORD_USERID","RECORD_DATE",
           "CONFIGID","REASONID","FORM_NO","RECORD_ID","REMARK","LOCATIONADDRESS"],
          desc="Raw timecard swipes (historical)",
          description_zh="原始打卡记录（历史）",
          tags=["timecard","raw","history"], pks=["RECORDID"], indexed=["PERSONID","TIMECARDDATE"],
          rows=2_000_000, is_hist=True, temporal=["TIMECARDDATE"],
          business_context="Forensic analysis / anomaly detection.",
          business_context_zh="法证分析/异常检测。",
          priority=2
        ),

        T("dbo.ATDHISNOTIMECARD",
          ["NOTIMECARDID","SHOULDTIME","ATTENDDATE","PERSONID","DEPARTMENTID","TIMECLASS",
           "REASONID","PROCSTATE","ALLEGEREASON","BUSINESSUNITID","SHOULDDATE","CREATIONTIME",
           "CREATEDBY","LASTUPDATETIME","LASTUPDATEDBY","PROCSTATEBEFORETODEAL","RECORD_USERID",
           "RECORD_DATE","TOPAYROLLDATE","PACKAGEID","ATDTYPE","EVENTRULECODE"],
          desc="Missing timecard events",
          description_zh="缺卡事件",
          tags=["attendance","notimecard","exceptions"], pks=["NOTIMECARDID"], indexed=["PERSONID","ATTENDDATE"],
          rows=50_000, temporal=["ATTENDDATE"],
          business_context="Exception management and root-cause analysis.",
          business_context_zh="异常管理与根因分析。",
          priority=2
        ),

        # External + rollups
        T("dbo.EDFATDLEAVEDATA",
          ["RECORDID","ATTENDANCETYPE","PERSONID","WORKDATES","WORKDATEE","STARTTIME","ENDTIME",
           "HOURS","DEPARTMENTID","VALIDATED","BUSINESSUNITID","PROCESSINSTANCEID","ISHANDLE"],
          desc="External (EDF) leave imports",
          description_zh="外部（EDF）请假导入",
          tags=["leave","edf","integration"], pks=["RECORDID"], indexed=["PERSONID"],
          rows=25_000, temporal=["WORKDATES","WORKDATEE"],
          business_context="Integration reconciliation and oversight.",
          business_context_zh="集成对账与监管。",
          relationships=["dbo.ATDLEAVEDATA"],
          priority=2
        ),

        T("dbo.ATDRESULTDATAIMPORT",
          ["PERSONID","DEPARTMENTID","ATTENDANCETYPE","BELONGYEARMONTH","RESULTDATA","ISPAY",
           "BUSINESSUNITID","CREATIONTIME","CREATEDBY","LASTUPDATETIME","LASTUPDATEDBY","IMPORTID"],
          desc="Monthly attendance/leave rollup import",
          description_zh="月度考勤/请假汇总导入",
          tags=["import","monthly","rollup"], pks=["IMPORTID"], indexed=["PERSONID","ATTENDANCETYPE"],
          rows=100_000, temporal=["BELONGYEARMONTH"],
          business_context="Executive rollups & payroll alignment.",
          business_context_zh="管理层汇总与薪资对齐。",
          priority=2
        ),

        # Person dimension (auto-resolved by constructor)
        T("dbo.BIPSNACCOUNTSP",
          ["PERIODID","ANALYZETYPEID","PERSONID","TRUENAME","EMPLOYEEID","HEADCOUNT","BRANCHID",
           "JOBCODE","JOBCODESERIALID","JOBCODEGROUPID","JOBCODEGRADEID","JOBCODETYPEID",
           "JOBCHARACTER","RESPONSIBILITYID","RESPONSIBILITYTYPEID","GRADEID","GRADETYPEID",
           "TITLEID","TITLETYPEID","GENDER","EDUCATIONALLEVELID","AGE","AGEOFSCOPEID",
           "NATIONALITYID","SERVICELENGTHCOMPANY","SERVICELENGTHCOMPANYSCOPEID",
           "SERVICELENGTHSOCIAL","SERVICELENGTHSOCIALSCOPEID","DLIDL","EMPLOYEETYPEID",
           "EMPLOYEECHARID","JOBTYPEID","ARRANGEMENTID","NATIVEPLACEPROPERTYID","BELONGCORPID",
           "MARRIAGEID","ACCESSIONSTATE","ATTENDONDATE"],
          desc="Person dimension (BI snapshot)",
          description_zh="人员维度（BI 快照）",
          tags=["person","employee","bi"], pks=["PERSONID"], indexed=["PERSONID"],
          business_context="Employee attributes enrichment for analytics.",
          business_context_zh="用于分析的员工属性信息。",
          relationships=["dbo.ATDLEAVEDATA","dbo.ATDHISLEAVEDATA"],
          priority=1
        ),
    ]

    return LeaveVectorDB(tables)


# ───────────────────────────────────────────────
# Query Analyzers (kept, lightly tuned)
# ───────────────────────────────────────────────

def analyze_query_complexity(query: str) -> Dict[str, Any]:
    q = query.lower()
    indicators = {
        "temporal_range": any(t in q for t in ["between", "range", "from", "to", "last", "past", "过去", "期间"]),
        "aggregation": any(t in q for t in ["count", "sum", "average", "total", "max", "min", "均值", "总计"]),
        "grouping": any(t in q for t in [" by ", "group", "department", "unit", "type", "按", "分组"]),
        "multiple_tables": len([t for t in ["leave", "attendance", "vacation", "请假", "考勤", "休假"] if t in q]) > 1,
        "person_lookup": any(t in q for t in ["who", "person", "employee", "name", "谁", "员工", "姓名"]),
        "current_data": any(t in q for t in ["current", "today", "now", "active", "今天", "当前"]),
        "historical_data": any(t in q for t in ["history", "historical", "trend", "past", "历史", "趋势"]),
    }
    hints = []
    if indicators["temporal_range"]:
        hints.append("Use indexed/filtered date columns for range queries")
    if indicators["multiple_tables"]:
        hints.append("Consider join order: smaller / filtered tables first")
    if indicators["historical_data"]:
        hints.append("Historical tables are large — always filter by date range")
    if indicators["aggregation"] and not indicators["temporal_range"]:
        hints.append("Add a date window to aggregations for performance")
    return {"complexity_score": sum(indicators.values()), "indicators": indicators, "optimization_hints": hints}


def suggest_query_improvements(query: str, selected_tables: List[str]) -> List[str]:
    suggestions = []
    q = query.lower()

    large = {"dbo.ATDHISTIMECARDDATA", "dbo.ATDHISLEAVEDATA", "dbo.ATDHISLATEEARLY"}
    if any(t in large for t in selected_tables):
        if not any(t in q for t in ["date", "between", "last", "past", "month", "year", "range", "日期", "期间"]):
            suggestions.append("Add a date range filter for historical tables")

    if any(t in q for t in ["who", "person", "employee", "name", "谁", "员工", "姓名"]):
        if not any("PSNACCOUNT" in t.upper() or "BIPSNACCOUNTSP" in t.upper() for t in selected_tables):
            suggestions.append("Include person dimension table for name/ID resolution")

    if "leave" in q or "请假" in q:
        if "dbo.ATDLEAVEDATA" in selected_tables:
            suggestions.append("Filter by VALIDATED = 1 for approved leave only")

    if any(t in q for t in ["department", "unit", "business", "部门", "单位"]):
        if not any("DEPARTMENTID" in t or "BUSINESSUNITID" in t for t in selected_tables):
            suggestions.append("Consider filtering/grouping by DEPARTMENTID or BUSINESSUNITID")

    return suggestions
