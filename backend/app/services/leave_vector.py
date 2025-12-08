# backend/app/services/leave_vector.py
# 修正版：移除 example_queries 參數，專注於 zh-tw
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
from datetime import datetime, timedelta
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

# 可設定的完整資料表名稱
ORG_TABLE = os.getenv("ORG_TABLE", "[eHRAntung_DB].[dbo].[ORGStdStruct]")
VAC_RESULT_TABLE = os.getenv("VAC_RESULT_TABLE", "[eHRAntung_DB].[dbo].[ATDCALCUVACATIONRESULT]")

# Schema 對應表
SCHEMA_MAP = {
    "PSNACCOUNT": {
        "PERSON_ID": ["PERSONID", "PersonID"],
        "NAME": ["TRUENAME", "CNAME", "NAME", "FULLNAME"],
        "EMP_NO": ["EMPLOYEEID", "EMPLOYEENO", "WORKNO", "EMPNO", "員工編號", "工號", "員編"],
        "DEPT_ID": ["BRANCHID", "DEPARTMENTID"],
    },
    "ATDLEAVEDATA": {
        "PERSON_ID": ["PERSONID", "PersonID"],
        "WORK_DATE": ["WORKDATE", "LEAVEDATE", "STARTDATE"],
        "START_DATE": ["STARTDATE"],
        "END_DATE": ["ENDDATE"],
        "HOURS": ["HOURS", "LEAVEHOURS"],
        "TYPE": ["ATTENDANCETYPE", "LEAVEID", "TIMECLASSID"],
        "VALIDATED": ["VALIDATED"],
    },
    "ATDCALCUVACATIONRESULT": {
        "PERSON_ID": ["PERSONID", "PersonID"],
        "YEAR": ["VACAYEAR", "YEAR"],
        "VAC_MONTH": ["VACAMONTH"],
        "TYPE": ["VACATIONTYPE", "LEAVETYPE", "LEAVETYPEID"],
        "REMAIN_DAYS": ["REMAINDAYS", "REMAININGDAYS"],
        "REMAIN_HOURS": ["REMAINHOURS", "REMAININGHOURS"],
        "VAC_DAYS": ["VACDAYS"],
        "USE_DAYS": ["USEDAYS"],
        "CAN_USE_DATE": ["CANUSEDATE"],
        "DISABLE_DATE": ["DISABLEDDATE"],
        "UPDATED_AT": ["LASTEDITTIME", "UPDATETIME"],
        "CREATED_AT": ["CREATIONTIME"],
    },
    "ORG": {
        "UNIT_ID": ["UNITID"],
        "UNIT_NAME": ["UNITNAME"],
        "UNIT_DISPLAY": ["UNITDISPLAYNAME"],
        "UNIT_CODE": ["UNITCODE"],
    },
}

# 意圖技能定義
INTENT_SKILLS = [
    {
        "id": "remaining_balance_by_person",
        "title": "依人員查詢剩餘年假（權威快照）",
        "tables": [VAC_RESULT_TABLE, "dbo.PSNACCOUNT", ORG_TABLE],
        "measure": ["annual_leave"],
        "action": ["remaining"],
        "phrases_zh": ["剩餘特休", "年假餘額", "還有年假", "未用年假"],
        "template_ref": "annual_balance_by_person",
    },
    {
        "id": "cancellations_detail",
        "title": "請假取消紀錄",
        "tables": ["dbo.ATDLEAVECANCELDATA", "dbo.ATDLEAVEDATA", "dbo.PSNACCOUNT"],
        "measure": ["any"],
        "action": ["cancelled"],
        "phrases_zh": ["取消", "撤銷", "改期"],
        "template_ref": "cancellations_detail",
    },
    {
        "id": "person_branch_map",
        "title": "人員對應部門",
        "tables": ["dbo.PSNACCOUNT", ORG_TABLE],
        "measure": ["org_map"],
        "action": ["lookup"],
        "phrases_zh": ["單位", "部門", "分公司"],
        "template_ref": "person_branch_map",
    },
]

LEXICON = {
    "measure": {
        "annual_leave": {"zh": ["特休", "年假", "年休", "年假時數", "年假天數"]},
        "org_map": {"zh": ["部門", "單位", "分公司"]},
    },
    "action": {
        "remaining": {"zh": ["剩餘", "餘額", "未用", "還有", "可用"]},
        "cancelled": {"zh": ["取消", "撤銷", "改期"]},
        "lookup": {"zh": ["查詢", "對應", "查看"]},
    },
}


def _extract_slots_from_text(q: str) -> Dict[str, Optional[Any]]:
    qlow = (q or "").lower()
    measure = None
    for key, val in LEXICON["measure"].items():
        if any(t in q for t in val.get("zh", [])):
            measure = key
            break
    action = None
    for key, val in LEXICON["action"].items():
        if any(t in q for t in val.get("zh", [])):
            action = key
            break
    m = re.search(r"(20\d{2})", q)
    year = int(m.group(1)) if m else None
    vtype = 1 if any(tok in q for tok in ["特休", "年假"]) else None
    th = None
    m2 = re.search(r"(?:大於|超過|>|≥|>=)\s*(\d+)\s*小時", q)
    if m2:
        th = int(m2.group(1))
    return {"measure": measure, "action": action, "year": year, "vacationtype": vtype, "threshold_hours": th}


def canonicalize_query_intent(q: str) -> Dict[str, Optional[Any]]:
    return _extract_slots_from_text(q)


# 同義詞擴充（zh-tw）
_ZH_SYNONYMS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"今天|今日"), "今天 今日"),
    (re.compile(r"昨天|昨日"), "昨天 昨日"),
    (re.compile(r"本週|這週"), "本週 這週"),
    (re.compile(r"本月|這個月"), "本月 這個月"),
    (re.compile(r"請假|休假"), "請假 休假"),
    (re.compile(r"員工|人員|同仁"), "員工 人員 同仁"),
    (re.compile(r"部門|單位"), "部門 單位"),
    (re.compile(r"假別|假種"), "假別 假種"),
    (re.compile(r"工號|員工編號|員編"), "工號 員工編號 員編"),
    (re.compile(r"已核准|已批准"), "已核准 已批准"),
    (re.compile(r"餘額|剩餘|可用"), "餘額 剩餘 可用"),
]


def _expand_zh_synonyms(q: str) -> str:
    out = q
    for pat, zh in _ZH_SYNONYMS:
        if pat.search(out):
            out += f" {zh}"
    return out


def detect_language(text: str) -> str:
    return "zh-tw"


class _LRUCache(OrderedDict):
    def __init__(self, maxsize: int = 256):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key):
        if key in self:
            self.move_to_end(key)
            return super().get(key)
        return None

    def set(self, key, value):
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)


def _today_date() -> datetime:
    return datetime.now()


def _iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _week_window(dt: Optional[datetime] = None) -> Tuple[str, str]:
    dt = dt or _today_date()
    start = dt - timedelta(days=(dt.weekday()))
    end = start + timedelta(days=6)
    return _iso(start), _iso(end)


def _parse_mmdd_range(txt: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})\s*[-~至到]\s*(\d{1,2})\s*/\s*(\d{1,2})", txt)
    if not m:
        return None
    y = _today_date().year
    m1, d1, m2, d2 = map(int, m.groups())
    try:
        start = datetime(y, m1, d1)
        end = datetime(y, m2, d2)
    except ValueError:
        return None
    return _iso(start), _iso(end)


class JoinType(Enum):
    INNER = "INNER"
    LEFT = "LEFT"


class Cardinality(Enum):
    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:M"
    MANY_TO_ONE = "M:1"


@dataclass
class TableJoin:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: JoinType = JoinType.LEFT
    cardinality: Cardinality = Cardinality.MANY_TO_ONE
    description_zh: str = ""
    tags: List[str] = field(default_factory=list)

    def on_clause(self) -> str:
        return f"{self.join_type.value} JOIN {self.right_table} ON {self.left_table}.{self.left_column} = {self.right_table}.{self.right_column}"


@dataclass
class QueryPattern:
    pattern: str
    description_zh: str
    primary_tables: List[str]
    suggested_joins: List[str] = field(default_factory=list)
    required_filters: List[str] = field(default_factory=list)
    examples_zh: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class TableSchema:
    full: str
    columns: List[str]
    description_zh: str = ""
    tags: List[str] = field(default_factory=list)
    row_estimate: Optional[int] = None
    temporal_columns: List[str] = field(default_factory=list)
    business_context_zh: str = ""
    common_queries_zh: List[str] = field(default_factory=list)
    key_columns: Dict[str, str] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    row_count_estimate: str = ""
    priority: int = 1
    kpi_relevance: List[str] = field(default_factory=list)


@dataclass
class KPIDef:
    name: str
    description_zh: str = ""
    formula_sql_hint: str = ""
    tables: List[str] = field(default_factory=list)
    grain: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class SQLRecipe:
    recipe_id: str
    title: str
    description_zh: str = ""
    sql_template: str = ""
    tables: List[str] = field(default_factory=list)
    expected_columns: List[str] = field(default_factory=list)
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
    text_zh: str
    priority: int = 2
    payload: Dict[str, Any] = field(default_factory=dict)


COLUMN_ALIASES: Dict[str, Set[str]] = {
    "BUSINESSUNITID": {"BUSINESSUINTID"},
    "EFFECTIVEDATE": {"EFFINIENTDATE", "EFFICIENTDATE"},
}


def _has_col(table: TableSchema, name: str) -> bool:
    cols = {c.upper() for c in table.columns}
    return name.upper() in cols

# ═══════════════════════════════════════════════════════════════════════════════
# LeaveVectorDB 類別
# ═══════════════════════════════════════════════════════════════════════════════
class LeaveVectorDB:
    def __init__(
        self,
        tables: List[TableSchema],
        db_path: str = "leave_schema_vectors.db",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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
        self.index_zh = None
        self._vector_items: List[VectorItem] = []
        self._id2item: Dict[int, VectorItem] = {}
        self.embeddings_zh: Optional[np.ndarray] = None
        self._search_cache: _LRUCache = _LRUCache(maxsize=512)

        self._load_model()
        self._build_vector_items()
        self._build_indexes()

    def record_outcome(self, **kwargs) -> None:
        pass

    def _pick(self, table_full: str, logical_key: str, default: Optional[str] = None) -> str:
        t = self._by_name.get(table_full.lower())
        if not t:
            return default or logical_key
        if "psnaccount" in table_full.lower():
            pool = SCHEMA_MAP["PSNACCOUNT"].get(logical_key, [])
        elif "orgstdstruct" in table_full.lower():
            pool = SCHEMA_MAP["ORG"].get(logical_key, [])
        elif "atdleavedata" in table_full.lower():
            pool = SCHEMA_MAP["ATDLEAVEDATA"].get(logical_key, [])
        elif "atdcalcuvacationresult" in table_full.lower():
            pool = SCHEMA_MAP["ATDCALCUVACATIONRESULT"].get(logical_key, [])
        else:
            pool = []
        colsU = {c.upper(): c for c in t.columns}
        for cand in pool:
            if cand.upper() in colsU:
                return colsU[cand.upper()]
        return default or logical_key

    def _emp_no_col(self) -> str:
        pt = self._by_name.get((self._person_table or "").lower())
        if not pt:
            return "EMPLOYEEID"
        for c in SCHEMA_MAP["PSNACCOUNT"]["EMP_NO"]:
            if c.upper() in {col.upper() for col in pt.columns}:
                return c
        return "EMPLOYEEID"

    def _classify_intent(self, q: str) -> str:
        t = (q or "").lower()
        if any(k in t for k in ["歷史", "紀錄", "記錄"]):
            if any(k in t for k in ["員工編號", "工號", "員編"]):
                return "PERSON_HISTORY_BY_EMP_NO"
            return "LEAVE_HISTORY_GENERIC"
        if any(k in t for k in ["剩餘", "餘額", "還有", "可用"]) and any(k in t for k in ["特休", "年假"]):
            if any(k in t for k in ["大於", ">", "超過"]):
                return "BALANCE_YEAR_THRESHOLD_HOURS"
            return "BALANCE_YEAR_REMAINING"
        if any(k in t for k in ["本週", "這週"]):
            if any(k in t for k in ["多少人", "人數"]):
                return "WEEKLY_COUNT"
            return "WEEKLY_WHO_ON_LEAVE"
        if any(k in t for k in ["今天", "今日"]):
            return "TODAY_WHO_ON_LEAVE"
        if any(k in t for k in ["本月", "這個月"]):
            return "MONTH_WHO_ON_LEAVE"
        return "GENERIC_LEAVE_LOOKUP"

    def get_intent_routing(self, query: str) -> Dict[str, Any]:
        q = (query or "").lower()
        plan: Dict[str, Any] = {
            "intent": "generic",
            "template_ref": None,
            "slots": {},
            "tables": [t.full for t in self.tables[:3]],
            "language": "zh-tw",
        }

        slots = canonicalize_query_intent(query)
        if slots.get("year"):
            plan["slots"]["year"] = slots["year"]
        if slots.get("threshold_hours"):
            plan["slots"]["threshold_hours"] = slots["threshold_hours"]

        rng = _parse_mmdd_range(q)
        if rng:
            plan["slots"]["start_date"], plan["slots"]["end_date"] = rng
        elif any(k in q for k in ["本週", "這週"]):
            s, e = _week_window()
            plan["slots"]["start_date"], plan["slots"]["end_date"] = s, e

        label = self._classify_intent(query)
        plan["intent"] = label

        psn = self._person_table or "dbo.PSNACCOUNT"
        if label in ("BALANCE_YEAR_REMAINING", "BALANCE_YEAR_THRESHOLD_HOURS"):
            plan.update({
                "template_ref": "annual_balance_by_person" if label == "BALANCE_YEAR_REMAINING" else "balance_year_threshold_hours",
                "tables": [VAC_RESULT_TABLE, psn, ORG_TABLE],
            })
        elif label in ("TODAY_WHO_ON_LEAVE", "WEEKLY_WHO_ON_LEAVE", "MONTH_WHO_ON_LEAVE", "LEAVE_HISTORY_GENERIC"):
            plan.update({
                "template_ref": "today_who_on_leave" if label == "TODAY_WHO_ON_LEAVE" else "range_who_on_leave",
                "tables": ["dbo.ATDLEAVEDATA", psn, ORG_TABLE],
            })
        else:
            plan.update({
                "template_ref": "range_who_on_leave",
                "tables": ["dbo.ATDLEAVEDATA", psn, ORG_TABLE],
            })

        return plan

    def _resolve_person_table(self) -> Optional[str]:
        for name in ["dbo.PSNACCOUNT", "dbo.PSNACCOUNT_D"]:
            if name.lower() in self._by_name:
                return name
        return None

    def _exists(self, full: str) -> bool:
        return (full or "").lower() in self._by_name

    def _load_model(self) -> None:
        if SentenceTransformer is None:
            logger.warning("sentence-transformers 未安裝")
            self.model = None
            return
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("已載入嵌入模型: %s", self.model_name)
        except Exception as e:
            logger.error("載入模型失敗: %s", e)
            self.model = None

    def _combine_text_zh(self, obj, obj_type: str) -> str:
        if obj_type == "table":
            t = obj
            return " ".join([t.description_zh, t.business_context_zh, " ".join(t.tags), " ".join(t.common_queries_zh), t.full, " ".join(t.columns[:30])])
        elif obj_type == "join":
            j = obj
            return " ".join([j.description_zh, " ".join(j.tags), j.left_table, j.right_table])
        elif obj_type == "pattern":
            p = obj
            return " ".join([p.pattern, p.description_zh, " ".join(p.tags), " ".join(p.examples_zh)])
        elif obj_type == "kpi":
            k = obj
            return " ".join([k.name, k.description_zh, " ".join(k.tags)])
        elif obj_type == "recipe":
            r = obj
            return " ".join([r.title, r.description_zh, " ".join(r.tags), " ".join(r.tables)])
        return ""

    def _build_vector_items(self) -> None:
        self._vector_items = []
        for t in self.tables:
            self._vector_items.append(VectorItem(key=t.full, item_type=VectorItemType.TABLE, text_zh=self._combine_text_zh(t, "table"), priority=t.priority, payload={"table": t}))
        for j in self._joins:
            self._vector_items.append(VectorItem(key=f"JOIN::{j.left_table}::{j.right_table}", item_type=VectorItemType.JOIN, text_zh=self._combine_text_zh(j, "join"), payload={"join": j}))
        for p in self._query_patterns:
            self._vector_items.append(VectorItem(key=f"PATTERN::{p.pattern}", item_type=VectorItemType.PATTERN, text_zh=self._combine_text_zh(p, "pattern"), payload={"pattern": p}))
        for k in self._kpis:
            self._vector_items.append(VectorItem(key=f"KPI::{k.name}", item_type=VectorItemType.KPI, text_zh=self._combine_text_zh(k, "kpi"), payload={"kpi": k}))
        for r in self._recipes:
            self._vector_items.append(VectorItem(key=f"RECIPE::{r.recipe_id}", item_type=VectorItemType.RECIPE, text_zh=self._combine_text_zh(r, "recipe"), payload={"recipe": r}))
        logger.info("VECTOR_ITEMS: 已建立=%d", len(self._vector_items))

    @staticmethod
    def _hashing_embed(texts: List[str], dim: int = 2048) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[i, hash(tok) % dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out

    def _build_indexes(self) -> None:
        if not self._vector_items:
            return
        texts_zh = [vi.text_zh for vi in self._vector_items]
        if self.model is not None:
            try:
                self.embeddings_zh = self.model.encode(texts_zh, normalize_embeddings=True, show_progress_bar=False).astype("float32")
            except Exception:
                self.embeddings_zh = self._hashing_embed(texts_zh)
        else:
            self.embeddings_zh = self._hashing_embed(texts_zh)

        if faiss is not None:
            try:
                dim = int(self.embeddings_zh.shape[1])
                self.index_zh = faiss.IndexFlatIP(dim)
                self.index_zh.add(self.embeddings_zh)
            except Exception:
                self.index_zh = None
        self._id2item = {i: vi for i, vi in enumerate(self._vector_items)}

    def _numpy_search(self, query_vec: np.ndarray, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        sims = (embeddings @ query_vec.T).squeeze()
        idxs = np.argsort(-sims)[:k]
        return sims[idxs], idxs

    def _encode_query(self, query: str) -> np.ndarray:
        if self.model is not None:
            try:
                return self.model.encode([query], normalize_embeddings=True).astype("float32")
            except Exception:
                pass
        return self._hashing_embed([query])

    def search(self, query: str, top_k: int = 8, min_score: float = 0.25) -> List[Tuple[VectorItem, float]]:
        if not self._vector_items or self.embeddings_zh is None:
            return []
        q_expanded = _expand_zh_synonyms(query)
        cache_key = (q_expanded, top_k, min_score)
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return cached

        qvec = self._encode_query(q_expanded).astype("float32")
        k = min(top_k * 3, len(self._vector_items))
        if self.index_zh is not None and faiss is not None:
            try:
                distances, indices = self.index_zh.search(qvec, k)
                sims, idxs = distances[0], indices[0]
            except Exception:
                sims, idxs = self._numpy_search(qvec, self.embeddings_zh, k)
        else:
            sims, idxs = self._numpy_search(qvec, self.embeddings_zh, k)

        results = []
        for sim, idx in zip(sims, idxs):
            if int(idx) < len(self._vector_items) and float(sim) >= min_score:
                results.append((self._id2item[int(idx)], float(sim)))
        out = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        self._search_cache.set(cache_key, out)
        return out

    def search_relevant_tables(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        mixed = self.search(query, top_k=top_k * 2)
        tables = [(vi.payload["table"].full, s) for vi, s in mixed if vi.item_type == VectorItemType.TABLE][:top_k]
        if not tables:
            ql = (query or "").lower()
            if any(k in ql for k in ["餘額", "年假", "特休"]) and self._exists(VAC_RESULT_TABLE):
                tables.append((VAC_RESULT_TABLE, 0.26))
            if any(k in ql for k in ["請假", "今天"]) and self._exists("dbo.ATDLEAVEDATA"):
                tables.append(("dbo.ATDLEAVEDATA", 0.24))
            if self._person_table:
                tables.append((self._person_table, 0.23))
        return tables

    def join_hints(self, tables: Iterable[str]) -> List[str]:
        table_set = {(t or "").lower() for t in tables if t}
        hints = []
        if self._person_table:
            hints.append(f"-- FACT.PERSONID → {self._person_table}.PERSONID (LEFT JOIN)")
            if self._exists(ORG_TABLE):
                hints.append(f"-- {self._person_table}.BRANCHID → {ORG_TABLE}.UNITID (LEFT JOIN, 需 CAST 為 NVARCHAR)")
        for j in self._joins:
            if j.left_table.lower() in table_set and j.right_table.lower() in table_set:
                hints.append(j.on_clause())
        return hints

    def is_ready(self) -> bool:
        return bool(self.tables) and (self.embeddings_zh is not None)

    def health_check(self) -> Dict[str, object]:
        return {
            "ready": self.is_ready(),
            "tables_indexed": len(self.tables),
            "vector_items": len(self._vector_items),
            "recipes": len(self._recipes),
            "person_table": self._person_table,
            "model_loaded": self.model is not None,
            "language": "zh-tw-only",
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Recipes 建構
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_recipes(self) -> List[SQLRecipe]:
        psn = self._person_table or "dbo.PSNACCOUNT"
        emp_no_col = self._emp_no_col()
        name_col = self._pick(psn, "NAME", default="TRUENAME")
        org_unitdisp = self._pick(ORG_TABLE, "UNIT_DISPLAY", default="UNITDISPLAYNAME")
        org_unitname = self._pick(ORG_TABLE, "UNIT_NAME", default="UNITNAME")
        dept_expr = f"COALESCE(org.{org_unitdisp}, org.{org_unitname})"

        ld = "dbo.ATDLEAVEDATA"
        vr = VAC_RESULT_TABLE

        recipes = []

        # 今天在休假的員工
        recipes.append(SQLRecipe(
            recipe_id="today_who_on_leave",
            title="今天在休假的員工",
            description_zh="列出今天有請假紀錄（已核准）的員工，含部門、員編、姓名。",
            tables=[ld, psn, ORG_TABLE],
            expected_columns=["department_name", "employee_id", "person_name", "LeaveType"],
            sql_template=f"""SELECT
  {dept_expr} AS department_name,
  p.{emp_no_col} AS employee_id,
  p.{name_col} AS person_name,
  ld.ATTENDANCETYPE AS LeaveType,
  COALESCE(ld.HOURS, 0) AS LeaveHours
FROM {ld} AS ld
LEFT JOIN {psn} AS p ON p.PERSONID = ld.PERSONID
LEFT JOIN {ORG_TABLE} AS org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))
WHERE ld.VALIDATED = 1
  AND CAST(GETDATE() AS date) BETWEEN CAST(ld.STARTDATE AS date) AND CAST(ld.ENDDATE AS date)
ORDER BY department_name, person_name;""",
            tags=["今天", "當前", "已核准"],
        ))

        # 日期區間內有請假的人員
        recipes.append(SQLRecipe(
            recipe_id="range_who_on_leave",
            title="日期區間內有請假的人員",
            description_zh="列出在指定日期期間有請假紀錄的人員。",
            tables=[ld, psn, ORG_TABLE],
            expected_columns=["department_name", "employee_id", "person_name", "WorkDate", "LeaveType"],
            sql_template=f"""SELECT
  {dept_expr} AS department_name,
  p.{emp_no_col} AS employee_id,
  p.{name_col} AS person_name,
  CAST(ld.WORKDATE AS date) AS WorkDate,
  ld.ATTENDANCETYPE AS LeaveType,
  COALESCE(ld.HOURS, 0) AS LeaveHours
FROM {ld} AS ld
LEFT JOIN {psn} AS p ON p.PERSONID = ld.PERSONID
LEFT JOIN {ORG_TABLE} AS org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))
WHERE ld.VALIDATED = 1
  AND CAST(ld.WORKDATE AS date) BETWEEN CAST(GETDATE() AS date) AND DATEADD(day, 7, CAST(GETDATE() AS date))
ORDER BY WorkDate, person_name;""",
            tags=["區間", "請假"],
        ))

        # 依人員查詢年度年假餘額
        recipes.append(SQLRecipe(
            recipe_id="annual_balance_by_person",
            title="依人員查詢年度年假餘額",
            description_zh="每人/年度/假別的最新有效快照，含給予/已用/剩餘天數。",
            tables=[vr, psn, ORG_TABLE],
            expected_columns=["department_name", "employee_id", "person_name", "VACAYEAR", "REMAINDAYS"],
            sql_template=f"""WITH latest AS (
  SELECT r.PERSONID, r.VACAYEAR, r.VACAMONTH, r.VACATIONTYPE,
         r.VACDAYS, r.USEDAYS, r.REMAINDAYS, r.CANUSEDATE, r.DISABLEDDATE,
         ROW_NUMBER() OVER (
           PARTITION BY r.PERSONID, r.VACAYEAR, r.VACATIONTYPE
           ORDER BY ISNULL(r.LASTEDITTIME, r.CREATIONTIME) DESC
         ) AS rn
  FROM {vr} AS r
  WHERE (r.CANUSEDATE IS NULL OR CAST(GETDATE() AS date) >= CAST(r.CANUSEDATE AS date))
    AND (r.DISABLEDDATE IS NULL OR CAST(GETDATE() AS date) <= CAST(r.DISABLEDDATE AS date))
)
SELECT
  {dept_expr} AS department_name,
  p.{emp_no_col} AS employee_id,
  p.{name_col} AS person_name,
  l.VACAYEAR, l.VACATIONTYPE, l.VACDAYS, l.USEDAYS, l.REMAINDAYS,
  l.CANUSEDATE, l.DISABLEDDATE
FROM latest AS l
LEFT JOIN {psn} AS p ON p.PERSONID = l.PERSONID
LEFT JOIN {ORG_TABLE} AS org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.UNITID AS NVARCHAR(100))
WHERE l.rn = 1
ORDER BY department_name, person_name;""",
            tags=["餘額", "年假", "快照"],
        ))

        # 剩餘年假超過門檻
        recipes.append(SQLRecipe(
            recipe_id="balance_year_threshold_hours",
            title="剩餘年假超過指定小時數的人員",
            description_zh="剩餘年假超過門檻（天數*8 估算小時）的人員。",
            tables=[vr, psn],
            expected_columns=["employee_id", "person_name", "VacYear", "REMAINDAYS"],
            sql_template=f"""SELECT
  p.{emp_no_col} AS employee_id,
  p.{name_col} AS person_name,
  r.VACAYEAR AS VacYear,
  r.REMAINDAYS,
  (r.REMAINDAYS * 8) AS REMAIN_HOURS_APPROX
FROM {vr} r
LEFT JOIN {psn} p ON p.PERSONID = r.PERSONID
WHERE r.VACAYEAR = YEAR(GETDATE())
  AND (r.REMAINDAYS * 8) > 200
ORDER BY REMAIN_HOURS_APPROX DESC;""",
            tags=["餘額", "門檻"],
        ))

        return recipes

    def _build_comprehensive_joins(self) -> List[TableJoin]:
        joins = []
        leave_tables = ["dbo.ATDLEAVEDATA", "dbo.ATDLEAVECANCELDATA", VAC_RESULT_TABLE]
        if self._person_table:
            for lt in leave_tables:
                if self._exists(lt):
                    joins.append(TableJoin(
                        left_table=lt,
                        left_column="PERSONID",
                        right_table=self._person_table,
                        right_column="PERSONID",
                        description_zh="以 PERSONID 關聯人員屬性",
                        tags=["人員", "維度"],
                    ))
        if self._exists(self._person_table or "") and self._exists(ORG_TABLE):
            joins.append(TableJoin(
                left_table=self._person_table,
                left_column="BRANCHID",
                right_table=ORG_TABLE,
                right_column="UNITID",
                description_zh="BRANCHID 連到部門",
                tags=["部門", "組織"],
            ))
        return joins

    def _build_query_patterns(self) -> List[QueryPattern]:
        return [
            QueryPattern(
                pattern="current_leave_status",
                description_zh="當前正在休假的人員（已批准）",
                primary_tables=["dbo.ATDLEAVEDATA"],
                suggested_joins=["PERSONID"],
                required_filters=["VALIDATED = 1", "今天日期在 STARTDATE 與 ENDDATE 之間"],
                examples_zh=["今天各部門有哪些員工在休假？"],
                tags=["今天", "已核准"],
            ),
            QueryPattern(
                pattern="annual_balance_snapshot",
                description_zh="從快照表取年度年假餘額",
                primary_tables=[VAC_RESULT_TABLE],
                suggested_joins=["PERSONID"],
                required_filters=["有效期窗口過濾"],
                examples_zh=["各部門人員年假剩餘天數"],
                tags=["餘額", "年假"],
            ),
        ]

    def _build_kpis(self) -> List[KPIDef]:
        return [
            KPIDef(
                name="total_leave_hours",
                description_zh="某時間段內已批准請假總小時數",
                formula_sql_hint="SUM(HOURS) WHERE VALIDATED=1",
                tables=["dbo.ATDLEAVEDATA"],
                grain="部門-日",
                tags=["小時"],
            ),
            KPIDef(
                name="annual_leave_remaining_days",
                description_zh="每人年假剩餘天數",
                formula_sql_hint="取快照表 REMAINDAYS",
                tables=[VAC_RESULT_TABLE],
                grain="人員-年度",
                tags=["餘額"],
            ),
        ]

    def get_business_prompt(self, query: str) -> str:
        routing = self.get_intent_routing(query)
        tr = routing.get("template_ref")
        selected_recipe_sql = None
        for r in self._recipes:
            if r.recipe_id == tr:
                selected_recipe_sql = r.sql_template
                break

        return f"""您是一位請假/考勤領域的 T-SQL 專家。

使用者查詢：{query}

意圖：{routing.get('intent')}
模板：{tr}
資料表：{','.join(routing.get('tables', []))}

=== 標準 SQL 範本 ===
{selected_recipe_sql or '-- (未匹配到模板)'}

【重要規則】
1. 不可使用 @ 開頭的變數（如 @today），請用 CAST(GETDATE() AS DATE)
2. 別名需先宣告才能使用
3. BRANCHID 與 UNITID 連接需 CAST 為 NVARCHAR
4. 核准請假需加 VALIDATED = 1

請只回傳 SQL，不要加 markdown 或說明。"""

    def get_schema_context(self, query: str, include_examples: bool = True) -> str:
        ranked = self.search(query, top_k=6)
        lines = ["=== 相關資料庫上下文 ==="]
        for vi, score in ranked:
            if vi.item_type == VectorItemType.TABLE:
                t = vi.payload["table"]
                lines.append(f"資料表: {t.full}")
                lines.append(f"  描述: {t.description_zh}")
                if t.key_columns:
                    lines.append("  關鍵欄位: " + ", ".join(t.key_columns.keys()))
        lines.append("")
        lines.append("【重要】不可使用 @ 變數，請用 GETDATE() 等函數。")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 全域單例與建構函數
# ═══════════════════════════════════════════════════════════════════════════════
_GLOBAL_LEAVE_VECTOR_DB: Optional[LeaveVectorDB] = None


def build_leave_index() -> LeaveVectorDB:
    global _GLOBAL_LEAVE_VECTOR_DB
    if _GLOBAL_LEAVE_VECTOR_DB is not None and _GLOBAL_LEAVE_VECTOR_DB.is_ready():
        return _GLOBAL_LEAVE_VECTOR_DB

    tables = [
        TableSchema(
            full="dbo.ATDLEAVEDATA",
            columns=["ATTENDANCETYPE", "PERSONID", "WORKDATE", "STARTTIME", "ENDTIME", "HOURS", "DEPARTMENTID", "VALIDATED", "BUSINESSUNITID", "STARTDATE", "ENDDATE", "TIMECLASSID", "TIMECLASSHOURS", "CREATIONTIME", "LASTUPDATETIME", "FROM_SOURCE", "FORM_NO", "LEAVEREASON"],
            description_zh="請假使用明細（小時/天），VALIDATED=1 表示已批准",
            tags=["請假", "使用", "已核准"],
            business_context_zh="日粒度請假紀錄，含核准狀態與時數",
            common_queries_zh=["今天有哪些員工在休假？", "各部門已核准請假數量"],
            key_columns={"PERSONID": "人員鍵", "WORKDATE": "請假日期", "VALIDATED": "核准狀態", "HOURS": "請假分鐘數"},
            relationships=["PERSONID → PSNACCOUNT.PERSONID"],
            priority=1,
        ),
        TableSchema(
            full="dbo.ATDLEAVECANCELDATA",
            columns=["OID", "ATTENDANCETYPE", "PERSONID", "WORKDATE", "STARTDATE", "ENDDATE", "HOURS", "VALIDATED", "REASON", "LEAVEREASON"],
            description_zh="請假取消資料（沖銷）",
            tags=["請假", "取消"],
            priority=2,
        ),
        TableSchema(
            full=VAC_RESULT_TABLE,
            columns=["VACATIONID", "PERSONID", "VACAYEAR", "VACAMONTH", "VACATIONTYPE", "USEDAYS", "REMAINDAYS", "VACDAYS", "CANUSEDATE", "DISABLEDDATE", "LASTEDITTIME", "CREATIONTIME"],
            description_zh="休假餘額快照表（權威來源）",
            tags=["餘額", "年假", "剩餘"],
            business_context_zh="餘額與給予之權威來源；取每人/年度/類型最新一筆",
            common_queries_zh=["各部門人員年假剩餘天數", "剩餘時數大於門檻的員工"],
            key_columns={"PERSONID": "人員鍵", "VACAYEAR": "年度", "REMAINDAYS": "剩餘天數", "VACDAYS": "給予天數"},
            priority=1,
        ),
        TableSchema(
            full="dbo.PSNACCOUNT",
            columns=["PERSONID", "EMPLOYEEID", "TRUENAME", "BRANCHID", "BUSINESSUNITID", "ACCESSIONSTATE", "LASTUPDATETIME"],
            description_zh="人員維度（權威來源）",
            tags=["人員", "員工"],
            business_context_zh="核心員工主檔；ACCESSIONSTATE IN (1,2,3,4,5,6) 篩選在職員工",
            key_columns={"PERSONID": "人員鍵", "EMPLOYEEID": "員編", "BRANCHID": "部門鍵", "TRUENAME": "姓名"},
            priority=1,
        ),
        TableSchema(
            full=ORG_TABLE,
            columns=["UNITID", "UNITCODE", "UNITNAME", "UNITDISPLAYNAME", "ISTEMPUNIT", "MANAGEUNIT", "LABELINDEX"],
            description_zh="組織/部門維度",
            tags=["組織", "部門"],
            business_context_zh="標準篩選：ISTEMPUNIT=0 AND MANAGEUNIT='0'",
            key_columns={"UNITID": "部門鍵", "UNITNAME": "部門名稱", "UNITDISPLAYNAME": "部門顯示名稱"},
            priority=1,
        ),
        TableSchema(
            full="dbo.ATDATTENDANCECLASS",
            columns=["ID", "CLASSCODE", "CLASSNAME", "CLASSTYPE", "BUSINESSRULEID"],
            description_zh="假別字典表",
            tags=["假別", "字典"],
            business_context_zh="連接條件：BUSINESSRULEID='0'",
            key_columns={"ID": "主鍵", "CLASSCODE": "假別代碼", "CLASSNAME": "假別名稱"},
            priority=2,
        ),
    ]

    _GLOBAL_LEAVE_VECTOR_DB = LeaveVectorDB(tables)
    return _GLOBAL_LEAVE_VECTOR_DB