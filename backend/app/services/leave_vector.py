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
from datetime import datetime, timedelta

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

# ───────────────────────────────────────────────────────────────────────────────
# Configurable FQNs (no duplicates)
# ───────────────────────────────────────────────────────────────────────────────
ORG_TABLE = os.getenv("ORG_TABLE", "[eHRAntung_DB].[dbo].[ORGStdStruct]")
VAC_RESULT_TABLE = os.getenv("VAC_RESULT_TABLE", "[eHRAntung_DB].[dbo].[ATDCALCUVACATIONRESULT]")

# ───────────────────────────────────────────────────────────────────────────────
# DEMO “cheat” schema map (logical → candidate physical names)
# These are resolved against your provided TableSchema.columns to avoid 42S22.
# ───────────────────────────────────────────────────────────────────────────────
SCHEMA_MAP = {
    "PSNACCOUNT": {
        "PERSON_ID": ["PERSONID", "PersonID"],
        "NAME": ["TRUENAME", "CNAME", "NAME", "FULLNAME"],
        "EMP_NO": ["EMPLOYEEID", "EMPLOYEENO", "WORKNO", "EMPNO", "員工編號", "工號", "員編"],
        "DEPT_ID": ["BRANCHID", "DEPARTMENTID"],
        "DEPT_NAME": ["DEPARTMENT_NAME"],  # not in table; resolved via ORG join
    },
    "ATDLEAVEDATA": {
        "PERSON_ID": ["PERSONID", "PersonID"],
        "WORK_DATE": ["WORKDATE", "LEAVEDATE", "STARTDATE"],
        "START_DATE": ["STARTDATE"],
        "END_DATE": ["ENDDATE"],
        "HOURS": ["HOURS", "LEAVEHOURS"],
        "DAYS": ["LEAVEDAYS", "DAYS"],
        "TYPE": ["ATTENDANCETYPE", "LEAVEID", "TIMECLASSID"],
        "VALIDATED": ["VALIDATED"],
        "UPDATED_AT": ["LASTUPDATETIME", "UPDATETIME", "LASTEDITTIME"],
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
    }
}

# ───────────────────────────────────────────────────────────────────────────────
# Intent-first scaffold (bilingual, lightweight) — untouched
# ───────────────────────────────────────────────────────────────────────────────
INTENT_SKILLS = [
    {
        "id": "remaining_balance_by_person",
        "title": "Remaining annual leave by person (authoritative snapshot)",
        "tables": [VAC_RESULT_TABLE, "dbo.PSNACCOUNT", ORG_TABLE],
        "measure": ["annual_leave"],
        "action": ["remaining"],
        "slots": ["year?", "vacationtype?"],
        "phrases_zh": ["剩餘特休", "剩餘特休假", "年假餘額", "還有年假", "未用年假"],
        "phrases_en": ["remaining annual leave", "unused pto", "annual leave balance"],
        "template_ref": "annual_balance_by_person",
    },
    {
        "id": "cancellations_detail",
        "title": "Leave cancellations with original record",
        "tables": ["dbo.ATDLEAVECANCELDATA", "dbo.ATDLEAVEDATA", "dbo.PSNACCOUNT"],
        "measure": ["any"],
        "action": ["cancelled"],
        "slots": ["date_range?"],
        "phrases_zh": ["取消", "撤銷", "改期"],
        "phrases_en": ["cancellation", "cancelled", "rescheduled", "voided"],
        "template_ref": "cancellations_detail",
    },
    {
        "id": "resolve_leave_class",
        "title": "Resolve leave type for records",
        "tables": ["dbo.ATDLEAVEDATA", "dbo.ATDATTENDANCECLASS", "dbo.PSNACCOUNT"],
        "measure": ["any"],
        "action": ["resolve_class"],
        "slots": [],
        "phrases_zh": ["假別", "請假類別", "類型"],
        "phrases_en": ["leave type", "class name", "category"],
        "template_ref": "resolve_leave_class",
    },
    {
        "id": "person_branch_map",
        "title": "Map employees to branch/department",
        "tables": ["dbo.PSNACCOUNT", ORG_TABLE],
        "measure": ["org_map"],
        "action": ["lookup"],
        "slots": [],
        "phrases_zh": ["單位", "部門", "分公司", "部門對應"],
        "phrases_en": ["branch", "department", "org mapping"],
        "template_ref": "person_branch_map",
    },
]

LEXICON = {
    "measure": {
        "annual_leave": {
            "zh": ["特休", "年假", "年休", "年假時數", "年假天數"],
            "en": ["annual leave", "pto"],
        },
        "org_map": {"zh": ["部門", "單位", "分公司"], "en": ["branch", "department", "org"]},
    },
    "action": {
        "remaining": {"zh": ["剩餘", "餘額", "未用", "還有", "可用"], "en": ["remaining", "unused", "balance", "available"]},
        "cancelled": {"zh": ["取消", "撤銷", "改期"], "en": ["cancelled", "voided", "rescheduled"]},
        "lookup": {"zh": ["查詢", "對應", "查看"], "en": ["lookup", "map"]},
        "resolve_class": {"zh": ["假別", "類別", "類型"], "en": ["class", "type", "category"]},
    },
}

def _extract_slots_from_text(q: str, lang_hint: Optional[str] = None) -> Dict[str, Optional[Any]]:
    qlow = (q or "").lower()
    # measure
    measure = None
    for key, val in LEXICON["measure"].items():
        terms = (val.get("zh", []) + val.get("en", [])) if not lang_hint else val.get(lang_hint, [])
        if any(t in q for t in terms) or any(t.lower() in qlow for t in terms):
            measure = key
            break
    # action
    action = None
    for key, val in LEXICON["action"].items():
        terms = (val.get("zh", []) + val.get("en", [])) if not lang_hint else val.get(lang_hint, [])
        if any(t in q for t in terms) or any(t.lower() in qlow for t in terms):
            action = key
            break
    # year
    m = re.search(r"(20\d{2})", q)
    year = int(m.group(1)) if m else None
    # vacationtype heuristic
    vtype = 1 if any(tok in q for tok in ["特休", "年假", "annual"]) else None
    # threshold hours (e.g., 大於200小時 / >200小時)
    th = None
    m2 = re.search(r"(?:大於|超過|>|≥|>=)\s*(\d+)\s*小時", q)
    if m2:
        th = int(m2.group(1))
    return {"measure": measure, "action": action, "year": year, "vacationtype": vtype, "threshold_hours": th}

def canonicalize_query_intent(q: str, lang: Optional[str]) -> Dict[str, Optional[Any]]:
    lang_hint = "zh" if (lang and "zh" in lang) else None
    return _extract_slots_from_text(q, lang_hint)

def rank_intent_candidates(q: str, lang: Optional[str]) -> List[Dict[str, Any]]:
    slots = canonicalize_query_intent(q, lang)
    cands = []
    txt = (q or "").lower()
    for skill in INTENT_SKILLS:
        score = 0.0
        if slots["action"] and slots["action"] in skill.get("action", []):
            score += 0.45
        if slots["measure"] and (("any" in skill.get("measure", [])) or slots["measure"] in skill.get("measure", [])):
            score += 0.35
        if any(p in q for p in skill.get("phrases_zh", [])) or any(p in txt for p in [s.lower() for s in skill.get("phrases_en", [])]):
            score += 0.20
        cands.append({
            "skill_id": skill["id"],
            "score": round(score, 4),
            "template_ref": skill["template_ref"],
            "slots": slots,
            "tables": skill["tables"],
            "title": skill["title"],
        })
    return sorted(cands, key=lambda x: x["score"], reverse=True)[:4]

# ───────────────────────────────────────────────────────────────────────────────
# zh synonyms (for recall)
# ───────────────────────────────────────────────────────────────────────────────
_ZH_TO_EN_SYNONYMS: List[Tuple[re.Pattern, str]] = [
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
    (re.compile(r"請假|休假"), "leave"),
    (re.compile(r"員工|人員|人力|同仁"), "employee person"),
    (re.compile(r"部門|單位|分行|分部"), "department branch unit"),
    (re.compile(r"部門名稱|單位名稱|分行名稱"), "department name branch name"),
    (re.compile(r"假別|假種|假期類型"), "leave type vacation type"),
    (re.compile(r"工號|員工編號|員編|人員代碼"), "employee id employeeid personid"),
    (re.compile(r"事業部|公司別|BU"), "business unit"),
    (re.compile(r"已核准|已批准|已驗證|已验证"), "validated approved"),
    (re.compile(r"餘額|余额|剩餘|剩下|可用"), "balance remaining available"),
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

# ───────────────────────────────────────────────────────────────────────────────
# Date helpers (Asia/Taipei assumed server; if not, still fine for demo)
# ───────────────────────────────────────────────────────────────────────────────
def _today_date() -> datetime:
    return datetime.now()

def _iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")

def _week_window(dt: Optional[datetime] = None) -> Tuple[str, str]:
    dt = dt or _today_date()
    # ISO week: Monday start
    start = dt - timedelta(days=(dt.weekday()))
    end = start + timedelta(days=6)
    return _iso(start), _iso(end)

def _parse_mmdd_range(txt: str) -> Optional[Tuple[str, str]]:
    # Accept: 9/22-9/26 or 09/22 ~ 09/26 or 9/22至9/26
    m = re.search(r'(\d{1,2})\s*/\s*(\d{1,2})\s*[-~至到]\s*(\d{1,2})\s*/\s*(\d{1,2})', txt)
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

# ───────────────────────────────────────────────────────────────────────────────
# Data model meta (unchanged)
# ───────────────────────────────────────────────────────────────────────────────
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

# Column alias heuristics (unchanged)
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

# ───────────────────────────────────────────────────────────────────────────────
# Main Vector DB
# ───────────────────────────────────────────────────────────────────────────────
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

    # ───────────────────────────────────────────────────────────────────────
    # DEMO-SAFE: public no-op to stop VDB_RECORD_OUTCOME_FAIL warnings
    # ───────────────────────────────────────────────────────────────────────
    def record_outcome(self, **kwargs) -> None:
        # Intentionally a no-op for demo stability
        try:
            logger.debug("LeaveVectorDB.record_outcome: %s", {k: v for k, v in kwargs.items() if k in ("query","success","tables")})
        except Exception:
            pass

    # ───────────────────────────────────────────────────────────────────────
    # Internal helpers: schema pickers
    # ───────────────────────────────────────────────────────────────────────
    def _pick(self, table_full: str, logical_key: str, default: Optional[str] = None) -> str:
        """Pick the first existing column mapped for logical_key on table_full."""
        t = self._by_name.get(table_full.lower())
        if not t:
            return default or logical_key
        # Resolve logical class
        if table_full.lower().endswith("psnaccount"):
            pool = SCHEMA_MAP["PSNACCOUNT"].get(logical_key, [])
        elif table_full.lower().endswith("orgstdstruct") or table_full.lower() == ORG_TABLE.lower():
            pool = SCHEMA_MAP["ORG"].get(logical_key, [])
        elif table_full.lower().endswith("atdleavedata"):
            pool = SCHEMA_MAP["ATDLEAVEDATA"].get(logical_key, [])
        elif table_full.lower() == VAC_RESULT_TABLE.lower():
            pool = SCHEMA_MAP["ATDCALCUVACATIONRESULT"].get(logical_key, [])
        else:
            pool = []
        colsU = {c.upper(): c for c in t.columns}
        for cand in pool:
            if cand.upper() in colsU:
                return colsU[cand.upper()]
        # fallback: first col if we must not fail
        if default:
            return default
        return next(iter(t.columns), logical_key)

    def _emp_no_col(self) -> str:
        pt = self._by_name.get((self._person_table or "").lower())
        if not pt:
            return "EMPLOYEEID"
        cand = SCHEMA_MAP["PSNACCOUNT"]["EMP_NO"]
        colsU = {c.upper(): c for c in pt.columns}
        for c in cand:
            if c.upper() in colsU:
                return colsU[c.upper()]
        return "EMPLOYEEID"

    # ───────────────────────────────────────────────────────────────────────
    # Intent classification (cheat router)
    # ───────────────────────────────────────────────────────────────────────
    def _classify_intent(self, q: str) -> str:
        t = (q or "").lower()
        # history with employee number
        if any(k in t for k in ["歷史", "紀錄", "記錄", "history", "past", "trend"]):
            if any(k in t for k in ["員工編號", "工號", "員編", "emp", "employee"]):
                return "PERSON_HISTORY_BY_EMP_NO"
            return "LEAVE_HISTORY_GENERIC"
        # remaining balance
        if any(k in t for k in ["剩餘", "餘額", "余额", "還有", "可用", "balance", "remaining", "available"]) and any(k in t for k in ["特休", "年假", "annual"]):
            if any(k in t for k in ["大於", ">", "超過", "至少", "more than"]):
                return "BALANCE_YEAR_THRESHOLD_HOURS"
            return "BALANCE_YEAR_REMAINING"
        # explicit ranges
        if "-" in t or "至" in t or "到" in t:
            return "RANGE_WHO_ON_LEAVE"
        # week
        if any(k in t for k in ["本週", "這週", "this week"]):
            if any(k in t for k in ["多少人", "人數", "count", "一人", "1人"]):
                return "WEEKLY_COUNT"
            return "WEEKLY_WHO_ON_LEAVE"
        # today / yesterday
        if any(k in t for k in ["今天", "今日", "today"]):
            return "TODAY_WHO_ON_LEAVE"
        if any(k in t for k in ["昨天", "昨日", "yesterday"]):
            return "YDAY_WHO_ON_LEAVE"
        # month
        if any(k in t for k in ["本月", "這個月", "this month"]):
            if "統計" in t or "總時數" in t:
                return "MONTH_SUM_HOURS"
            return "MONTH_WHO_ON_LEAVE"
        return "GENERIC_LEAVE_LOOKUP"

    # ───────────────────────────────────────────────────────────────────────
    # Intent routing (kept API, returns template_ref for vector_search_service)
    # ───────────────────────────────────────────────────────────────────────
    def get_intent_routing(self, query: str) -> Dict[str, Any]:
        lang = detect_language(query)
        q = (query or "").lower()

        plan: Dict[str, Any] = {
            "intent": "generic",
            "template_ref": None,
            "slots": {},
            "tables": [t.full for t in self.tables[:3]],
            "language": lang,
        }

        # year + threshold extraction
        slots = canonicalize_query_intent(query, lang)
        if slots.get("year"):
            plan["slots"]["year"] = slots["year"]
        if slots.get("threshold_hours"):
            plan["slots"]["threshold_hours"] = slots["threshold_hours"]

        # date anchoring
        rng = _parse_mmdd_range(q)
        if rng:
            plan["slots"]["start_date"], plan["slots"]["end_date"] = rng
        elif any(k in q for k in ["本週", "這週", "this week"]):
            s, e = _week_window()
            plan["slots"]["start_date"], plan["slots"]["end_date"] = s, e

        # employee number (keep leading zeros)
        m_emp = re.search(r"(?:員工編號|工號|員編|employee\s*no\.?|empid|emp\s*no\.?)\s*[:：]?\s*([A-Za-z0-9\-]+)", query)
        if m_emp:
            plan["slots"]["emp_no"] = m_emp.group(1).strip()

        # classify
        label = self._classify_intent(query)
        plan["intent"] = label

        # table routing + template_ref
        psn = self._person_table or "dbo.PSNACCOUNT"
        if label in ("BALANCE_YEAR_REMAINING", "BALANCE_YEAR_THRESHOLD_HOURS"):
            plan.update({
                "template_ref": "annual_balance_by_person" if label == "BALANCE_YEAR_REMAINING" else "balance_year_threshold_hours",
                "tables": [VAC_RESULT_TABLE, psn, ORG_TABLE],
            })
        elif label in ("PERSON_HISTORY_BY_EMP_NO", "LEAVE_HISTORY_GENERIC", "RANGE_WHO_ON_LEAVE",
                       "WEEKLY_WHO_ON_LEAVE", "TODAY_WHO_ON_LEAVE", "YDAY_WHO_ON_LEAVE", "MONTH_WHO_ON_LEAVE"):
            plan.update({
                "template_ref": {
                    "PERSON_HISTORY_BY_EMP_NO": "person_history_by_empno",
                    "LEAVE_HISTORY_GENERIC": "person_history_generic",
                    "RANGE_WHO_ON_LEAVE": "range_who_on_leave",
                    "WEEKLY_WHO_ON_LEAVE": "weekly_who_on_leave",
                    "TODAY_WHO_ON_LEAVE": "today_who_on_leave",
                    "YDAY_WHO_ON_LEAVE": "yday_who_on_leave",
                    "MONTH_WHO_ON_LEAVE": "month_who_on_leave",
                }[label],
                "tables": ["dbo.ATDLEAVEDATA", psn, ORG_TABLE],
            })
        elif label == "WEEKLY_COUNT":
            plan.update({
                "template_ref": "weekly_count_people",
                "tables": ["dbo.ATDLEAVEDATA", psn],
            })
        else:
            # generic fallback
            if any(k in q for k in ["balance", "餘額", "年假", "特休"]) and self._exists(VAC_RESULT_TABLE):
                plan.update({"template_ref": "annual_balance_by_person", "tables": [VAC_RESULT_TABLE, psn, ORG_TABLE]})
            else:
                plan.update({"template_ref": "range_who_on_leave", "tables": ["dbo.ATDLEAVEDATA", psn, ORG_TABLE]})

        return plan

    def _resolve_person_table(self) -> Optional[str]:
        for name in ["dbo.PSNACCOUNT", "dbo.PSNACCOUNT_D"]:
            if name.lower() in self._by_name:
                return name
        return None

    def _exists(self, full: str) -> bool:
        return (full or "").lower() in self._by_name

    # ---------------- embeddings/index
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
            en_parts = [j.description, j.purpose, " ".join(j.tags), j.left_table, j.right_table,
                        f"{j.left_table}.{j.left_column}={j.right_table}.{j.right_column}",
                        j.join_type.value, j.cardinality.value]
            zh_parts = [j.description_zh or j.description, j.purpose, " ".join(j.tags), j.left_table, j.right_table,
                        f"{j.left_table}.{j.left_column}={j.right_table}.{j.right_column}",
                        j.join_type.value, j.cardinality.value]
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
                item_type=VectorItemType.JOIN, text_en=en_text, text_zh=zh_text, priority=2, payload={"join": j}
            ))
        for p in self._query_patterns:
            en_text, zh_text = self._combine_text_by_language(p, "pattern")
            self._vector_items.append(VectorItem(
                key=f"PATTERN::{p.pattern}", item_type=VectorItemType.PATTERN,
                text_en=en_text, text_zh=zh_text, priority=1, payload={"pattern": p}
            ))
        for k in self._kpis:
            en_text, zh_text = self._combine_text_by_language(k, "kpi")
            self._vector_items.append(VectorItem(
                key=f"KPI::{k.name}", item_type=VectorItemType.KPI,
                text_en=en_text, text_zh=zh_text, priority=1, payload={"kpi": k}
            ))
        for r in self._recipes:
            en_text, zh_text = self._combine_text_by_language(r, "recipe")
            self._vector_items.append(VectorItem(
                key=f"RECIPE::{r.recipe_id}", item_type=VectorItemType.RECIPE,
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
            results += self._do_search_once(q_expanded, other_lang, top_k)
        dedup: Dict[str, Tuple[VectorItem, float]] = {}
        for vi, s in results:
            if s < min_score:
                continue
            if vi.key not in dedup or s > dedup[vi.key][1]:
                dedup[vi.key] = (vi, s)
        out = sorted(dedup.values(), key=lambda x: x[1], reverse=True)[:top_k]
        logger.info("VDB_SEARCH: final_hits=%d", len(out))
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
            if any(k in ql for k in ["balance", "餘額", "年假", "特休"]):
                if self._exists(VAC_RESULT_TABLE):
                    likely.append((VAC_RESULT_TABLE, 0.26))
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
            hints.append(f"-- e.g. LEFT JOIN {self._person_table} p ON p.PERSONID = <FACT>.PERSONID  (declare your fact alias first)")
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

    # ───────────────────────────────────────────────────────────────────────
    # Canonical recipes — now include demo cheats for history/range/week
    # ───────────────────────────────────────────────────────────────────────
    def _build_recipes(self) -> List[SQLRecipe]:
        psn = self._person_table or "dbo.PSNACCOUNT"

        # helper lambdas for column picking
        def P(table: str, key: str, default: Optional[str] = None):
            return self._pick(table, key, default=default)

        person_id_psn = P(psn, "PERSON_ID", default="PERSONID")
        name_col = P(psn, "NAME", default="TRUENAME")
        emp_no_col = self._emp_no_col()

        org_unitdisp = self._pick(ORG_TABLE, "UNIT_DISPLAY", default="UNITDISPLAYNAME")
        org_unitname = self._pick(ORG_TABLE, "UNIT_NAME", default="UNITNAME")
        dept_expr = f"COALESCE(org.{org_unitdisp}, org.{org_unitname})"

        # ATDLEAVEDATA picks
        ld = "dbo.ATDLEAVEDATA"
        ld_person = P(ld, "PERSON_ID", default="PERSONID")
        ld_workdate = P(ld, "WORK_DATE", default="WORKDATE")
        ld_hours = P(ld, "HOURS", default="HOURS")
        ld_type = P(ld, "TYPE", default="ATTENDANCETYPE")
        ld_validated = P(ld, "VALIDATED", default="VALIDATED")
        ld_start = P(ld, "START_DATE", default="STARTDATE")
        ld_end = P(ld, "END_DATE", default="ENDDATE")

        # VAC RESULT picks
        vr = VAC_RESULT_TABLE
        vr_person = self._pick(vr, "PERSON_ID", default="PERSONID")
        vr_year = self._pick(vr, "YEAR", default="VACAYEAR")
        vr_type = self._pick(vr, "TYPE", default="VACATIONTYPE")
        vr_remain_days = self._pick(vr, "REMAIN_DAYS", default="REMAINDAYS")
        vr_remain_hours = self._pick(vr, "REMAIN_HOURS", default=None)  # may be missing
        vr_vacdays = self._pick(vr, "VAC_DAYS", default="VACDAYS")
        vr_usedays = self._pick(vr, "USE_DAYS", default="USEDAYS")
        vr_canuse = self._pick(vr, "CAN_USE_DATE", default="CANUSEDATE")
        vr_disable = self._pick(vr, "DISABLE_DATE", default="DISABLEDDATE")
        vr_updated = self._pick(vr, "UPDATED_AT", default="LASTEDITTIME")
        vr_created = self._pick(vr, "CREATED_AT", default="CREATIONTIME")

        recipes: List[SQLRecipe] = []

        # 1) Current on leave by department
        recipes.append(SQLRecipe(
            recipe_id="current_on_leave_by_dept",
            title="Current on-leave employees by department",
            description="Lists employees currently on validated leave grouped by department.",
            description_zh="按部門列出當前已批准且在休假的員工（含部門＋員編＋姓名）",
            tables=[ld, psn, ORG_TABLE],
            expected_columns=["department_name", "employee_id", "person_name", ld_type, ld_start, ld_end],
            sql_template=(
                "SELECT "
                f"  {dept_expr} AS department_name,\n"
                f"  p.{emp_no_col} AS employee_id,\n"
                f"  p.{name_col}   AS person_name,\n"
                f"  fact.{ld_type} AS AttendanceType,\n"
                f"  CAST(fact.{ld_start} AS date) AS STARTDATE,\n"
                f"  CAST(fact.{ld_end}   AS date) AS ENDDATE\n"
                f"FROM {ld} AS fact\n"
                f"LEFT JOIN {psn} AS p ON p.{person_id_psn} = fact.{ld_person}\n"
                f"LEFT JOIN {ORG_TABLE} AS org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.{self._pick(ORG_TABLE,'UNIT_ID','UNITID')} AS NVARCHAR(100))\n"
                f"WHERE fact.{ld_validated} = 1\n"
                f"  AND CAST(GETDATE() AS date) BETWEEN CAST(fact.{ld_start} AS date) AND CAST(fact.{ld_end} AS date)\n"
                "ORDER BY department_name, person_name;"
            ),
            caution_notes=["Ensure BRANCHID matches ORG.UNITID via NVARCHAR CAST."],
            tags=["current","validated","preview"]
        ))

        # 2) Annual balance by person
        recipes.append(SQLRecipe(
            recipe_id="annual_balance_by_person",
            title="Annual leave balance by person (snapshot table)",
            description="Latest valid snapshot per person/year/type with entitlement/used/remaining days.",
            description_zh="每人/年度/類型之最新有效快照（給予/已用/剩餘天數）。",
            tables=[vr, psn, ORG_TABLE],
            expected_columns=["department_name", "employee_id", "person_name", vr_year, vr_type, vr_vacdays, vr_usedays, vr_remain_days, vr_canuse, vr_disable],
            sql_template=(
                "WITH latest AS (\n"
                f"  SELECT r.{vr_person}, r.{vr_year}, r.{self._pick(vr,'VAC_MONTH','VACAMONTH')}, r.{vr_type},\n"
                f"         r.{vr_vacdays}, r.{vr_usedays}, r.{vr_remain_days}, r.{vr_canuse}, r.{vr_disable},\n"
                f"         r.{vr_updated}, r.{vr_created},\n"
                "         ROW_NUMBER() OVER (\n"
                f"           PARTITION BY r.{vr_person}, r.{vr_year}, r.{vr_type}\n"
                f"           ORDER BY ISNULL(r.{vr_updated}, r.{vr_created}) DESC,\n"
                f"                    ISNULL(r.{vr_disable}, '9999-12-31') DESC,\n"
                f"                    r.{self._pick(vr,'VAC_MONTH','VACAMONTH')} DESC\n"
                "         ) AS rn\n"
                f"  FROM {vr} AS r\n"
                "  WHERE (@year IS NULL OR r." + vr_year + " = @year)\n"
                "    AND (@vacationtype IS NULL OR r." + vr_type + " = @vacationtype)\n"
                "    AND (r." + vr_canuse + " IS NULL OR CAST(@today AS date) >= CAST(r." + vr_canuse + " AS date))\n"
                "    AND (r." + vr_disable + " IS NULL OR CAST(@today AS date) <= CAST(r." + vr_disable + " AS date))\n"
                ")\n"
                "SELECT " + dept_expr + " AS department_name,\n"
                f"       p.{emp_no_col} AS employee_id,\n"
                f"       p.{name_col}   AS person_name,\n"
                f"       l.{vr_year} AS VACAYEAR, l.{vr_type} AS VACATIONTYPE, l.{vr_vacdays} AS VACDAYS, l.{vr_usedays} AS USEDAYS, l.{vr_remain_days} AS REMAINDAYS,\n"
                f"       l.{vr_canuse} AS CANUSEDATE, l.{vr_disable} AS DISABLEDDATE\n"
                "FROM latest AS l\n"
                f"LEFT JOIN {psn} AS p ON p.{person_id_psn} = l.{vr_person}\n"
                f"LEFT JOIN {ORG_TABLE} AS org ON CAST(p.BRANCHID AS NVARCHAR(100)) = CAST(org.{self._pick(ORG_TABLE,'UNIT_ID','UNITID')} AS NVARCHAR(100))\n"
                "WHERE l.rn = 1\n"
                "ORDER BY department_name, person_name;"
            ),
            caution_notes=[
                "Use @year/@vacationtype; respect validity window via @today.",
                "VACATIONTYPE code for annual leave is system-specific (often 1)."
            ],
            tags=["balance","annual","snapshot","days","force-balance"]
        ))

        # 3) Balance with hours threshold (uses REMAIN_HOURS if available, else days*8)
        if vr_remain_hours:
            thr_expr = f"r.{vr_remain_hours}"
            hours_alias = vr_remain_hours
        else:
            thr_expr = f"(r.{vr_remain_days} * 8)"
            hours_alias = "REMAIN_HOURS_APPROX"
        recipes.append(SQLRecipe(
            recipe_id="balance_year_threshold_hours",
            title="Annual balance over hour threshold",
            description="Remaining annual leave over threshold (hours); uses hours if available, else days*8.",
            description_zh="剩餘年假超過門檻（小時）；若無小時欄位，改用天數*8。",
            tables=[vr, psn],
            expected_columns=[person_id_psn, "TrueName", "VacYear", hours_alias],
            sql_template=(
                "SELECT r." + vr_person + " AS PersonID,\n"
                f"       p.{name_col} AS TrueName,\n"
                f"       r.{vr_year} AS VacYear,\n"
                f"       {thr_expr} AS {hours_alias}\n"
                f"FROM {vr} r\n"
                f"LEFT JOIN {psn} p ON p.{person_id_psn} = r.{vr_person}\n"
                "WHERE r." + vr_year + " = @year\n"
                "  AND " + thr_expr + " > @threshold_hours\n"
                "ORDER BY " + hours_alias + " DESC;"
            ),
            caution_notes=["If REMAIN_HOURS is missing, conversion assumes 1 day = 8 hours."],
            tags=["balance","threshold","hours"]
        ))

        # 4) Range who-on-leave
        recipes.append(SQLRecipe(
            recipe_id="range_who_on_leave",
            title="Employees on leave within date range",
            description="Distinct people with leave records between @start_date and @end_date.",
            description_zh="在 @start_date..@end_date 期間有請假的人員（去重）。",
            tables=[ld, psn],
            expected_columns=["PersonID", "TrueName", "WorkDate", "LeaveType", "LeaveHours"],
            sql_template=(
                "SELECT DISTINCT\n"
                f"  ld.{ld_person} AS PersonID,\n"
                f"  p.{name_col}   AS TrueName,\n"
                f"  CAST(ld.{ld_workdate} AS date) AS WorkDate,\n"
                f"  ld.{ld_type}   AS LeaveType,\n"
                f"  COALESCE(ld.{ld_hours}, 0) AS LeaveHours\n"
                f"FROM {ld} AS ld\n"
                f"LEFT JOIN {psn} AS p ON p.{person_id_psn} = ld.{ld_person}\n"
                "WHERE CAST(ld." + ld_workdate + " AS date) BETWEEN CAST(@start_date AS date) AND CAST(@end_date AS date)\n"
                "ORDER BY WorkDate ASC, TrueName ASC;"
            ),
            tags=["range","who","leave"]
        ))

        # 5) Weekly count
        recipes.append(SQLRecipe(
            recipe_id="weekly_count_people",
            title="Count of people on leave (weekly window)",
            description="COUNT DISTINCT people with any leave record within @week_start..@week_end.",
            description_zh="@week_start..@week_end 期間請假之人次（去重）。",
            tables=[ld],
            expected_columns=["PeopleOnLeave"],
            sql_template=(
                "SELECT COUNT(DISTINCT ld." + ld_person + ") AS PeopleOnLeave\n"
                f"FROM {ld} AS ld\n"
                "WHERE CAST(ld." + ld_workdate + " AS date) BETWEEN CAST(@week_start AS date) AND CAST(@week_end AS date);"
            ),
            tags=["weekly","count","kpi"]
        ))

        # 6) Person history by employee number
        recipes.append(SQLRecipe(
            recipe_id="person_history_by_empno",
            title="Leave history by employee number",
            description="All leave records for a given employee number (string, keep leading zeros).",
            description_zh="指定員工編號之歷史請假紀錄（字串，比對時保留前導零）。",
            tables=[ld, psn],
            expected_columns=["PersonID", "TrueName", "WorkDate", "LeaveType", "LeaveHours"],
            sql_template=(
                "SELECT\n"
                f"  ld.{ld_person} AS PersonID,\n"
                f"  p.{name_col}   AS TrueName,\n"
                f"  CAST(ld.{ld_workdate} AS date) AS WorkDate,\n"
                f"  ld.{ld_type}   AS LeaveType,\n"
                f"  COALESCE(ld.{ld_hours}, 0) AS LeaveHours\n"
                f"FROM {ld} AS ld\n"
                f"LEFT JOIN {psn} AS p ON p.{person_id_psn} = ld.{ld_person}\n"
                f"WHERE p.{emp_no_col} = @emp_no  -- NVARCHAR, do not cast; keep leading zeros\n"
                "ORDER BY WorkDate DESC;"
            ),
            caution_notes=["Bind @emp_no as NVARCHAR; do not trim/cast."],
            tags=["person","history"]
        ))

        # 7) Today / Yesterday / Month who-on-leave (simple variants)
        recipes.append(SQLRecipe(
            recipe_id="today_who_on_leave",
            title="Who is on leave today",
            description="People on leave for today.",
            description_zh="今天請假的人員。",
            tables=[ld, psn],
            expected_columns=["PersonID", "TrueName", "WorkDate"],
            sql_template=(
                "SELECT DISTINCT ld." + ld_person + " AS PersonID, p." + name_col + " AS TrueName,\n"
                "       CAST(ld." + ld_workdate + " AS date) AS WorkDate\n"
                f"FROM {ld} ld\n"
                f"LEFT JOIN {psn} p ON p.{person_id_psn} = ld.{ld_person}\n"
                "WHERE CAST(ld." + ld_workdate + " AS date) = CAST(@today AS date)\n"
                "ORDER BY TrueName;"
            ),
            tags=["today","who"]
        ))
        recipes.append(SQLRecipe(
            recipe_id="yday_who_on_leave",
            title="Who was on leave yesterday",
            description="People on leave for yesterday.",
            description_zh="昨天請假的人員。",
            tables=[ld, psn],
            expected_columns=["PersonID", "TrueName", "WorkDate"],
            sql_template=(
                "SELECT DISTINCT ld." + ld_person + " AS PersonID, p." + name_col + " AS TrueName,\n"
                "       CAST(ld." + ld_workdate + " AS date) AS WorkDate\n"
                f"FROM {ld} ld\n"
                f"LEFT JOIN {psn} p ON p.{person_id_psn} = ld.{ld_person}\n"
                "WHERE CAST(ld." + ld_workdate + " AS date) = CAST(@yesterday AS date)\n"
                "ORDER BY TrueName;"
            ),
            tags=["yesterday","who"]
        ))
        recipes.append(SQLRecipe(
            recipe_id="month_who_on_leave",
            title="Who is on leave this month",
            description="People with leave in current month.",
            description_zh="本月請假的人員。",
            tables=[ld, psn],
            expected_columns=["PersonID", "TrueName", "WorkDate"],
            sql_template=(
                "SELECT DISTINCT ld." + ld_person + " AS PersonID, p." + name_col + " AS TrueName,\n"
                "       CAST(ld." + ld_workdate + " AS date) AS WorkDate\n"
                f"FROM {ld} ld\n"
                f"LEFT JOIN {psn} p ON p.{person_id_psn} = ld.{ld_person}\n"
                "WHERE FORMAT(CAST(ld." + ld_workdate + " AS date), 'yyyy-MM') = FORMAT(CAST(@today AS date), 'yyyy-MM')\n"
                "ORDER BY TrueName;"
            ),
            tags=["month","who"]
        ))

        return recipes

    # ---------------- readiness, health, persistence
    def is_ready(self) -> bool:
        return bool(self.tables) and (self.embeddings_en is not None or self.embeddings_zh is not None)

    def health_check(self) -> Dict[str, object]:
        faiss_used = bool(faiss is not None and self.index_en is not None and self.index_zh is not None)
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
            "version": "6.0",  # bumped for demo cheat router + recipes
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
            model_name=data.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            joins=joins,
            patterns=patterns,
            kpis=kpis,
            recipes=recipes,
        )
        return inst

    # ---------------- convenience
    def get_query_pattern(self, query: str) -> Optional[QueryPattern]:
        q = (query or "").lower()
        if any(x in q for x in ["當前", "今天", "現在", "今日", "目前"]) or ("leave" in q and any(x in q for x in ["current", "today", "now"])):
            return next((p for p in self._query_patterns if p.pattern == "current_leave_status"), None)
        if any(x in q for x in ["取消", "已取消"]) or any(x in q for x in ["cancel", "cancelled", "cancellation"]):
            return next((p for p in self._query_patterns if p.pattern == "leave_cancellations"), None)
        if (
            any(x in q for x in ["餘額", "余额", "剩餘", "剩下", "可用"]) and
            any(x in q for x in ["年假", "特休", "annual"])
        ) or any(x in q for x in ["annual leave balance", "remaining annual", "vacation balance"]):
            return next((p for p in self._query_patterns if p.pattern == "annual_balance_snapshot"), None)
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

    def debug_intent_block(self, query: str) -> str:
        routing = self.get_intent_routing(query)
        lines = [f"lang={routing.get('lang') or routing.get('language')}",
                 f"intent={routing.get('intent')}", f"template_ref={routing.get('template_ref')}"]
        slots = routing.get("slots") or {}
        if slots:
            lines.append(f"slots={json.dumps(slots, ensure_ascii=False)}")
        for i, c in enumerate(routing.get("candidates", []), 1):
            lines.append(f"[{i}] {c['skill_id']} score={c['score']} slots={json.dumps(c['slots'], ensure_ascii=False)} tables={','.join(c['tables'])}")
        return "\n".join(lines)

    # ───────────────────────────────────────────────────────────────────────
    # Schema context + DEMO CHEAT: inject the selected canonical recipe
    # ───────────────────────────────────────────────────────────────────────
    @staticmethod
    def preview_projection_sql(lang: Literal["zh-tw","en"] = "en") -> str:
        dept_expr = "COALESCE(org.UNITDISPLAYNAME, org.UNITNAME)"
        if lang == "zh-tw":
            return f"{dept_expr} AS 部門, p.EMPLOYEEID AS 員編, p.TRUENAME AS 姓名"
        else:
            return f"{dept_expr} AS department_name, p.EMPLOYEEID AS employee_id, p.TRUENAME AS person_name"

    def get_business_prompt(self, query: str) -> str:
        lang = detect_language(query)
        routing = self.get_intent_routing(query)
        context = self.get_schema_context(query, include_examples=True)
        pv = self.preview_projection_sql("zh-tw" if lang == "zh-tw" else "en")

        # DEMO: inject the exact canonical recipe matching template_ref
        selected_recipe_sql = None
        tr = routing.get("template_ref")
        if tr:
            for r in self._recipes:
                if r.recipe_id == tr:
                    selected_recipe_sql = r.sql_template
                    break

        intent_lines = []
        top_intent = routing.get("intent")
        if top_intent:
            intent_lines.append(f"intent={top_intent}")
            intent_lines.append(f"template_ref={tr}")
            intent_lines.append(f"slots={json.dumps(routing.get('slots', {}), ensure_ascii=False)}")
            intent_lines.append(f"tables_hint={','.join(routing.get('tables', []))}")
        intent_block = ("\n".join(intent_lines)) if intent_lines else "(no strong intent match)"

        # Build prompt
        if lang == "zh-tw":
            prompt = f"""
您是一位請假/考勤領域的專業分析工程師。請提供可執行且正確的 SQL 與推理（以業務可讀輸出為目標）。

使用者查詢：
{query}

[意圖與路由]
{intent_block}

{context}

=== DEMO CANONICAL SQL（請嚴格遵循此模板，僅替換變數）===
{(selected_recipe_sql or '-- (未匹配到模板；請參考上方上下文)')}

必須遵守（SQL SAFETY）：
1) 別名需先宣告；避免引用未宣告的別名。
2) 盡量使用完整表名與欄位限定（table.column）。
3) 餘額/給予以 {VAC_RESULT_TABLE} 為權威；於有效期內（CANUSEDATE..DISABLEDDATE）加入 @today 條件。
4) 大表 ATD* 務必加日期過濾；核准請假含 VALIDATED = 1。
5) 人員/部門 JOIN 僅在需要時；BRANCHID 與 ORG.UNITID 以 NVARCHAR CAST 對齊。

預覽欄位（若需顯示人員/部門）：
{pv}

回傳格式：
- 簡短推理要點
- SQL
- 預期欄位與資料粒度
- 後續驗證建議（可選）
""".strip()
        else:
            prompt = f"""
You are an expert analytics engineer for the leave/attendance domain. Provide executable, correct SQL with reasoning.

USER QUERY:
{query}

[INTENT & ROUTING]
{intent_block}

{context}

=== DEMO CANONICAL SQL (FOLLOW THIS TEMPLATE; ONLY SUBSTITUTE VARIABLES) ===
{(selected_recipe_sql or '-- (no matched template; rely on context above)')}

SQL SAFETY:
1) Declare aliases before use; never reference undeclared aliases.
2) Prefer fully-qualified tables and qualified columns.
3) For balances/entitlement, prefer {VAC_RESULT_TABLE}; respect @today in CANUSEDATE..DISABLEDDATE.
4) For large ATD* facts, always add date filters; use VALIDATED = 1 for approved.
5) Join person/department only when needed; align BRANCHID↔ORG.UNITID via NVARCHAR casts.

Preview fields (if showing person/department):
{pv}

Return:
- Short reasoning bullets
- SQL
- Expected columns and grain
- Optional follow-up checks
""".strip()
        return prompt

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
                f"• 餘額/給予：以 {VAC_RESULT_TABLE} 為權威；需套用 @today 在可用區間（CANUSEDATE ≤ @today ≤ DISABLEDDATE）。",
                "• 使用明細/取消：以 ATDLEAVEDATA / ATDLEAVECANCELDATA。",
                "• 歷史大表務必加日期範圍過濾；核准資料包含 VALIDATED = 1。",
            ])
        else:
            lines.extend([
                "=== Query Construction Tips ===",
                f"• Balances/entitlement: authoritative via {VAC_RESULT_TABLE}; respect @today within CANUSEDATE..DISABLEDDATE.",
                "• Usage/cancellation detail: ATDLEAVEDATA / ATDLEAVECANCELDATA.",
                "• Filter large facts by date; include VALIDATED = 1 for approved records.",
            ])
        return "\n".join(lines)

    # ---------------- metadata builders
    def _build_comprehensive_joins(self) -> List[TableJoin]:
        joins: List[TableJoin] = []
        leave_core = ["dbo.ATDLEAVEDATA", "dbo.ATDLEAVECANCELDATA", VAC_RESULT_TABLE]

        if self._person_table:
            for lt in leave_core:
                if self._exists(lt) and _has_col(self._by_name[lt.lower()], "PERSONID"):
                    joins.append(TableJoin(
                        left_table=lt, left_column="PERSONID",
                        right_table=self._person_table, right_column="PERSONID",
                        join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                        description="Resolve PERSONID to person attributes (TRUENAME, EMPLOYEEID, BRANCHID). VAC snapshot is authoritative for balances.",
                        description_zh="以 PERSONID 關聯人員屬性（姓名/員編/BRANCHID）。餘額以快照表為權威。",
                        purpose="Preview needs names/employee_id and department joins.",
                        tags=["person","dimension","name","employee_id","branch","balance","authoritative"]
                    ))
        if self._exists(self._person_table or "") and self._exists(ORG_TABLE):
            joins.append(TableJoin(
                left_table=self._person_table, left_column="BRANCHID",
                right_table=ORG_TABLE, right_column="UNITID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                description="Resolve BRANCHID to department/branch (UNITID → UNITDISPLAYNAME/UNITNAME/UNITCODE).",
                description_zh="BRANCHID 連到部門（UNITID → UNITDISPLAYNAME/UNITNAME/UNITCODE）。",
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
            QueryPattern(
                pattern="annual_balance_snapshot",
                description="Annual leave balance (days) from authoritative snapshot with validity window",
                description_zh="從權威快照表取年度年假餘額（天），包含有效期過濾",
                primary_tables=[VAC_RESULT_TABLE],
                suggested_joins=["PERSONID"],
                required_filters=["VACAYEAR=@year", "CANUSEDATE..DISABLEDDATE contains @today"],
                performance_notes=["Pick latest row per PERSONID/YEAR/TYPE"],
                examples=["Annual remaining days by person and department"],
                examples_zh=["各部門人員年假剩餘天數"],
                tags=["balance","annual","snapshot","force-balance"]
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
            KPIDef(
                name="annual_leave_remaining_days",
                description="Remaining annual leave days per person (from snapshot table).",
                description_zh="每人年假剩餘天數（取快照表 REMAINDAYS）。",
                formula_sql_hint="Pick latest valid row per person/year/type=1; use REMAINDAYS.",
                tables=[VAC_RESULT_TABLE],
                grain="person-year",
                tags=["balance","days","force-balance"]
            ),
        ]

# ───────────────────────────────────────────────────────────────────────────────
# Builder
# ───────────────────────────────────────────────────────────────────────────────
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
        # Usage fact (lines/hours/days)
        T("dbo.ATDLEAVEDATA",
          ["ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","STARTDATE","ENDDATE",
           "AutoRevise","TIMECLASSID","TIMECLASSHOURS","CREATIONTIME","CREATEDBY",
           "LASTUPDATETIME","LASTUPDATEDBY","LEAVEID","FORMKIND","FROM_SOURCE",
           "FORM_NO","RECORD_ID","FLEAVEBYDAYTYPE","TLEAVEBYDAYTYPE","LEAVEREASON"],
          desc="Leave usage lines (hours/days) including date span; VALIDATED=1 → approved.",
          description_zh="請假使用明細（小時/天），包含起迄；VALIDATED=1 表示已批准。",
          tags=["leave","usage","validated"],
          priority=1,
          business_context="Operational day-level leave records",
          business_context_zh="日粒度的作業性請假紀錄",
          common_queries=["Who is on leave today?","Count validated leaves by department"],
          common_queries_zh=["今天有哪些員工在休假？","各部門已核准請假數量"],
          key_cols={"PERSONID": "Person key", "DEPARTMENTID": "Department key"},
        ),
        # Cancellations
        T("dbo.ATDLEAVECANCELDATA",
          ["OID","ATTENDANCETYPE","PERSONID","WORKDATE","STARTDATE","STARTTIME","ENDDATE","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","ISHISDATA","AutoRevise","CREATEDATE","CREATEUSERID",
           "LASTEDITUSERID","LASTEDITTIME","REASON","FLEAVEBYDAYTYPE","TLEAVEBYDAYTYPE","LEAVEREASON",
           "FROM_SOURCE","FORM_NO","RECORD_ID","FORMKIND"],
          desc="Leave cancellation records (reversals).",
          description_zh="請假取消資料（沖銷）。",
          tags=["leave","cancel","reason"],
          priority=2,
          temporal=["WORKDATE","CREATEDATE","LASTEDITTIME"],
        ),
        # Authoritative snapshot: balances/entitlement (days) per person/year/month/type
        T(VAC_RESULT_TABLE,
          ["VACATIONID","PERSONID","VACAYEAR","VACAMONTH","VACATIONTYPE","USEDAYS","REMAINDAYS","VACDAYS",
           "CANUSEDATE","LASTYEARREMAINDAYS","BUSINESSUNITID","DISABLEDDATE","LASTEDITUSERID","LASTEDITTIME",
           "CREATIONTIME","CREATEDBY","TOPAYROLLDATE","PACKAGEID","PROCSTATE"],
          desc="Computed vacation snapshot (balances/usage/entitlement) by person, year, month, type.",
          description_zh="每人/年/月/類型之已計算休假快照（使用/剩餘/給予等）。",
          tags=["balance","usage","entitlement","vacation"],
          key_cols={
              "PERSONID": "人員鍵（連接 PSNACCOUNT）",
              "VACAYEAR": "休假年度",
              "VACAMONTH": "休假月份",
              "VACATIONTYPE": "休假類型（年假通常為 1）",
              "USEDAYS": "已使用天數（累計）",
              "REMAINDAYS": "剩餘天數（系統計算）",
              "VACDAYS": "給予天數（年度或方案）",
              "CANUSEDATE": "可使用起始日",
              "DISABLEDDATE": "失效日（到期）"
          },
          business_context="Authoritative source for remaining balance & entitlement; pick latest row per person/year/type within validity window.",
          business_context_zh="餘額與給予之權威來源；於有效期內，取每人/年度/類型最新一筆。",
          common_queries=["Annual leave remaining by person for a given year",
                          "Entitlement vs used vs remaining by department"],
          common_queries_zh=["指定年度每位員工年假餘額","部門別給予/已用/剩餘對照"],
          priority=1
        ),
        # Dimensions
        T("dbo.PSNACCOUNT",
          ["CARDNUM","TRUENAME","PERSONID","EMPLOYEEID","COMPANYEMAIL","BRANCHID","BUSINESSUNITID","FIRSTNAME",
           "MIDDLENAME","LASTNAME","ENGNAME","LASTUPDATETIME","MOBILE","EXT_NO"],
          desc="Person dimension (authoritative; includes BRANCHID for department).",
          description_zh="人員維度（權威來源；含BRANCHID以解析部門）。",
          tags=["person","dimension","branch"],
          key_cols={"PERSONID": "人員鍵 (join to facts)", "EMPLOYEEID": "員編", "BRANCHID": "部門/單位鍵 → ORGStdStruct.UNITID"},
          priority=1
        ),
        T(ORG_TABLE,
          ["UNITID","UNITCODE","UNITNAME","UNITDISPLAYNAME","ISDELETE"],
          desc="Organization structure / branch dimension (UNITID as key).",
          description_zh="組織/單位維度（以 UNITID 為鍵）。",
          tags=["org","department","branch"],
          key_cols={"UNITID": "部門鍵 ← PSNACCOUNT.BRANCHID", "UNITNAME": "部門名稱", "UNITDISPLAYNAME": "部門顯示名稱", "UNITCODE": "部門代碼"},
          priority=1
        ),
        # Leave type dictionary (for name lookups)
        T("dbo.ATDATTENDANCECLASS",
          ["CLASSCODE","ID","CLASSNAME","CLASSTYPE"],
          desc="Attendance/leave class dictionary (maps LEAVEID/VACATIONTYPE → readable type).",
          description_zh="出勤/請假類別字典（LEAVEID/VACATIONTYPE → 假別名稱）。",
          tags=["dictionary","leave-type"],
          key_cols={"ID": "LEAVEID ↔ ATDLEAVEDATA.LEAVEID"},
          priority=1
        ),
    ]

    return LeaveVectorDB(tables)
