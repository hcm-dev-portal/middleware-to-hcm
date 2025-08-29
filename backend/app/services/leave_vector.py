# backend/app/services/leave_vector.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Set, Optional, Any
import re
from enum import Enum

# ───────────────────────────────── Enums & Models ─────────────────────────────────

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

@dataclass
class QueryPattern:
    pattern: str
    description: str
    primary_tables: List[str]
    suggested_joins: List[str]
    required_filters: List[str] = field(default_factory=list)
    performance_notes: List[str] = field(default_factory=list)

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

# ───────────────────────────────── Helper Aliases ─────────────────────────────────

COLUMN_ALIASES: Dict[str, Set[str]] = {
    # Frequent schema spelling variations
    "BUSINESSUNITID": {"BUSINESSUINTID"},
    "EFFECTIVEDATE": {"EFFINIENTDATE", "EFFICIENTDATE", "EFFDATE"},
    # Timecard spelling variations
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

# ───────────────────────────────── Leave Vector DB ─────────────────────────────────

class LeaveVectorDB:
    def __init__(self, tables: List[TableSchema]):
        self.tables = tables
        self._by_name: Dict[str, TableSchema] = {t.full.lower(): t for t in tables}

        # Resolve person dimension table once (PSNACCOUNT_D or BIPSNACCOUNTSP)
        self._person_table = self._resolve_person_table()

        # Build data
        self._joins = self._build_comprehensive_joins()
        self._query_patterns = self._build_query_patterns()
        self._semantic_keywords = self._build_semantic_keywords()

    # ---------- Resolution helpers ----------

    def _resolve_person_table(self) -> Optional[str]:
        candidates = [
            "dbo.PSNACCOUNT_D",       # classic person master (if present)
            "dbo.BIPSNACCOUNTSP",     # BI person snapshot view (present in your dump)
            "BIPSNACCOUNTSP",         # fallback if schema-less naming is used
        ]
        for name in candidates:
            if name.lower() in self._by_name:
                return name
        return None

    def _exists(self, full: str) -> bool:
        return full.lower() in self._by_name

    # ---------- Joins ----------

    def _build_comprehensive_joins(self) -> List[TableJoin]:
        joins: List[TableJoin] = []

        # Core leave tables (current, historical, deleted, transferred)
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
                        join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE
                    ))

        # LeaveEX ↔ Vacation balance (reconciliation path)
        if self._exists("dbo.ATDLEAVEDATAEX") and self._exists("dbo.ATDNONCALCULATEDVACATION"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVEDATAEX", left_column="VACATIONID",
                right_table="dbo.ATDNONCALCULATEDVACATION", right_column="OID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE
            ))

        # Leave (current) ↔ LeaveEX (1:1-ish by LEAVEID, LEAVEID in EX may be NULL)
        if self._exists("dbo.ATDLEAVEDATA") and self._exists("dbo.ATDLEAVEDATAEX"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVEDATA", left_column="LEAVEID",
                right_table="dbo.ATDLEAVEDATAEX", right_column="LEAVEID",
                join_type=JoinType.LEFT, cardinality=Cardinality.ONE_TO_ONE
            ))

        # Historical/Deleted ↔ Original by LEAVEID (audit trace)
        if self._exists("dbo.ATDHISLEAVEDATA_D") and self._exists("dbo.ATDHISLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.ATDHISLEAVEDATA_D", left_column="LEAVEID",
                right_table="dbo.ATDHISLEAVEDATA", right_column="LEAVEID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE
            ))
        if self._exists("dbo.ATDLEAVEDATA_D") and self._exists("dbo.ATDLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVEDATA_D", left_column="LEAVEID",
                right_table="dbo.ATDLEAVEDATA", right_column="LEAVEID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE
            ))

        # Leave ↔ Cancel (composite ON since cancel table has no LEAVEID)
        if self._exists("dbo.ATDLEAVECANCELDATA") and self._exists("dbo.ATDLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVECANCELDATA", left_column="PERSONID",
                right_table="dbo.ATDLEAVEDATA", right_column="PERSONID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                condition=(
                    "({left}.ATTENDANCETYPE = {right}.ATTENDANCETYPE "
                    "AND {left}.STARTDATE = {right}.STARTDATE "
                    "AND {left}.ENDDATE = {right}.ENDDATE)"
                )
            ))

        # Leave ↔ Legal calendar (holiday detection)
        if self._exists("dbo.ATDLEAVEDATA") and self._exists("dbo.ATDLEGALCALENDAR"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVEDATA", left_column="WORKDATE",
                right_table="dbo.ATDLEGALCALENDAR", right_column="CALENDARDATE",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE
            ))

        # Dept-day operation state ↔ Leave (validate/calc checkpoints by day & dept)
        if self._exists("dbo.ATDDEPTOPERSTATE") and self._exists("dbo.ATDLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.ATDLEAVEDATA", left_column="DEPARTMENTID",
                right_table="dbo.ATDDEPTOPERSTATE", right_column="DEPARTMENTID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                condition="({left}.WORKDATE = {right}.WORKDATE)"
            ))

        # EDF external leave ↔ Leave (same person + overlapping date window)
        if self._exists("dbo.EDFATDLEAVEDATA") and self._exists("dbo.ATDLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.EDFATDLEAVEDATA", left_column="PERSONID",
                right_table="dbo.ATDLEAVEDATA", right_column="PERSONID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                condition=(
                    "( {right}.WORKDATE BETWEEN {left}.WORKDATES AND {left}.WORKDATEE )"
                )
            ))

        # Monthly import results ↔ Leave by PERSONID + ATTENDANCETYPE (for rollups)
        if self._exists("dbo.ATDRESULTDATAIMPORT") and self._exists("dbo.ATDLEAVEDATA"):
            joins.append(TableJoin(
                left_table="dbo.ATDRESULTDATAIMPORT", left_column="PERSONID",
                right_table="dbo.ATDLEAVEDATA", right_column="PERSONID",
                join_type=JoinType.LEFT, cardinality=Cardinality.MANY_TO_ONE,
                condition="({left}.ATTENDANCETYPE = {right}.ATTENDANCETYPE)"
            ))

        return joins

    # ---------- Patterns ----------

    def _build_query_patterns(self) -> List[QueryPattern]:
        patterns = [
            QueryPattern(
                pattern="current_leave_status",
                description="Who is currently on leave",
                primary_tables=["dbo.ATDLEAVEDATA"],
                suggested_joins=["PERSONID"],
                required_filters=[
                    "CAST(GETDATE() AS date) BETWEEN CAST(STARTDATE AS date) AND CAST(ENDDATE AS date)",
                    "VALIDATED = 1"
                ],
                performance_notes=[
                    "Index/Filter on STARTDATE, ENDDATE",
                    "Group by PERSONID to collapse multiple days"
                ],
            ),
            QueryPattern(
                pattern="upcoming_leave",
                description="Who will be on leave in a future window",
                primary_tables=["dbo.ATDLEAVEDATA"],
                suggested_joins=["PERSONID"],
                required_filters=["STARTDATE >= CAST(GETDATE() AS date)"],
                performance_notes=["Index on STARTDATE, ENDDATE; include VALIDATED = 1"],
            ),
            QueryPattern(
                pattern="historical_leave_analysis",
                description="Leave patterns over time periods",
                primary_tables=["dbo.ATDHISLEAVEDATA"],
                suggested_joins=["PERSONID"],
                required_filters=["Date range filter required (WORKDATE/STARTDATE/ENDDATE)"],
                performance_notes=["Large table — always filter by date range"],
            ),
            QueryPattern(
                pattern="leave_balance_reconciliation",
                description="Reconcile used/remaining leave with vacation balance",
                primary_tables=["dbo.ATDLEAVEDATAEX", "dbo.ATDNONCALCULATEDVACATION"],
                suggested_joins=["PERSONID", "VACATIONID"],
                required_filters=[
                    "VACATIONTYPE filter recommended",
                    "WORKDATE or STARTDATE/ENDDATE window"
                ],
                performance_notes=["Join VACATIONID ↔ OID; group by PERSONID, VACATIONTYPE"],
            ),
            QueryPattern(
                pattern="leave_cancellations",
                description="Cancelled leave requests and reasons",
                primary_tables=["dbo.ATDLEAVECANCELDATA"],
                suggested_joins=["PERSONID"],
                required_filters=["WORKDATE range filter"],
                performance_notes=["Filter VALIDATED if needed; use REASON/LEAVEREASON"],
            ),
        ]
        return patterns

    # ---------- Semantics ----------

    def _build_semantic_keywords(self) -> Dict[str, List[str]]:
        return {
            # Leave-first focus
            "current_leave": ["dbo.ATDLEAVEDATA"],
            "historical_leave": ["dbo.ATDHISLEAVEDATA"],
            "deleted_leave": ["dbo.ATDLEAVEDATA_D", "dbo.ATDHISLEAVEDATA_D"],
            "transferred_leave": ["dbo.ATDLEAVEDATA_T"],
            "leave_extensions": ["dbo.ATDLEAVEDATAEX", "dbo.ATDLEAVEDATAEX_D"],
            "cancelled_leave": ["dbo.ATDLEAVECANCELDATA"],
            "vacation_balance": ["dbo.ATDNONCALCULATEDVACATION", "dbo.ATDHISNONCALCULATEDVACATION"],
            "holiday_calendar": ["dbo.ATDLEGALCALENDAR"],

            # Auxiliary context (lower scoring)
            "edf_leave": ["dbo.EDFATDLEAVEDATA"],
            "historical_attendance": ["dbo.ATDHISLATEEARLY"],
            "no_timecard": ["dbo.ATDHISNOTIMECARD"],
            "timecard_data": ["dbo.ATDHISTIMECARDDATA"],
            "dept_ops": ["dbo.ATDDEPTOPERSTATE"],
            "monthly_import": ["dbo.ATDRESULTDATAIMPORT"],
            "people": [self._person_table] if self._person_table else [],
        }

    # ---------- Health ----------

    def is_ready(self) -> bool:
        return bool(self.tables)

    def health_check(self) -> Dict[str, object]:
        return {
            "ready": self.is_ready(),
            "tables_indexed": len(self.tables),
            "join_relationships": len(self._joins),
            "query_patterns": len(self._query_patterns),
            "person_table": self._person_table,
        }

    # ---------- Search & Hints ----------

    def search_relevant_tables(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q = query.lower()
        q_terms = set(re.findall(r"\w+", q))
        if not q_terms:
            return []

        def score(table: TableSchema) -> float:
            s = 0.0
            text = " ".join([table.full, table.description, " ".join(table.tags), " ".join(table.columns)]).lower()

            for term in q_terms:
                if term in text:
                    if term in table.full.lower():
                        s += 3.0
                    elif term in table.description.lower():
                        s += 2.0
                    elif term in " ".join(table.tags).lower():
                        s += 2.0
                    else:
                        s += 1.0

            # semantic boost
            for keyword, tabs in self._semantic_keywords.items():
                if keyword in q and table.full in tabs:
                    s += 4.0

            # domain boosts
            boosts = {
                ("leave", "vacation", "absence", "sick", "annual"): [
                    "atdleavedata", "atdhisleavedata", "atdleavedataex",
                    "atdleavedata_d", "atdhisleavedata_d", "atdleavercanceldata",
                    "atdnocalculatedvacation", "atdhisnoncalculatedvacation",
                ],
                ("cancel", "cancelled"): ["atdleavercanceldata"],
                ("deleted", "removed"): ["_d"],
                ("history", "historical", "trend"): ["his"],
                ("person", "employee", "who", "name"): [
                    "psnaccount_d", "bipsnaccountsp"
                ],
                ("department", "unit", "business"): [
                    "departmentid", "businessunitid"
                ],
            }
            for kws, pats in boosts.items():
                if any(k in q for k in kws):
                    for p in pats:
                        if p in table.full.lower():
                            s += 2.0

            # temporal detection
            if any(t in q for t in ["today", "yesterday", "week", "month", "year", "date", "current"]):
                if table.temporal_columns:
                    s += 1.5
                if "his" in table.full.lower():
                    s += 1.0
                if table.full.lower() == "dbo.atdleavedata":
                    s += 2.0

            # downweight deleted/historical unless requested
            if table.is_deleted_data and not any(t in q for t in ["deleted", "cancelled", "removed", "history", "historical"]):
                s *= 0.5
            if table.is_historical and not any(t in q for t in ["history", "historical", "past", "trend"]):
                s *= 0.7

            return s

        ranked = sorted(((t, score(t)) for t in self.tables), key=lambda x: x[1], reverse=True)
        return [(t.full, s) for t, s in ranked if s > 0][:max(1, top_k)]

    def join_hints(self, tables: Iterable[str]) -> List[str]:
        table_set = {t.lower() for t in tables}
        hints: List[str] = []

        # Filter applicable joins
        for j in self._joins:
            if j.left_table.lower() in table_set and j.right_table.lower() in table_set:
                cond = ""
                if j.condition:
                    cond = " AND " + j.condition.format(
                        left=j.left_table, right=j.right_table
                    )
                hints.append(
                    f"{j.join_type.value} JOIN {j.right_table} "
                    f"ON {j.left_table}.{j.left_column} = {j.right_table}.{j.right_column}"
                    f"{cond} -- {j.cardinality.value}"
                )

        # Perf notes
        for tname in table_set:
            t = self._by_name.get(tname)
            if t and t.row_estimate and t.row_estimate > 100_000:
                hints.append(f"-- Performance: filter {t.full} by date range when possible")

        # Deduplicate preserve order
        return list(dict.fromkeys(hints))

    def get_query_pattern(self, query: str) -> Optional[QueryPattern]:
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

    # ---------- Validation ----------

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

# ───────────────────────────────── Index Builder ─────────────────────────────────
# Keep this lightweight; include only the tables you actively query for leave.

def build_leave_index() -> LeaveVectorDB:
    def T(full: str, cols: List[str], desc: str = "", tags: List[str] = None,
          pks: List[str] = None, indexed: List[str] = None, rows: int = None,
          is_hist: bool = False, is_del: bool = False, temporal: List[str] = None) -> TableSchema:
        return TableSchema(
            full=full, columns=cols, description=desc, tags=tags or [],
            primary_keys=pks or [], indexed_columns=indexed or [], row_estimate=rows,
            is_historical=is_hist, is_deleted_data=is_del, temporal_columns=temporal or []
        )

    tables: List[TableSchema] = [
        # Leave core
        T("dbo.ATDLEAVEDATA",
          ["ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","STARTDATE","ENDDATE","AutoRevise",
           "TIMECLASSID","TIMECLASSHOURS","CREATIONTIME","CREATEDBY","LASTUPDATETIME",
           "LASTUPDATEDBY","LEAVEID","FORMKIND","FROM_SOURCE","FORM_NO","RECORD_ID",
           "FLEAVEBYDAYTYPE","TLEAVEBYDAYTYPE","LEAVEREASON"],
          "Current validated leave data", ["leave","current","validated"],
          ["LEAVEID"], ["PERSONID","STARTDATE","ENDDATE"], rows=50_000,
          temporal=["WORKDATE","STARTDATE","ENDDATE"]),

        T("dbo.ATDHISLEAVEDATA",
          ["LEAVEID","ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","STARTDATE","ENDDATE","AutoRevise",
           "TIMECLASSID","TIMECLASSHOURS","CREATIONTIME","CREATEDBY","LASTUPDATETIME",
           "LASTUPDATEDBY","FORMKIND","FROM_SOURCE","FORM_NO","RECORD_ID","RECORD_USERID",
           "RECORD_DATE","FLEAVEBYDAYTYPE","TLEAVEBYDAYTYPE","LEAVEREASON"],
          "Historical leave data", ["leave","history"], ["LEAVEID"], ["PERSONID","WORKDATE"],
          rows=500_000, is_hist=True, temporal=["WORKDATE","STARTDATE","ENDDATE"]),

        T("dbo.ATDLEAVEDATA_D",
          ["DELETEID","ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","STARTDATE","ENDDATE","AUTOREVISE",
           "TIMECLASSID","TIMECLASSHOURS","CREATIONTIME","CREATEDBY","LASTUPDATETIME",
           "LASTUPDATEDBY","LEAVEID","DELETE_USER_ID","DELETE_TIME","FORMKIND","FROM_SOURCE",
           "FORM_NO","RECORD_ID","FLEAVEBYDAYTYPE","TLEAVEBYDAYTYPE","LEAVEREASON"],
          "Deleted leave data (audit)", ["leave","deleted","audit"], ["DELETEID"],
          ["PERSONID","DELETE_TIME"], rows=25_000, is_del=True,
          temporal=["WORKDATE","DELETE_TIME"]),

        T("dbo.ATDHISLEAVEDATA_D",
          ["DELETEID","ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "DEPARTMENTID","VALIDATED","BUSINESSUNITID","STARTDATE","ENDDATE","AUTOREVISE",
           "TIMECLASSID","TIMECLASSHOURS","CREATIONTIME","CREATEDBY","LASTUPDATETIME",
           "LASTUPDATEDBY","LEAVEID","DELETE_USER_ID","DELETE_TIME","FORMKIND","FROM_SOURCE",
           "FORM_NO","RECORD_ID","RECORD_USERID","RECORD_DATE","FLEAVEBYDAYTYPE",
           "TLEAVEBYDAYTYPE","LEAVEREASON"],
          "Deleted historical leave (audit)", ["leave","deleted","audit"], ["DELETEID"],
          ["PERSONID","DELETE_TIME"], rows=25_000, is_del=True,
          temporal=["WORKDATE","DELETE_TIME"]),

        T("dbo.ATDLEAVEDATA_T",
          ["ATTENDANCETYPE","PERSONID","WORKDATE","STARTTIME","ENDTIME","HOURS",
           "BUSINESSUNITID","STARTDATE","ENDDATE","LEAVEID","TRANSFERUNITID","LEAVEREASON"],
          "Transferred leave across units", ["leave","transferred"], ["LEAVEID"],
          ["PERSONID","TRANSFERUNITID"], rows=5_000, temporal=["WORKDATE"]),

        T("dbo.ATDLEAVEDATAEX",
          ["ATTENDANCETYPE","PERSONID","WORKDATE","HOURS","BUSINESSUNITID","VACATIONID",
           "MINUSDAYS","LEAVETYPE","STARTTIME","ENDTIME","STARTDATE","ENDDATE",
           "TIMECLASSID","TIMECLASSHOURS","LEAVEID"],
          "Extended leave accounting / vacation linkage", ["leave","extended","balance"],
          ["LEAVEID"], ["PERSONID","VACATIONID"], rows=50_000, temporal=["WORKDATE"]),

        T("dbo.ATDLEAVEDATAEX_D",
          ["DELETEID","ATTENDANCETYPE","PERSONID","WORKDATE","HOURS","BUSINESSUNITID",
           "VACATIONID","MINUSDAYS","LEAVETYPE","STARTTIME","ENDTIME","STARTDATE","ENDDATE",
           "TIMECLASSID","TIMECLASSHOURS","LEAVEID","DELETE_USER_ID","DELETE_TIME"],
          "Deleted extended leave (audit)", ["leave","extended","deleted","audit"], ["DELETEID"],
          ["PERSONID","DELETE_TIME"], rows=10_000, is_del=True,
          temporal=["WORKDATE","DELETE_TIME"]),

        T("dbo.ATDLEAVECANCELDATA",
          ["OID","ATTENDANCETYPE","PERSONID","WORKDATE","STARTDATE","STARTTIME","ENDDATE",
           "ENDTIME","HOURS","DEPARTMENTID","VALIDATED","BUSINESSUNITID","ISHISDATA",
           "AutoRevise","CREATEDATE","CREATEUSERID","LASTEDITUSERID","LASTEDITTIME","REASON",
           "FLEAVEBYDAYTYPE","TLEAVEBYDAYTYPE","LEAVEREASON","FROM_SOURCE","FORM_NO",
           "RECORD_ID","FORMKIND"],
          "Cancelled leave requests", ["leave","cancelled"], ["OID"], ["PERSONID","WORKDATE"],
          rows=10_000, temporal=["WORKDATE","STARTDATE","ENDDATE"]),

        # Balance / holiday / ops
        T("dbo.ATDNONCALCULATEDVACATION",
          ["OID","PERSONID","BUSINESSUINTID","VACATIONDAYS","USEDDAYS","REMAINDAYS","VACATIONTYPE",
           "LASTUPDATETIME","EFFINIENTDATE","INVALIDATIONDATE","DESCRIPTION","CREATIONTIME",
           "CREATEDBY","LASTUPDATEDBY","FLAG"],
          "Current vacation balances", ["vacation","balance","current"], ["OID"],
          ["PERSONID","VACATIONTYPE"], rows=25_000,
          temporal=["EFFINIENTDATE","INVALIDATIONDATE"]),

        T("dbo.ATDHISNONCALCULATEDVACATION",
          ["OID","PERSONID","BUSINESSUINTID","VACATIONDAYS","USEDDAYS","REMAINDAYS","VACATIONTYPE",
           "LASTUPDATETIME","EFFINIENTDATE","INVALIDATIONDATE","DESCRIPTION","CREATIONTIME",
           "CREATEDBY","LASTUPDATEDBY"],
          "Historical vacation balances", ["vacation","balance","history"], ["OID"],
          ["PERSONID"], rows=100_000, is_hist=True,
          temporal=["EFFINIENTDATE","INVALIDATIONDATE"]),

        T("dbo.ATDLEGALCALENDAR",
          ["CALENDARDATE","LEGALID","CALENDARTYPE"],
          "Legal holidays / calendar", ["calendar","holidays"], ["CALENDARDATE","LEGALID"],
          ["CALENDARDATE"], rows=1_000, temporal=["CALENDARDATE"]),

        T("dbo.ATDDEPTOPERSTATE",
          ["DEPARTMENTID","WORKDATE","STATEROLLCALL","STATEOVERTIME","STATEALLOWANCE",
           "CALCULATOR","CALCULATETIME","SUBMITER","SUBMITTIME","CHECKER","CHECKTIME",
           "VALIDATER","VALIDATETIME","BUSINESSUNITID"],
          "Department-day operation status (calc/validate checkpoints)",
          ["dept","ops","status"], ["DEPARTMENTID","WORKDATE"], ["WORKDATE"], rows=50_000,
          temporal=["WORKDATE"]),

        # Aux attendance/timecard
        T("dbo.ATDHISLATEEARLY",
          ["LATEEARLYID","ATTENDDATE","PERSONID","SHOULDTIME","DEPARTMENTID","TIMECLASS",
           "CARDTIME","MINUTES","FUNCFLAG","PROCSTATE","ATDTYPE","ALLEGEREASON",
           "BUSINESSUNITID","SHOULDDATE","CREATIONTIME","CREATEDBY","LASTUPDATETIME",
           "LASTUPDATEDBY","PROCSTATEBEFORETODEAL","RECORD_USERID","RECORD_DATE",
           "TOPAYROLLDATE","PACKAGEID","CARDDATE","EVENTRULECODE"],
          "Historical late/early events", ["attendance","late","early","history"],
          ["LATEEARLYID"], ["PERSONID","ATTENDDATE"], rows=750_000, is_hist=True,
          temporal=["ATTENDDATE","CARDDATE"]),

        T("dbo.ATDHISTIMECARDDATA",
          ["RECORDID","DATAID","TIMECARDDATE","TIMECARDTIME","RECEIVEDATE","MACHINEID",
           "DEPARTMENTID","PERSONID","REGIONID","DATAFROM","BUSINESSUNITID","CREATIONTIME",
           "CREATEDBY","LASTUPDATETIME","LASTUPDATEDBY","RECORD_USERID","RECORD_DATE",
           "CONFIGID","REASONID","FORM_NO","RECORD_ID","REMARK","LOCATIONADDRESS"],
          "Raw timecard swipes (historical)", ["timecard","raw","history"],
          ["RECORDID"], ["PERSONID","TIMECARDDATE"], rows=2_000_000, is_hist=True,
          temporal=["TIMECARDDATE"]),

        T("dbo.ATDHISNOTIMECARD",
          ["NOTIMECARDID","SHOULDTIME","ATTENDDATE","PERSONID","DEPARTMENTID","TIMECLASS",
           "REASONID","PROCSTATE","ALLEGEREASON","BUSINESSUNITID","SHOULDDATE","CREATIONTIME",
           "CREATEDBY","LASTUPDATETIME","LASTUPDATEDBY","PROCSTATEBEFORETODEAL","RECORD_USERID",
           "RECORD_DATE","TOPAYROLLDATE","PACKAGEID","ATDTYPE","EVENTRULECODE"],
          "Missing timecard events", ["attendance","notimecard","exceptions"], ["NOTIMECARDID"],
          ["PERSONID","ATTENDDATE"], rows=50_000, temporal=["ATTENDDATE"]),

        # External + rollups
        T("dbo.EDFATDLEAVEDATA",
          ["RECORDID","ATTENDANCETYPE","PERSONID","WORKDATES","WORKDATEE","STARTTIME","ENDTIME",
           "HOURS","DEPARTMENTID","VALIDATED","BUSINESSUNITID","PROCESSINSTANCEID","ISHANDLE"],
          "External (EDF) leave imports", ["leave","edf","integration"], ["RECORDID"],
          ["PERSONID"], rows=25_000, temporal=["WORKDATES","WORKDATEE"]),

        T("dbo.ATDRESULTDATAIMPORT",
          ["PERSONID","DEPARTMENTID","ATTENDANCETYPE","BELONGYEARMONTH","RESULTDATA","ISPAY",
           "BUSINESSUNITID","CREATIONTIME","CREATEDBY","LASTUPDATETIME","LASTUPDATEDBY","IMPORTID"],
          "Monthly attendance/leave rollup import", ["import","monthly","rollup"],
          ["IMPORTID"], ["PERSONID","ATTENDANCETYPE"], rows=100_000,
          temporal=["BELONGYEARMONTH"]),

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
          "Person dimension (BI snapshot)", ["person","employee","bi"], ["PERSONID"], ["PERSONID"]),
    ]

    return LeaveVectorDB(tables)

# ───────────────────────────────── Query Analyzers ─────────────────────────────────

def analyze_query_complexity(query: str) -> Dict[str, Any]:
    q = query.lower()
    indicators = {
        "temporal_range": any(t in q for t in ["between", "range", "from", "to", "last", "past"]),
        "aggregation": any(t in q for t in ["count", "sum", "average", "total", "max", "min"]),
        "grouping": any(t in q for t in [" by ", "group", "department", "unit", "type"]),
        "multiple_tables": len([t for t in ["leave", "attendance", "vacation"] if t in q]) > 1,
        "person_lookup": any(t in q for t in ["who", "person", "employee", "name"]),
        "current_data": any(t in q for t in ["current", "today", "now", "active"]),
        "historical_data": any(t in q for t in ["history", "historical", "trend", "past"]),
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
        if not any(t in q for t in ["date", "between", "last", "past", "month", "year", "range"]):
            suggestions.append("Add a date range filter for historical tables")

    if any(t in q for t in ["who", "person", "employee", "name"]):
        if not any("PSNACCOUNT" in t.upper() or "BIPSNACCOUNTSP" in t.upper() for t in selected_tables):
            suggestions.append("Include person dimension table for name/ID resolution")

    if "leave" in q and "dbo.ATDLEAVEDATA" in selected_tables:
        suggestions.append("Filter by VALIDATED = 1 for approved leave only")

    if any(t in q for t in ["department", "unit", "business"]):
        if not any("DEPARTMENTID" in t or "BUSINESSUNITID" in t for t in selected_tables):
            suggestions.append("Consider filtering/grouping by DEPARTMENTID or BUSINESSUNITID")

    return suggestions
