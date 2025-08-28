# backend/app/services/leave_vector.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Set, Optional, Any
import re
from enum import Enum

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
    condition: Optional[str] = None  # Additional join conditions

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

class LeaveVectorDB:
    def __init__(self, tables: List[TableSchema]):
        self.tables = tables
        self._by_name = {t.full.lower(): t for t in tables}
        
        # Comprehensive join relationships
        self._joins = self._build_comprehensive_joins()
        
        # Query patterns for common HR scenarios
        self._query_patterns = self._build_query_patterns()
        
        # Semantic keywords mapping
        self._semantic_keywords = self._build_semantic_keywords()

    def _build_comprehensive_joins(self) -> List[TableJoin]:
        """Build comprehensive join relationships with cardinality and performance hints."""
        joins = []
        
        # Core person relationships - all attendance tables link to person master
        person_tables = [
            "dbo.ATDLEAVEDATA", "dbo.ATDHISLEAVEDATA", "dbo.ATDLEAVEDATA_D", "dbo.ATDLEAVEDATA_T",
            "dbo.ATDLEAVEDATAEX", "dbo.ATDLEAVEDATAEX_D", "dbo.ATDLEAVECANCELDATA",
            "dbo.ATDLATEEARLY", "dbo.ATDHISLATEEARLY", "dbo.ATDLATEEARLY_D", "dbo.ATDLATEEARLY_C",
            "dbo.ATDHISNOTIMECARD", "dbo.ATDHISTIMECARDDATA", "dbo.ATDHISOVERTIME",
            "dbo.ATDHISOVERTIMEEXCEPTION", "dbo.ATDHISOVERTIMEORDER", "dbo.EDFATDLEAVEDATA",
            "dbo.ATDJOBCONTENT", "dbo.ATDJOBCONTENT_T", "dbo.ATDNONCALCULATEDVACATION",
            "dbo.ATDHISNONCALCULATEDVACATION", "dbo.ATDNONCALCULATEDVACATION_D"
        ]
        
        for table in person_tables:
            joins.append(TableJoin(
                left_table=table,
                left_column="PERSONID",
                right_table="dbo.PSNACCOUNT_D",
                right_column="PERSONID",
                join_type=JoinType.LEFT,
                cardinality=Cardinality.MANY_TO_ONE
            ))
        
        # Leave data extensions via LEAVEID
        leave_extensions = [
            ("dbo.ATDLEAVEDATA", "dbo.ATDLEAVEDATAEX"),
            ("dbo.ATDLEAVEDATA", "dbo.ATDLEAVEDATA_D"),
            ("dbo.ATDHISLEAVEDATA", "dbo.ATDHISLEAVEDATA_D"),
            ("dbo.ATDLEAVEDATAEX", "dbo.ATDLEAVEDATAEX_D"),
        ]
        
        for left, right in leave_extensions:
            joins.append(TableJoin(
                left_table=left,
                left_column="LEAVEID",
                right_table=right,
                right_column="LEAVEID",
                join_type=JoinType.LEFT,
                cardinality=Cardinality.ONE_TO_ONE
            ))
        
        # Overtime relationships
        joins.append(TableJoin(
            left_table="dbo.ATDHISOVERTIME",
            left_column="OVERTIMEID",
            right_table="dbo.ATDHISOVERTIMEEXCEPTION",
            right_column="OVERTIMEID",
            join_type=JoinType.LEFT,
            cardinality=Cardinality.ONE_TO_MANY
        ))
        
        # Job content type relationships
        joins.append(TableJoin(
            left_table="dbo.ATDJOBCONTENT",
            left_column="CONTENTTYPEID",
            right_table="dbo.ATDJOBCONTENTTYPE",
            right_column="CONTENTTYPEID",
            join_type=JoinType.LEFT,
            cardinality=Cardinality.MANY_TO_ONE
        ))
        
        # Family information
        joins.append(TableJoin(
            left_table="dbo.PSNFAMILYINFO",
            left_column="PERSONID",
            right_table="dbo.PSNACCOUNT_D",
            right_column="PERSONID",
            join_type=JoinType.LEFT,
            cardinality=Cardinality.MANY_TO_ONE
        ))
        
        # Calendar relationships
        joins.append(TableJoin(
            left_table="dbo.ATDHISTIMEORDERCALENDAR",
            left_column="TIMEORDERID",
            right_table="dbo.ATDHISOVERTIMEORDER",
            right_column="OVERTIMEORDERID",
            join_type=JoinType.LEFT,
            cardinality=Cardinality.MANY_TO_ONE
        ))
        
        return joins

    def _build_query_patterns(self) -> List[QueryPattern]:
        """Define common HR query patterns with optimization hints."""
        return [
            QueryPattern(
                pattern="current_leave_status",
                description="Who is currently on leave",
                primary_tables=["dbo.ATDLEAVEDATA", "dbo.PSNACCOUNT_D"],
                suggested_joins=["PERSONID"],
                required_filters=["CAST(GETDATE() AS date) BETWEEN CAST(STARTDATE AS date) AND CAST(ENDDATE AS date)"],
                performance_notes=["Index on STARTDATE, ENDDATE recommended", "Consider VALIDATED = 1 filter"]
            ),
            QueryPattern(
                pattern="historical_leave_analysis",
                description="Leave patterns over time periods",
                primary_tables=["dbo.ATDHISLEAVEDATA", "dbo.PSNACCOUNT_D"],
                suggested_joins=["PERSONID"],
                required_filters=["Date range filter required for performance"],
                performance_notes=["Large table - always filter by date range", "Consider BUSINESSUNITID filter"]
            ),
            QueryPattern(
                pattern="department_attendance",
                description="Attendance by department/business unit",
                primary_tables=["dbo.ATDLEAVEDATA", "dbo.PSNACCOUNT_D"],
                suggested_joins=["PERSONID"],
                required_filters=["DEPARTMENTID or BUSINESSUNITID filter recommended"],
                performance_notes=["Group by department for better performance"]
            ),
            QueryPattern(
                pattern="overtime_analysis",
                description="Overtime hours and patterns",
                primary_tables=["dbo.ATDHISOVERTIME", "dbo.PSNACCOUNT_D"],
                suggested_joins=["PERSONID"],
                required_filters=["Date range required"],
                performance_notes=["Join with ATDHISOVERTIMEORDER for approved overtime only"]
            ),
            QueryPattern(
                pattern="attendance_exceptions",
                description="Late arrivals, early departures, no-card events",
                primary_tables=["dbo.ATDLATEEARLY", "dbo.ATDHISNOTIMECARD", "dbo.PSNACCOUNT_D"],
                suggested_joins=["PERSONID"],
                required_filters=["ATTENDDATE filter required"],
                performance_notes=["Consider PROCSTATE for processed vs pending"]
            ),
            QueryPattern(
                pattern="vacation_balance",
                description="Vacation days available and used",
                primary_tables=["dbo.ATDNONCALCULATEDVACATION", "dbo.PSNACCOUNT_D"],
                suggested_joins=["PERSONID"],
                required_filters=["EFFINIENTDATE and INVALIDATIONDATE for active balances"],
                performance_notes=["Check FLAG column for record status"]
            ),
            QueryPattern(
                pattern="leave_cancellations",
                description="Cancelled leave requests and reasons",
                primary_tables=["dbo.ATDLEAVECANCELDATA", "dbo.PSNACCOUNT_D"],
                suggested_joins=["PERSONID"],
                required_filters=["WORKDATE range filter"],
                performance_notes=["Include REASON column for analysis"]
            ),
            QueryPattern(
                pattern="timecard_raw_data",
                description="Raw timecard swipes and machine data",
                primary_tables=["dbo.ATDHISTIMECARDDATA", "dbo.PSNACCOUNT_D"],
                suggested_joins=["PERSONID"],
                required_filters=["TIMECARDDATE range required"],
                performance_notes=["Very large table - narrow date ranges essential"]
            )
        ]

    def _build_semantic_keywords(self) -> Dict[str, List[str]]:
        """Map semantic concepts to relevant tables and columns."""
        return {
            "current_leave": ["dbo.ATDLEAVEDATA"],
            "historical_leave": ["dbo.ATDHISLEAVEDATA"],
            "deleted_leave": ["dbo.ATDLEAVEDATA_D", "dbo.ATDHISLEAVEDATA_D"],
            "transferred_leave": ["dbo.ATDLEAVEDATA_T"],
            "leave_extensions": ["dbo.ATDLEAVEDATAEX", "dbo.ATDLEAVEDATAEX_D"],
            "cancelled_leave": ["dbo.ATDLEAVECANCELDATA"],
            "edf_leave": ["dbo.EDFATDLEAVEDATA"],
            "current_attendance": ["dbo.ATDLATEEARLY"],
            "historical_attendance": ["dbo.ATDHISLATEEARLY"],
            "deleted_attendance": ["dbo.ATDLATEEARLY_D"],
            "calculated_attendance": ["dbo.ATDLATEEARLY_C"],
            "no_timecard": ["dbo.ATDHISNOTIMECARD"],
            "timecard_data": ["dbo.ATDHISTIMECARDDATA"],
            "current_overtime": ["dbo.ATDHISOVERTIME"],
            "overtime_exceptions": ["dbo.ATDHISOVERTIMEEXCEPTION"],
            "overtime_orders": ["dbo.ATDHISOVERTIMEORDER"],
            "calendar": ["dbo.ATDHISTIMEORDERCALENDAR", "dbo.ATDLEGALCALENDAR"],
            "vacation_balance": ["dbo.ATDNONCALCULATEDVACATION", "dbo.ATDHISNONCALCULATEDVACATION"],
            "job_content": ["dbo.ATDJOBCONTENT", "dbo.ATDJOBCONTENT_T"],
            "job_types": ["dbo.ATDJOBCONTENTTYPE"],
            "longevity_rules": ["dbo.ATDLONGEVITYRULE"],
            "no_card_reasons": ["dbo.ATDNOCARDREASON"],
            "person_master": ["dbo.PSNACCOUNT_D"],
            "family_info": ["dbo.PSNFAMILYINFO"],
            "people": ["dbo.PSNACCOUNT_D", "dbo.PSNFAMILYINFO"],
            "employees": ["dbo.PSNACCOUNT_D"],
            "staff": ["dbo.PSNACCOUNT_D"]
        }

    def is_ready(self) -> bool:
        return bool(self.tables)

    def health_check(self) -> Dict[str, object]:
        return {
            "ready": self.is_ready(),
            "tables_indexed": len(self.tables),
            "join_relationships": len(self._joins),
            "query_patterns": len(self._query_patterns)
        }

    def search_relevant_tables(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Enhanced table relevance scoring with semantic understanding."""
        q = query.lower()
        q_terms = set(re.findall(r"\w+", q))
        if not q_terms:
            return []
        
        def calculate_score(table: TableSchema) -> float:
            score = 0.0
            
            # Basic term matching in table name, description, tags, columns
            searchable_text = " ".join([
                table.full, table.description,
                " ".join(table.tags), " ".join(table.columns)
            ]).lower()
            
            # Term frequency scoring
            for term in q_terms:
                if term in searchable_text:
                    # Boost score based on where the term appears
                    if term in table.full.lower():
                        score += 3.0
                    elif term in table.description.lower():
                        score += 2.0
                    elif term in " ".join(table.tags).lower():
                        score += 2.0
                    elif term in " ".join(table.columns).lower():
                        score += 1.0
            
            # Semantic keyword matching
            for keyword, relevant_tables in self._semantic_keywords.items():
                if keyword in q and table.full in relevant_tables:
                    score += 4.0
            
            # Domain-specific bonuses
            domain_bonuses = {
                ("leave", "vacation", "absence", "sick", "annual"): [
                    "atdleavedata", "atdhisleavedata", "atdleavecanceldata", 
                    "atdleavedataex", "edfatdleavedata", "atdnoncalculatedvacation"
                ],
                ("overtime", "ot"): [
                    "atdhisovertime", "atdhisovertimeexception", "atdhisovertimeorder"
                ],
                ("attendance", "late", "early", "timecard"): [
                    "atdlateearly", "atdhislateearly", "atdhistimecarddata", "atdhisnotimecard"
                ],
                ("current", "active", "now", "today"): [
                    "atdleavedata", "atdlateearly", "atdnoncalculatedvacation"
                ],
                ("history", "historical", "past", "previous"): [
                    "atdhisleavedata", "atdhislateearly", "atdhistimecarddata", 
                    "atdhisovertime", "atdhisnoncalculatedvacation"
                ],
                ("deleted", "cancelled", "removed"): [
                    "_d", "atdleavecanceldata"
                ],
                ("person", "employee", "staff", "people", "who"): [
                    "psnaccount_d", "psnfamilyinfo"
                ],
                ("department", "unit", "business"): [
                    "departmentid", "businessunitid"
                ],
                ("job", "content", "work"): [
                    "atdjobcontent", "atdjobcontenttype"
                ]
            }
            
            for keywords, table_patterns in domain_bonuses.items():
                if any(kw in q for kw in keywords):
                    for pattern in table_patterns:
                        if pattern.lower() in table.full.lower():
                            score += 2.0
            
            # Temporal query detection
            temporal_terms = ["today", "yesterday", "week", "month", "year", "date", "when", "current"]
            if any(term in q for term in temporal_terms):
                if table.temporal_columns:
                    score += 1.5
                if "his" in table.full.lower():  # Historical tables
                    score += 1.0
                if table.full == "dbo.ATDLEAVEDATA":  # Current data
                    score += 2.0
            
            # Penalize deleted/historical tables unless specifically requested
            if table.is_deleted_data and not any(term in q for term in ["deleted", "cancelled", "removed", "history"]):
                score *= 0.5
            
            if table.is_historical and not any(term in q for term in ["history", "historical", "past", "trend"]):
                score *= 0.7
            
            # Performance penalty for very large tables without specific targeting
            if table.row_estimate and table.row_estimate > 1000000:
                if not any(term in q for term in ["timecard", "historical", "all"]):
                    score *= 0.8
            
            return score
        
        # Score and rank tables
        scored_tables = [(table, calculate_score(table)) for table in self.tables]
        ranked = sorted(scored_tables, key=lambda x: x[1], reverse=True)
        
        # Filter out zero scores and return top_k
        relevant = [(t.full, score) for t, score in ranked if score > 0]
        return relevant[:max(1, top_k)]

    def join_hints(self, tables: Iterable[str]) -> List[str]:
        """Generate optimized join hints with performance considerations."""
        table_set = {t.lower() for t in tables}
        hints: List[str] = []
        
        # Find applicable joins
        applicable_joins = []
        for join in self._joins:
            if join.left_table.lower() in table_set and join.right_table.lower() in table_set:
                applicable_joins.append(join)
        
        # Generate join hints with type and performance notes
        for join in applicable_joins:
            join_hint = f"{join.join_type.value} JOIN {join.right_table} ON {join.left_table}.{join.left_column} = {join.right_table}.{join.right_column}"
            
            if join.condition:
                join_hint += f" AND {join.condition}"
            
            # Add cardinality comment for performance awareness
            join_hint += f" -- {join.cardinality.value}"
            
            hints.append(join_hint)
        
        # Add performance recommendations
        performance_hints = []
        for table_name in table_set:
            table = self._by_name.get(table_name)
            if table and table.row_estimate and table.row_estimate > 100000:
                performance_hints.append(f"-- Performance: Filter {table.full} by date range when possible")
        
        return list(dict.fromkeys(hints + performance_hints))

    def get_query_pattern(self, query: str) -> Optional[QueryPattern]:
        """Match query to known patterns for optimization hints."""
        q = query.lower()
        
        # Pattern matching logic
        if any(term in q for term in ["current", "today", "now"]) and "leave" in q:
            return next((p for p in self._query_patterns if p.pattern == "current_leave_status"), None)
        
        if any(term in q for term in ["history", "trend", "past"]) and "leave" in q:
            return next((p for p in self._query_patterns if p.pattern == "historical_leave_analysis"), None)
        
        if any(term in q for term in ["department", "unit", "team"]):
            return next((p for p in self._query_patterns if p.pattern == "department_attendance"), None)
        
        if "overtime" in q:
            return next((p for p in self._query_patterns if p.pattern == "overtime_analysis"), None)
        
        if any(term in q for term in ["late", "early", "timecard"]):
            return next((p for p in self._query_patterns if p.pattern == "attendance_exceptions"), None)
        
        if any(term in q for term in ["vacation", "balance", "remaining", "available"]):
            return next((p for p in self._query_patterns if p.pattern == "vacation_balance"), None)
        
        if any(term in q for term in ["cancel", "cancelled"]):
            return next((p for p in self._query_patterns if p.pattern == "leave_cancellations"), None)
        
        return None

def build_leave_index() -> LeaveVectorDB:
    """Build comprehensive leave index with full schema information."""
    
    def T(full: str, cols: List[str], desc: str = "", tags: List[str] = None, 
          pks: List[str] = None, indexed: List[str] = None, 
          rows: int = None, is_hist: bool = False, is_del: bool = False,
          temporal: List[str] = None) -> TableSchema:
        return TableSchema(
            full=full, 
            columns=cols, 
            description=desc, 
            tags=tags or [], 
            primary_keys=pks or [],
            indexed_columns=indexed or [],
            row_estimate=rows,
            is_historical=is_hist,
            is_deleted_data=is_del,
            temporal_columns=temporal or []
        )
    
    tables: List[TableSchema] = [
        # Core leave tables
        T("dbo.ATDLEAVEDATA", 
          ["ATTENDANCETYPE", "PERSONID", "WORKDATE", "STARTTIME", "ENDTIME", "HOURS", 
           "DEPARTMENTID", "BUSINESSUNITID", "STARTDATE", "ENDDATE", "LEAVEID", "LEAVEREASON"],
          "Primary validated leave data - core for current leave status queries",
          ["leave", "current", "validated", "active"],
          ["LEAVEID"], ["PERSONID", "STARTDATE", "ENDDATE", "WORKDATE"], 50000,
          temporal=["WORKDATE", "STARTDATE", "ENDDATE"]),
        
        T("dbo.ATDHISLEAVEDATA",
          ["LEAVEID", "ATTENDANCETYPE", "PERSONID", "WORKDATE", "STARTTIME", "ENDTIME", 
           "HOURS", "DEPARTMENTID", "BUSINESSUNITID", "STARTDATE", "ENDDATE", "LEAVEREASON"],
          "Historical leave data snapshot for trend analysis",
          ["leave", "history", "trends"],
          ["LEAVEID"], ["PERSONID", "WORKDATE"], 500000, is_hist=True,
          temporal=["WORKDATE", "STARTDATE", "ENDDATE"]),
        
        T("dbo.ATDLEAVEDATA_D",
          ["DELETEID", "ATTENDANCETYPE", "PERSONID", "WORKDATE", "STARTTIME", "ENDTIME", 
           "HOURS", "LEAVEID", "DELETE_USER_ID", "DELETE_TIME"],
          "Deleted leave records with audit trail",
          ["leave", "deleted", "audit"],
          ["DELETEID"], ["PERSONID", "DELETE_TIME"], 25000, is_del=True,
          temporal=["WORKDATE", "DELETE_TIME"]),
        
        T("dbo.ATDLEAVEDATAEX",
          ["ATTENDANCETYPE", "PERSONID", "WORKDATE", "HOURS", "VACATIONID", "MINUSDAYS", 
           "LEAVETYPE", "LEAVEID"],
          "Extended leave accounting and vacation balance tracking",
          ["leave", "extended", "vacation", "balance"],
          ["LEAVEID"], ["PERSONID", "VACATIONID"], 50000,
          temporal=["WORKDATE"]),
        
        T("dbo.ATDLEAVECANCELDATA",
          ["OID", "ATTENDANCETYPE", "PERSONID", "WORKDATE", "STARTDATE", "STARTTIME", 
           "ENDDATE", "ENDTIME", "HOURS", "REASON", "LEAVEREASON"],
          "Cancelled leave requests with cancellation reasons",
          ["leave", "cancelled", "reasons"],
          ["OID"], ["PERSONID", "WORKDATE"], 10000,
          temporal=["WORKDATE", "STARTDATE", "ENDDATE"]),
        
        # Attendance and timecard tables
        T("dbo.ATDLATEEARLY",
          ["LATEEARLYID", "ATTENDDATE", "PERSONID", "SHOULDTIME", "CARDTIME", "MINUTES", 
           "ATDTYPE", "DEPARTMENTID", "BUSINESSUNITID"],
          "Current late arrivals and early departures",
          ["attendance", "late", "early", "current"],
          ["LATEEARLYID"], ["PERSONID", "ATTENDDATE"], 75000,
          temporal=["ATTENDDATE", "CARDDATE"]),
        
        T("dbo.ATDHISLATEEARLY",
          ["LATEEARLYID", "ATTENDDATE", "PERSONID", "SHOULDTIME", "CARDTIME", "MINUTES", 
           "ATDTYPE", "DEPARTMENTID", "BUSINESSUNITID"],
          "Historical late/early records for trend analysis",
          ["attendance", "late", "early", "history"],
          ["LATEEARLYID"], ["PERSONID", "ATTENDDATE"], 750000, is_hist=True,
          temporal=["ATTENDDATE"]),
        
        T("dbo.ATDHISTIMECARDDATA",
          ["RECORDID", "PERSONID", "TIMECARDDATE", "TIMECARDTIME", "MACHINEID", 
           "DEPARTMENTID", "BUSINESSUNITID"],
          "Raw timecard swipe data - very large table, filter by date essential",
          ["timecard", "raw", "swipes", "machine"],
          ["RECORDID"], ["PERSONID", "TIMECARDDATE"], 2000000, is_hist=True,
          temporal=["TIMECARDDATE"]),
        
        T("dbo.ATDHISNOTIMECARD",
          ["NOTIMECARDID", "ATTENDDATE", "PERSONID", "DEPARTMENTID", "REASONID", 
           "ATDTYPE", "BUSINESSUNITID"],
          "Missing timecard events with reasons",
          ["attendance", "notimecard", "missing", "exceptions"],
          ["NOTIMECARDID"], ["PERSONID", "ATTENDDATE"], 50000,
          temporal=["ATTENDDATE"]),
        
        # Overtime tables
        T("dbo.ATDHISOVERTIME",
          ["OVERTIMEID", "PERSONID", "OVERTIMEDATE", "TIMEFROM", "TIMETO", "HOURS", 
           "OVERTIMETYPE", "DEPARTMENTID", "BUSINESSUNITID"],
          "Overtime hours worked with approval status",
          ["overtime", "hours"],
          ["OVERTIMEID"], ["PERSONID", "OVERTIMEDATE"], 100000,
          temporal=["OVERTIMEDATE"]),
        
        T("dbo.ATDHISOVERTIMEORDER",
          ["OVERTIMEORDERID", "PERSONID", "OVERTIMEDATE", "STARTTIME", "ENDTIME", 
           "HOURS", "OVERTIMETYPE", "CONTRASTSTATE"],
          "Overtime orders/requests with approval workflow",
          ["overtime", "orders", "requests", "approval"],
          ["OVERTIMEORDERID"], ["PERSONID", "OVERTIMEDATE"], 50000,
          temporal=["OVERTIMEDATE"]),
        
        T("dbo.ATDHISOVERTIMEEXCEPTION",
          ["PERSONID", "WORKDATE", "BEGINTIME", "ENDTIME", "HOURS", "TYPE", 
           "DEPARTMENTID", "BUSINESSUNITID", "OVERTIMEID"],
          "Overtime exceptions and adjustments",
          ["overtime", "exceptions", "adjustments"],
          [], ["PERSONID", "WORKDATE"], 25000,
          temporal=["WORKDATE"]),
        
        # Vacation and balance tables
        T("dbo.ATDNONCALCULATEDVACATION",
          ["OID", "PERSONID", "BUSINESSUINTID", "VACATIONDAYS", "USEDDAYS", "REMAINDAYS", 
           "VACATIONTYPE", "EFFINIENTDATE", "INVALIDATIONDATE", "FLAG"],
          "Current vacation balances and entitlements",
          ["vacation", "balance", "entitlements", "current"],
          ["OID"], ["PERSONID", "VACATIONTYPE"], 25000,
          temporal=["EFFINIENTDATE", "INVALIDATIONDATE"]),
        
        T("dbo.ATDHISNONCALCULATEDVACATION",
          ["OID", "PERSONID", "BUSINESSUINTID", "VACATIONDAYS", "USEDDAYS", "REMAINDAYS", 
           "VACATIONTYPE", "EFFINIENTDATE", "INVALIDATIONDATE"],
          "Historical vacation balance snapshots",
          ["vacation", "balance", "history"],
          ["OID"], ["PERSONID"], 100000, is_hist=True,
          temporal=["EFFINIENTDATE", "INVALIDATIONDATE"]),
        
        # Job content tables
        T("dbo.ATDJOBCONTENT",
          ["JOBCONTENTID", "PERSONID", "WORKDATE", "STARTDATE", "STARTTIME", 
           "ENDDATE", "ENDTIME", "CONTENTTYPEID"],
          "Job content and work activity tracking",
          ["job", "content", "activity", "work"],
          ["JOBCONTENTID"], ["PERSONID", "WORKDATE"], 200000,
          temporal=["WORKDATE", "STARTDATE", "ENDDATE"]),
        
        T("dbo.ATDJOBCONTENTTYPE",
          ["CONTENTTYPEID", "CONTENTTYPECODE", "CONTENTTYPENAME", "CONTENTTYPECLASS", 
           "BUSINESSUNITID"],
          "Job content type definitions and classifications",
          ["job", "content", "types", "definitions"],
          ["CONTENTTYPEID"], ["BUSINESSUNITID"], 100),
        
        # Person and organizational tables
        T("dbo.PSNACCOUNT_D",
          ["PERSONID", "TRUENAME", "EMPLOYEEID", "COMPANYEMAIL", "BELONGCORPID", 
           "BRANCHID", "POSITIONID", "MOBILE"],
          "Person master data including soft-deleted records - central for all people queries",
          ["person", "employee", "master", "people"],
          ["PERSONID"], ["EMPLOYEEID", "TRUENAME"], 10000),
        
        T("dbo.PSNFAMILYINFO",
          ["PERSONID", "RELATION", "NAME", "PHONE", "EMAIL", "ADDRESS"],
          "Employee family contact information",
          ["person", "family", "contacts"],
          [], ["PERSONID"], 15000),
        
        # Calendar and rules tables
        T("dbo.ATDLEGALCALENDAR",
          ["CALENDARDATE", "LEGALID", "CALENDARTYPE"],
          "Legal holidays and calendar definitions",
          ["calendar", "holidays", "legal"],
          ["CALENDARDATE", "LEGALID"], ["CALENDARDATE"], 1000,
          temporal=["CALENDARDATE"]),
        
        T("dbo.ATDHISTIMEORDERCALENDAR",
          ["RECORDID", "CALENDARDATE", "TIMEORDERID", "TIMECLASSID", "CALENDARTYPE", "BUSINESSUNITID"],
          "Time order calendar scheduling",
          ["calendar", "timeorder", "scheduling"],
          ["RECORDID"], ["TIMEORDERID", "CALENDARDATE"], 50000,
          temporal=["CALENDARDATE"]),
        
        T("dbo.ATDLONGEVITYRULE",
          ["RULEID", "YEARMORETHAN", "YEARLESSTHAN", "VACATIONDAYS", "BUSINESSUNITID"],
          "Longevity-based vacation day rules and calculations",
          ["vacation", "rules", "longevity", "calculations"],
          ["RULEID"], ["BUSINESSUNITID"], 50),
        
        T("dbo.ATDNOCARDREASON",
          ["REASONID", "REASONCODE", "REASONVALUE", "REASONDESC", "BUSINESSRULEID"],
          "Predefined reasons for missing timecard events",
          ["reasons", "notimecard", "codes"],
          ["REASONID"], ["BUSINESSRULEID"], 100),
        
        # EDF integration table
        T("dbo.EDFATDLEAVEDATA",
          ["RECORDID", "ATTENDANCETYPE", "PERSONID", "WORKDATES", "WORKDATEE", 
           "STARTTIME", "ENDTIME", "HOURS", "DEPARTMENTID", "BUSINESSUNITID"],
          "EDF-integrated leave data from external systems",
          ["leave", "edf", "integration", "external"],
          ["RECORDID"], ["PERSONID"], 25000,
          temporal=["WORKDATES", "WORKDATEE"]),
        
        # Additional deleted/historical tables
        T("dbo.ATDLATEEARLY_D",
          ["DELETEID", "ATTENDDATE", "PERSONID", "SHOULDTIME", "CARDTIME", "MINUTES", 
           "ATDTYPE", "LATEEARLYID", "DELETE_USER_ID", "DELETE_TIME"],
          "Deleted late/early records with audit information",
          ["attendance", "late", "early", "deleted", "audit"],
          ["DELETEID"], ["PERSONID", "DELETE_TIME"], 50000, is_del=True,
          temporal=["ATTENDDATE", "DELETE_TIME"]),
        
        T("dbo.ATDLATEEARLY_C",
          ["CALGUID", "ATTENDDATE", "PERSONID", "SHOULDTIME", "CARDTIME", "MINUTES", 
           "ATDTYPE", "LATEEARLYID"],
          "Calculated late/early attendance data",
          ["attendance", "late", "early", "calculated"],
          ["CALGUID"], ["PERSONID", "ATTENDDATE"], 75000,
          temporal=["ATTENDDATE"]),
        
        T("dbo.ATDLEAVEDATA_T",
          ["ATTENDANCETYPE", "PERSONID", "WORKDATE", "STARTTIME", "ENDTIME", "HOURS", 
           "LEAVEID", "TRANSFERUNITID", "LEAVEREASON"],
          "Transferred leave data between business units",
          ["leave", "transferred", "businessunit"],
          ["LEAVEID"], ["PERSONID", "TRANSFERUNITID"], 5000,
          temporal=["WORKDATE"]),
        
        T("dbo.ATDLEAVEDATAEX_D",
          ["DELETEID", "ATTENDANCETYPE", "PERSONID", "WORKDATE", "HOURS", "VACATIONID", 
           "LEAVEID", "DELETE_USER_ID", "DELETE_TIME"],
          "Deleted extended leave data with audit trail",
          ["leave", "extended", "deleted", "audit"],
          ["DELETEID"], ["PERSONID", "DELETE_TIME"], 10000, is_del=True,
          temporal=["WORKDATE", "DELETE_TIME"]),
        
        T("dbo.ATDNONCALCULATEDVACATION_D",
          ["DELETEID", "OID", "PERSONID", "BUSINESSUINTID", "VACATIONDAYS", "USEDDAYS", 
           "REMAINDAYS", "VACATIONTYPE", "DELETE_USER_ID", "DELETE_TIME"],
          "Deleted vacation balance records",
          ["vacation", "balance", "deleted", "audit"],
          ["DELETEID"], ["PERSONID", "DELETE_TIME"], 5000, is_del=True,
          temporal=["DELETE_TIME"]),
        
        T("dbo.ATDJOBCONTENT_T",
          ["JOBCONTENTID", "PERSONID", "WORKDATE", "STARTDATE", "STARTTIME", 
           "ENDDATE", "ENDTIME", "CONTENTTYPEID", "TRANSFERUNITID"],
          "Transferred job content records",
          ["job", "content", "transferred"],
          ["JOBCONTENTID"], ["PERSONID", "TRANSFERUNITID"], 10000,
          temporal=["WORKDATE", "STARTDATE", "ENDDATE"])
    ]
    
    return LeaveVectorDB(tables)


# Additional utility functions for enhanced query optimization

def analyze_query_complexity(query: str) -> Dict[str, Any]:
    """Analyze query complexity and provide optimization suggestions."""
    q = query.lower()
    
    complexity_indicators = {
        "temporal_range": any(term in q for term in ["between", "range", "from", "to", "last", "past"]),
        "aggregation": any(term in q for term in ["count", "sum", "average", "total", "max", "min"]),
        "grouping": any(term in q for term in ["by", "group", "department", "unit", "type"]),
        "multiple_tables": len([term for term in ["leave", "overtime", "attendance", "vacation"] if term in q]) > 1,
        "person_lookup": any(term in q for term in ["who", "person", "employee", "name"]),
        "current_data": any(term in q for term in ["current", "today", "now", "active"]),
        "historical_data": any(term in q for term in ["history", "historical", "trend", "past"])
    }
    
    optimization_hints = []
    
    if complexity_indicators["temporal_range"]:
        optimization_hints.append("Use indexed date columns for range queries")
    
    if complexity_indicators["multiple_tables"]:
        optimization_hints.append("Consider join order: smaller tables first")
    
    if complexity_indicators["historical_data"]:
        optimization_hints.append("Historical tables are large - always filter by date range")
    
    if complexity_indicators["aggregation"] and not complexity_indicators["temporal_range"]:
        optimization_hints.append("Add date filters to aggregation queries for better performance")
    
    return {
        "complexity_score": sum(complexity_indicators.values()),
        "indicators": complexity_indicators,
        "optimization_hints": optimization_hints
    }


def suggest_query_improvements(query: str, selected_tables: List[str]) -> List[str]:
    """Suggest improvements based on selected tables and query pattern."""
    suggestions = []
    q = query.lower()
    
    # Check for missing essential filters
    large_tables = ["dbo.ATDHISTIMECARDDATA", "dbo.ATDHISLEAVEDATA", "dbo.ATDHISLATEEARLY"]
    if any(table in selected_tables for table in large_tables):
        if not any(term in q for term in ["date", "between", "last", "past", "month", "year"]):
            suggestions.append("Add date range filter for better performance on historical tables")
    
    # Check for person queries without name resolution
    if any(term in q for term in ["who", "person", "employee"]) and "dbo.PSNACCOUNT_D" not in selected_tables:
        suggestions.append("Include PSNACCOUNT_D table for person name resolution")
    
    # Check for department/unit queries
    if any(term in q for term in ["department", "unit", "team"]):
        if not any("DEPARTMENTID" in table or "BUSINESSUNITID" in table for table in selected_tables):
            suggestions.append("Consider filtering by DEPARTMENTID or BUSINESSUNITID")
    
    # Check for validation status
    if "leave" in q and "dbo.ATDLEAVEDATA" in selected_tables:
        suggestions.append("Consider filtering by VALIDATED = 1 for approved leave only")
    
    return suggestions