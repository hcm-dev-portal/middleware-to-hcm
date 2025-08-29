# backend/app/tests/test_leave_vector.py
import pathlib, sys
import pytest

# ── Put backend/ on sys.path so "app" is a top-level package ────────────────
THIS_FILE = pathlib.Path(__file__).resolve()

BACKEND_DIR = None
for p in THIS_FILE.parents:
    cand = p / "backend" / "app"
    if cand.is_dir():
        BACKEND_DIR = p / "backend"
        break
if BACKEND_DIR is None:
    raise RuntimeError("Could not find 'backend/app' relative to test file")
sys.path.insert(0, str(BACKEND_DIR))

# Now these imports work because top-level "app" exists
from app.services.leave_vector import (
    build_leave_index,
    LeaveVectorDB,
    suggest_query_improvements,
)

def _has_join(db: LeaveVectorDB, left, lcol, right, rcol, condition_substr=None):
    for j in db._joins:  # touching internals is fine in tests
        if (j.left_table == left and j.left_column == lcol and
            j.right_table == right and j.right_column == rcol):
            if condition_substr is None:
                return True
            if j.condition and condition_substr in j.condition:
                return True
    return False

@pytest.fixture(scope="module")
def db():
    return build_leave_index()

def test_health_and_sanity(db: LeaveVectorDB):
    hc = db.health_check()
    assert hc["ready"] is True
    assert hc["tables_indexed"] > 0
    assert hc["person_table"] in (None, "dbo.PSNACCOUNT_D", "dbo.BIPSNACCOUNTSP")

    report = db.relationships_sanity_check()
    assert report["errors"] == [], f"Join/column errors: {report['errors']}"
    if report["warnings"]:
        print("WARNINGS:", report["warnings"])

def test_core_joins_exist(db: LeaveVectorDB):
    person = db.health_check()["person_table"]

    assert _has_join(db,
        "dbo.ATDLEAVEDATA", "LEAVEID",
        "dbo.ATDLEAVEDATAEX", "LEAVEID"
    )

    assert _has_join(db,
        "dbo.ATDLEAVEDATAEX", "VACATIONID",
        "dbo.ATDNONCALCULATEDVACATION", "OID"
    )

    assert _has_join(db,
        "dbo.ATDLEAVECANCELDATA", "PERSONID",
        "dbo.ATDLEAVEDATA", "PERSONID",
        condition_substr="ATTENDANCETYPE"
    )

    assert _has_join(db,
        "dbo.ATDLEAVEDATA", "WORKDATE",
        "dbo.ATDLEGALCALENDAR", "CALENDARDATE"
    )

    if person:
        assert _has_join(db,
            "dbo.ATDLEAVEDATA", "PERSONID",
            person, "PERSONID"
        )

def test_join_hints_render_conditions(db: LeaveVectorDB):
    hints = db.join_hints(["dbo.ATDLEAVEDATA", "dbo.ATDLEGALCALENDAR"])
    rendered = [h for h in hints if
                "JOIN dbo.ATDLEGALCALENDAR" in h and
                "dbo.ATDLEAVEDATA.WORKDATE = dbo.ATDLEGALCALENDAR.CALENDARDATE" in h]
    assert rendered, f"Expected a rendered join hint, got: {hints}"

@pytest.mark.parametrize("q,expected", [
    ("who is currently on leave today?", "current_leave_status"),
    ("show upcoming leave next month", "upcoming_leave"),
    ("leave trend last year by department", "historical_leave_analysis"),
    ("reconcile vacation balance for annual leave", "leave_balance_reconciliation"),
    ("which leaves were cancelled last week", "leave_cancellations"),
])
def test_query_patterns(db: LeaveVectorDB, q, expected):
    pat = db.get_query_pattern(q)
    assert pat and pat.pattern == expected

def test_search_biases_leave_tables(db: LeaveVectorDB):
    top = db.search_relevant_tables("current leave today", top_k=3)
    names = [n for n, _ in top]
    assert any("ATDLEAVEDATA" in n for n in names), f"Top tables: {names}"

def test_suggestions_person_resolution():
    suggs = suggest_query_improvements("who is on leave today", ["dbo.ATDLEAVEDATA"])
    assert any("Include person dimension table" in s for s in suggs)
