# backend/app/tests/test_leave_vector_aliases.py
import pathlib, sys
from typing import List
import pytest

THIS_FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # repo/
sys.path.insert(0, str(PROJECT_ROOT / "backend"))  # make "app" importable

from app.services.leave_vector import build_leave_index, LeaveVectorDB

@pytest.fixture(scope="module")
def db() -> LeaveVectorDB:
    return build_leave_index()

def test_alias_businessunitid_mapped(db: LeaveVectorDB):
    # If the table exposes BUSINESSUINTID (typo) but not BUSINESSUNITID,
    # relationships_sanity_check should still be clean (alias bridges it).
    report = db.relationships_sanity_check()
    assert report["errors"] == [], f"Alias mapping failed: {report['errors']}"

def test_deleted_and_historical_not_overranked(db: LeaveVectorDB):
    top = db.search_relevant_tables("current leave today", top_k=5)
    names = [n for n, _ in top]
    # ensure current table ranks above deleted/historical in a "current" query
    assert any("ATDLEAVEDATA" in n for n in names)
    assert not (names and names[0].endswith("_D")), f"Deleted table overranked: {names}"

def test_join_hint_contains_cardinality(db: LeaveVectorDB):
    hints = db.join_hints(["dbo.ATDLEAVEDATA", "dbo.ATDLEAVEDATAEX"])
    assert any("-- 1:1" in h or "-- 1:M" in h or "-- M:1" in h for h in hints), hints
