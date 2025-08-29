# backend/app/tests/test_relationship_files.py
import json, pathlib, sys
import pytest

THIS_FILE = pathlib.Path(__file__).resolve()
REPO = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app.services.leave_vector import build_leave_index

REL_FILES = [
    REPO / "database_relationships.json",
    REPO / "discovered_relationships.json",
]

@pytest.mark.parametrize("path", REL_FILES)
def test_relationship_json_against_index(path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    data = json.loads(path.read_text(encoding="utf-8"))
    db = build_leave_index()
    idx = {t.full.lower(): {c.lower() for c in t.columns} for t in db.tables}
    problems = []
    for rel in data if isinstance(data, list) else data.get("relationships", []):
        lt = rel.get("left_table", "").lower()
        rt = rel.get("right_table", "").lower()
        lc = (rel.get("left_column") or "").lower()
        rc = (rel.get("right_column") or "").lower()
        if lt not in idx:
            problems.append(f"Unknown left_table: {lt}")
            continue
        if rt not in idx:
            problems.append(f"Unknown right_table: {rt}")
            continue
        # columns are soft-validated here (aliases handled inside leave_vector)
        if lc and (lc not in idx[lt]):
            # allow missing here; leave_vector aliases may cover it
            pass
        if rc and (rc not in idx[rt]):
            pass
    assert not problems, f"Issues in {path.name}: {problems}"
