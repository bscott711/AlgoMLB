import pytest

@pytest.mark.functional
def test_fk_targets_exist(db_inspector):
    """Every FK must reference a table that actually exists in the schema."""
    all_tables = {t.name for t in db_inspector.list_tables()}
    fks = db_inspector.foreign_keys()
    
    broken = [fk for fk in fks if fk.to_table not in all_tables]
    assert broken == [], f"Broken FK references (target table missing): {broken}"
