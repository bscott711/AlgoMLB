import pytest
from algomlb.config import get_settings

@pytest.mark.functional
def test_no_all_null_columns(db_inspector):
    """Flag any column where 100% of populated rows are NULL, unless allow-listed."""
    settings = get_settings()
    allowed = {
        (table, col)
        for table, cols in settings.db_health.allow_null_columns.items()
        for col in cols
    }
    
    violations = db_inspector.all_null_columns()
    unexpected = [
        (col.table, col.name)
        for col in violations
        if (col.table, col.name) not in allowed
    ]
    
    assert unexpected == [], (
        "The following columns are 100% NULL and not in the allow-list:\n"
        + "\n".join(f"  - {t}.{c}" for t, c in unexpected)
    )
