import pytest
import re
from algomlb.config import get_settings


@pytest.mark.functional
def test_table_naming_convention(db_inspector):
    """All tables must follow the configured snake_case convention."""
    settings = get_settings()
    pattern = re.compile(settings.db_health.table_naming_pattern)

    violations = [
        t.name for t in db_inspector.list_tables() if not pattern.match(t.name)
    ]
    assert violations == [], f"Tables with invalid naming conventions: {violations}"
