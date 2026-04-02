import pytest
from algomlb.config import get_settings

@pytest.mark.functional
def test_empty_tables_are_intentional(db_inspector):
    """Every empty table must be a known placeholder."""
    settings = get_settings()
    placeholders = set(settings.db_health.known_placeholders)
    
    empty = db_inspector.empty_tables()
    orphaned = [t.name for t in empty if t.name not in placeholders]
    
    assert orphaned == [], (
        "Empty tables not in known-placeholder list (possibly orphaned):\n"
        + "\n".join(f"  - {n}" for n in orphaned)
        + "\nEither add to 'known_placeholders' in config.yaml or drop them."
    )
