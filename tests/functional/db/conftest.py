import pytest
from algomlb.db.introspection import SchemaInspector
from algomlb.db.session import get_engine


@pytest.fixture(scope="session")
def db_inspector():
    return SchemaInspector(get_engine())


@pytest.fixture(scope="session", autouse=True)
def print_schema_health(db_inspector):
    """Prints a quick health summary at the start of the functional test session."""
    tables = db_inspector.list_tables()
    empty = [t for t in tables if t.is_empty]
    null_cols = db_inspector.all_null_columns()
    
    print(f"\n{'='*60}")
    print("📡 DB Functional Health Summary")
    print(f"  Total Tables:    {len(tables)}")
    print(f"  Empty Tables:    {len(empty)}")
    print(f"  All-NULL Cols:   {len(null_cols)}")
    print(f"{'='*60}\n")
