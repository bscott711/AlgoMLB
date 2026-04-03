import pytest
from algomlb.db.introspection import SchemaInspector


@pytest.mark.functional
def test_statcast_raw_table_exists(db_inspector: SchemaInspector):
    tables = {t.name for t in db_inspector.list_tables()}
    assert "statcast_raw" in tables, "statcast_raw table not found — run the migration."


@pytest.mark.functional
def test_statcast_raw_has_primary_key_columns(db_inspector: SchemaInspector):
    cols = {c.name for c in db_inspector.column_report("statcast_raw")}
    required = {"game_pk", "at_bat_number", "pitch_number"}
    assert required.issubset(cols), (
        f"Missing required PK columns in statcast_raw. Found: {cols}"
    )


@pytest.mark.functional
def test_statcast_raw_coordinate_columns_present(db_inspector: SchemaInspector):
    cols = {c.name for c in db_inspector.column_report("statcast_raw")}
    assert "hc_x" in cols and "hc_y" in cols, (
        "Missing hit coordinates (hc_x, hc_y) in statcast_raw"
    )


@pytest.mark.functional
def test_statcast_raw_launch_metrics_present(db_inspector: SchemaInspector):
    cols = {c.name for c in db_inspector.column_report("statcast_raw")}
    metrics = {
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
        "estimated_ba_using_speedangle",
    }
    assert metrics.issubset(cols), (
        f"Missing launch metrics in statcast_raw. Found: {cols}"
    )
