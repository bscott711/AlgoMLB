from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Engine

from algomlb.db.introspection import SchemaInspector, TableMeta, ColumnMeta


@pytest.fixture
def mock_engine():
    return MagicMock(spec=Engine)


def test_is_expired(mock_engine):
    inspector = SchemaInspector(mock_engine, ttl_minutes=5)
    now = datetime(2024, 1, 1, 12, 0, 0)

    with patch("algomlb.db.introspection.datetime") as mock_datetime:
        mock_datetime.now.return_value = now
        # timestamp 4 mins ago -> not expired
        assert not inspector._is_expired(now - timedelta(minutes=4))
        # timestamp 6 mins ago -> expired
        assert inspector._is_expired(now - timedelta(minutes=6))


def test_list_tables_caching(mock_engine):
    inspector = SchemaInspector(mock_engine, ttl_minutes=5)

    # Mock connection and result
    mock_conn = mock_engine.connect.return_value.__enter__.return_value
    mock_result = mock_conn.execute.return_value
    mock_result.__iter__.return_value = [("table1", 100), ("table2", 0)]

    # First call - executes query
    tables = inspector.list_tables()
    assert len(tables) == 2
    assert tables[0].name == "table1"
    assert tables[1].is_empty is True
    assert mock_conn.execute.call_count == 1

    # Second call - returns cached (within TTL)
    with patch("algomlb.db.introspection.datetime") as mock_dt:
        mock_dt.now.return_value = datetime.now()
        tables2 = inspector.list_tables()
        assert tables2 == tables
        assert mock_conn.execute.call_count == 1  # No new execute

        # Expire cache
        mock_dt.now.return_value = datetime.now() + timedelta(minutes=10)
        inspector.list_tables()
        assert mock_conn.execute.call_count == 2


def test_column_report_missing_table(mock_engine):
    inspector = SchemaInspector(mock_engine)
    mock_conn = mock_engine.connect.return_value.__enter__.return_value
    mock_conn.execute.return_value.fetchall.return_value = []

    report = inspector.column_report("nonexistent")
    assert report == []


def test_column_report_stats_exception(mock_engine):
    inspector = SchemaInspector(mock_engine)
    mock_conn = mock_engine.connect.return_value.__enter__.return_value

    # Mock base info result
    mock_base_res = MagicMock()
    mock_row = MagicMock()
    mock_row.column_name = "col1"
    mock_row.data_type = "integer"
    mock_row.is_nullable = "YES"
    mock_base_res.fetchall.return_value = [mock_row]

    # Mock stats query to raise exception
    mock_conn.execute.side_effect = [
        mock_base_res,  # for base_info
        Exception("DB Error"),  # for stats_query
    ]

    report = inspector.column_report("table1")
    assert len(report) == 1
    assert report[0].name == "col1"
    assert report[0].null_pct == 0.0  # Default fallback


def test_column_report_caching(mock_engine):
    inspector = SchemaInspector(mock_engine)
    mock_conn = mock_engine.connect.return_value.__enter__.return_value

    # Mock base info result
    mock_base_res = MagicMock()
    mock_row = MagicMock()
    mock_row.column_name = "col1"
    mock_row.data_type = "text"
    mock_row.is_nullable = "NO"
    mock_base_res.fetchall.return_value = [mock_row]

    # Mock stats result
    mock_stats_res = MagicMock()
    mock_stats_res.fetchone.return_value = [0.5]

    mock_conn.execute.side_effect = [mock_base_res, mock_stats_res]

    # Call 1
    report = inspector.column_report("table1")
    assert report[0].null_pct == 0.5
    assert mock_conn.execute.call_count == 2

    # Call 2 - cached
    count_before = mock_conn.execute.call_count
    report2 = inspector.column_report("table1")
    assert report2 == report
    assert mock_conn.execute.call_count == count_before


def test_foreign_keys_caching(mock_engine):
    inspector = SchemaInspector(mock_engine)
    mock_conn = mock_engine.connect.return_value.__enter__.return_value

    mock_row = MagicMock()
    mock_row._asdict.return_value = {
        "from_table": "f",
        "from_col": "fc",
        "to_table": "t",
        "to_col": "tc",
    }
    mock_conn.execute.return_value = [mock_row]

    fks = inspector.foreign_keys()
    assert len(fks) == 1
    assert fks[0].from_table == "f"

    # Cached call
    inspector.foreign_keys()
    assert mock_conn.execute.call_count == 1


def test_empty_tables(mock_engine):
    inspector = SchemaInspector(mock_engine)
    with patch.object(inspector, "list_tables") as mock_list:
        mock_list.return_value = [TableMeta("t1", 100, False), TableMeta("t2", 0, True)]
        empty = inspector.empty_tables()
        assert len(empty) == 1
        assert empty[0].name == "t2"


def test_all_null_columns(mock_engine):
    inspector = SchemaInspector(mock_engine)
    with patch.object(inspector, "list_tables") as mock_list:
        mock_list.return_value = [
            TableMeta("t1", 100, False),
            TableMeta("t2", 0, True),  # Empty table should be skipped
        ]
        with patch.object(inspector, "column_report") as mock_report:
            mock_report.return_value = [
                ColumnMeta("t1", "c1", "int", True, 1.0, True),
                ColumnMeta("t1", "c2", "int", True, 0.5, False),
            ]
            violations = inspector.all_null_columns()
            assert len(violations) == 1
            assert violations[0].name == "c1"
            mock_report.assert_called_once_with("t1")


def test_clear_cache(mock_engine):
    inspector = SchemaInspector(mock_engine)
    inspector._table_cache = (datetime.now(), [])
    inspector._column_cache = {"t1": (datetime.now(), [])}
    inspector._fk_cache = (datetime.now(), [])

    inspector.clear_cache()
    assert inspector._table_cache is None
    assert inspector._column_cache == {}
    assert inspector._fk_cache is None
