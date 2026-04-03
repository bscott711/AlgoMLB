import pytest
from unittest.mock import MagicMock
from algomlb.db.introspection import SchemaInspector


@pytest.fixture
def mock_engine():
    return MagicMock()


@pytest.fixture
def inspector(mock_engine):
    return SchemaInspector(mock_engine)


def test_list_tables(inspector, mock_engine):
    mock_conn = mock_engine.connect.return_value.__enter__.return_value
    mock_result = MagicMock()
    mock_result.__iter__.return_value = [
        MagicMock(table_name="test_table", row_count=10),
        MagicMock(table_name="empty_table", row_count=0),
    ]
    mock_conn.execute.return_value = mock_result

    tables = inspector.list_tables()

    assert len(tables) == 2
    assert tables[0].name == "test_table"
    assert tables[0].row_count == 10
    assert not tables[0].is_empty
    assert tables[1].name == "empty_table"
    assert tables[1].is_empty


def test_column_report(inspector, mock_engine):
    mock_conn = mock_engine.connect.return_value.__enter__.return_value

    # Mock base info results
    mock_base_info = [
        MagicMock(column_name="col1", data_type="integer", is_nullable="NO"),
        MagicMock(column_name="col2", data_type="text", is_nullable="YES"),
    ]

    # Mock stats query result (null percentages)
    mock_stats_row = [0.0, 1.0]  # col1 is 0% null, col2 is 100% null

    def side_effect(query, params=None):
        query_str = str(query)
        if "information_schema.columns" in query_str:
            mock_result = MagicMock()
            mock_result.fetchall.return_value = mock_base_info
            return mock_result
        if "COUNT(*) FILTER" in query_str:
            mock_result = MagicMock()
            mock_result.fetchone.return_value = mock_stats_row
            return mock_result
        return MagicMock()

    mock_conn.execute.side_effect = side_effect

    columns = inspector.column_report("test_table")

    assert len(columns) == 2
    assert columns[0].name == "col1"
    assert not columns[0].nullable
    assert columns[0].null_pct == 0.0

    assert columns[1].name == "col2"
    assert columns[1].nullable
    assert columns[1].null_pct == 1.0
    assert columns[1].is_all_null


def test_foreign_keys(inspector, mock_engine):
    mock_conn = mock_engine.connect.return_value.__enter__.return_value
    mock_result = [
        MagicMock(
            _asdict=lambda: {
                "from_table": "t1",
                "from_col": "c1",
                "to_table": "t2",
                "to_col": "c2",
            }
        )
    ]
    mock_conn.execute.return_value = mock_result

    fks = inspector.foreign_keys()

    assert len(fks) == 1
    assert fks[0].from_table == "t1"
    assert fks[0].to_table == "t2"
