import datetime
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from algomlb.ml.quant_processor import (
    _fetch_raw_batch,
    _fetch_baseline,
    _upsert_quant,
    process_quant_for_date,
    process_quant_for_game,
)


@pytest.fixture
def mock_engine():
    return MagicMock()


def test_fetch_raw_batch(mock_engine):
    _mock_conn = mock_engine.connect.return_value.__enter__.return_value
    with patch("pandas.read_sql", return_value=pd.DataFrame({"game_pk": [1]})):
        df = _fetch_raw_batch(mock_engine, datetime.date(2025, 4, 1))
        assert not df.empty
        assert "game_pk" in df.columns


def test_fetch_baseline(mock_engine):
    _mock_conn = mock_engine.connect.return_value.__enter__.return_value
    with patch("pandas.read_sql", return_value=pd.DataFrame({"game_pk": [1]})):
        df = _fetch_baseline(mock_engine, datetime.date(2025, 4, 1), 7)
        assert not df.empty


def test_upsert_quant_empty(mock_engine):
    assert _upsert_quant(mock_engine, pd.DataFrame()) == 0


def test_upsert_quant_values(mock_engine):
    df = pd.DataFrame(
        [{"game_pk": 1, "at_bat_number": 1, "pitch_number": 1, "val": 1.0}]
    )
    # Mocking pg_insert and execute
    with patch("algomlb.ml.quant_processor.pg_insert") as _mock_insert:
        count = _upsert_quant(mock_engine, df)
        assert count == 1
        assert mock_engine.begin.called


def test_upsert_quant_with_nans(mock_engine):
    """Test NaN to None conversion in _upsert_quant."""
    df = pd.DataFrame(
        [{"game_pk": 1, "at_bat_number": 1, "pitch_number": 1, "val": float("nan")}]
    )
    with patch("algomlb.ml.quant_processor.pg_insert") as mock_insert:
        _upsert_quant(mock_engine, df)
        # Check the values passed to insert
        passed_values = mock_insert.return_value.values.call_args[0][0]
        assert passed_values[0]["val"] is None


def test_process_quant_for_date_no_rows(mock_engine):
    with patch(
        "algomlb.ml.quant_processor._fetch_raw_batch", return_value=pd.DataFrame()
    ):
        res = process_quant_for_date(datetime.date(2025, 4, 1), engine=mock_engine)
        assert res == 0


def test_process_quant_for_date_dry_run(mock_engine):
    raw = pd.DataFrame({"game_pk": [1]})
    baseline = pd.DataFrame({"game_pk": [2]})
    quant = pd.DataFrame({"game_pk": [1], "q": [0.5]})

    with patch("algomlb.ml.quant_processor._fetch_raw_batch", return_value=raw):
        with patch("algomlb.ml.quant_processor._fetch_baseline", return_value=baseline):
            with patch(
                "algomlb.ml.quant_processor.build_quant_features", return_value=quant
            ):
                res = process_quant_for_date(
                    datetime.date(2025, 4, 1), engine=mock_engine, dry_run=True
                )
                assert res == 0


def test_process_quant_for_date_success(mock_engine):
    raw = pd.DataFrame({"game_pk": [1]})
    baseline = pd.DataFrame({"game_pk": [2]})
    quant = pd.DataFrame({"game_pk": [1], "q": [0.5]})

    with patch("algomlb.ml.quant_processor._fetch_raw_batch", return_value=raw):
        with patch("algomlb.ml.quant_processor._fetch_baseline", return_value=baseline):
            with patch(
                "algomlb.ml.quant_processor.build_quant_features", return_value=quant
            ):
                with patch("algomlb.ml.quant_processor._upsert_quant", return_value=1):
                    res = process_quant_for_date(
                        datetime.date(2025, 4, 1), engine=mock_engine
                    )
                    assert res == 1


def test_process_quant_for_date_empty_baseline(mock_engine):
    raw = pd.DataFrame({"game_pk": [1]})
    baseline = pd.DataFrame()
    quant = pd.DataFrame({"game_pk": [1], "q": [0.5]})

    with patch("algomlb.ml.quant_processor._fetch_raw_batch", return_value=raw):
        with patch("algomlb.ml.quant_processor._fetch_baseline", return_value=baseline):
            with patch(
                "algomlb.ml.quant_processor.build_quant_features", return_value=quant
            ):
                with patch("algomlb.ml.quant_processor._upsert_quant", return_value=1):
                    res = process_quant_for_date(
                        datetime.date(2025, 4, 1), engine=mock_engine
                    )
                    assert res == 1


def test_process_quant_for_game_not_found(mock_engine):
    mock_conn = mock_engine.connect.return_value.__enter__.return_value
    mock_conn.execute.return_value.fetchone.return_value = None
    res = process_quant_for_game(1, engine=mock_engine)
    assert res == 0


def test_process_quant_for_game_success(mock_engine):
    mock_conn = mock_engine.connect.return_value.__enter__.return_value
    mock_conn.execute.return_value.fetchone.return_value = [datetime.date(2025, 4, 1)]
    with patch("algomlb.ml.quant_processor.process_quant_for_date", return_value=1):
        res = process_quant_for_game(1, engine=mock_engine)
        assert res == 1


def test_process_quant_for_game_iso_date(mock_engine):
    mock_conn = mock_engine.connect.return_value.__enter__.return_value
    mock_conn.execute.return_value.fetchone.return_value = ["2025-04-01"]
    with patch("algomlb.ml.quant_processor.process_quant_for_date", return_value=1):
        res = process_quant_for_game(1, engine=mock_engine)
        assert res == 1
