from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from typer.testing import CliRunner
from algomlb.cli.process import app
import datetime

runner = CliRunner()


@pytest.fixture
def mock_ml_funcs():
    with (
        patch("algomlb.ml.quant_processor.process_quant_for_game") as m1,
        patch("algomlb.ml.quant_processor.process_quant_for_date") as m2,
        patch("algomlb.ml.silver_processor.process_silver_incremental") as m3,
        patch("algomlb.ml.silver_processor.summarize_to_silver") as m4,
        patch("algomlb.ml.silver_processor._upsert_silver") as m5,
    ):
        yield {"pqfg": m1, "pqfd": m2, "psi": m3, "sts": m4, "us": m5}


def test_quant_game_pk(mock_ml_funcs):
    result = runner.invoke(app, ["quant", "--game-pk", "123"])
    assert result.exit_code == 0
    mock_ml_funcs["pqfg"].assert_called_with(123, dry_run=False)


def test_quant_date(mock_ml_funcs):
    result = runner.invoke(app, ["quant", "--date", "2025-04-01"])
    assert result.exit_code == 0
    mock_ml_funcs["pqfd"].assert_called()


def test_quant_range(mock_ml_funcs):
    result = runner.invoke(
        app, ["quant", "--start-date", "2025-04-01", "--end-date", "2025-04-02"]
    )
    assert result.exit_code == 0
    assert mock_ml_funcs["pqfd"].call_count == 2


def test_quant_error(mock_ml_funcs):
    result = runner.invoke(app, ["quant"])
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_silver_incremental(mock_ml_funcs):
    result = runner.invoke(app, ["silver", "--incremental"])
    assert result.exit_code == 0
    mock_ml_funcs["psi"].assert_called()


def test_silver_date(mock_ml_funcs):
    with (
        patch("algomlb.cli.process.get_engine"),
        patch(
            "algomlb.cli.process.pd.read_sql",
            return_value=pd.DataFrame({"game_pk": [1]}),
        ),
    ):
        result = runner.invoke(app, ["silver", "--date", "2025-04-01"])
        assert result.exit_code == 0
        mock_ml_funcs["us"].assert_called()


def test_silver_year(mock_ml_funcs):
    mock_engine = MagicMock()
    with (
        patch("algomlb.cli.process.get_engine", return_value=mock_engine),
        patch(
            "algomlb.cli.process.pd.read_sql",
            return_value=pd.DataFrame({"game_pk": [1]}),
        ),
    ):
        mock_engine.connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
            (datetime.date(2025, 4, 1),)
        ]
        result = runner.invoke(app, ["silver", "--year", "2025"])
        assert result.exit_code == 0
        mock_ml_funcs["us"].assert_called()


def test_silver_year_empty(mock_ml_funcs):
    mock_engine = MagicMock()
    with patch("algomlb.cli.process.get_engine", return_value=mock_engine):
        mock_engine.connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = []
        result = runner.invoke(app, ["silver", "--year", "2025"])
        assert result.exit_code == 0
        assert "No game data found" in result.stdout


def test_silver_error(mock_ml_funcs):
    result = runner.invoke(app, ["silver"])
    assert result.exit_code == 1


def test_silver_date_empty(mock_ml_funcs):
    """Test 'silver --date' when no raw data is available."""
    with (
        patch("algomlb.cli.process.get_engine"),
        patch("algomlb.cli.process.pd.read_sql", return_value=pd.DataFrame()),
    ):
        result = runner.invoke(app, ["silver", "--date", "2025-04-01"])
        assert result.exit_code == 0
        assert "No data found" in result.stdout
        mock_ml_funcs["us"].assert_not_called()
