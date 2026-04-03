import pytest
from typer.testing import CliRunner
from unittest.mock import patch
from datetime import date
from algomlb.cli.process import app

runner = CliRunner()


@pytest.fixture
def mock_service():
    with patch("algomlb.cli.process.RollingService") as mock:
        yield mock.return_value


@pytest.fixture
def mock_processor():
    with patch("algomlb.cli.process.RollingProcessor") as mock:
        yield mock.return_value


@patch("algomlb.cli.process.get_settings")
@patch("algomlb.cli.process.DatabaseRepository")
def test_process_rolling_date(mock_repo, mock_settings, mock_processor, mock_service):
    mock_service.process_date_range.return_value = 10

    result = runner.invoke(app, ["rolling", "--date", "2024-04-01"])

    assert result.exit_code == 0
    assert "Processed 10 rolling records" in result.stdout
    mock_service.process_date_range.assert_called_once_with(
        date(2024, 4, 1), date(2024, 4, 1), dry_run=False
    )


@patch("algomlb.cli.process.get_settings")
@patch("algomlb.cli.process.DatabaseRepository")
def test_process_rolling_range(mock_repo, mock_settings, mock_processor, mock_service):
    mock_service.process_date_range.return_value = 50
    result = runner.invoke(
        app, ["rolling", "--start", "2024-04-01", "--end", "2024-04-05", "--dry-run"]
    )

    assert result.exit_code == 0
    mock_service.process_date_range.assert_called_once_with(
        date(2024, 4, 1), date(2024, 4, 5), dry_run=True
    )


def test_process_rolling_no_args():
    result = runner.invoke(app, ["rolling"])
    assert result.exit_code == 1
    assert "Error: Please provide --date or --start" in result.stderr
