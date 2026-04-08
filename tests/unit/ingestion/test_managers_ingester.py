import pytest
from unittest.mock import MagicMock, patch
from algomlb.ingestion.managers_ingester import (
    _fetch_teams,
    _fetch_manager,
    backfill_team_managers,
)


@pytest.fixture
def teams_json():
    return {
        "teams": [
            {
                "id": 119,
                "abbreviation": "LAN",
                "name": "Los Angeles Dodgers",
                "sport": {"id": 1},
            },
            {
                "id": 999,
                "abbreviation": "AAA",
                "name": "Minor League Team",
                "sport": {"id": 11},  # Should be filtered out
            },
        ]
    }


@pytest.fixture
def roster_json():
    return {
        "roster": [
            {
                "jobId": "MNGR",
                "person": {"id": 123, "fullName": "Dave Roberts"},
                "jerseyNumber": "30",
            },
            {
                "jobId": "COACH",  # Should be skipped
                "person": {"id": 456, "fullName": "Coach Smith"},
            },
        ]
    }


def test_fetch_teams(teams_json):
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = teams_json

        teams = _fetch_teams(2023)
        assert len(teams) == 1
        assert teams[0]["abbreviation"] == "LAN"


def test_fetch_manager(roster_json):
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = roster_json

        managers = _fetch_manager(119, 2023)
        assert len(managers) == 1
        assert managers[0]["manager_name"] == "Dave Roberts"
        assert managers[0]["manager_id"] == 123


def test_backfill_team_managers_flow(teams_json, roster_json):
    mock_engine = MagicMock()
    # Mocking the internal fetch functions
    with (
        patch(
            "algomlb.ingestion.managers_ingester._fetch_teams",
            return_value=[{"id": 119, "abbreviation": "LAN", "name": "Dodgers"}],
        ),
        patch(
            "algomlb.ingestion.managers_ingester._fetch_manager",
            return_value=[
                {
                    "manager_id": 123,
                    "manager_name": "Dave Roberts",
                    "jersey_number": "30",
                }
            ],
        ),
        patch(
            "algomlb.ingestion.managers_ingester.get_engine", return_value=mock_engine
        ),
    ):
        # Test single year backfill
        backfill_team_managers(start_year=2023, end_year=2023, engine=mock_engine)

        # Verify DB interaction
        assert mock_engine.begin.called
        # The engine context manager was entered, and then conn.execute(upsert) was called inside.


def test_backfill_team_managers_error_handling():
    mock_engine = MagicMock()
    with (
        patch(
            "algomlb.ingestion.managers_ingester._fetch_teams",
            return_value=[{"id": 119, "abbreviation": "LAN", "name": "Dodgers"}],
        ),
        patch(
            "algomlb.ingestion.managers_ingester._fetch_manager",
            side_effect=Exception("API Error"),
        ),
        patch(
            "algomlb.ingestion.managers_ingester.get_engine", return_value=mock_engine
        ),
    ):
        # Should catch exception and continue (log warning)
        backfill_team_managers(start_year=2023, end_year=2023, engine=mock_engine)
        assert not mock_engine.begin.called


def test_backfill_no_managers_found():
    mock_engine = MagicMock()
    with (
        patch(
            "algomlb.ingestion.managers_ingester._fetch_teams",
            return_value=[{"id": 119, "abbreviation": "LAN", "name": "Dodgers"}],
        ),
        patch("algomlb.ingestion.managers_ingester._fetch_manager", return_value=[]),
        patch(
            "algomlb.ingestion.managers_ingester.get_engine", return_value=mock_engine
        ),
    ):
        # Should log warning about no manager and continue
        backfill_team_managers(start_year=2023, end_year=2023, engine=mock_engine)
        assert not mock_engine.begin.called
