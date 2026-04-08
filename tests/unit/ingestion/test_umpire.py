import pytest
from unittest.mock import MagicMock, patch
from algomlb.ingestion.umpire_ingester import UmpireScorecardIngester
from algomlb.db.models import UmpireScorecardORM


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def ingester(mock_session):
    return UmpireScorecardIngester(mock_session, since_year=2023)


def test_safe_helpers(ingester):
    # Float
    assert ingester._safe_float("98.5") == 98.5
    assert ingester._safe_float("ND") == 0.0
    assert ingester._safe_float(None) == 0.0
    assert ingester._safe_float("bad") == 0.0

    # Float or None
    assert ingester._safe_float_or_none("1.2") == 1.2
    assert ingester._safe_float_or_none("ND") is None

    # Int
    assert ingester._safe_int("5") == 5
    assert ingester._safe_int("ND") is None


def test_find_game_id(ingester, mock_session):
    mock_game = MagicMock()
    mock_game.game_id = "101"
    mock_session.query.return_value.filter_by.return_value.first.return_value = (
        mock_game
    )

    assert ingester._find_game_id(101) == "101"

    mock_session.query.return_value.filter_by.return_value.first.return_value = None
    assert ingester._find_game_id(999) is None


def test_api_row_to_orm(ingester):
    row = {
        "game_pk": 123,
        "date": "2023-04-01",
        "umpire": "Joe West",
        "home_team": "ANA",
        "away_team": "HOU",
        "overall_accuracy": "95.0",
        "consistency": "94.0",
        "favor": "0.1",
        "home_score": 5,
        "away_score": 3,
        "abs_away_A": "ND",
    }

    with patch.object(ingester, "_find_game_id", return_value="G123"):
        sc = ingester._api_row_to_orm(row)
        assert isinstance(sc, UmpireScorecardORM)
        assert sc.game_pk == 123
        assert sc.umpire_name == "Joe West"
        assert sc.abs_away_a is None  # ND case
        assert sc.actual_runs == 8.0


def test_api_row_filtering(ingester):
    # Missing PK
    assert ingester._api_row_to_orm({"date": "2023-04-01"}) is None
    # Old Date
    assert ingester._api_row_to_orm({"game_pk": 1, "date": "2020-01-01"}) is None


def test_ingest_from_api(ingester):
    mock_resp = MagicMock()
    # One valid row, one failed row
    mock_resp.json.return_value = {
        "rows": [
            {"game_pk": 123, "date": "2023-04-01", "umpire": "U1"},
            {"failed": True, "game_pk": 124},
        ]
    }

    with (
        patch("httpx.get", return_value=mock_resp),
        patch("time.sleep"),
        patch.object(ingester, "_find_game_id", return_value=None),
        patch.object(ingester.repo, "save_umpire_scorecards") as mock_save,
    ):
        # Test specific year loop
        total = ingester.ingest_from_api(seasons=[2023])
        assert total == 1
        assert mock_save.called


def test_ingest_from_api_default_seasons(ingester):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"rows": []}

    with (
        patch("httpx.get", return_value=mock_resp),
        patch("datetime.datetime") as mock_dt,
    ):
        mock_dt.now.return_value.year = 2023
        # Should call with [2023] (since_year=2023 to current 2023)
        ingester.ingest_from_api()
        assert mock_resp.json.called
