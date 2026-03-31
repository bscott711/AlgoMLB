import datetime
from unittest.mock import MagicMock, patch

import pytest

from algomlb.domain import Odds
from algomlb.ingestion.historical_odds import HistoricalOddsIngester
from algomlb.db.models import GameResultORM


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.session = MagicMock()
    return repo


@pytest.fixture
def mock_client():
    client = MagicMock()
    return client


@pytest.fixture
def ingester(mock_repo, mock_client):
    return HistoricalOddsIngester(mock_repo, mock_client)


def test_find_game_with_mapping(ingester, mock_repo):
    # Setup
    mock_game = GameResultORM(
        id=1,
        home_team="Oakland Athletics",
        away_team="Boston Red Sox",
        game_date=datetime.date(2024, 4, 1),
    )
    mock_repo.session.query.return_value.filter_by.return_value.first.return_value = (
        mock_game
    )

    # Test mapping: "Oakland A's" should map to "Oakland Athletics"
    game = ingester._find_game(
        "Oakland A's", "Boston Red Sox", datetime.date(2024, 4, 1)
    )

    # Verify
    assert game.id == 1
    mock_repo.session.query.return_value.filter_by.assert_called_with(
        home_team="Oakland Athletics",
        away_team="Boston Red Sox",
        game_date=datetime.date(2024, 4, 1),
    )


def test_create_orm_h2h(ingester):
    data = {
        "outcomes": {"Home Team": {"price": -150}, "Away Team": {"price": 130}},
        "snapshot_at": datetime.datetime(2024, 4, 1, 12, 0),
    }

    orm = ingester._create_orm(
        1, "Home Team", "Away Team", "DraftKings", "h2h", "opening", data
    )

    assert orm.game_id == 1
    assert orm.home_price == -150
    assert orm.away_price == 130
    assert orm.spread is None
    assert orm.total is None


def test_create_orm_spreads(ingester):
    data = {
        "outcomes": {
            "Home Team": {"price": -110, "point": -1.5},
            "Away Team": {"price": -110, "point": 1.5},
        },
        "snapshot_at": datetime.datetime(2024, 4, 1, 12, 0),
    }

    orm = ingester._create_orm(
        1, "Home Team", "Away Team", "DraftKings", "spreads", "opening", data
    )

    assert orm.home_price == -110
    assert orm.away_price == -110
    assert orm.spread == -1.5


def test_create_orm_totals(ingester):
    data = {
        "outcomes": {
            "Over": {"price": -120, "point": 8.5},
            "Under": {"price": 100, "point": 8.5},
        },
        "snapshot_at": datetime.datetime(2024, 4, 1, 12, 0),
    }

    orm = ingester._create_orm(
        1, "Home Team", "Away Team", "DraftKings", "totals", "opening", data
    )

    assert orm.home_price == -120  # Over
    assert orm.away_price == 100  # Under
    assert orm.total == 8.5


@patch("time.sleep", return_value=None)
def test_run_backfill_reverse(mock_sleep, ingester, mock_client):
    start_date = datetime.date(2024, 4, 1)
    end_date = datetime.date(2024, 4, 3)

    # We expect ingest_day_snapshots to be called for 4/3, 4/2, 4/1
    with patch.object(ingester, "ingest_day_snapshots") as mock_ingest:
        ingester.run_backfill(start_date, end_date, reverse=True)

        assert mock_ingest.call_count == 3
        calls = [c[0][0] for c in mock_ingest.call_args_list]
        assert calls == [
            datetime.date(2024, 4, 3),
            datetime.date(2024, 4, 2),
            datetime.date(2024, 4, 1),
        ]


def test_group_raw_odds_with_point(ingester):
    mock_odds = [
        Odds(
            odds_game_id="g1",
            home_team="Home",
            away_team="Away",
            game_date=datetime.date(2024, 4, 1),
            sportsbook="Book",
            market_type="spreads",
            outcome="Home",
            price=-110.0,
            point=-1.5,
            timestamp=datetime.datetime(2024, 4, 1, 10, 0),
        )
    ]

    grouped = ingester._group_raw_odds(mock_odds)
    key = ("Home", "Away", datetime.date(2024, 4, 1))

    assert key in grouped
    assert grouped[key]["Book"]["spreads"]["outcomes"]["Home"]["point"] == -1.5
    assert grouped[key]["Book"]["spreads"]["outcomes"]["Home"]["price"] == -110.0


@patch("time.sleep", return_value=None)
def test_run_backfill_forward(mock_sleep, ingester):
    start_date = datetime.date(2024, 4, 1)
    end_date = datetime.date(2024, 4, 3)

    with patch.object(ingester, "ingest_day_snapshots") as mock_ingest:
        ingester.run_backfill(start_date, end_date, reverse=False)

        assert mock_ingest.call_count == 3
        calls = [c[0][0] for c in mock_ingest.call_args_list]
        assert calls == [
            datetime.date(2024, 4, 1),
            datetime.date(2024, 4, 2),
            datetime.date(2024, 4, 3),
        ]


@patch("time.sleep", return_value=None)
def test_run_backfill_exception(mock_sleep, ingester):
    start_date = datetime.date(2024, 4, 1)
    end_date = datetime.date(2024, 4, 1)

    with patch.object(ingester, "ingest_day_snapshots", side_effect=Exception("Oops")):
        # Should not raise exception but log it (covered by capturing logs if needed, but here simple execution is enough for coverage)
        ingester.run_backfill(start_date, end_date)


def test_process_snapshot_exception(ingester, mock_client):
    # Test exception in _process_snapshot
    mock_client.fetch_historical_odds.side_effect = Exception("API Error")
    # Should not raise
    ingester._process_snapshot("2024-04-01T10:00:00Z", "opening")
