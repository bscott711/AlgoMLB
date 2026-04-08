import pytest
import datetime
from unittest.mock import MagicMock, patch
from algomlb.ingestion.lineup_ingester import LineupIngester


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def boxscore_json():
    return {
        "teams": {
            "home": {
                "players": {
                    "ID1": {
                        "battingOrder": "100",
                        "person": {"id": 1, "fullName": "Player 1"},
                        "position": {"abbreviation": "SS"},
                    },
                    "ID2": {
                        "battingOrder": "101",
                        "person": {"id": 2, "fullName": "Sub 1"},
                        "position": {"abbreviation": "PH"},
                    },
                    "ID3": {
                        "battingOrder": "200",
                        "person": {"id": 3, "fullName": "Player 2"},
                        "position": {"abbreviation": "2B"},
                    },
                }
            },
            "away": {
                "players": {
                    "ID4": {
                        "battingOrder": "100",
                        "person": {"id": 4, "fullName": "Player 3"},
                    },
                }
            },
        }
    }


def test_parse_starters(mock_session, boxscore_json):
    ingester = LineupIngester(mock_session)
    game_pk = 123
    game_date = datetime.date(2023, 4, 1)

    records = ingester._parse_starters(boxscore_json, game_pk, game_date)

    # 2 starters for home (100, 200), 1 for away (100). Sub (101) skipped.
    assert len(records) == 3

    home_slots = [r["batting_order"] for r in records if r["team_side"] == "home"]
    assert sorted(home_slots) == [1, 2]

    player1 = next(r for r in records if r["player_id"] == 1)
    assert player1["player_name"] == "Player 1"
    assert player1["position"] == "SS"


def test_ingest_game_success(mock_session, boxscore_json):
    ingester = LineupIngester(mock_session)

    with (
        patch.object(ingester, "_fetch_boxscore", return_value=boxscore_json),
        patch("algomlb.ingestion.lineup_ingester.insert"),
    ):
        count = ingester.ingest_game(123, datetime.date(2023, 4, 1))
        assert count == 3
        assert mock_session.execute.called
        assert mock_session.commit.called


def test_ingest_game_failed_fetch(mock_session):
    ingester = LineupIngester(mock_session)
    with patch.object(ingester, "_fetch_boxscore", return_value=None):
        assert ingester.ingest_game(123, datetime.date(2023, 4, 1)) == 0


def test_backfill_range_logic(mock_session):
    ingester = LineupIngester(mock_session)
    # Mock database returning 2 games to process
    mock_session.execute.return_value.fetchall.return_value = [
        (101, datetime.date(2023, 4, 1)),
        (102, datetime.date(2023, 4, 2)),
    ]

    with patch.object(ingester, "ingest_game", return_value=9) as mock_ingest:
        total = ingester.backfill_range(
            datetime.date(2023, 4, 1), datetime.date(2023, 4, 2), throttle_ms=0
        )
        assert total == 18
        assert mock_ingest.call_count == 2
