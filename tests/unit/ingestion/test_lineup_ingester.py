import pytest
import datetime
from unittest.mock import MagicMock, patch
from algomlb.ingestion.lineup_ingester import LineupIngester


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def ingester(mock_session):
    return LineupIngester(mock_session)


def test_fetch_boxscore_success(ingester):
    with patch("httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"teams": {}}
        mock_get.return_value = mock_resp

        data = ingester._fetch_boxscore(123)
        assert data == {"teams": {}}
        mock_get.assert_called_with(
            "https://statsapi.mlb.com/api/v1/game/123/boxscore",
            timeout=30.0,
            follow_redirects=True,
        )


def test_fetch_boxscore_fail(ingester):
    with patch("httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        assert ingester._fetch_boxscore(123) is None

        mock_get.side_effect = Exception("network error")
        assert ingester._fetch_boxscore(123) is None


def test_parse_starters(ingester):
    boxscore = {
        "teams": {
            "home": {
                "players": {
                    "ID1": {
                        "battingOrder": "100",
                        "person": {"id": 1, "fullName": "P1"},
                        "position": {"abbreviation": "SS"},
                    },
                    "ID2": {"battingOrder": "101", "person": {"id": 2}},  # Sub
                    "ID3": {
                        "battingOrder": "200",
                        "person": {"id": 3, "fullName": "P3"},
                        "position": {"abbreviation": "2B"},
                    },
                }
            },
            "away": {
                "players": {
                    "ID4": {
                        "battingOrder": "100",
                        "person": {"id": 4, "fullName": "P4"},
                        "position": {"abbreviation": "CF"},
                    },
                }
            },
        }
    }
    game_date = datetime.date(2023, 4, 1)
    records = ingester._parse_starters(boxscore, 123, game_date)

    assert len(records) == 3  # P1, P3, P4
    assert records[0]["player_id"] == 1
    assert records[0]["batting_order"] == 1
    assert records[1]["batting_order"] == 2
    assert records[2]["team_side"] == "away"


def test_ingest_game_success(ingester):
    boxscore = {
        "teams": {
            "home": {
                "players": {
                    "ID1": {
                        "battingOrder": "100",
                        "person": {"id": 1, "fullName": "P1"},
                        "position": {"abbreviation": "SS"},
                    }
                }
            },
            "away": {"players": {}},
        }
    }
    with patch.object(ingester, "_fetch_boxscore", return_value=boxscore):
        n = ingester.ingest_game(123, datetime.date(2023, 4, 1))
        assert n == 1
        assert ingester.session.execute.called
        assert ingester.session.commit.called


def test_ingest_game_empty(ingester):
    with patch.object(ingester, "_fetch_boxscore", return_value=None):
        assert ingester.ingest_game(123, datetime.date(2023, 4, 1)) == 0
    with patch.object(ingester, "_fetch_boxscore", return_value={"teams": {}}):
        assert ingester.ingest_game(123, datetime.date(2023, 4, 1)) == 0


def test_backfill_range(ingester):
    ingester.session.execute.return_value.fetchall.return_value = [
        (123, datetime.date(2023, 4, 1)),
        (124, datetime.date(2023, 4, 2)),
    ]
    with patch.object(ingester, "ingest_game", return_value=9) as mock_ingest:
        with patch("time.sleep"):  # Skip throttling
            total = ingester.backfill_range(
                datetime.date(2023, 4, 1), datetime.date(2023, 4, 2), throttle_ms=0
            )
            assert total == 18
            assert mock_ingest.call_count == 2
