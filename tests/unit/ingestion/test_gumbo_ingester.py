import pytest
from unittest.mock import MagicMock, patch
from algomlb.ingestion.gumbo_ingester import GumboIngester


@pytest.fixture
def sample_gumbo_json():
    return {
        "liveData": {
            "plays": {
                "allPlays": [
                    {
                        "about": {"atBatIndex": 0},
                        "playEvents": [
                            {
                                "isPitch": True,
                                "pitchNumber": 1,
                                "playId": "uuid-1",
                                "startTime": "2023-04-01T20:00:00Z",
                                "endTime": "2023-04-01T20:00:05Z",
                            },
                            {"isPitch": False, "playId": "uuid-action"},
                        ],
                    }
                ]
            }
        }
    }


def test_ingest_game_success(sample_gumbo_json):
    mock_session = MagicMock()
    ingester = GumboIngester(mock_session)

    with patch("httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_gumbo_json
        mock_get.return_value = mock_resp

        count = ingester.ingest_game(718000)

        assert count == 1
        assert mock_session.execute.called
        assert mock_session.commit.called


def test_ingest_game_http_error():
    mock_session = MagicMock()
    ingester = GumboIngester(mock_session)

    with patch("httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        count = ingester.ingest_game(718000)
        assert count == 0
        assert not mock_session.commit.called


def test_ingest_game_exception():
    mock_session = MagicMock()
    ingester = GumboIngester(mock_session)
    with patch("httpx.get", side_effect=Exception("Timeout")):
        assert ingester.ingest_game(123) == 0


def test_ingest_game_no_pitches(sample_gumbo_json):
    mock_session = MagicMock()
    ingester = GumboIngester(mock_session)
    # Remove pitches from JSON
    sample_gumbo_json["liveData"]["plays"]["allPlays"][0]["playEvents"] = []
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = sample_gumbo_json
        assert ingester.ingest_game(123) == 0


def test_parse_iso_time_edge_cases():
    ingester = GumboIngester(MagicMock())
    assert ingester._parse_iso_time(None) is None
    assert ingester._parse_iso_time("invalid") is None
    t = ingester._parse_iso_time("2023-04-01T20:00:00Z")
    assert t is not None
    assert t.year == 2023


def test_parse_all_plays_missing_pitch_num():
    ingester = GumboIngester(MagicMock())
    plays = [{"playEvents": [{"isPitch": True, "pitchNumber": None}]}]
    assert ingester._parse_all_plays(123, plays) == []


def test_ingest_games_plural():
    mock_session = MagicMock()
    ingester = GumboIngester(mock_session)
    with patch.object(ingester, "ingest_game", return_value=10) as mock_single:
        total = ingester.ingest_games([1, 2, 3])
        assert total == 30
        assert mock_single.call_count == 3
