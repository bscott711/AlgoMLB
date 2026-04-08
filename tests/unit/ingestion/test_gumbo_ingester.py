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
