import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from algomlb.ingestion.ballpark_ingester import BallparkIngester


@pytest.fixture
def session():
    return MagicMock()


@pytest.fixture
def ingester(session):
    return BallparkIngester(session)


def test_ingest_from_csv_success(ingester, session, tmp_path):
    # Mock the JSON data
    mock_synonym_map = {
        "chase field": {
            "ballpark": "Chase Field",
            "city": "Phoenix",
            "state": "AZ",
            "lat": 33.4453,
            "long": -112.0667,
            "synonyms": ["Chase Field"],
        }
    }

    csv_file = tmp_path / "ballparks.csv"
    df = pd.DataFrame(
        [
            {
                "team_name": "ARI",
                "ballpark": "Chase Field",
                "left_field": 330,
                "center_field": 413,
                "right_field": 334,
                "min_wall_height": 7.6,
                "max_wall_height": 25.0,
                "hr_park_effects": 99.0,
                "extra_distance": 0.6,
                "avg_temp": 71.3,
                "elevation": 1082,
                "roof": 1.0,
                "daytime": 0.4,
            }
        ]
    )
    df.to_csv(csv_file, index=False)

    with patch.object(BallparkIngester, "_load_geo_map", return_value=mock_synonym_map):
        ingester.ingest_from_csv(str(csv_file))

    # Verify session additions
    orm = session.add.call_args[0][0]
    assert orm.team_name == "ARI"
    assert orm.ballpark == "Chase Field"
    assert orm.city == "Phoenix"
    assert orm.latitude == 33.4453


def test_ingest_from_csv_no_geo_data(ingester, session, tmp_path):
    # Mock the JSON data with a different ballpark
    mock_synonym_map = {
        "truist park": {
            "ballpark": "Truist Park",
            "city": "Atlanta",
            "state": "GA",
            "lat": 33.8907,
            "long": -84.4678,
            "synonyms": ["Truist Park"],
        }
    }

    csv_file = tmp_path / "ballparks.csv"
    df = pd.DataFrame(
        [
            {
                "team_name": "ARI",
                "ballpark": "Unknown Park",
                "left_field": 330,
                "center_field": 413,
                "right_field": 334,
                "min_wall_height": 7.6,
                "max_wall_height": 25.0,
                "hr_park_effects": 99.0,
                "extra_distance": 0.6,
                "avg_temp": 71.3,
                "elevation": 1082,
                "roof": 1.0,
                "daytime": 0.4,
            }
        ]
    )
    df.to_csv(csv_file, index=False)

    with patch.object(BallparkIngester, "_load_geo_map", return_value=mock_synonym_map):
        ingester.ingest_from_csv(str(csv_file))

    # Verify session additions (should still create ORM but with no city)
    orm = session.add.call_args[0][0]
    assert orm.ballpark == "Unknown Park"
    assert orm.city is None
