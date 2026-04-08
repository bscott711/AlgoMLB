import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from algomlb.ml.decoupler_pipeline import load_decoupler_dataset, run_decoupler_pipeline


@pytest.fixture
def mock_session_factory():
    factory = MagicMock()
    session = factory.return_value.__enter__.return_value
    session.bind = MagicMock()
    return factory


def test_load_decoupler_dataset(mock_session_factory):
    # Mock data returned by SQL
    mock_df = pd.DataFrame([{"game_pk": 1, "batter": 101, "pitcher": 202}])

    with patch("pandas.read_sql", return_value=mock_df):
        df = load_decoupler_dataset(mock_session_factory, 2023, 2023)
        assert len(df) == 1
        # Check renaming (batter -> batter_id)
        assert "batter_id" in df.columns
        assert df["batter_id"].iloc[0] == 101


def test_run_decoupler_pipeline_full_flow(mock_session_factory):
    # Mocking the decoupler class and dataset loader
    mock_coords = pd.DataFrame(
        [
            {
                "id": 1,
                "hp_lat": 34.0,
                "hp_lon": -118.0,
                "pm_lat": 34.1,
                "pm_lon": -118.1,
                "hp_bearing_deg": 0.0,
            }
        ]
    )

    with (
        patch(
            "algomlb.ml.decoupler_pipeline.get_session_factory",
            return_value=mock_session_factory,
        ),
        patch(
            "algomlb.ml.decoupler_pipeline.BattedBallFlightDecoupler"
        ) as mock_dec_cls,
        patch("algomlb.ml.decoupler_pipeline.load_decoupler_dataset") as mock_load,
        patch("pandas.read_sql", return_value=mock_coords),
    ):
        mock_decoupler = mock_dec_cls.return_value
        # Mocking returns for preprocess/decouple
        input_df = pd.DataFrame(
            [
                {
                    "game_pk": 1,
                    "batter_id": 101,
                    "pitcher_id": 202,
                    "venue_id": 1,
                    "stand": "R",
                    "launch_speed": 100.0,
                    "launch_angle": 10.0,
                    "hc_x": 125.0,
                    "hc_y": 125.0,
                    "hit_distance_sc": 300.0,
                    "bb_type": "line_drive",
                    "events": "single",
                    "game_date": "2023-01-01",
                    "temperature_f": 70,
                    "pressure_hpa": 1013,
                    "relative_humidity": 50,
                    "wind_speed_mph": 5,
                    "wind_direction_deg": 180,
                    "precipitation_mm_hr": 0,
                }
            ]
        )
        mock_load.return_value = input_df

        # We need more columns for the backfill/ORM construction
        calc_df = input_df.copy()
        for col in [
            "spray_angle",
            "is_rhb",
            "cf_bearing_deg",
            "air_density_ratio",
            "tailwind_component",
            "baseline_distance",
            "total_delta",
            "delta_density",
            "delta_wind",
            "delta_precip",
            "environmental_factor",
            "spin_contact_factor",
        ]:
            calc_df[col] = 0.0
        calc_df["is_rhb"] = 1

        mock_decoupler.preprocess.return_value = calc_df
        mock_decoupler.decouple.return_value = calc_df

        # Test "full" action
        run_decoupler_pipeline("full")

        assert mock_decoupler.train_baseline.called
        assert mock_decoupler.calibrate.called
        assert mock_decoupler.save.called
        assert mock_decoupler.decouple.called

        # Verify TRUNCATE and batch insert
        session = mock_session_factory.return_value.__enter__.return_value
        assert session.execute.called
        assert session.commit.called


def test_run_decoupler_pipeline_calibrate_only(mock_session_factory):
    with (
        patch(
            "algomlb.ml.decoupler_pipeline.get_session_factory",
            return_value=mock_session_factory,
        ),
        patch(
            "algomlb.ml.decoupler_pipeline.BattedBallFlightDecoupler"
        ) as mock_dec_cls,
        patch("algomlb.ml.decoupler_pipeline.load_decoupler_dataset") as mock_load,
        patch("pandas.read_sql", return_value=pd.DataFrame([{"id": 1}])),
    ):
        mock_decoupler = mock_dec_cls.return_value
        mock_load.return_value = pd.DataFrame([])

        run_decoupler_pipeline("calibrate")

        assert mock_decoupler.load.called
        assert mock_decoupler.calibrate.called
        assert not mock_decoupler.train_baseline.called
