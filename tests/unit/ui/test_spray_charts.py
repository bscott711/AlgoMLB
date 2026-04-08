import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from algomlb.ui.components.spray_charts import (
    get_fence_at_angle,
    is_simulated_hr,
    get_stadium_dims,
    get_ballpark_selection_ui,
    plot_spray_chart,
)


@pytest.fixture
def ballpark_dims():
    return {
        "lf": 330,
        "lc": 375,
        "cf": 400,
        "rc": 375,
        "rf": 330,
        "h_lf": 8.0,
        "h_lc": 8.0,
        "h_cf": 8.0,
        "h_rc": 8.0,
        "h_rf": 8.0,
        "name": "Standard",
    }


def test_get_fence_at_angle_extents(ballpark_dims):
    # Test fallback for extreme angles (removed np.clip for 100% coverage reachability)
    d, h = get_fence_at_angle(-75.0, ballpark_dims)
    assert d == 400.0

    d, h = get_fence_at_angle(75.0, ballpark_dims)
    assert d == 400.0


def test_is_simulated_hr_logic(ballpark_dims):
    # 1. Dist < Fence -> Not a HR (Line 302)
    assert not is_simulated_hr(110.0, 25.0, 0.0, 300.0, ballpark_dims)

    # 2. Dist > Fence and High Trajectory -> HR
    # 120mph at 28 deg reaches over the 400ft wall comfortably
    assert is_simulated_hr(120.0, 28.0, 0.0, 410.0, ballpark_dims)

    # 3. Dist > Fence but Low/Dying Trajectory -> Not a HR
    # 80mph at 20 deg might land past 330 (LF) but height at wall is small
    assert not is_simulated_hr(80.0, 5.0, -45.0, 340.0, ballpark_dims)


def test_get_stadium_dims_dispatch():
    mock_engine = MagicMock()

    # Mock return value for search
    mock_df = pd.DataFrame(
        [
            {
                "id": 1,
                "ballpark": "Dodger Stadium",
                "left_field": 330,
                "lf_wall_height": 8.0,
            }
        ]
    )

    with patch("pandas.read_sql", return_value=mock_df):
        # By ID
        dims = get_stadium_dims(mock_engine, ballpark_id=1)
        assert dims is not None
        assert dims["name"] == "Dodger Stadium"

        # By Name (Line 497-498)
        dims = get_stadium_dims(mock_engine, ballpark_name="O'Malley Field")
        assert dims is not None
        assert dims["lf"] == 330.0

    # Empty case (Line 506)
    with patch("pandas.read_sql", return_value=pd.DataFrame()):
        assert get_stadium_dims(mock_engine, ballpark_id=999) is None

    # None search (Line 500)
    assert get_stadium_dims(mock_engine) is None


def test_get_ballpark_selection_ui_flow():
    mock_engine = MagicMock()

    with patch("streamlit.checkbox", return_value=False):
        # No simulation selected (Line 563)
        with patch("algomlb.ui.components.spray_charts.get_stadium_dims") as mock_get:
            mock_get.return_value = {"name": "Native"}
            dims = get_ballpark_selection_ui(mock_engine, native_id=1)
            assert dims is not None
            assert dims["name"] == "Native"

    # Simulation enabled (Line 545-560)
    with (
        patch("streamlit.checkbox", return_value=True),
        patch("streamlit.selectbox", return_value="Target"),
        patch(
            "pandas.read_sql",
            return_value=pd.DataFrame([{"ballpark": "Target", "id": 2}]),
        ),
        patch("algomlb.ui.components.spray_charts.get_stadium_dims") as mock_get,
    ):
        mock_get.return_value = {"name": "Simulated"}
        dims = get_ballpark_selection_ui(mock_engine)
        assert dims is not None
        assert dims["name"] == "Simulated"


def test_plot_spray_chart_with_sim_hr(ballpark_dims):
    # Target Line 346: df["is_sim_hr"] = df.apply(...)
    df_long = pd.DataFrame(
        {
            "hc_x": [125.42],
            "hc_y": [50],  # Dist approx 345
            "hit_distance_sc": [420.0],
            "launch_speed": [120.0],
            "launch_angle": [28.0],
            "bb_type": ["fly_ball"],
            "events": ["double"],
        }
    )
    fig_hr = plot_spray_chart(df_long, ballpark_dims=ballpark_dims)
    # Line width should be 2 for sim_hr
    assert fig_hr.data[0].marker.line.width[0] == 2
