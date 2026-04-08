import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from algomlb.ui.components.field_equations import (
    arizona_diamondbacks,
    athletics,
    atlanta_braves,
    baltimore_orioles,
    boston_red_sox,
    chicago_cubs,
    chicago_white_sox,
    cincinnati_reds,
    cleveland_guardians,
    colorado_rockies,
    detroit_tigers,
    houston_astros,
    kansas_city_royals,
    los_angeles_angels,
    los_angeles_dodgers,
    miami_marlins,
    milwaukee_brewers,
    minnesota_twins,
    new_york_mets,
    new_york_yankees,
    philadelphia_phillies,
    pittsburgh_pirates,
    san_diego_padres,
    san_francisco_giants,
    seattle_mariners,
    st_louis_cardinals,
    tampa_bay_rays,
    texas_rangers,
    toronto_blue_jays,
    washington_nationals,
    get_stadium_points,
)
from algomlb.ui.components.spray_charts import (
    get_baseball_field_shapes,
    get_fence_at_angle,
    is_simulated_hr,
    transform_coordinates,
    plot_spray_chart,
    get_stadium_dims,
    get_ballpark_selection_ui,
)
from algomlb.ui.styles import get_plotly_template, apply_premium_styles, get_color_scale


def test_field_equations_targeted_buckets():
    """Exercise specific ELIF branches for complex stadium geometry."""
    # Atlanta (146): 79.2 <= theta_deg < 81.3
    assert atlanta_braves(80.0) is not None
    # KC (422): 33.0 <= theta_deg < 42.6
    assert kansas_city_royals(35.0) is not None
    # Yankees (612): 16.2 <= theta_deg < 25.1
    assert new_york_yankees(20.0) is not None
    # Seattle (962): 12.5 <= theta_deg < 14.1
    assert seattle_mariners(13.0) is not None
    # Toronto (1078): 74.5 <= theta_deg < 90.0
    assert toronto_blue_jays(80.0) is not None
    # Washington (1121): 74.1 <= theta_deg < 74.2
    assert washington_nationals(74.15) is not None


def test_field_equations_out_of_bounds():
    """Exercise return None/out-of-bounds paths and fallback points."""
    stadiums = [
        arizona_diamondbacks,
        athletics,
        atlanta_braves,
        baltimore_orioles,
        boston_red_sox,
        chicago_cubs,
        chicago_white_sox,
        cincinnati_reds,
        cleveland_guardians,
        colorado_rockies,
        detroit_tigers,
        houston_astros,
        kansas_city_royals,
        los_angeles_angels,
        los_angeles_dodgers,
        miami_marlins,
        milwaukee_brewers,
        minnesota_twins,
        new_york_mets,
        new_york_yankees,
        philadelphia_phillies,
        pittsburgh_pirates,
        san_diego_padres,
        san_francisco_giants,
        seattle_mariners,
        st_louis_cardinals,
        tampa_bay_rays,
        texas_rangers,
        toronto_blue_jays,
        washington_nationals,
    ]
    for func in stadiums:
        assert func(180) is None

    from algomlb.ui.components.field_equations import _get_fallback_points

    assert _get_fallback_points(None) == []

    with patch.dict(
        "algomlb.ui.components.field_equations.STADIUM_EQUATIONS",
        {"Mock": lambda a: None if a >= 90 else 400.0},
    ):
        pts = get_stadium_points("Mock")
        assert len(pts) == 91

    assert get_stadium_points("Nowhere", fallback_dims=None) == []


def test_spray_charts_edge_cases():
    """Exercise fallback paths and edge cases in spray charts."""
    # 1. safe_val np.nan
    shapes = get_baseball_field_shapes(lf=np.nan, cf=None)
    assert len(shapes) > 0

    # 2. get_fence_at_angle fallback (Line 288)
    # Use 180 to definitively hit the final return 400.0, 8.0 because abs(180) > 90
    d, h = get_fence_at_angle(
        180.0,
        {
            "lf": 330,
            "lc": 375,
            "cf": 400,
            "rc": 375,
            "rf": 330,
            "h_lf": 8,
            "h_lc": 8,
            "h_cf": 8,
            "h_rc": 8,
            "h_rf": 8,
        },
    )
    assert d == 400.0
    assert h == 8.0

    # 3. transform_coordinates already has field_x
    df_existing = pd.DataFrame({"field_x": [0], "field_y": [0]})
    df_result = transform_coordinates(df_existing)
    assert id(df_result) == id(df_existing)

    # 4. hit_distance_sc missing
    df_no_dist = pd.DataFrame({"hc_x": [125], "hc_y": [198]})
    df_result = transform_coordinates(df_no_dist)
    assert "dist_ft" in df_result.columns

    # 5. Simulation & Simulated HR
    ballpark_dims = {
        "lf": 330,
        "lc": 375,
        "cf": 400,
        "rc": 375,
        "rf": 330,
        "h_lf": 8,
        "h_lc": 8,
        "h_cf": 8,
        "h_rc": 8,
        "h_rf": 8,
    }

    assert bool(is_simulated_hr(120, 25, 0, 450, ballpark_dims))
    assert not bool(is_simulated_hr(50, 25, 0, 450, ballpark_dims))
    assert not bool(is_simulated_hr(120, 25, 0, 100, ballpark_dims))

    df_sim = pd.DataFrame(
        {
            "hc_x": [125],
            "hc_y": [100],
            "launch_speed": [120],
            "launch_angle": [25],
            "dist_ft": [450],
            "events": ["field_out"],
            "bb_type": ["fly_ball"],
        }
    )
    fig = plot_spray_chart(df_sim, ballpark_dims=ballpark_dims)
    assert fig is not None

    # 6. Categorical coloring with None
    df_events = pd.DataFrame(
        {
            "hc_x": [125, 125, 125, 125],
            "hc_y": [100, 150, 200, 250],
            "events": ["single", "double", "home_run", None],
            "launch_speed": [90, 100, 110, 80],
            "launch_angle": [10, 20, 30, 5],
            "bb_type": ["line_drive", "double", "fly_ball", "ground_ball"],
        }
    )
    fig2 = plot_spray_chart(df_events, color_col="events")
    assert fig2 is not None


def test_stadium_retrieval_logic():
    """Exercise database retrieval paths for stadium dimensions."""
    mock_engine = MagicMock()

    with patch("pandas.read_sql", return_value=pd.DataFrame()):
        assert get_stadium_dims(mock_engine, ballpark_id=1) is None
        assert get_stadium_dims(mock_engine, ballpark_name="Test") is None
        assert get_stadium_dims(mock_engine) is None

    df_val = pd.DataFrame(
        {
            "ballpark": ["Test Park"],
            "id": [1],
            "left_field": [330],
            "center_field": [400],
            "right_field": [330],
            "lf_wall_height": [8.0],
        }
    )
    with patch("pandas.read_sql", return_value=df_val):
        res = get_stadium_dims(mock_engine, ballpark_name="Test's Park")
        assert res is not None
        assert res["name"] == "Test Park"


def test_stadium_ui_logic():
    """Exercise Streamlit UI logic for ballpark selection."""
    mock_engine = MagicMock()

    def robust_read_sql(query, _engine):
        q = str(query).lower()
        if "from ballparks" in q and "where" not in q:
            return pd.DataFrame(
                {"ballpark": ["Target Park", "Other Park"], "id": [99, 100]}
            )
        elif "where" in q:
            return pd.DataFrame(
                {
                    "ballpark": ["Target Park"],
                    "id": [99],
                    "left_field": [330],
                    "center_field": [400],
                    "right_field": [330],
                    "lf_wall_height": [8.0],
                }
            )
        return pd.DataFrame()

    with (
        patch("streamlit.subheader"),
        patch("streamlit.checkbox", return_value=True),
        patch("streamlit.selectbox", return_value="Target Park"),
        patch("streamlit.cache_data", lambda **kw: lambda f: f),
        patch("algomlb.ui.components.spray_charts.get_stadium_dims") as mock_dims,
        patch("pandas.read_sql", side_effect=robust_read_sql),
    ):
        mock_dims.return_value = {
            "lf": 330,
            "lc": 375,
            "cf": 400,
            "rc": 375,
            "rf": 330,
            "h_lf": 8,
            "h_lc": 8,
            "h_cf": 8,
            "h_rc": 8,
            "h_rf": 8,
            "name": "Target Park",
        }
        res = get_ballpark_selection_ui(mock_engine, native_id=None, key_prefix="test")
        assert res is not None
        assert res["name"] == "Target Park"

    with patch("streamlit.subheader"), patch("streamlit.checkbox", return_value=False):
        res = get_ballpark_selection_ui(mock_engine, native_id=None)
        assert res is not None
        assert res["name"] == "Standard Field"


def test_styles_fallbacks():
    """Exercise fallback paths in styles."""
    assert get_plotly_template() == "plotly_dark"
    scale = get_color_scale("non_existent")
    assert scale == ["#3D5AFE", "#00E5FF"]

    with patch("streamlit.markdown") as mock_st_md:
        apply_premium_styles()
        assert mock_st_md.called
