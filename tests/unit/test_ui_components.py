import pytest
import pandas as pd
from algomlb.ui.components.spray_charts import transform_coordinates
from algomlb.ui.styles import COLORS


def test_coordinate_transform():
    """
    Verify the Statcast-to-Field coordinate transform logic using the 2.33x scale.
    Standard:
        angle = atan((hc_x - 125.42) / (198.27 - hc_y))
        dist = scaled_distance or hit_distance_sc
    """
    # Sample hit: hc_x=181.22, hc_y=71.69
    # dx=55.8, dy=126.58 -> angle=0.415 rad
    # distance = approx 138.35 * 2.33 = 322.36 ft
    data = {
        "hc_x": [181.22, 125.42],
        "hc_y": [71.69, 198.27],
        "hit_distance_sc": [None, None],
    }
    df = pd.DataFrame(data)

    df_transformed = transform_coordinates(df)

    # Values scaled by 2.33
    assert df_transformed["field_x"].iloc[0] == pytest.approx(130.0, abs=1.0)
    assert df_transformed["field_y"].iloc[0] == pytest.approx(294.9, abs=1.0)

    # Home Plate (125.42, 198.27) -> (0, 0)
    assert df_transformed["field_x"].iloc[1] == pytest.approx(0.0, abs=0.1)
    assert df_transformed["field_y"].iloc[1] == pytest.approx(0.0, abs=0.1)


def test_distance_sc_priority():
    """
    Ensure 'hit_distance_sc' correctly overrides the calculated distance.
    """
    # Center of the grid (Home Plate) but with a distance of 400ft
    data = {
        "hc_x": [125.42],
        "hc_y": [98.27],  # Straight past center (relative dy=100)
        "hit_distance_sc": [400.0],
    }
    df = pd.DataFrame(data)
    df_transformed = transform_coordinates(df)

    # Even though dy is 100 (which * 2.33 = 233), the 400.0ft should win.
    assert df_transformed["field_y"].iloc[0] == pytest.approx(400.0, abs=0.1)


def test_null_handling_in_transform():
    """
    Ensure the transform drops records with missing coordinates to prevent Plotly crashes.
    """
    data = {"hc_x": [100, None], "hc_y": [100, 100]}
    df = pd.DataFrame(data)
    df_transformed = transform_coordinates(df)

    assert len(df_transformed) == 1
    assert "field_x" in df_transformed.columns


def test_field_shapes_generation():
    """
    Verify the 5-point field shapes generation logic.
    """
    from algomlb.ui.components.spray_charts import get_baseball_field_shapes

    # Passing lf=None to trigger the safe_val default branch (coverage line 66)
    shapes = get_baseball_field_shapes(
        lf=None,
        lc=379,
        cf=390,
        rc=380,
        rf=302,
        h_lf=37.0,
        h_lc=37.0,
        h_cf=17.0,
        h_rc=3.0,
        h_rf=3.0,
    )

    # Should contain infield + 2 foul lines + 4 fence segments
    assert len(shapes) == 7

    # Check that fence segments have higher width for Fenway's Green Monster
    # Segment 0: LF to LC (Green Monster 37ft) -> width 37/4=9.25
    assert shapes[3]["line"]["width"] >= 8.0
    # Segment 3: RC to RF (Standard 3ft) -> width max(2.0, 3/4)=2.0
    assert shapes[6]["line"]["width"] == 2.0


def test_style_tokens():
    """
    Verify design system tokens are present and valid.
    """

    assert "primary" in COLORS
    assert "secondary" in COLORS
    assert "background" in COLORS


def test_style_functions(monkeypatch):
    """
    Verify styling helper functions.
    """
    from algomlb.ui.styles import (
        apply_premium_styles,
        get_plotly_template,
        get_color_scale,
    )

    # Mock st.markdown
    mock_markdown_called = False

    def mock_markdown(*args, **kwargs):
        nonlocal mock_markdown_called
        mock_markdown_called = True

    monkeypatch.setattr("streamlit.markdown", mock_markdown)
    apply_premium_styles()
    assert mock_markdown_called

    assert get_plotly_template() == "plotly_dark"
    assert get_color_scale("hot") == ["#FFEA00", "#FF1744"]
    assert get_color_scale("default") == ["#3D5AFE", "#00E5FF"]
    assert get_color_scale("unknown") == ["#3D5AFE", "#00E5FF"]


def test_plot_spray_chart_integration():
    """
    Verify plot_spray_chart generation for both numeric and categorical color modes.
    """
    from algomlb.ui.components.spray_charts import plot_spray_chart
    import pandas as pd
    import plotly.graph_objects as go

    df = pd.DataFrame(
        {
            "hc_x": [125.42, 100.0, 150.0],
            "hc_y": [198.27, 150.0, 150.0],
            "launch_speed": [100.0, 95.0, 105.0],
            "bb_type": ["fly_ball", "ground_ball", "line_drive"],
            "launch_angle": [25.0, -10.0, 15.0],
            "events": ["home_run", "single", "double"],
        }
    )

    # 1. Test Continuous Coloring (Default)
    fig_cont = plot_spray_chart(df, title="Test Continuous")
    assert isinstance(fig_cont, go.Figure)
    assert fig_cont.layout["title"]["text"] == "Test Continuous"
    # One trace for points, plus shapes added in layout
    assert len(fig_cont.data) == 1

    # 2. Test Categorical Coloring
    fig_cat = plot_spray_chart(df, color_col="events", title="Test Categorical")
    assert isinstance(fig_cat, go.Figure)
    # 3 unique events -> 3 traces
    assert len(fig_cat.data) == 3

    # 3. Test with custom ballpark dims and Nulls
    df_null = df.copy()
    df_null.loc[0, "events"] = None
    # Passing None to ballpark_dims to trigger that default block too
    fig_null = plot_spray_chart(df_null, color_col="events", ballpark_dims=None)
    assert len(fig_null.data) == 2  # home_run skip


def test_transform_coordinates_idempotent():
    """
    Verify transform_coordinates returns early if already transformed.
    """
    from algomlb.ui.components.spray_charts import transform_coordinates
    import pandas as pd

    df = pd.DataFrame({"field_x": [10.0], "field_y": [20.0]})
    df_transformed = transform_coordinates(df)
    assert df_transformed is df
