import pandas as pd
from unittest.mock import patch, MagicMock
from algomlb.ui.views.player_health import show_health_analytics
from algomlb.ui.views.data import show_data_health
from algomlb.ui.components.field_equations import get_stadium_points


def test_field_equations_isolated_angles():
    """Target the last 5 missing geometric buckets in field_equations.py."""
    dims_tuple = (330, 375, 400, 375, 330)
    assert len(get_stadium_points("PNC Park", dims_tuple)) > 0
    assert len(get_stadium_points("Petco Park", dims_tuple)) > 0
    assert len(get_stadium_points("Unknown Stadium", dims_tuple)) > 0


def test_data_view_comprehensive():
    """Target both success and failure paths in the modularized data.py."""
    mock_engine = MagicMock()

    def mock_read_sql(query, engine):
        if "game_results" in str(query):
            return pd.DataFrame(
                {"season": [2024], "status": ["completed"], "count": [162]}
            )
        return pd.DataFrame({"season": [2024], "count": [100]})

    # Dynamic column handler for both n (int) and weight lists [1,1]
    def dynamic_columns(n):
        count = len(n) if isinstance(n, list) else n
        return [MagicMock() for _ in range(count)]

    # SUCCESS PATH
    with (
        patch("streamlit.title"),
        patch("streamlit.markdown"),
        patch("streamlit.subheader"),
        patch("streamlit.write"),
        patch("streamlit.error"),
        patch("streamlit.table"),
        patch("streamlit.info"),
        patch("streamlit.warning"),
        patch("streamlit.success"),
        patch("streamlit.plotly_chart"),
        patch("streamlit.columns", side_effect=dynamic_columns),
        patch("pandas.read_sql", side_effect=mock_read_sql),
    ):
        show_data_health(mock_engine)

    # FAILURE PATH
    mock_engine.connect.side_effect = Exception("Database Connection Refused")
    with (
        patch("streamlit.title"),
        patch("streamlit.markdown"),
        patch("streamlit.subheader"),
        patch("streamlit.write"),
        patch("streamlit.error") as mock_st_error,
        patch("streamlit.table"),
        patch("streamlit.columns", side_effect=dynamic_columns),
        patch("pandas.read_sql", side_effect=Exception("SQL Error")),
    ):
        show_data_health(mock_engine)
        assert mock_st_error.called


def test_player_health_comprehensive():
    """Target both success and failure paths in player_health.py."""
    mock_engine = MagicMock()
    df_with_days = pd.DataFrame(
        {
            "player_id": [1],
            "raw_description": ["placed"],
            "days_on_il": [15],
            "transaction_date": ["2024-04-01"],
        }
    )

    def dynamic_columns(n):
        count = len(n) if isinstance(n, list) else n
        return [MagicMock() for _ in range(count)]

    # Patch internal sub-functions and main view
    with (
        patch("algomlb.ui.views.player_health._render_league_trends"),
        patch("algomlb.ui.views.player_health._render_temporal_trends"),
        patch("algomlb.ui.views.player_health._render_gold_metrics"),
        patch("streamlit.title"),
        patch("streamlit.markdown"),
        patch("streamlit.subheader"),
        patch("streamlit.write"),
        patch("streamlit.metric"),
        patch("streamlit.plotly_chart"),
        patch("streamlit.dataframe"),
        patch("streamlit.success") as mock_st_success,
        patch("streamlit.text_input", return_value="605141"),
        patch("streamlit.columns", side_effect=dynamic_columns),
        patch("pandas.read_sql", return_value=df_with_days),
    ):
        show_health_analytics(mock_engine)
        assert mock_st_success.called

    # Exercise the internal sub-functions ALONE with a FULLY populated mock dataframe
    from algomlb.ui.views.player_health import (
        _render_league_trends,
        _render_temporal_trends,
        _render_gold_metrics,
    )

    df_full_render = pd.DataFrame(
        {
            "injury_body_part": ["Arm"],
            "injury_descriptor": ["Soreness"],
            "count": [1],
            "month_name": ["Apr"],
            "month_num": [4],
            "fatigue_index_7d": [10.5],
            "roll_avg_spin_rate": [2500],
            "delta_spin_rate_3g": [10],
            "roll_avg_release_speed": [95],
            "delta_fb_velo_3g": [0.5],
            "roll_avg_release_extension": [6.5],
            "delta_extension_3g": [0.1],
            "status": ["Completed"],
            "season": [2024],
        }
    )

    with (
        patch("streamlit.columns", side_effect=dynamic_columns),
        patch("streamlit.write"),
        patch("streamlit.plotly_chart"),
        patch("streamlit.metric"),
        patch("streamlit.subheader"),
        patch("streamlit.markdown"),
        patch("streamlit.info"),
        patch("pandas.read_sql", return_value=df_full_render),
    ):
        _render_league_trends(mock_engine)
        _render_temporal_trends(mock_engine)
        _render_gold_metrics(mock_engine, "605141", df_with_days)
