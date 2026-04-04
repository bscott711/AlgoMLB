import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Optional
from algomlb.ui.styles import get_plotly_template


@st.cache_data(ttl=300)
def load_rolling_features(player_id: int, role: str, _engine) -> pd.DataFrame:
    """
    Loads rolling features for a specific player and role from the Gold layer.
    """
    query = f"""
        SELECT * FROM player_rolling_features 
        WHERE player_id = {player_id} AND role = '{role}'
        ORDER BY game_date ASC
    """
    return pd.read_sql(query, _engine)


def plot_rolling_trend(
    df: pd.DataFrame,
    metric: str,
    league_mean: Optional[float] = None,
    title: Optional[str] = None,
):
    """
    Generates a trend line chart for a rolling metric with a league-mean reference.
    """
    if df.empty:
        return None

    # Semantic Labels
    label_map = {
        "roll_avg_pitcher_xwoba": "Roll xwOBA (Pitcher)",
        "roll_avg_batter_xwoba": "Roll xwOBA (Batter)",
        "roll_k_pct": "K% (Rolling)",
        "roll_bb_pct": "BB% (Rolling)",
        "roll_whiff_pct": "Whiff% (Rolling)",
        "roll_barrel_pct": "Barrel% (Rolling)",
    }
    y_label = label_map.get(metric, metric.replace("_", " ").title())

    fig = go.Figure()

    # Add League Mean Reference Line (Shrinkage Baseline)
    if league_mean is not None:
        fig.add_hline(
            y=league_mean,
            line_dash="dash",
            line_color="rgba(255, 234, 0, 0.4)",
            annotation_text=f"League Mean: {league_mean:.3f}",
            annotation_position="bottom right",
        )

    # Add Player Rolling Trend
    fig.add_trace(
        go.Scatter(
            x=df["game_date"],
            y=df[metric],
            mode="lines+markers",
            line=dict(width=3, color="#3D5AFE"),
            marker=dict(size=6, color="#00E5FF"),
            name="Rolling Average",
            text=df.apply(
                lambda row: (
                    f"Date: {row['game_date']}<br>{y_label}: {row[metric]:.3f}<br>Games: {row['n_games_used']}"
                ),
                axis=1,
            ),
            hoverinfo="text",
        )
    )

    # Shaded confidence/window area (optional if we had SE, but showing 'Partial' vs 'Full' baseline instead)
    # Highlight points where baseline_quality is NOT 'FULL'
    cold_starts = df[df["baseline_quality"] != "FULL"]
    if not cold_starts.empty:
        fig.add_trace(
            go.Scatter(
                x=cold_starts["game_date"],
                y=cold_starts[metric],
                mode="markers",
                marker=dict(size=10, symbol="x", color="rgba(255, 23, 68, 0.6)"),
                name="Cold Start / Partial",
                hoverinfo="skip",
            )
        )

    # Layout
    fig.update_layout(
        template=get_plotly_template(),
        title=title or f"{y_label} Trend",
        xaxis=dict(title="Game Date", showgrid=False),
        yaxis=dict(title=y_label, showgrid=True, zeroline=False),
        width=800,
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig
