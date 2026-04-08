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
    metrics: list[str] | str,
    league_mean: Optional[float] = None,
    title: Optional[str] = None,
    volatility_metric: Optional[str] = None,
):
    """
    Generates a trend chart with support for multiple lines (EMAs) and volatility bands.
    """
    if df.empty:
        return None

    if isinstance(metrics, str):
        metrics = [metrics]

    # Semantic Labels
    label_map = {
        "roll_avg_pitcher_xwoba": "Standard Rolling xwOBA",
        "roll_avg_batter_xwoba": "Standard Rolling xwOBA",
        "ema_pitcher_xwoba_3g": "Momentum (3-Game EMA)",
        "ema_pitcher_xwoba_7g": "Trend (7-Game EMA)",
        "ema_batter_xwoba_3g": "Momentum (3-Game EMA)",
        "ema_batter_xwoba_7g": "Trend (7-Game EMA)",
        "roll_k_pct": "K% (Rolling)",
        "roll_bb_pct": "BB% (Rolling)",
        "roll_whiff_pct": "Whiff% (Rolling)",
        "roll_barrel_pct": "Barrel% (Rolling)",
    }

    fig = go.Figure()

    # 1. Add Volatility Band (Shaded STD region)
    # This must be added first so it sits behind the lines
    if volatility_metric and volatility_metric in df.columns and len(metrics) == 1:
        main_metric = metrics[0]
        upper = df[main_metric] + df[volatility_metric]
        lower = df[main_metric] - df[volatility_metric]

        fig.add_trace(
            go.Scatter(
                x=pd.concat([df["game_date"], df["game_date"][::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(61, 90, 254, 0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Volatility Band (±1 STD)",
            )
        )

    # 2. Add League Mean Reference Line
    if league_mean is not None:
        fig.add_hline(
            y=league_mean,
            line_dash="dash",
            line_color="rgba(255, 234, 0, 0.4)",
            annotation_text=f"League Mean: {league_mean:.3f}",
            annotation_position="bottom right",
        )

    # 3. Add Metric Lines
    colors = ["#3D5AFE", "#00E5FF", "#76FF03", "#FF1744"]
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue

        y_label = label_map.get(metric, metric.replace("_", " ").title())
        is_main = "ema" not in metric

        fig.add_trace(
            go.Scatter(
                x=df["game_date"],
                y=df[metric],
                mode="lines" if not is_main else "lines+markers",
                line=dict(
                    width=4 if is_main else 2,
                    color=colors[i % len(colors)],
                    dash="solid" if is_main else "dot",
                ),
                name=y_label,
                text=df.apply(
                    lambda row: (
                        f"Date: {row['game_date']}<br>{y_label}: {row[metric]:.3f}"
                    ),
                    axis=1,
                ),
                hoverinfo="text",
            )
        )

    # Highlight points where baseline_quality is NOT 'FULL' (only for main metric)
    cold_starts = df[df["baseline_quality"] != "FULL"]
    if not cold_starts.empty:
        main_val = df.loc[cold_starts.index, metrics[0]]
        fig.add_trace(
            go.Scatter(
                x=cold_starts["game_date"],
                y=main_val,
                mode="markers",
                marker=dict(size=10, symbol="x", color="rgba(255, 23, 68, 0.6)"),
                name="Cold Start Period",
                hoverinfo="skip",
            )
        )

    # Layout
    fig.update_layout(
        template=get_plotly_template(),
        title=title or "Performance Trends",
        xaxis=dict(title="Game Date", showgrid=False),
        yaxis=dict(title="Metric Value", showgrid=True, zeroline=False),
        width=800,
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig
