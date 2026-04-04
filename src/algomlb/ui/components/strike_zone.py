import plotly.graph_objects as go
import pandas as pd
from algomlb.ui.styles import get_plotly_template


def plot_strike_zone(
    df: pd.DataFrame, title: str = "Pitch Locations", color_col: str = "launch_speed"
):
    """
    Generates an interactive Plotly strike zone scatter plot.
    Coordinates: plate_x (feet left/right), plate_z (feet height).
    """
    # Guard against non-NULL samples and dirt pitches (< 0.5 ft height)
    df = df.dropna(subset=["plate_x", "plate_z"])
    df = df[df["plate_z"] >= 0.5]

    # Defaults for sz_top/sz_bot if not in dataframe
    sz_top = df["sz_top"].mean() if "sz_top" in df.columns else 3.5
    sz_bot = df["sz_bot"].mean() if "sz_bot" in df.columns else 1.5

    fig = go.Figure()

    # Add Strike Zone Box (Standard 17-inch plate = ±0.708 ft)
    fig.add_shape(
        type="rect",
        x0=-0.708,
        y0=sz_bot,
        x1=0.708,
        y1=sz_top,
        line=dict(color="rgba(255, 234, 0, 0.8)", width=3),
        fillcolor="rgba(255,234,0,0.1)",
    )

    # Add Pitch Locations
    fig.add_trace(
        go.Scatter(
            x=df["plate_x"],
            y=df["plate_z"],
            mode="markers",
            marker=dict(
                size=7,
                color=df[color_col],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=color_col.replace("_", " ").title()),
                opacity=0.7,
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            text=df.apply(
                lambda row: (
                    f"Type: {row['pitch_type']}<br>Velo: {row.get('release_speed', 0):.1f}<br>Outcome: {row['description']}"
                ),
                axis=1,
            ),
            hoverinfo="text",
        )
    )

    # Set Axes & Layout
    fig.update_layout(
        template=get_plotly_template(),
        title=title,
        xaxis=dict(
            title="Plate Side (ft)", range=[-2.5, 2.5], zeroline=True, zerolinewidth=2
        ),
        yaxis=dict(title="Plate Height (ft)", range=[0, 5], zeroline=False),
        width=500,
        height=600,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig
