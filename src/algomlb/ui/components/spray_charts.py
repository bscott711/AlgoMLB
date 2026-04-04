import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List, Dict
from algomlb.ui.styles import get_plotly_template


def get_baseball_field_shapes(
    lf: float | int | None = 330,
    lc: float | int | None = 375,
    cf: float | int | None = 400,
    rc: float | int | None = 375,
    rf: float | int | None = 330,
    h_lf: float | None = 8.0,
    h_lc: float | None = 8.0,
    h_cf: float | None = 8.0,
    h_rc: float | None = 8.0,
    h_rf: float | None = 8.0,
) -> List[Dict]:
    """
    Returns Plotly shapes to draw a high-fidelity baseball field outline.
    Uses 5 anchor points (LF, LC, CF, RC, RF) and wall heights.
    """
    import numpy as np

    shapes = []

    # Robustly handle None/NaN from DB at the very start
    def safe_val(v, default):
        import pandas as pd

        if v is None or pd.isna(v):
            return default
        return float(v)

    lf = safe_val(lf, 330.0)
    cf = safe_val(cf, 400.0)
    rf = safe_val(rf, 330.0)
    lc = safe_val(lc, (lf + cf) / 1.95)
    rc = safe_val(rc, (rf + cf) / 1.95)

    h_lf = safe_val(h_lf, 8.0)
    h_lc = safe_val(h_lc, 8.0)
    h_cf = safe_val(h_cf, 8.0)
    h_rc = safe_val(h_rc, 8.0)
    h_rf = safe_val(h_rf, 8.0)

    # 1. Infield Diamond (Standard 90ft bases)
    shapes.append(
        dict(
            type="path",
            path="M 0,0 L -63.6,63.6 L 0,127.2 L 63.6,63.6 Z",
            fillcolor="rgba(139, 69, 19, 0.2)",
            line=dict(color="rgba(139, 69, 19, 0.5)", width=1),
        )
    )

    # 2. Foul Lines (Home to LF, Home to RF)
    shapes.append(
        dict(
            type="line",
            x0=0,
            y0=0,
            x1=-lf * 0.707,
            y1=lf * 0.707,
            line=dict(color="white", width=2),
        )
    )
    shapes.append(
        dict(
            type="line",
            x0=0,
            y0=0,
            x1=rf * 0.707,
            y1=rf * 0.707,
            line=dict(color="white", width=2),
        )
    )

    # 3. Outfield Fence (5-Point Spline)
    # Coordinates in feet relative to home plate (0,0)
    def pol_to_cart(dist: float, angle_deg: float):
        rad = np.radians(angle_deg)
        return dist * np.sin(rad), dist * np.cos(rad)

    p_lf = pol_to_cart(lf, -45)
    p_lc = pol_to_cart(lc, -22.5)
    p_cf = (0, cf)
    p_rc = pol_to_cart(rc, 22.5)
    p_rf = pol_to_cart(rf, 45)

    # Draw the fence as a sequence of Quadratic Bezier curves for smoothness
    # Using multiple path segments lets us vary the line thickness per section
    sections = [
        (p_lf, p_lc, h_lf),
        (p_lc, p_cf, h_lc),
        (p_cf, p_rc, h_cf),
        (p_rc, p_rf, h_rc),
    ]

    for p_start, p_end, height in sections:
        # Calculate control point for a smooth arc
        # Midpoint shifted outwards slightly to create a curve
        mid_x = (p_start[0] + p_end[0]) / 1.9
        mid_y = (p_start[1] + p_end[1]) / 1.9

        shapes.append(
            dict(
                type="path",
                path=f"M {p_start[0]},{p_start[1]} Q {mid_x},{mid_y} {p_end[0]},{p_end[1]}",
                line=dict(
                    color="#00E5FF",
                    # Thicker line for higher walls (Max Fenway 37ft -> approx 8px)
                    width=max(2.0, height / 4.0),
                    dash="dash" if height < 10 else "solid",
                ),
            )
        )

    return shapes


def transform_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms Statcast hc_x, hc_y (pixel-ish) to Field Coordinates (feet).
    Standard Statcast Transform:
        field_x = (hc_x - 125.42)
        field_y = (198.27 - hc_y)
    """
    # Guard against already transformed data
    if "field_x" in df.columns:
        return df

    # Standard Statcast Scale: 2.33 feet per unit
    SCALE = 2.33

    # Standard Home Plate Anchor (Statcast 0-250 grid center)
    CENTER_X = 125.42
    CENTER_Y = 198.27

    # Guard against NULLs
    df = df.dropna(subset=["hc_x", "hc_y"]).copy()

    # Calculate components in feet
    # 1. Base spray angle
    import numpy as np

    # Ensure hit_distance_sc exists (even if all Null) to avoid KeyError
    if "hit_distance_sc" not in df.columns:
        df["hit_distance_sc"] = None

    # Use arctan2 for robust handling of dy=0 and signs
    # Statcast y is distance FROM home (positive y is outfield), so dy = CENTER_Y - hc_y
    dx = df["hc_x"] - CENTER_X
    dy = CENTER_Y - df["hc_y"]
    df["angle_rad"] = np.arctan2(dx, dy)

    # 2. Distance: Prioritize 'hit_distance_sc' if available, otherwise use scaled HC distance
    fallback_dist = np.sqrt(dx**2 + dy**2) * SCALE
    df["dist_ft"] = df["hit_distance_sc"].fillna(
        pd.Series(fallback_dist, index=df.index)
    )

    # 3. Final Field Coordinates (relative to home plate at 0,0)
    df["field_x"] = np.sin(df["angle_rad"]) * df["dist_ft"]
    df["field_y"] = np.cos(df["angle_rad"]) * df["dist_ft"]

    return df


def plot_spray_chart(
    df: pd.DataFrame,
    title: str = "Player Spray Chart",
    color_col: str = "launch_speed",
    ballpark_dims: Optional[Dict] = None,
):
    """
    Generates an interactive Plotly spray chart.
    """
    df = transform_coordinates(df)

    # 1. Handle Categorical vs Numeric Coloring
    is_categorical = color_col == "events"

    label_map = {
        "launch_speed": "Exit Velo",
        "launch_angle": "Launch Angle",
        "bb_type": "Hit Type",
        "events": "Outcome",
    }
    display_label = label_map.get(color_col, color_col.replace("_", " ").title())

    fig = go.Figure()

    # Add Field Shapes
    dims = ballpark_dims or {
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
    }
    fig.update_layout(shapes=get_baseball_field_shapes(**dims))

    if is_categorical:
        # Discrete colors for hit outcomes
        event_colors = {
            "home_run": "#FF00E5",  # Neon Pink
            "triple": "#FF8C00",  # Dark Orange
            "double": "#00E5FF",  # Cyan
            "single": "#00FF41",  # Matrix Green
            "field_out": "rgba(255,255,255,0.4)",
            "strikeout": "rgba(255,255,255,0.2)",
            "walk": "rgba(255,255,255,0.6)",
        }

        # Plot each category as a separate trace for the legend
        for event in df[color_col].unique():
            # Skip Nulls
            if pd.isna(event):
                continue

            mask = df[color_col] == event
            fig.add_trace(
                go.Scatter(
                    x=df[mask]["field_x"],
                    y=df[mask]["field_y"],
                    mode="markers",
                    name=event.replace("_", " ").title(),
                    marker=dict(
                        size=8,
                        color=event_colors.get(str(event), "gray"),
                        line=dict(width=1, color="rgba(255,255,255,0.3)"),
                    ),
                    text=df[mask].apply(
                        lambda row: (
                            f"Type: {row['bb_type']}<br>Velo: {row['launch_speed']:.1f}<br>Angle: {row['launch_angle']:.1f}°"
                        ),
                        axis=1,
                    ),
                    hoverinfo="text",
                )
            )
    else:
        # Continuous numeric coloring (Launch Speed, etc.)
        fig.add_trace(
            go.Scatter(
                x=df["field_x"],
                y=df["field_y"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=df[color_col],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=display_label),
                    line=dict(width=1, color="rgba(255,255,255,0.3)"),
                ),
                text=df.apply(
                    lambda row: (
                        f"Type: {row['bb_type']}<br>Velo: {row['launch_speed']:.1f}<br>Angle: {row['launch_angle']:.1f}°"
                    ),
                    axis=1,
                ),
                hoverinfo="text",
                name="Hits",
            )
        )

    # Set Axes & Layout
    fig.update_layout(
        template=get_plotly_template(),
        title=title,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[-250, 250],
            fixedrange=True,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[-50, 500],
            fixedrange=True,
            scaleanchor="x",  # Force 1:1 aspect ratio
            scaleratio=1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        width=700,
        height=600,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig
