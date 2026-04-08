import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List, Dict
from algomlb.ui.styles import get_plotly_template

pd.set_option("future.no_silent_downcasting", True)


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
    **kwargs,
) -> List[Dict]:
    """
    Returns Plotly shapes to draw a professional, high-fidelity 2019-2026 baseball field.
    """
    import numpy as np

    shapes = []

    def safe_val(v, default):
        if v is None or (isinstance(v, float) and np.isnan(v)):
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

    C_DARK_GREEN = "#2C4C34"
    C_LIGHT_GREEN = "#3A6142"
    C_DIRT = "#8B5A2B"
    C_WHITE = "#FFFFFF"
    C_FOUL_TERRITORY = "#15261A"
    C_FENCE = "#00E5FF"

    MOUND_Y = 60.5
    TRACK_W = 15.0

    def pol_to_cart(dist: float, angle_deg: float):
        rad = np.radians(angle_deg)
        return dist * np.sin(rad), dist * np.cos(rad)

    from algomlb.ui.components.field_equations import get_stadium_points

    stadium_name = kwargs.get("name")
    raw_points = get_stadium_points(stadium_name, fallback_dims=(lf, lc, cf, rc, rf))

    p_f = [pol_to_cart(r, theta) for r, theta in raw_points]
    p_t = [pol_to_cart(max(r - TRACK_W, 0), theta) for r, theta in raw_points]

    shapes.append(
        dict(
            type="rect",
            x0=-500,
            y0=-500,
            x1=500,
            y1=500,
            fillcolor=C_FOUL_TERRITORY,
            line=dict(width=0),
            layer="below",
        )
    )

    fair_path = f"M 0,0 L {p_f[0][0]},{p_f[0][1]}"
    for i in range(1, len(p_f)):
        fair_path += f" L {p_f[i][0]},{p_f[i][1]}"
    fair_path += " Z"
    shapes.append(
        dict(
            type="path",
            path=fair_path,
            fillcolor=C_DARK_GREEN,
            line=dict(width=0),
            layer="below",
        )
    )

    track_path = f"M {p_t[0][0]},{p_t[0][1]}"
    for i in range(1, len(p_t)):
        track_path += f" L {p_t[i][0]},{p_t[i][1]}"
    track_path += f" L {p_f[-1][0]},{p_f[-1][1]}"
    for i in range(len(p_f) - 1, -1, -1):
        track_path += f" L {p_f[i][0]},{p_f[i][1]}"
    track_path += " Z"
    shapes.append(
        dict(
            type="path",
            path=track_path,
            fillcolor=C_DIRT,
            line=dict(width=0),
            layer="below",
        )
    )

    # --- 5. Infield Geometry (Professional Hub Restoration) ---
    # Centered at Pitcher's Mound (0, 60.5) for absolute broadcast-quality rounding.
    def get_mound_arc_pts(radius, start_ang_deg, end_ang_deg, num=91):
        pts = []
        for i in range(num):
            a_deg = start_ang_deg + (end_ang_deg - start_ang_deg) * i / (num - 1)
            rad = np.radians(a_deg)
            # x = r*sin(a), y = 60.5 + r*cos(a) [In spray angle space]
            pts.append((radius * np.sin(rad), MOUND_Y + radius * np.cos(rad)))
        return pts

    # 5a. Total Infield Dirt Skin (Layer 1: The 'fielder area' arc)
    # Radius 145ft from mound. This is a massive sector from Home (0,0).
    # Angle from mound to foul lines (127, 127) is ~62.3 deg.
    # To avoid crossing through the mound, we order the points: Home -> Left Foul -> Arc -> Right Foul -> Home.
    skin_pts = get_mound_arc_pts(145, -62.3, 62.3)

    # Left Foul Side (Negative X) / Right Foul Side (Positive X)
    # Using sort to ensure correct orientation regardless of point calculation order
    p_f_sorted = sorted([p_f[0], p_f[-1]], key=lambda p: p[0])
    p_left_foul = p_f_sorted[0]
    p_right_foul = p_f_sorted[-1]

    skin_path = "M 0,0"  # Home
    skin_path += f" L {p_left_foul[0]},{p_left_foul[1]}"  # Left Foul
    for x, y in skin_pts:
        skin_path += f" L {x},{y}"
    skin_path += f" L {p_right_foul[0]},{p_right_foul[1]}"  # Right Foul
    skin_path += " Z"

    shapes.append(
        dict(
            type="path",
            path=skin_path,
            fillcolor=C_DIRT,
            line=dict(width=0),
            layer="below",
        )
    )

    # 5b. Unified Infield Grass (Layer 2: The shrunken hub)
    # This uses the 'Double-Parallel Diamond ARC' model:
    # 1. Edges parallel to baselines (y=x+15 and y=-x+15) for 'straight box' dirt paths (width ~10.6ft).
    # 2. Shrunken Radius (55ft from mound) to ensure 2nd base (at y=127.3) is in the dirt (11.8ft clear).
    # Intersections occur at x ~ 54.3, y ~ 69.3 (Angle ~80.8 deg from mound).
    hub_pts = get_mound_arc_pts(55, -80.8, 80.8)

    hub_path = "M 0,15"  # Catcher's dirt cutout
    hub_path += " L -54.3,69.3"  # Segment parallel to Home-3rd (Left)
    for x, y in hub_pts:
        hub_path += f" L {x},{y}"  # Arc from Left to Right
    hub_path += " L 54.3,69.3"  # Segment parallel to Home-1st (Right)
    hub_path += " Z"

    shapes.append(
        dict(
            type="path",
            path=hub_path,
            fillcolor=C_LIGHT_GREEN,
            line=dict(color="rgba(255,255,255,0.22)", width=1),
            layer="below",
        )
    )

    shapes.append(
        dict(
            type="circle",
            x0=-9,
            y0=MOUND_Y - 9,
            x1=9,
            y1=MOUND_Y + 9,
            fillcolor=C_DIRT,
            line=dict(width=0),
            layer="below",
        )
    )

    for bx, by in [(63.6, 63.6), (0.0, 127.3), (-63.6, 63.6)]:
        shapes.append(
            dict(
                type="rect",
                x0=bx - 1.25,
                y0=by - 1.25,
                x1=bx + 1.25,
                y1=by + 1.25,
                fillcolor=C_WHITE,
                line=dict(width=0),
                layer="below",
            )
        )

    shapes.append(
        dict(
            type="rect",
            x0=-2.0,
            y0=MOUND_Y - 0.5,
            x1=2.0,
            y1=MOUND_Y + 0.5,
            fillcolor=C_WHITE,
            line=dict(width=0),
            layer="below",
        )
    )
    shapes.append(
        dict(
            type="path",
            path="M -0.8,0 L 0.8,0 L 1.0,-1.0 L 0,-1.5 L -1.0,-1.0 Z",
            fillcolor=C_WHITE,
            line=dict(width=0),
            layer="below",
        )
    )

    for px, py in [p_f[0], p_f[-1]]:
        shapes.append(
            dict(
                type="line",
                x0=0,
                y0=0,
                x1=px,
                y1=py,
                line=dict(color=C_WHITE, width=2),
                layer="below",
            )
        )

    for i in range(0, len(p_f) - 1, 2):
        f_s, f_e = p_f[i], p_f[i + 1]
        theta = raw_points[i][1]
        from algomlb.ui.components.spray_charts import get_fence_at_angle

        dims_local = {
            "lf": lf,
            "lc": lc,
            "cf": cf,
            "rc": rc,
            "rf": rf,
            "h_lf": h_lf,
            "h_lc": h_lc,
            "h_cf": h_cf,
            "h_rc": h_rc,
            "h_rf": h_rf,
        }
        _, h = get_fence_at_angle(theta, dims_local)
        shapes.append(
            dict(
                type="path",
                path=f"M {f_s[0]},{f_s[1]} L {f_e[0]},{f_e[1]}",
                line=dict(
                    color=C_FENCE,
                    width=max(2.5, h / 3.0),
                    dash="dash" if h < 9 else "solid",
                ),
                layer="below",
            )
        )

    return shapes


def get_fence_at_angle(angle_deg: float, dims: Dict) -> tuple[float, float]:
    import numpy as np

    anchors = [
        (-45.0, float(dims["lf"]), float(dims["h_lf"])),
        (-22.5, float(dims["lc"]), float(dims["h_lc"])),
        (0.0, float(dims["cf"]), float(dims["h_cf"])),
        (22.5, float(dims["rc"]), float(dims["h_rc"])),
        (45.0, float(dims["rf"]), float(dims["h_rf"])),
    ]
    angle_deg = np.clip(angle_deg, -45.0, 45.0)
    for i in range(len(anchors) - 1):
        a1, a2 = anchors[i], anchors[i + 1]
        if a1[0] <= angle_deg <= a2[0]:
            t = (angle_deg - a1[0]) / (a2[0] - a1[0])
            return a1[1] + t * (a2[1] - a1[1]), a1[2] + t * (a2[2] - a1[2])
    return 400.0, 8.0


def is_simulated_hr(
    launch_speed_mph: float,
    launch_angle_deg: float,
    spray_angle_deg: float,
    dist_ft: float,
    dims: Dict,
) -> bool:
    import numpy as np

    fence_dist, wall_height = get_fence_at_angle(spray_angle_deg, dims)
    if dist_ft < fence_dist:
        return False
    v0 = launch_speed_mph * 1.467
    theta = np.radians(launch_angle_deg)
    g_eff = 32.2 * 1.5
    x = fence_dist
    ball_height_at_wall = np.tan(theta) * x - (g_eff * x**2) / (
        2 * (v0 * np.cos(theta)) ** 2
    )
    return ball_height_at_wall > wall_height


def transform_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    if "field_x" in df.columns:
        return df
    SCALE = 2.33
    CENTER_X = 125.42
    CENTER_Y = 198.27
    df = df.dropna(subset=["hc_x", "hc_y"]).copy()
    import numpy as np

    if "hit_distance_sc" not in df.columns:
        df["hit_distance_sc"] = None
    dx = df["hc_x"] - CENTER_X
    dy = CENTER_Y - df["hc_y"]
    df["angle_rad"] = np.arctan2(dx, dy)
    fallback_dist = np.sqrt(dx**2 + dy**2) * SCALE
    df["dist_ft"] = df["hit_distance_sc"].fillna(
        pd.Series(fallback_dist, index=df.index)
    )
    df["field_x"] = np.sin(df["angle_rad"]) * df["dist_ft"]
    df["field_y"] = np.cos(df["angle_rad"]) * df["dist_ft"]
    return df


def plot_spray_chart(
    df: pd.DataFrame,
    title: str = "Player Spray Chart",
    color_col: str = "launch_speed",
    ballpark_dims: Optional[Dict] = None,
):
    import numpy as np

    df = transform_coordinates(df)
    if ballpark_dims:
        df["is_sim_hr"] = df.apply(
            lambda r: is_simulated_hr(
                float(r["launch_speed"]),
                float(r["launch_angle"]),
                np.degrees(r["angle_rad"]),
                float(r["dist_ft"]),
                ballpark_dims,
            ),
            axis=1,
        )
    else:
        df["is_sim_hr"] = False
    is_categorical = color_col == "events"
    label_map = {
        "launch_speed": "Exit Velo",
        "launch_angle": "Launch Angle",
        "bb_type": "Hit Type",
        "events": "Outcome",
    }
    display_label = label_map.get(color_col, color_col.replace("_", " ").title())
    fig = go.Figure()
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
        event_colors = {
            "home_run": "#FF00E5",
            "triple": "#FF8C00",
            "double": "#00E5FF",
            "single": "#00FF41",
            "field_out": "rgba(255,255,255,0.4)",
            "strikeout": "rgba(255,255,255,0.2)",
            "walk": "rgba(255,255,255,0.6)",
        }
        for event in df[color_col].unique():
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
                        line=dict(
                            width=df[mask]["is_sim_hr"].map(lambda x: 2 if x else 1),
                            color=df[mask]["is_sim_hr"].map(
                                lambda x: "#00E5FF" if x else "rgba(255,255,255,0.3)"
                            ),
                        ),
                    ),
                    text=df[mask].apply(
                        lambda row: (
                            f"Type: {row['bb_type']}<br>Velo: {row['launch_speed']:.1f}<br>Angle: {row['launch_angle']:.1f}°{'<br><b>✨ SIMULATED HR</b>' if row['is_sim_hr'] else ''}"
                        ),
                        axis=1,
                    ),
                    hoverinfo="text",
                )
            )
    else:
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
                    line=dict(
                        width=df["is_sim_hr"].map(lambda x: 2 if x else 1),
                        color=df["is_sim_hr"].map(
                            lambda x: "#FF00E5" if x else "rgba(255,255,255,0.3)"
                        ),
                    ),
                ),
                text=df.apply(
                    lambda row: (
                        f"Type: {row['bb_type']}<br>Velo: {row['launch_speed']:.1f}<br>Angle: {row['launch_angle']:.1f}°{'<br><b>✨ SIMULATED HR</b>' if row['is_sim_hr'] else ''}"
                    ),
                    axis=1,
                ),
                hoverinfo="text",
                name="Hits",
            )
        )
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
            scaleanchor="x",
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


def get_stadium_dims(
    engine, ballpark_id: int | None = None, ballpark_name: str | None = None
) -> dict | None:
    """
    Unified helper to fetch ballpark dimensions from the database.
    Supports fetching by 'id' or 'ballpark' (name). Escapes quotes for safety.
    """
    import pandas as pd

    where_clause = ""
    if ballpark_id is not None:
        where_clause = f"WHERE id = {int(ballpark_id)}"
    elif ballpark_name is not None:
        safe_name = str(ballpark_name).replace("'", "''")
        where_clause = f"WHERE ballpark = '{safe_name}'"
    else:
        return None

    query = f"SELECT * FROM ballparks {where_clause} LIMIT 1"
    df = pd.read_sql(query, engine)

    if df.empty:
        return None

    row = df.iloc[0]
    return {
        "lf": float(row.get("left_field", 330)),
        "lc": float(row.get("left_center", 375)),
        "cf": float(row.get("center_field", 400)),
        "rc": float(row.get("right_center", 375)),
        "rf": float(row.get("right_field", 330)),
        "h_lf": float(row.get("lf_wall_height", 8.0)),
        "h_lc": float(row.get("lc_wall_height", 8.0)),
        "h_cf": float(row.get("cf_wall_height", 8.0)),
        "h_rc": float(row.get("rc_wall_height", 8.0)),
        "h_rf": float(row.get("rf_wall_height", 8.0)),
        "name": row.get("ballpark"),
    }


def get_ballpark_selection_ui(
    engine, native_id: int | None = None, key_prefix: str = ""
):
    """
    SOLID HELPER: Centrally manages the Streamlit UI and logic for ballpark selection.
    Encapsulates the 'Stadium Simulator' experience and ensures consistent geometry.
    """
    import streamlit as st
    import pandas as pd

    # 1. Establish native/default baseline
    native_dims = None
    if native_id:
        native_dims = get_stadium_dims(engine, ballpark_id=native_id)

    # 2. Render Sidebar UI
    st.subheader("🧪 Stadium Simulator")
    simulate = st.checkbox(
        "Swap Ballpark Fences", value=False, key=f"{key_prefix}_sim_chk"
    )

    if simulate:
        # Fetch list of ballparks for selection
        @st.cache_data(ttl=3600)
        def _get_all_ballparks(_engine):
            return pd.read_sql("SELECT ballpark, id FROM ballparks", _engine)

        all_bp_df = _get_all_ballparks(engine)
        target_ballpark = st.selectbox(
            "Target Ballpark",
            all_bp_df["ballpark"].sort_values().unique(),
            key=f"{key_prefix}_target_bp",
        )
        selected_bp_id = all_bp_df[all_bp_df["ballpark"] == target_ballpark].iloc[0][
            "id"
        ]
        return get_stadium_dims(engine, ballpark_id=int(selected_bp_id))
    else:
        # Fallback to native or standard standard
        return native_dims or {
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
            "name": "Standard Field",
        }
