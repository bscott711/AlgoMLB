import streamlit as st
import pandas as pd
import plotly.express as px
from algomlb.db.session import get_engine

st.set_page_config(page_title="Batter & Player Analytics", layout="wide")

st.title("⚾ Player Performance Analytics")
st.markdown("---")

engine = get_engine()

# --- Player Type Selector ---
tab1, tab2, tab3 = st.tabs(
    ["🔥 Pitcher Breakout", "🚀 Batter Power", "🧤 Defensive Impact"]
)

with tab1:
    st.markdown("### Pitcher Arsenal & Health")
    with engine.connect() as conn:
        q_pitch = """
            SELECT 
                pitcher_id,
                count(*) as pitches,
                avg(release_speed) as avg_velo,
                avg(release_spin_rate) as avg_spin,
                avg(release_extension) as avg_extension,
                count(DISTINCT pitch_type) as pitch_count,
                avg(effective_speed) as eff_velo
            FROM pitch_events 
            GROUP BY 1 
            HAVING count(*) > 100
            ORDER BY avg_velo DESC
        """
        df_p = pd.read_sql(q_pitch, engine)

    if not df_p.empty:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Top Velocity", f"{df_p['avg_velo'].max():.1f} mph")
        col_m2.metric("Max Spin", f"{df_p['avg_spin'].max():,.0f} rpm")
        col_m3.metric("Avg Extension", f"{df_p['avg_extension'].mean():.2f} ft")

        st.dataframe(
            df_p.style.background_gradient(
                subset=["avg_velo", "avg_spin"], cmap="Oranges"
            ),
            use_container_width=True,
        )

        st.markdown("#### Velo vs. Spin Profile")
        fig_p = px.scatter(
            df_p,
            x="avg_velo",
            y="avg_spin",
            size="pitches",
            hover_name="pitcher_id",
            color="avg_extension",
            template="plotly_dark",
            title="Pitcher Core Metrics (Speed vs Spin)",
        )
        st.plotly_chart(fig_p, width="stretch")

with tab2:
    st.markdown("### Batter Launch Profile")
    with engine.connect() as conn:
        q_bat = """
            SELECT 
                batter_id,
                count(*) as balls_in_play,
                avg(launch_speed) as avg_ev,
                max(launch_speed) as max_ev,
                avg(launch_angle) as avg_la,
                count(CASE WHEN launch_speed > 95 THEN 1 END) / CAST(count(*) as float) as hard_hit_rate
            FROM pitch_events 
            WHERE launch_speed IS NOT NULL
            GROUP BY 1 
            HAVING count(*) > 20
            ORDER BY avg_ev DESC
        """
        df_b = pd.read_sql(q_bat, engine)

    if not df_b.empty:
        st.dataframe(
            df_b.style.format({"hard_hit_rate": "{:.1%}"}).background_gradient(
                subset=["avg_ev", "hard_hit_rate"], cmap="YlGn"
            ),
            width="stretch",
        )

        fig_b = px.scatter(
            df_b,
            x="avg_ev",
            y="avg_la",
            color="hard_hit_rate",
            size="max_ev",
            hover_name="batter_id",
            template="plotly_dark",
            title="Averaged Launch Profiles vs. Hard Hit Rate",
        )
        st.plotly_chart(fig_b, width="stretch")

with tab3:
    st.markdown("### Defensive Workload & Zone Distribution")
    with engine.connect() as conn:
        q_zone = """
            SELECT 
                zone,
                count(*) as frequency,
                avg(launch_speed) as avg_ev_in_zone
            FROM pitch_events 
            WHERE zone IS NOT NULL
            GROUP BY 1 
            ORDER BY 1
        """
        df_z = pd.read_sql(q_zone, engine)

    if not df_z.empty:
        st.markdown("#### Strike Zone Frequency Map")
        # Simple heatmap for zones 1-9 (the standard grid)
        zone_grid = df_z[df_z["zone"] <= 9].copy()
        if not zone_grid.empty:
            # Map zones (1-9) to 3x3 grid
            # 1 2 3
            # 4 5 6
            # 7 8 9
            zone_grid["row"] = (zone_grid["zone"] - 1) // 3
            zone_grid["col"] = (zone_grid["zone"] - 1) % 3

            fig_z = px.density_heatmap(
                zone_grid,
                x="col",
                y="row",
                z="frequency",
                color_continuous_scale="Viridis",
                labels={"col": "Width", "row": "Height"},
                title="Ball Frequency by Zone Grid (Top-Level Defensive Context)",
                template="plotly_dark",
            )
            # Flip y to make 1 2 3 top row
            fig_z.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_z, width="stretch")

        st.write("#### Hit Distribution (Fielding Context)")
        with engine.connect() as conn:
            q_type = "SELECT bb_type, count(*) FROM pitch_events WHERE bb_type IS NOT NULL GROUP BY 1"
            df_type = pd.read_sql(q_type, engine)

        fig_pie = px.sunburst(
            df_type, path=["bb_type"], values="count", template="plotly_dark"
        )
        st.plotly_chart(fig_pie, width="stretch")
