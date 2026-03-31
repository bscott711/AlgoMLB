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
    ["🔥 Pitcher Analytics", "🚀 Batter Power", "🧤 Defensive context"]
)

with tab1:
    st.markdown("### Pitcher Arsenal & Umpire Favor")
    with engine.connect() as conn:
        q_pitch = """
            WITH pitcher_stats AS (
                SELECT 
                    p.pitcher_id,
                    count(*) as pitches,
                    avg(p.release_speed) as avg_velo,
                    avg(p.release_spin_rate) as avg_spin,
                    avg(p.release_extension) as avg_extension,
                    avg(p.pfx_x) * 12 as avg_horiz_break_in,
                    avg(p.pfx_z) * 12 as avg_vert_break_in,
                    count(CASE WHEN p.description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 END)::float / 
                        NULLIF(count(CASE WHEN p.description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play') THEN 1 END), 0) as whiff_rate,
                    count(CASE WHEN p.description = 'called_strike' THEN 1 END)::float / count(*) as called_strike_rate,
                    count(CASE WHEN p.events = 'home_run' THEN 1 END) as hr_allowed
                FROM pitch_events p
                GROUP BY 1
            ),
            ump_favor AS (
                SELECT 
                    p.pitcher_id,
                    avg(u.home_pitcher_impact) as avg_ump_favor
                FROM pitch_events p
                JOIN umpire_scorecards u ON p.game_id = u.game_id
                GROUP BY 1
            )
            SELECT 
                ps.*,
                uf.avg_ump_favor
            FROM pitcher_stats ps
            LEFT JOIN ump_favor uf ON ps.pitcher_id = uf.pitcher_id
            WHERE ps.pitches > 100
            ORDER BY ps.avg_velo DESC
        """
        df_p = pd.read_sql(q_pitch, engine)

    if not df_p.empty:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Top Velocity", f"{df_p['avg_velo'].max():.1f} mph")
        col_m2.metric("Max Spin", f"{df_p['avg_spin'].max():,.0f} rpm")
        col_m3.metric("Avg Whiff %", f"{df_p['whiff_rate'].mean():.1%}")
        col_m4.metric("Avg Ump Favor", f"{df_p['avg_ump_favor'].mean():.2f} runs")

        st.dataframe(
            df_p.style.format(
                {
                    "whiff_rate": "{:.1%}",
                    "called_strike_rate": "{:.1%}",
                    "avg_ump_favor": "{:+.2f}",
                }
            ).background_gradient(
                subset=["avg_velo", "whiff_rate", "avg_ump_favor"], cmap="RdBu_r"
            ),
            use_container_width=True,
        )

        st.markdown("#### Pitch Movement Profile")
        fig_p = px.scatter(
            df_p,
            x="avg_horiz_break_in",
            y="avg_vert_break_in",
            size="pitches",
            hover_name="pitcher_id",
            color="avg_velo",
            template="plotly_dark",
            title="Horizontal vs Vertical Break (Inches)",
            labels={
                "avg_horiz_break_in": "Horiz Break (in)",
                "avg_vert_break_in": "Vert Break (in)",
            },
        )
        st.plotly_chart(fig_p, use_container_width=True)

with tab2:
    st.markdown("### Batter Launch Profile & Discipline")
    with engine.connect() as conn:
        q_bat = """
            SELECT 
                batter_id,
                count(*) as pitches_seen,
                count(DISTINCT game_id) as games,
                avg(launch_speed) as avg_ev,
                max(launch_speed) as max_ev,
                avg(launch_angle) as avg_la,
                count(CASE WHEN launch_speed > 95 THEN 1 END)::float / 
                    NULLIF(count(CASE WHEN launch_speed IS NOT NULL THEN 1 END), 0) as hard_hit_rate,
                count(CASE WHEN events = 'strikeout' THEN 1 END)::float / 
                    NULLIF(count(DISTINCT at_bat_number), 0) as k_rate,
                count(CASE WHEN events = 'walk' THEN 1 END)::float / 
                    NULLIF(count(DISTINCT at_bat_number), 0) as bb_rate
            FROM pitch_events 
            GROUP BY 1 
            HAVING count(*) > 50
            ORDER BY avg_ev DESC
        """
        df_b = pd.read_sql(q_bat, engine)

    if not df_b.empty:
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Max Exit Velo", f"{df_b['max_ev'].max():.1f} mph")
        col_b2.metric("Avg Hard Hit", f"{df_b['hard_hit_rate'].mean():.1%}")
        col_b3.metric("League Avg K%", f"{df_b['k_rate'].mean():.1%}")

        st.dataframe(
            df_b.style.format(
                {"hard_hit_rate": "{:.1%}", "k_rate": "{:.1%}", "bb_rate": "{:.1%}"},
                na_rep="-",
            ).background_gradient(subset=["avg_ev", "hard_hit_rate"], cmap="YlGn"),
            use_container_width=True,
        )

        df_b["max_ev"] = df_b["max_ev"].fillna(0)
        st.markdown("#### Exit Velo vs Launch Angle")
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
        st.plotly_chart(fig_b, use_container_width=True)

with tab3:
    st.markdown("### Hit Distribution & Zone Frequency")
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
        zone_grid = df_z[df_z["zone"] <= 9].copy()
        if not zone_grid.empty:
            zone_grid["row"] = (zone_grid["zone"] - 1) // 3
            zone_grid["col"] = (zone_grid["zone"] - 1) % 3

            fig_z = px.density_heatmap(
                zone_grid,
                x="col",
                y="row",
                z="frequency",
                color_continuous_scale="Viridis",
                labels={"col": "Width", "row": "Height"},
                title="Ball Frequency by Zone Grid",
                template="plotly_dark",
            )
            fig_z.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_z, use_container_width=True)

        st.write("#### Hit Type Distribution")
        with engine.connect() as conn:
            q_type = "SELECT bb_type, count(*) FROM pitch_events WHERE bb_type IS NOT NULL GROUP BY 1"
            df_type = pd.read_sql(q_type, engine)

        fig_pie = px.sunburst(
            df_type, path=["bb_type"], values="count", template="plotly_dark"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

st.success("Analysis Engine: 100% Online")
