import streamlit as st
import pandas as pd
import plotly.express as px
from algomlb.db.session import get_engine

st.set_page_config(page_title="Ballpark Context", layout="wide")

st.title("🏟️ Ballpark Structural & Environmental Factors")
st.markdown("---")

engine = get_engine()

# --- 1. Ballpark Overview ---
with engine.connect() as conn:
    q_ballparks = "SELECT * FROM ballparks ORDER BY team_name ASC"
    df_ballparks = pd.read_sql(q_ballparks, engine)

if not df_ballparks.empty:
    st.subheader("📍 Ballpark Dimensions & Elevation")
    # Specific column ordering to show location first
    cols = [
        "team_name",
        "ballpark",
        "city",
        "state",
        "latitude",
        "longitude",
        "elevation",
        "avg_temp",
        "left_field",
        "center_field",
        "right_field",
        "hr_park_effects",
    ]
    # Filter only available columns to prevent errors if schema is slightly off
    display_cols = [c for c in cols if c in df_ballparks.columns]

    st.dataframe(
        df_ballparks[display_cols]
        .style.format(
            {
                "latitude": "{:.4f}",
                "longitude": "{:.4f}",
                "hr_park_effects": "{:.2f}",
                "avg_temp": "{:.1f}",
                "elevation": "{:,.0f} ft",
            },
            na_rep="N/A",
        )
        .background_gradient(subset=["elevation", "hr_park_effects"], cmap="Spectral"),
        width='stretch',
    )

st.markdown("---")

# --- 2. Structural Analysis ---
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("#### HR Park Effects vs. Elevation")
    df_ballparks["center_field"] = df_ballparks["center_field"].fillna(0)
    fig_hr = px.scatter(
        df_ballparks,
        x="elevation",
        y="hr_park_effects",
        size="center_field",
        hover_name="ballpark",
        color="team_name",
        template="plotly_dark",
        title="High Elevation Impact on HR Propensity",
    )
    st.plotly_chart(fig_hr, width='stretch')

with c_right:
    st.markdown("#### Field Dimensions Comparison")
    # Reshape for melting
    df_melt = df_ballparks.melt(
        id_vars=["ballpark"],
        value_vars=["left_field", "center_field", "right_field"],
        var_name="Field Position",
        value_name="Distance (ft)",
    )
    fig_dist = px.box(
        df_melt,
        x="Field Position",
        y="Distance (ft)",
        color="Field Position",
        points="all",
        template="plotly_dark",
        title="League-wide Outfield Dimensions Distribution",
    )
    st.plotly_chart(fig_dist, width='stretch')

st.markdown("---")

# --- 3. Environmental Analysis (Pitch Velo/Spin vs. Elevation) ---
st.subheader("🏔️ Altitude Impact on Pitch Metrics")
with engine.connect() as conn:
    q_env = """
        SELECT 
            b.ballpark,
            b.elevation,
            avg(p.release_speed) as avg_velo,
            avg(p.release_spin_rate) as avg_spin
        FROM pitch_events p
        JOIN game_results g ON p.game_id = g.game_id
        JOIN ballparks b ON g.ballpark_id = b.id
        GROUP BY 1, 2
        ORDER BY 2 DESC
    """
    try:
        df_env = pd.read_sql(q_env, engine)
        df_env["avg_spin"] = df_env["avg_spin"].fillna(0)
        if not df_env.empty:
            fig_env = px.scatter(
                df_env,
                x="elevation",
                y="avg_velo",
                size="avg_spin",
                hover_name="ballpark",
                color="avg_spin",
                template="plotly_dark",
                title="Elevation vs. Release Velocity & Spin",
                labels={"avg_velo": "Avg Velo (mph)", "avg_spin": "Avg Spin (rpm)"},
            )
            st.plotly_chart(fig_env, width='stretch')
        else:
            st.info(
                "💡 Link `game_results.ballpark_id` to `ballparks.id` to see elevation impact."
            )
    except Exception:
        st.warning("⚠️ Environmental cross-analysis requires linked game IDs.")

st.info("💡 Ballpark structural data sourced from Kaggle mlb-ballparks dataset.")
