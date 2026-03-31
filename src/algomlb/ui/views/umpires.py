import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from algomlb.db.session import get_engine

st.set_page_config(page_title="Umpire Analytics", layout="wide")

st.title("👨‍⚖️ Umpire Accuracy & Bias Analytics")
st.markdown("---")

engine = get_engine()

# --- 1. Top Level Metrics ---
with engine.connect() as conn:
    q_metrics = """
        SELECT 
            avg(accuracy) as avg_acc,
            avg(consistency) as avg_con,
            count(*) as total_cards,
            sum(abs(favoritism_home)) as total_favor_impact
        FROM umpire_scorecards
    """
    metrics = conn.execute(text(q_metrics)).fetchone()

if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("League Avg Accuracy", f"{metrics.avg_acc / 100:.1%}")
    c2.metric("League Avg Consistency", f"{metrics.avg_con / 100:.1%}")
    c3.metric("Total Scorecards", f"{metrics.total_cards:,}")
    c4.metric("Bias Run Impact", f"{metrics.total_favor_impact:.1f} runs")

st.markdown("---")

# --- 2. Umpire Performance Table ---
st.subheader("📊 Umpire Performance Leaderboard")
with engine.connect() as conn:
    q_leaderboard = """
        SELECT 
            umpire_name,
            count(*) as games,
            avg(accuracy) as avg_accuracy,
            avg(consistency) as avg_consistency,
            avg(favoritism_home) as avg_favor_home,
            avg(total_run_impact) as avg_run_impact,
            sum(n_overturned) as total_overturned
        FROM umpire_scorecards
        GROUP BY 1
        HAVING count(*) >= 5
        ORDER BY avg_accuracy DESC
    """
    df_umps = pd.read_sql(q_leaderboard, engine)

if not df_umps.empty:
    df_umps["avg_accuracy"] /= 100
    df_umps["avg_consistency"] /= 100
    st.dataframe(
        df_umps.style.format(
            {
                "avg_accuracy": "{:.1%}",
                "avg_consistency": "{:.1%}",
                "avg_favor_home": "{:+.2f}",
                "avg_run_impact": "{:.2f}",
            },
            na_rep="-",
        ).background_gradient(
            subset=["avg_accuracy", "avg_consistency"], cmap="RdYlGn"
        ),
        use_container_width=True,
    )

st.markdown("---")

# --- 3. Visual Analysis ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Accuracy vs. Consistency")
    fig_acc = px.scatter(
        df_umps,
        x="avg_accuracy",
        y="avg_consistency",
        size="games",
        hover_name="umpire_name",
        color="avg_run_impact",
        template="plotly_dark",
        title="Umpire Reliability Matrix",
    )
    st.plotly_chart(fig_acc, use_container_width=True)

with col_right:
    st.markdown("#### Home Favoritism Bias")
    # Show top 20 by absolute favor
    df_bias = df_umps.copy()
    df_bias["abs_favor"] = df_bias["avg_favor_home"].abs()
    df_bias = df_bias.sort_values("abs_favor", ascending=False).head(20)

    fig_bias = px.bar(
        df_bias,
        x="umpire_name",
        y="avg_favor_home",
        color="avg_favor_home",
        color_continuous_scale="RdBu",
        template="plotly_dark",
        title="Top 20 Umpires by Home/Away Favoritism (Run Impact)",
    )
    st.plotly_chart(fig_bias, use_container_width=True)

st.info("💡 Umpire data is synchronized from umpscorecards.us API.")
