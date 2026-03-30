import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from algomlb.db.session import get_engine

st.set_page_config(page_title="Data & Ingest Health", layout="wide")

st.title("📡 Data Ingest & Storage Health")
st.markdown("---")

engine = get_engine()

# 1. High Level Metrics
col1, col2, col3, col4 = st.columns(4)

with engine.connect() as conn:
    total_pitches = int(
        conn.execute(text("SELECT count(*) FROM pitch_events")).scalar() or 0
    )
    unique_games = int(
        conn.execute(text("SELECT count(DISTINCT game_id) FROM pitch_events")).scalar()
        or 0
    )
    unique_pitchers = int(
        conn.execute(
            text("SELECT count(DISTINCT pitcher_id) FROM pitch_events")
        ).scalar()
        or 0
    )
    last_date = conn.execute(text("SELECT max(game_date) FROM pitch_events")).scalar()

col1.metric("Total Pitch Events", f"{total_pitches:,}")
col2.metric("Unique Games", f"{unique_games:,}")
col3.metric("Active Pitchers", f"{unique_pitchers:,}")
col4.metric("Last Ingest Date", str(last_date))

st.markdown("### 📈 Ingestion Volume by Day")

# 2. Daily Volume Chart
query = """
    SELECT game_date as date, count(*) as count 
    FROM pitch_events 
    GROUP BY 1 
    ORDER BY 1
"""
df_daily = pd.read_sql(query, engine)
if not df_daily.empty:
    fig_daily = px.area(
        df_daily,
        x="date",
        y="count",
        title="Daily Pitch Event Volume",
        labels={"count": "Number of Pitches", "date": "Game Date"},
        template="plotly_dark",
    )
    st.plotly_chart(fig_daily, width="stretch")

# 3. Data Distribution
st.markdown("### 🔍 Data Quality & Distributions")
c1, c2 = st.columns(2)

with c1:
    st.write("#### Pitch Type Distribution")
    query_types = """
        SELECT pitch_type, count(*) as count 
        FROM pitch_events 
        WHERE pitch_type IS NOT NULL
        GROUP BY 1 
        ORDER BY 2 DESC
    """
    df_types = pd.read_sql(query_types, engine)
    if not df_types.empty:
        fig_types = px.bar(
            df_types, x="pitch_type", y="count", color="count", template="plotly_dark"
        )
        st.plotly_chart(fig_types, width="stretch")

with c2:
    st.write("#### Missing Values Report")
    cols_to_check = [
        "release_speed",
        "release_spin_rate",
        "launch_speed",
        "launch_angle",
    ]
    missing_data = []

    with engine.connect() as conn:
        for col in cols_to_check:
            null_count = int(
                conn.execute(
                    text(f"SELECT count(*) FROM pitch_events WHERE {col} IS NULL")
                ).scalar()
                or 0
            )
            missing_data.append(
                {
                    "Column": col,
                    "Missing": null_count,
                    "Percentage": (null_count / total_pitches) * 100
                    if total_pitches > 0
                    else 0,
                }
            )

    df_missing = pd.DataFrame(missing_data)
    st.table(df_missing)
    st.caption(
        "💡 **Note:** High missing values for `launch_speed/angle` are expected as they only apply to batted ball events (~33% of pitches)."
    )

# 4. Glossary
with st.expander("📝 Pitch Type Glossary (Statcast Codes)"):
    col_a, col_b = st.columns(2)
    glossary = {
        "FF": "Four-Seam Fastball",
        "SI": "Sinker",
        "SL": "Slider",
        "CH": "Changeup",
        "FC": "Cutter",
        "ST": "Sweeper",
        "CU": "Curveball",
        "FS": "Splitter",
        "KC": "Knuckle Curve",
        "SV": "Slurve",
        "KN": "Knuckleball",
        "FA": "Other Fastball",
        "EP": "Eephus",
        "SC": "Screwball",
        "FO": "Forkball",
        "CS": "Slow Curve",
        "PO": "Pitchout",
    }

    # Split for readability
    items = list(glossary.items())
    midpoint = len(items) // 2 + 1

    with col_a:
        for code, name in items[:midpoint]:
            st.markdown(f"**{code}**: {name}")
    with col_b:
        for code, name in items[midpoint:]:
            st.markdown(f"**{code}**: {name}")

st.success("Database Connection: OK")
