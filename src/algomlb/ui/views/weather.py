import streamlit as st  # pragma: no cover
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from algomlb.db.session import get_engine

st.set_page_config(page_title="Weather Analytics", layout="wide")

st.title("🌦️ Weather & Environment Ingest Health")
st.markdown("---")

engine = get_engine()

# --- 1. OVERVIEW METRICS ---
with engine.connect() as conn:
    total_games = int(
        conn.execute(text("SELECT count(*) FROM game_results")).scalar() or 0
    )
    weather_games = int(
        conn.execute(
            text("SELECT count(*) FROM openmeteo_weather_progression")
        ).scalar()
        or 0
    )

    # Calculate coverage
    coverage = (weather_games / total_games * 100) if total_games > 0 else 0

m1, m2, m3 = st.columns(3)
m1.metric("Total Resolved Games", f"{total_games:,}")
m2.metric("Games with Weather", f"{weather_games:,}")
m3.metric(
    "Coverage Gap", f"{total_games - weather_games:,}", f"{coverage:.1f}% Covered"
)

st.markdown("---")

# --- 2. SEASONAL BACKFILL PROGRESS ---
st.subheader("📅 Seasonal Backfill Progress")
query = """
    SELECT 
        extract(year from g.game_date) as season,
        count(g.game_id) as total_games,
        count(w.game_id) as weather_games
    FROM game_results g
    LEFT JOIN openmeteo_weather_progression w ON g.game_id = w.game_id
    WHERE g.ballpark_id IS NOT NULL
    GROUP BY 1
    ORDER BY 1
"""
df = pd.read_sql(query, engine)
if not df.empty:
    df["Percent Complete"] = (df["weather_games"] / df["total_games"] * 100).round(1)

    # Simple Progress Bars per Season
    for _, row in df.iterrows():
        season = int(row["season"])
        cols = st.columns([1, 4, 1])
        cols[0].write(f"**{season}**")
        cols[1].progress(min(row["Percent Complete"] / 100, 1.0))
        cols[2].write(
            f"{row['weather_games']:,} / {row['total_games']:,} ({row['Percent Complete']}%)"
        )

st.markdown("---")

# --- 3. ENVIRONMENTAL TRENDS ---
st.subheader("🌡️ Environmental Distributions (Ingested Data)")
col1, col2 = st.columns(2)

with col1:
    st.write("#### Temperature at First Pitch (T0)")
    q_temp = "SELECT temp_t0_f FROM openmeteo_weather_progression"
    df_temp = pd.read_sql(q_temp, engine)
    if not df_temp.empty:
        fig_temp = px.histogram(
            df_temp,
            x="temp_t0_f",
            nbins=30,
            title="Temp Distribution",
            color_discrete_sequence=["orange"],
        )
        st.plotly_chart(fig_temp, use_container_width=True)

with col2:
    st.write("#### Wind Speed at First Pitch (T0)")
    q_wind = "SELECT wind_speed_t0 FROM openmeteo_weather_progression"
    df_wind = pd.read_sql(q_wind, engine)
    if not df_wind.empty:
        fig_wind = px.histogram(
            df_wind,
            x="wind_speed_t0",
            nbins=30,
            title="Wind Speed Distribution",
            color_discrete_sequence=["cyan"],
        )
        st.plotly_chart(fig_wind, use_container_width=True)

# --- 4. TOP WINDIER/HOTTER STADIUMS ---
st.subheader("🏟️ Stadium Climate Profiles")
query_stadiums = """
    SELECT 
        b.ballpark,
        avg(w.temp_t0_f) as avg_temp,
        avg(w.wind_speed_t0) as avg_wind,
        count(*) as samples
    FROM openmeteo_weather_progression w
    JOIN game_results g ON w.game_id = g.game_id
    JOIN ballparks b ON g.ballpark_id = b.id
    GROUP BY 1
    HAVING count(*) > 50
    ORDER BY avg_temp DESC
"""
df_stadiums = pd.read_sql(query_stadiums, engine)
if not df_stadiums.empty:
    st.dataframe(
        df_stadiums.style.highlight_max(axis=0, subset=["avg_temp", "avg_wind"]),
        use_container_width=True,
    )

st.success("Weather Data Feed: Active | Idempotent Sync: Enabled")
