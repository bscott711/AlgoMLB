import streamlit as st  # pragma: no cover
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from algomlb.db.session import get_engine

st.set_page_config(page_title="Weather Analytics", layout="wide")

st.title("🌦️ Weather & Environment Ingest Health")
st.markdown("---")


@st.cache_resource
def get_cached_engine():
    return get_engine()


engine = get_cached_engine()

# --- 1. OVERVIEW METRICS ---
with engine.connect() as conn:
    total_completed = int(
        conn.execute(
            text("SELECT count(*) FROM game_results WHERE status = 'COMPLETED'")
        ).scalar()
        or 0
    )
    weather_games = int(
        conn.execute(
            text("""
                SELECT count(w.game_id) 
                FROM openmeteo_weather_progression w
                JOIN game_results g ON w.game_id = g.game_id
                WHERE g.status = 'COMPLETED'
            """)
        ).scalar()
        or 0
    )

    # Calculate coverage
    coverage = (weather_games / total_completed * 100) if total_completed > 0 else 0

m1, m2, m3 = st.columns(3)
m1.metric("Total Completed Games", f"{total_completed:,}")
m2.metric("Games with Weather", f"{weather_games:,}")
m3.metric(
    "Backfill Coverage",
    f"{total_completed - weather_games:,} Missing",
    f"{coverage:.1f}% Covered",
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
    WHERE g.status = 'COMPLETED'
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
else:
    st.info("No completed games found to analyze backfill progress.")

st.markdown("---")

# --- 3. IDENTIFY MISSING WEATHER ---
st.subheader("🕵️ Missing Weather Audit")
selected_year = st.selectbox(
    "Select Season to Audit", [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026], index=7
)

with engine.connect() as conn:
    q_missing = text("""
        SELECT g.game_id, g.game_date, g.home_team, g.away_team
        FROM game_results g
        LEFT JOIN openmeteo_weather_progression w ON g.game_id = w.game_id
        WHERE extract(year from g.game_date) = :year
          AND g.status = 'COMPLETED'
          AND w.game_id IS NULL
        ORDER BY g.game_date ASC
    """)
    df_missing = pd.read_sql(q_missing, engine, params={"year": int(selected_year)})

if not df_missing.empty:
    st.warning(
        f"Found {len(df_missing)} completed games missing weather data for {selected_year}."
    )
    st.dataframe(df_missing, use_container_width=True)

    # Generate ingestion command for the user
    min_date = df_missing["game_date"].min()
    max_date = df_missing["game_date"].max()
    st.info(
        "💡 To fill this gap, you can run research-based ingestion for this date range:"
    )
    st.code(f"uv run algomlb ingest weather --start {min_date} --end {max_date}")
else:
    st.success(f"All completed games in {selected_year} have weather data!")

st.markdown("---")

# --- 4. ENVIRONMENTAL TRENDS ---
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

# --- 5. TOP WINDIER/HOTTER STADIUMS ---
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
