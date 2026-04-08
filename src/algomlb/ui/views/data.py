import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from algomlb.db.session import get_engine

st.set_page_config(page_title="Data & Ingest Health", layout="wide")

st.title("📡 Data Ingest & Storage Health")
st.markdown("---")


@st.cache_resource
def get_cached_engine():
    return get_engine()


engine = get_cached_engine()


# --- 1. SYSTEM OVERVIEW METRICS ---
st.subheader("🚀 System Overview")
m1, m2, m3, m4 = st.columns(4)

with engine.connect() as conn:
    total_pitches = int(
        conn.execute(text("SELECT count(*) FROM pitch_events")).scalar() or 0
    )
    total_games = int(
        conn.execute(text("SELECT count(*) FROM game_results")).scalar() or 0
    )
    total_umps = int(
        conn.execute(text("SELECT count(*) FROM umpire_scorecards")).scalar() or 0
    )
    total_odds = int(
        conn.execute(text("SELECT count(*) FROM historical_odds")).scalar() or 0
    )
    total_txs = int(
        conn.execute(text("SELECT count(*) FROM player_transactions")).scalar() or 0
    )

m1.metric("Pitch Events", f"{total_pitches:,}", "Historical")
m2.metric("Game IDs", f"{total_games:,}", "Parsed")
m3.metric("Umpire Cards", f"{total_umps:,}", "Scraped API")
m4.metric("Transactions", f"{total_txs:,}", "Player Health")

st.markdown("---")

# --- 2. STORAGE HEALTH SUMMARY ---
st.subheader("📊 Storage Health Summary")

tables_to_check = [
    ("pitch_events", "game_date"),
    ("game_results", "game_date"),
    (
        "umpire_scorecards",
        "id",
    ),  # Ump card doesn't have a direct date col yet in ORM, using ID for count
    ("retrosheet_events", "date"),
    ("historical_odds", "snapshot_at"),
    ("player_transactions", "transaction_date"),
    ("ballparks", "id"),
    ("bankroll_ledger", "timestamp"),
    ("openmeteo_weather_progression", "game_id"),
]

health_data = []
with engine.connect() as conn:
    for table, date_col in tables_to_check:
        count = int(conn.execute(text(f"SELECT count(*) FROM {table}")).scalar() or 0)

        # Determine the best query for Last Update date
        last_date = "N/A"
        try:
            if table == "umpire_scorecards":
                # Join with games to get actual date
                last_date = conn.execute(
                    text("""
                    SELECT max(g.game_date) 
                    FROM umpire_scorecards u 
                    JOIN game_results g ON u.game_id = g.game_id
                """)
                ).scalar()
            elif table == "openmeteo_weather_progression":
                last_date = conn.execute(
                    text("""
                    SELECT max(g.game_date) 
                    FROM openmeteo_weather_progression w 
                    JOIN game_results g ON w.game_id = g.game_id
                """)
                ).scalar()
            else:
                last_date = conn.execute(
                    text(f"SELECT max({date_col}) FROM {table}")
                ).scalar()
        except Exception:  # pragma: no cover
            pass  # pragma: no cover

        status = "🟢 Healthy" if count > 0 else "🔴 Empty"
        if count > 0 and table == "game_results":
            # Check for 2019 coverage
            min_year = conn.execute(
                text("SELECT extract(year from min(game_date)) FROM game_results")
            ).scalar()
            if min_year and min_year > 2019:
                status = "🟡 Incomplete (Missing 2019-2021)"  # pragma: no cover

        health_data.append(
            {
                "Table": table,
                "Records": f"{count:,}",
                "Last Update": str(last_date).split(".")[0] if last_date else "N/A",
                "Health Status": status,
            }
        )

st.table(pd.DataFrame(health_data))

st.markdown("---")

# --- 3. SEASONAL COVERAGE ANALYSIS ---
st.subheader("📅 Seasonal Coverage (2019 - 2026)")
col_a, col_b = st.columns(2)

with col_a:
    st.write("#### Games per Season")
    query = """
        SELECT extract(year from game_date) as season, status, count(*) as count 
        FROM game_results 
        GROUP BY 1, 2 
        ORDER BY season, status
    """
    df_seasons = pd.read_sql(query, engine)
    if not df_seasons.empty:
        # Prettify the status for the legend
        df_seasons["Status"] = df_seasons["status"].astype(str).str.title()

        fig_seasons = px.bar(
            df_seasons,
            x="season",
            y="count",
            color="Status",
            title="Game ID Coverage By Status",
            labels={"count": "Number of Games", "season": "Year"},
            template="plotly_dark",
            barmode="stack",
            color_discrete_map={
                "Completed": "#00CC96",
                "Scheduled": "#636EFA",
                "In_Progress": "#FFA15A",
                "Postponed": "#EF553B",
                "Cancelled": "#D3D3D3",
            },
        )
        st.plotly_chart(fig_seasons, width="stretch")
    else:
        st.info("No season data found. Run `algomlb ingest schedule`.")

with col_b:
    st.write("#### Umpire Data Coverage")
    # Join cards with games to get dates
    query_umps = """
        SELECT extract(year from g.game_date) as season, count(u.id) as count
        FROM umpire_scorecards u
        JOIN game_results g ON u.game_id = g.game_id
        GROUP BY 1 ORDER BY 1
    """
    try:
        df_umps = pd.read_sql(query_umps, engine)
        if not df_umps.empty:
            fig_umps = px.line(
                df_umps,
                x="season",
                y="count",
                title="Umpire Scorecards per Year",
                markers=True,
                template="plotly_dark",
            )
            st.plotly_chart(fig_umps, width="stretch")
        else:
            st.warning(
                "Umpire data not yet linked. Game IDs might be missing for 2019-2022."
            )
    except Exception:
        st.error("Umpire scorecard table schema mismatch or missing.")

st.markdown("---")

# --- 4. TRANSACTION COVERAGE ---
st.subheader("🩹 Transaction Data Coverage (2019 - 2025)")
q_tx = """
    SELECT extract(year from transaction_date) as season, count(*) as count
    FROM player_transactions
    GROUP BY 1 ORDER BY 1
"""
df_tx = pd.read_sql(q_tx, engine)
if not df_tx.empty:
    fig_tx = px.bar(
        df_tx,
        x="season",
        y="count",
        title="Transactions per Season",
        template="plotly_dark",
        color_discrete_sequence=["#FF4B4B"],
    )
    st.plotly_chart(fig_tx, width="stretch")
else:
    st.info("No transaction data found. Run `algomlb ingest transactions`.")

# --- 4. DATA QUALITY ---
st.subheader("🔍 Data Quality Reports")
c1, c2 = st.columns(2)

with c1:
    st.write("#### Pitch Events Density")
    query_density = """
        SELECT game_date as date, count(*) as count 
        FROM pitch_events 
        GROUP BY 1 ORDER BY 1
    """
    df_density = pd.read_sql(query_density, engine)
    if not df_density.empty:
        fig_density = px.area(df_density, x="date", y="count", template="plotly_dark")
        st.plotly_chart(fig_density, width="stretch")

with c2:
    st.write("#### Missing Values (Pitch Analytics)")
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
    st.table(pd.DataFrame(missing_data))

st.success("Database Connection: OK")
