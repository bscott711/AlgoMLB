import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from algomlb.db.session import get_engine


def _get_table_health(conn, table, date_col):
    """Calculate health metrics for a single table."""
    count = int(conn.execute(text(f"SELECT count(*) FROM {table}")).scalar() or 0)
    last_date = "N/A"
    try:
        if table == "umpire_scorecards":
            q = "SELECT max(g.game_date) FROM umpire_scorecards u JOIN game_results g ON u.game_id = g.game_id"
            last_date = conn.execute(text(q)).scalar()
        elif table == "openmeteo_weather_progression":
            q = "SELECT max(g.game_date) FROM openmeteo_weather_progression w JOIN game_results g ON w.game_id = g.game_id"
            last_date = conn.execute(text(q)).scalar()
        else:
            last_date = conn.execute(
                text(f"SELECT max({date_col}) FROM {table}")
            ).scalar()
    except Exception:
        pass

    status = "🟢 Healthy" if count > 0 else "🔴 Empty"
    return {
        "Table": table,
        "Records": f"{count:,}",
        "Last Update": str(last_date).split(".")[0] if last_date else "N/A",
        "Health Status": status,
    }


def _render_storage_health(engine):
    """Render a table summarizing record counts and last update dates per table."""
    st.subheader("📊 Storage Health Summary")
    tables_to_check = [
        ("pitch_events", "game_date"),
        ("game_results", "game_date"),
        ("umpire_scorecards", "id"),
        ("retrosheet_events", "date"),
        ("historical_odds", "snapshot_at"),
        ("player_transactions", "transaction_date"),
        ("ballparks", "id"),
        ("bankroll_ledger", "timestamp"),
        ("openmeteo_weather_progression", "game_id"),
    ]

    health_data = []
    try:
        with engine.connect() as conn:
            for table, date_col in tables_to_check:
                health_data.append(_get_table_health(conn, table, date_col))
        st.table(pd.DataFrame(health_data))
    except Exception:
        st.error("Error generating storage health summary.")


def _render_system_metrics(engine):
    """Render high-level record counts from the database."""
    st.subheader("🚀 System Overview")
    m1, m2, m3, m4 = st.columns(4)

    total_pitches = 0
    try:
        with engine.connect() as conn:
            total_pitches = int(
                conn.execute(text("SELECT count(*) FROM pitch_events")).scalar() or 0
            )
            total_games = int(
                conn.execute(text("SELECT count(*) FROM game_results")).scalar() or 0
            )
            total_umps = int(
                conn.execute(text("SELECT count(*) FROM umpire_scorecards")).scalar()
                or 0
            )
            total_txs = int(
                conn.execute(text("SELECT count(*) FROM player_transactions")).scalar()
                or 0
            )

        m1.metric("Pitch Events", f"{total_pitches:,}", "Historical")
        m2.metric("Game IDs", f"{total_games:,}", "Parsed")
        m3.metric("Umpire Cards", f"{total_umps:,}", "Scraped API")
        m4.metric("Transactions", f"{total_txs:,}", "Player Health")
    except Exception:
        st.error("Error connecting to database for metrics.")
    return total_pitches


def _render_seasonal_analysis(engine):
    """Render charts for season and umpire coverage."""
    st.subheader("📅 Seasonal Coverage (2019 - 2026)")
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("#### Games per Season")
        try:
            df = pd.read_sql(
                "SELECT extract(year from game_date) as season, status, count(*) as count FROM game_results GROUP BY 1, 2 ORDER BY 1, 2",
                engine,
            )
            if not df.empty:
                df["Status"] = df["status"].astype(str).str.title()
                fig = px.bar(
                    df,
                    x="season",
                    y="count",
                    color="Status",
                    template="plotly_dark",
                    barmode="stack",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No season data found.")
        except Exception:
            st.error("Error loading seasonal game data.")

    with col_b:
        st.write("#### Umpire Data Coverage")
        try:
            df = pd.read_sql(
                "SELECT extract(year from g.game_date) as season, count(u.id) as count FROM umpire_scorecards u JOIN game_results g ON u.game_id = g.game_id GROUP BY 1 ORDER BY 1",
                engine,
            )
            if not df.empty:
                fig = px.line(
                    df, x="season", y="count", markers=True, template="plotly_dark"
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Umpire data not yet linked.")
        except Exception:
            st.error("Umpire scorecard data missing or error.")


def show_data_health(engine=None):
    """Main entry point for the Data Ingest & Storage Health view."""
    if engine is None:
        engine = get_engine()

    st.title("📡 Data Ingest & Storage Health")
    st.markdown("---")

    total_pitches = _render_system_metrics(engine)
    st.markdown("---")
    _render_storage_health(engine)
    st.markdown("---")
    _render_seasonal_analysis(engine)

    # Simple Quality Table
    st.subheader("🔍 Quality: Pitch Nulls")
    cols = ["release_speed", "release_spin_rate", "launch_speed", "launch_angle"]
    quality_list = []
    try:
        with engine.connect() as conn:
            for col in cols:
                nulls = int(
                    conn.execute(
                        text(f"SELECT count(*) FROM pitch_events WHERE {col} IS NULL")
                    ).scalar()
                    or 0
                )
                pct = (nulls / total_pitches * 100) if total_pitches > 0 else 0
                quality_list.append(
                    {"Column": col, "Missing": nulls, "Percentage": f"{pct:.1f}%"}
                )
        st.table(pd.DataFrame(quality_list))
    except Exception:
        st.error("Error loading quality metrics.")

    st.success("Database Connection: OK")


if __name__ == "__main__":
    st.set_page_config(page_title="Data & Ingest Health", layout="wide")
    show_data_health()
