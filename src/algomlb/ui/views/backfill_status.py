import streamlit as st
import pandas as pd
import os
from sqlalchemy import text
from algomlb.db.session import get_engine


def _get_backfill_matrix(engine):
    """Computes a completion matrix for all layers across years."""
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    layers = ["Bronze", "Silver", "Gold", "Evals", "Props"]
    matrix = []

    with engine.connect() as conn:
        for yr in years:
            row = {"Year": str(yr)}
            row["Bronze"] = _check_bronze(conn, yr)
            row["Silver"] = _check_silver(conn, yr)
            row["Gold"] = _check_gold(conn, yr)
            row["Evals"] = _check_evals(conn, yr)
            row["Props"] = _check_props(conn, yr)
            matrix.append(row)

    return pd.DataFrame(matrix, columns=["Year"] + layers)


def _check_bronze(conn, yr):
    cnt = (
        conn.execute(
            text(
                f"SELECT count(*) FROM statcast_raw WHERE extract(year from game_date) = {yr}"
            )
        ).scalar()
        or 0
    )
    return "✅" if cnt > 200000 else ("⚠️" if cnt > 0 else "❌")


def _check_silver(conn, yr):
    cnt = (
        conn.execute(
            text(
                f"SELECT count(*) FROM statcast_player_game_logs WHERE extract(year from game_date) = {yr} AND fb_speed IS NOT NULL"
            )
        ).scalar()
        or 0
    )
    return "✅" if cnt > 10000 else ("⚠️" if cnt > 0 else "❌")


def _check_gold(conn, yr):
    cnt = (
        conn.execute(
            text(f"SELECT count(*) FROM player_rolling_features WHERE season = {yr}")
        ).scalar()
        or 0
    )
    return "✅" if cnt > 10000 else ("⚠️" if cnt > 0 else "❌")


def _check_evals(conn, yr):
    cnt = (
        conn.execute(
            text(
                f"SELECT count(*) FROM uranium_eval_history WHERE extract(year from fold_date) = {yr}"
            )
        ).scalar()
        or 0
    )
    return "✅" if cnt > 0 else "❌"


def _check_props(conn, yr):
    cnt = (
        conn.execute(
            text(
                f"SELECT count(*) FROM uranium_simulated_player_props WHERE season = {yr}"
            )
        ).scalar()
        or 0
    )
    return "✅" if cnt > 0 else "❌"


def _render_live_tail(log_path, lines=20):
    """Displays the last N lines of the log file with auto-refresh."""
    st.subheader("📝 Live Orchestration Log")
    if not os.path.exists(log_path):
        st.warning(f"Log file not found: {log_path}")
        return

    try:
        with open(log_path, "r") as f:
            # Simple tail implementation
            log_lines = f.readlines()[-lines:]
            st.code("".join(log_lines), language="text")
    except Exception as e:
        st.error(f"Error reading log: {e}")


def show_backfill_status():
    st.title("🔋 Backfill & Orchestration Status")
    st.markdown("---")

    engine = get_engine()

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("🏗️ Pipeline Matrix")
        df = _get_backfill_matrix(engine)
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("✅ Complete | ⚠️ Missing Metrics | ❌ Empty")

        if st.button("Refresh Matrix"):
            st.rerun()

    with col2:
        _render_live_tail("logs/master_backfill.log")

    st.markdown("---")
    st.info(
        "💡 Tip: Multi-core XGBoost training is currently active. Logs may 'pause' during 20-trial Optuna sweeps."
    )


if __name__ == "__main__":
    show_backfill_status()
