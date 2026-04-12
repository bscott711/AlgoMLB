import importlib
import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
from pathlib import Path

import algomlb.ui.utils as ui_utils
import algomlb.ml.monte_carlo.loader as mc_loader

# Force deep reloads for Monte Carlo components
importlib.reload(ui_utils)
importlib.reload(mc_loader)

from algomlb.db.session import get_session_factory
from algomlb.ui.utils import (
    get_upcoming_games,
    load_simulation_results,
    run_and_persist_simulation
)
from algomlb.ui.styles import apply_premium_styles
from algomlb.ml.monte_carlo.loader import MatchupLoader


def show_simulation_lab():
    apply_premium_styles()
    st.title("🤖 Monte Carlo Simulation Lab")
    st.markdown(
        "Explore high-fidelity game projections and player prop distributions powered by our Uranium simulation engine."
    )
    st.markdown("---")

    session_factory = get_session_factory()
    session = session_factory()

    # --- Sidebar: Control Panel ---
    with st.sidebar:
        st.header("⚙️ Simulation Controls")
        
        selected_date = st.date_input("Matchup Date", value=datetime.date.today())
        
        games_df = get_upcoming_games(session, selected_date)
        
        if games_df.empty:
            st.warning(f"No games found for {selected_date}")
            session.close()
            return

        games_df["display"] = games_df.apply(
            lambda r: f"{r['away_team']} @ {r['home_team']} ({r['status']})", axis=1
        )
        selected_game_display = st.selectbox("Select Game", options=games_df["display"].tolist())
        game_row = games_df[games_df["display"] == selected_game_display].iloc[0]
        game_pk = int(game_row["game_id"])
        
        st.divider()
        trials = st.slider("Monte Carlo Trials", 1000, 20000, 10000, step=1000)
        run_sim = st.button("🚀 Run New Simulation", use_container_width=True)
        st.info("Note: New simulations are persisted to the database.")

    # --- Main Content ---
    
    # 1. Matchup Header
    col_date, col_status = st.columns([2, 1])
    with col_date:
        st.subheader(f"📅 Simulation for {game_row['game_date']}")
    with col_status:
        st.info(f"Status: {game_row['status']}")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"<h3 style='text-align: center;'>🏟️ {game_row['away_team']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Starter: <b>{game_row['away_pitcher'] if game_row['away_pitcher'] else 'TBD'}</b></p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
        # Handle nan scores
        a_score = game_row["away_score"] if pd.notna(game_row["away_score"]) else 0
        h_score = game_row["home_score"] if pd.notna(game_row["home_score"]) else 0
        st.markdown(f"<h3 style='text-align: center;'>{a_score} - {h_score}</h3>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<h3 style='text-align: center;'>🏟️ {game_row['home_team']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Starter: <b>{game_row['home_pitcher'] if game_row['home_pitcher'] else 'TBD'}</b></p>", unsafe_allow_html=True)

    st.markdown("---")

    # 2. Logic: Load or Run
    sim_df = pd.DataFrame()
    if run_sim:
        with st.spinner(f"Executing {trials} Markov Chain trials..."):
            try:
                sim_df = run_and_persist_simulation(session, game_pk, trials)
                st.success("Simulation complete and persisted.")
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                st.exception(e)
    else:
        sim_df = load_simulation_results(session, game_pk)

    if sim_df.empty:
        st.warning("No simulation results found for this game.")
        st.info("Click 'Run New Simulation' in the sidebar to generate projections.")
        session.close()
        return

    # 3. Visualization
    
    # Win Probability Card
    win_props = sim_df[sim_df["stat_type"] == "WIN"]
    if not win_props.empty:
        home_win = win_props[win_props["player_id"] == 1]["mean"].iloc[0]
        away_win = win_props[win_props["player_id"] == 0]["mean"].iloc[0]
        
        st.subheader("🏆 Win Probability")
        w_col1, w_col2 = st.columns(2)
        w_col1.metric(f"{game_row['away_team']} Win %", f"{away_win:.1%}")
        w_col2.metric(f"{game_row['home_team']} Win %", f"{home_win:.1%}")
        
        st.progress(home_win, text=f"Home Advantage: {home_win:.1%}")

    # Tabs for Props
    tab1, tab2, tab3 = st.tabs(["🔥 Batter Props", "🧤 Pitcher Props", "📈 Distributions"])

    # Load player names and team assignments for display
    loader = MatchupLoader(session)
    context = loader.load_matchup(game_pk)
    player_names = {}
    away_pids = set()
    home_pids = set()
    pitcher_pids = set()

    if context:
        for b in context.away_lineup:
            player_names[b.player_id] = b.player_name
            away_pids.add(b.player_id)
        for b in context.home_lineup:
            player_names[b.player_id] = b.player_name
            home_pids.add(b.player_id)
        
        player_names[context.away_starter.pitcher_id] = context.away_starter.player_name
        away_pids.add(context.away_starter.pitcher_id)
        pitcher_pids.add(context.away_starter.pitcher_id)
        
        player_names[context.home_starter.pitcher_id] = context.home_starter.player_name
        home_pids.add(context.home_starter.pitcher_id)
        pitcher_pids.add(context.home_starter.pitcher_id)

    with tab1:
        st.subheader("Batter Market Projections")
        batter_stats = ["H", "HR", "RBI", "R", "TB", "HRR"]
        df_bat = sim_df[sim_df["stat_type"].isin(batter_stats)].copy()
        df_bat["Player"] = df_bat["player_id"].map(player_names).fillna("Unknown")
        
        col_bat_a, col_bat_h = st.columns(2)
        
        with col_bat_a:
            st.markdown(f"#### ⚾ {game_row['away_team']}")
            df_bat_a = df_bat[df_bat["player_id"].isin(away_pids)]
            if not df_bat_a.empty:
                pivot_a = df_bat_a.pivot(index="Player", columns="stat_type", values="mean")
                st.dataframe(pivot_a.style.background_gradient(cmap="Greens"), use_container_width=True)
            else:
                st.caption("No data")

        with col_bat_h:
            st.markdown(f"#### ⚾ {game_row['home_team']}")
            df_bat_h = df_bat[df_bat["player_id"].isin(home_pids)]
            if not df_bat_h.empty:
                pivot_h = df_bat_h.pivot(index="Player", columns="stat_type", values="mean")
                st.dataframe(pivot_h.style.background_gradient(cmap="Greens"), use_container_width=True)
            else:
                st.caption("No data")

    with tab2:
        st.subheader("Pitching Market Projections")
        pitcher_stats = ["K", "PO"]
        # CRITICAL: Filter to only include Pitcher PIDs
        df_pit = sim_df[sim_df["stat_type"].isin(pitcher_stats)].copy()
        df_pit = df_pit[df_pit["player_id"].isin(pitcher_pids)]
        df_pit["Player"] = df_pit["player_id"].map(player_names).fillna("Unknown")
        
        col_pit_a, col_pit_h = st.columns(2)
        
        with col_pit_a:
            st.markdown(f"#### 🧤 {game_row['away_team']} (Starter)")
            df_pit_a = df_pit[df_pit["player_id"].isin(away_pids)]
            if not df_pit_a.empty:
                pivot_pit_a = df_pit_a.pivot(index="Player", columns="stat_type", values="mean")
                st.dataframe(pivot_pit_a.style.background_gradient(cmap="Blues"), use_container_width=True)
            else:
                st.caption("No data")

        with col_pit_h:
            st.markdown(f"#### 🧤 {game_row['home_team']} (Starter)")
            df_pit_h = df_pit[df_pit["player_id"].isin(home_pids)]
            if not df_pit_h.empty:
                pivot_pit_h = df_pit_h.pivot(index="Player", columns="stat_type", values="mean")
                st.dataframe(pivot_pit_h.style.background_gradient(cmap="Blues"), use_container_width=True)
            else:
                st.caption("No data")

    with tab3:
        st.subheader("Outcome Frequency")
        selected_player = st.selectbox("Select Player for Distribution", options=list(player_names.values()))
        p_id = [k for k, v in player_names.items() if v == selected_player][0]
        
        p_data = sim_df[sim_df["player_id"] == p_id]
        if not p_data.empty:
            fig = px.bar(
                p_data, 
                x="stat_type", 
                y="mean", 
                error_y="p90", # Stub for range
                title=f"Mean Projected Stats: {selected_player}",
                template="plotly_dark",
                color="mean",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No distribution data for selected player.")

    session.close()


if __name__ == "__main__":
    show_simulation_lab()
