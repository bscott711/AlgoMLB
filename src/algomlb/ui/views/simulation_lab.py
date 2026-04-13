import importlib
import streamlit as st
import pandas as pd
import datetime
import plotly.express as px

import algomlb.ui.utils as ui_utils
import algomlb.ml.monte_carlo.loader as mc_loader

# Force deep reloads for Monte Carlo components
importlib.reload(ui_utils)
importlib.reload(mc_loader)

from algomlb.db.session import get_session_factory  # noqa: E402
from algomlb.ui.utils import (  # noqa: E402
    get_upcoming_games,
    load_simulation_results,
    run_and_persist_simulation,
)
from algomlb.ui.styles import apply_premium_styles  # noqa: E402
from algomlb.ml.monte_carlo.loader import MatchupLoader  # noqa: E402


def show_simulation_lab():
    apply_premium_styles()
    st.title("🤖 Monte Carlo Simulation Lab")
    st.markdown(
        "Explore high-fidelity game projections powered by our Uranium simulation engine."
    )
    st.markdown("---")

    session = get_session_factory()()
    game_row, trials, run_sim = _render_sidebar(session)

    if game_row is None:
        session.close()
        return

    _render_matchup_header(game_row)
    st.markdown("---")

    sim_df = _get_simulation_data(session, int(game_row["game_id"]), trials, run_sim)

    if not sim_df.empty:
        _render_win_probability(sim_df, game_row)
        _render_tabs(session, sim_df, int(game_row["game_id"]), game_row)
    else:
        st.warning("No simulation results found for this game.")
        st.info("Click 'Run New Simulation' in the sidebar to generate projections.")

    session.close()


def _render_sidebar(session):
    """Render the sidebar and return selected game/config."""
    with st.sidebar:
        st.header("⚙️ Simulation Controls")
        sel_date = st.date_input("Matchup Date", value=datetime.date.today())
        games_df = get_upcoming_games(session, sel_date)

        if games_df.empty:
            st.warning(f"No games found for {sel_date}")
            return None, 10000, False

        games_df["display"] = games_df.apply(
            lambda r: f"{r['away_team']} @ {r['home_team']} ({r['status']})", axis=1
        )
        sel_display = st.selectbox("Select Game", options=games_df["display"].tolist())
        game_row = games_df[games_df["display"] == sel_display].iloc[0]

        st.divider()
        trials = st.slider("Monte Carlo Trials", 1000, 20000, 10000, step=1000)
        run_sim = st.button("🚀 Run New Simulation", use_container_width=True)
        return game_row, trials, run_sim


def _render_matchup_header(game_row):
    """Render the visual VS header for the game."""
    st.subheader(
        f"📅 Simulation for {game_row['game_date']} (Status: {game_row['status']})"
    )
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(
            f"<h3 style='text-align: center;'>🏟️ {game_row['away_team']}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align: center;'>Starter: <b>{game_row['away_pitcher'] or 'TBD'}</b></p>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
        a_score = game_row["away_score"] if pd.notna(game_row["away_score"]) else 0
        h_score = game_row["home_score"] if pd.notna(game_row["home_score"]) else 0
        st.markdown(
            f"<h3 style='text-align: center;'>{a_score} - {h_score}</h3>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<h3 style='text-align: center;'>🏟️ {game_row['home_team']}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align: center;'>Starter: <b>{game_row['home_pitcher'] or 'TBD'}</b></p>",
            unsafe_allow_html=True,
        )


def _get_simulation_data(session, game_pk, trials, run_sim):
    """Load existing results or execute a new simulation."""
    if run_sim:
        with st.spinner(f"Executing {trials} Markov Chain trials..."):
            try:
                return run_and_persist_simulation(session, game_pk, trials)
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                return pd.DataFrame()
    return load_simulation_results(session, game_pk)


def _render_win_probability(sim_df, game_row):
    """Display the win probability section."""
    win_props = sim_df[sim_df["stat_type"] == "WIN"]
    if not win_props.empty:
        h_win = win_props[win_props["player_id"] == 1]["mean"].iloc[0]
        a_win = win_props[win_props["player_id"] == 0]["mean"].iloc[0]
        st.subheader("🏆 Win Probability")
        w1, w2 = st.columns(2)
        w1.metric(f"{game_row['away_team']} Win %", f"{a_win:.1%}")
        w2.metric(f"{game_row['home_team']} Win %", f"{h_win:.1%}")
        st.progress(h_win, text=f"Home Advantage: {h_win:.1%}")


def _render_tabs(session, sim_df, game_pk, game_row):
    """Render the detailed prop distribution tabs."""
    tab1, tab2, tab3 = st.tabs(
        ["🔥 Batter Props", "🧤 Pitcher Props", "📈 Distributions"]
    )
    loader = MatchupLoader(session)
    ctx = loader.load_matchup(game_pk)
    if not ctx:
        return

    names, a_pids, h_pids, p_pids = {}, set(), set(), set()
    for b in ctx.away_lineup:
        names[b.player_id] = b.player_name
        a_pids.add(b.player_id)
    for b in ctx.home_lineup:
        names[b.player_id] = b.player_name
        h_pids.add(b.player_id)
    names[ctx.away_starter.pitcher_id] = ctx.away_starter.player_name
    names[ctx.home_starter.pitcher_id] = ctx.home_starter.player_name
    a_pids.add(ctx.away_starter.pitcher_id)
    h_pids.add(ctx.home_starter.pitcher_id)
    p_pids.update([ctx.away_starter.pitcher_id, ctx.home_starter.pitcher_id])

    with tab1:
        _render_player_table(
            sim_df,
            ["H", "HR", "RBI", "R", "TB", "HRR"],
            names,
            a_pids,
            h_pids,
            game_row,
            "Greens",
        )
    with tab2:
        _render_player_table(
            sim_df, ["K", "PO"], names, a_pids, h_pids, game_row, "Blues", p_pids
        )
    with tab3:
        sel_player = st.selectbox(
            "Select Player for Distribution", options=list(names.values())
        )
        p_id = [k for k, v in names.items() if v == sel_player][0]
        p_data = sim_df[sim_df["player_id"] == p_id]
        if not p_data.empty:
            fig = px.bar(
                p_data,
                x="stat_type",
                y="mean",
                error_y="p90",
                title=f"Mean Projected Stats: {sel_player}",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No distribution data for selected player.")


def _render_player_table(
    df, stats, names, a_pids, h_pids, game_row, cmap, filter_pids=None
):
    """Generic helper to render a side-by-side player stat table."""
    df_filtered = df[df["stat_type"].isin(stats)].copy()
    if filter_pids:
        df_filtered = df_filtered[df_filtered["player_id"].isin(filter_pids)]
    df_filtered["Player"] = df_filtered["player_id"].map(names).fillna("Unknown")

    c1, c2 = st.columns(2)
    for col, side, pids, team in zip(
        [c1, c2],
        ["Away", "Home"],
        [a_pids, h_pids],
        [game_row["away_team"], game_row["home_team"]],
    ):
        with col:
            st.markdown(f"#### {team}")
            side_df = df_filtered[df_filtered["player_id"].isin(pids)]
            if not side_df.empty:
                st.dataframe(
                    side_df.pivot(
                        index="Player", columns="stat_type", values="mean"
                    ).style.background_gradient(cmap=cmap),
                    use_container_width=True,
                )
            else:
                st.caption("No data")


if __name__ == "__main__":
    show_simulation_lab()
