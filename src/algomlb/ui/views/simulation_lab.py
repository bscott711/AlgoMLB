import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
from pathlib import Path
from algomlb.ml.model import MLBModel
from algomlb.ml.monte_carlo.engine import SimulationEngine


def _load_model(target, version="v1.0"):
    path = Path(f".data/models/{target}_{version}.joblib")
    if not path.exists():
        return None
    return MLBModel.load(path)


def show_simulation_lab():
    st.title("🤖 Monte Carlo Simulation Lab")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Simulation Params")
        trials = st.slider("Trials", 1000, 20000, 10000, step=1000)

        st.text_input("Home Team ID", "NYY")
        st.text_input("Away Team ID", "BOS")

        run_sim = st.button("🚀 Run 10k Trial Simulation", use_container_width=True)

    with col2:
        st.subheader("📊 Projected Outcomes")

        pa_model = _load_model("pa_outcome")
        if not pa_model:
            st.warning("⚠️ **PA Outcome Model (v1.0) Not Found.**")
            st.info(
                "The Uranium backfill is currently calibrating this model. Please check the `Backfill & Orchestration` page for progress."
            )
            return

        if run_sim:
            with st.spinner("Executing Markov Chain trials..."):
                # Note: Simulation Lab currently uses a simplified trial loop
                # This needs a MatchupContext to run properly with the new engine.
                # For now, we stub a minimal context to satisfy the API.
                from algomlb.ml.monte_carlo.loader import MatchupContext
                from algomlb.ml.monte_carlo.state import BatterSimState, PitcherSimState

                stub_context = MatchupContext(
                    game_pk=0,
                    home_lineup=[BatterSimState(player_id=i) for i in range(9)],
                    away_lineup=[BatterSimState(player_id=i + 10) for i in range(9)],
                    home_starter=PitcherSimState(pitcher_id=99),
                    away_starter=PitcherSimState(pitcher_id=88),
                    batter_features={},
                    pitcher_features={},
                    game_date=datetime.date.today(),
                    game_context={},
                )

                engine = SimulationEngine(pa_model=pa_model)

                # Run simulations
                results = engine.run_trials(context=stub_context, trials=trials)
                # Results is List[Dict[int, State]]
                # For the metric display below, we need home/away scores.
                # We can compute these from the state or just sum them.
                # Since the registry includes all players, we need to sum by side.
                # This logic is a bit more involved now, so let's simplify for the UI display.
                sim_data = []
                home_ids = [b.player_id for b in stub_context.home_lineup]
                away_ids = [b.player_id for b in stub_context.away_lineup]
                for trial in results:
                    h_score = sum(
                        s.runs
                        for s in trial.values()
                        if isinstance(s, BatterSimState) and s.player_id in home_ids
                    )
                    a_score = sum(
                        s.runs
                        for s in trial.values()
                        if isinstance(s, BatterSimState) and s.player_id in away_ids
                    )
                    sim_data.append({"home_runs": h_score, "away_runs": a_score})

                df = pd.DataFrame(sim_data)

                # Visualizations
                df["total_runs"] = df["home_runs"] + df["away_runs"]
                df["home_win"] = df["home_runs"] > df["away_runs"]

                home_win_prob = df["home_win"].mean()

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Home Win Prob", f"{home_win_prob:.1%}")
                m2.metric("Avg Total Runs", f"{df['total_runs'].mean():.2f}")
                m3.metric("Home Avg Score", f"{df['home_runs'].mean():.2f}")

                # Distribution Chart
                fig = px.histogram(
                    df,
                    x="total_runs",
                    nbins=20,
                    title="Score Distribution (Total Runs)",
                    template="plotly_dark",
                    color_discrete_sequence=["#00CC96"],
                )
                st.plotly_chart(fig, use_container_width=True)

                st.success("Simulation Complete.")


if __name__ == "__main__":
    show_simulation_lab()
