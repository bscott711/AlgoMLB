import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from algomlb.ml.model import MLBModel
from algomlb.ml.monte_carlo.engine import SimulationEngine
from algomlb.ml.monte_carlo.bullpen import BullpenManager

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
        
        home_team = st.text_input("Home Team ID", "NYY")
        away_team = st.text_input("Away Team ID", "BOS")
        
        run_sim = st.button("🚀 Run 10k Trial Simulation", use_container_width=True)

    with col2:
        st.subheader("📊 Projected Outcomes")
        
        pa_model = _load_model("pa_outcome")
        if not pa_model:
            st.warning("⚠️ **PA Outcome Model (v1.0) Not Found.**")
            st.info("The Uranium backfill is currently calibrating this model. Please check the `Backfill & Orchestration` page for progress.")
            return

        if run_sim:
            with st.spinner("Executing Markov Chain trials..."):
                # Initialize Engine with trained model and default bullpen
                manager = BullpenManager(bullpen_df=pd.DataFrame(), hook_profiles=pd.DataFrame())
                engine = SimulationEngine(pa_model=pa_model, bullpen_manager=manager)
                
                # Run simulations
                results = engine.run_trials(trials=trials)
                df = pd.DataFrame(results)
                
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
                    color_discrete_sequence=["#00CC96"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("Simulation Complete.")

if __name__ == "__main__":
    show_simulation_lab()
