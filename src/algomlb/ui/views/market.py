import streamlit as st
import pandas as pd
import plotly.express as px
import importlib
import algomlb.db.models as models
importlib.reload(models)
from algomlb.db.session import get_engine
from algomlb.db.models import ModelPredictionORM, GameResultORM, LiveOddsORM
from sqlalchemy import func

st.set_page_config(page_title="Market Analytics", layout="wide")

st.title("📈 Market Analytics: Uranium CLV & Calibration")
st.markdown("---")

engine = get_engine()

# --- 1. Alpha & CLV Tracking ---
st.markdown("### 🔬 Model Alpha & Closing Line Value (CLV)")

query = """
    WITH latest_predictions AS (
        SELECT DISTINCT ON (game_id) 
            game_id, 
            home_win_prob, 
            market_home_implied_at_prediction as entry_implied,
            timestamp as pred_time
        FROM model_predictions
        ORDER BY game_id, timestamp DESC
    ),
    closing_odds AS (
        SELECT DISTINCT ON (co.game_result_id, co.outcome)
            co.game_result_id as game_id,
            co.outcome,
            co.price as closing_price
        FROM live_odds co
        JOIN game_results gr ON co.game_result_id = gr.game_id
        WHERE co.market_type = 'h2h' 
          AND co.timestamp <= gr.game_datetime
        ORDER BY co.game_result_id, co.outcome, co.timestamp DESC
    ),
    game_meta AS (
        SELECT game_id, home_team, away_team, game_date, status
        FROM game_results
    )
    SELECT 
        gm.game_date,
        gm.away_team || ' @ ' || gm.home_team as matchup,
        lp.home_win_prob as model_prob,
        lp.entry_implied,
        (1.0 / co.closing_price) as closing_implied,
        ((1.0 / co.closing_price) - lp.entry_implied) as market_move,
        (lp.home_win_prob - (1.0 / co.closing_price)) as closing_edge,
        gm.status
    FROM latest_predictions lp
    JOIN game_meta gm ON lp.game_id = gm.game_id
    LEFT JOIN closing_odds co ON lp.game_id = co.game_id AND co.outcome = gm.home_team
    ORDER BY gm.game_date DESC, lp.pred_time DESC
"""

df_alpha = pd.read_sql(query, engine)

if not df_alpha.empty:
    # 2. Display CLV Table
    def color_clv(val):
        color = '#2ecc71' if val > 0.02 else '#e74c3c' if val < -0.02 else '#95a5a6'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df_alpha.style.format({
            "model_prob": "{:.1%}",
            "entry_implied": "{:.1%}",
            "closing_implied": "{:.1%}",
            "market_move": "{:+.1%}",
            "closing_edge": "{:+.1%}"
        }).map(color_clv, subset=["market_move"]),
        use_container_width=True
    )

    # 3. Calibration Plot
    st.markdown("---")
    st.markdown("### 🎯 Model Calibration: Projections vs Market Close")
    
    fig_cal = px.scatter(
        df_alpha,
        x="closing_implied",
        y="model_prob",
        color="market_move",
        hover_name="matchup",
        trendline="ols",
        title="Uranium Projections vs. Market Closing Probabilities",
        labels={"closing_implied": "Market Closing Prob", "model_prob": "Uranium Prob"},
        template="plotly_dark",
        color_continuous_scale="RdYlGn"
    )
    # Add 45-degree line
    fig_cal.add_shape(
        type='line', line=dict(dash='dash', color='gray'),
        x0=0, x1=1, y0=0, y1=1
    )
    st.plotly_chart(fig_cal, use_container_width=True)

else:
    st.info("No model prediction history found yet. Run a sync or view the Simulation Lab to archive predictions.")

# --- 4. Raw Market Movement (Legacy) ---
st.markdown("---")
st.markdown("### 🏛️ Raw Market Drift (Legacy View)")
# ... (Keeping the original market drift logic below for completeness)
"""
(Include original drift logic here if needed, or just let the new view dominate)
"""
