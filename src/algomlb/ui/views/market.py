import streamlit as st
import pandas as pd
import plotly.express as px
import importlib
import algomlb.db.models as models

importlib.reload(models)
from algomlb.db.session import get_engine
from algomlb.ml.clv import summarize_clv

st.set_page_config(page_title="Market Analytics", layout="wide")

st.title("📈 Market Analytics: Uranium CLV & Calibration")
st.markdown("---")

engine = get_engine()

# --- 1. Alpha & CLV Tracking ---
st.markdown("### 🔬 Model Alpha & Closing Line Value (CLV)")
st.caption(
    "Closing line = de-vigged consensus across all books with a live_odds "
    "snapshot at/before first pitch (see `algomlb ml clv`). Not a single "
    "sharp-book close — see clv_results for the underlying picks."
)

query = """
    SELECT
        cr.game_date,
        gr.away_team || ' @ ' || gr.home_team AS matchup,
        cr.model_prob,
        cr.entry_implied,
        cr.closing_implied,
        (cr.closing_implied - cr.entry_implied) AS market_move,
        cr.closing_edge,
        cr.num_books_at_close,
        cr.clv,
        gr.status
    FROM clv_results cr
    JOIN game_results gr ON cr.game_id = gr.game_id
    ORDER BY cr.game_date DESC, cr.closing_snapshot_at DESC
"""

df_alpha = pd.read_sql(query, engine)

if not df_alpha.empty:
    # 2. Display CLV Table
    def color_clv(val):
        color = "#2ecc71" if val > 0.02 else "#e74c3c" if val < -0.02 else "#95a5a6"
        return f"color: {color}; font-weight: bold"

    st.dataframe(
        df_alpha.style.format(
            {
                "model_prob": "{:.1%}",
                "entry_implied": "{:.1%}",
                "closing_implied": "{:.1%}",
                "market_move": "{:+.1%}",
                "closing_edge": "{:+.1%}",
                "clv": "{:+.2%}",
            }
        ).map(color_clv, subset=["clv"]),
        use_container_width=True,
    )

    summary = summarize_clv()

    st.markdown("---")
    st.markdown("### 📊 CLV & Calibration Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "CLV Beat Rate",
        f"{summary['clv_beat_rate']:.1%}" if summary["n"] else "—",
        help="% of games where the closing line moved toward our model's entry-time side.",
    )
    m2.metric(
        "Average CLV",
        f"{summary['avg_clv']:+.2%}" if summary["n"] else "—",
        help="Average probability points gained by beating the closing line.",
    )
    m3.metric(
        "Avg Closing Edge",
        f"{summary['avg_closing_edge']:+.2%}" if summary["n"] else "—",
        help="Average (model probability − closing implied probability) across all picks.",
    )

    # 3. Calibration Plot
    st.markdown("---")
    st.markdown("### 🎯 Model Calibration: Projections vs Market Close")

    fig_cal = px.scatter(
        df_alpha,
        x="closing_implied",
        y="model_prob",
        color="clv",
        hover_name="matchup",
        trendline="ols",
        title="Uranium Projections vs. Market Closing Probabilities",
        labels={"closing_implied": "Market Closing Prob", "model_prob": "Uranium Prob"},
        template="plotly_dark",
        color_continuous_scale="RdYlGn",
    )
    # Add 45-degree line
    fig_cal.add_shape(
        type="line", line=dict(dash="dash", color="gray"), x0=0, x1=1, y0=0, y1=1
    )
    st.plotly_chart(fig_cal, use_container_width=True)

else:
    st.info(
        "No CLV results yet. Run `algomlb ml clv --start-date ... --end-date ...` "
        "(or wait for the daily sync's clv stage) once games have closing odds."
    )

