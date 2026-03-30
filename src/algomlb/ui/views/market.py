import streamlit as st
import pandas as pd
import plotly.express as px
from algomlb.db.session import get_engine

st.set_page_config(page_title="Market Analytics", layout="wide")

st.title("📈 Market Analytics: Opening vs Closing Odds")
st.markdown("---")

engine = get_engine()

# --- 1. Top Level Market Stats ---
st.markdown("### 🏛️ Market Movement Summary")

with engine.connect() as conn:
    query = """
        WITH game_extrema AS (
            SELECT 
                odds_game_id,
                min(timestamp) as open_time,
                max(timestamp) as close_time
            FROM live_odds
            GROUP BY 1
        ),
        game_names AS (
            -- Resolve real names by looking for non-'Unknown' values across all ticks
            SELECT 
                odds_game_id,
                max(CASE WHEN home_team NOT IN ('unknown', 'Unknown') THEN home_team END) as home_team,
                max(CASE WHEN away_team NOT IN ('unknown', 'Unknown') THEN away_team END) as away_team,
                max(game_date) as game_date
            FROM live_odds
            GROUP BY 1
        ),
        opening_odds AS (
            SELECT l.odds_game_id, l.sportsbook, l.outcome, l.price as opening_price, l.timestamp as o_time
            FROM live_odds l
            JOIN game_extrema g ON l.odds_game_id = g.odds_game_id AND l.timestamp = g.open_time
            WHERE l.market_type = 'h2h'
        ),
        closing_odds AS (
            SELECT l.odds_game_id, l.sportsbook, l.outcome, l.price as closing_price, l.timestamp as c_time
            FROM live_odds l
            JOIN game_extrema g ON l.odds_game_id = g.odds_game_id AND l.timestamp = g.close_time
            WHERE l.market_type = 'h2h'
        )
        SELECT 
            gn.game_date,
            o.sportsbook,
            COALESCE(gn.home_team, 'N/A') as home,
            COALESCE(gn.away_team, 'N/A') as away,
            o.outcome,
            o.opening_price,
            c.closing_price,
            (c.closing_price - o.opening_price) as drift,
            o.o_time as opened,
            c.c_time as closed
        FROM opening_odds o
        JOIN closing_odds c ON o.odds_game_id = c.odds_game_id AND o.sportsbook = c.sportsbook AND o.outcome = c.outcome
        JOIN game_names gn ON o.odds_game_id = gn.odds_game_id
        ORDER BY opened DESC
    """
    df_market = pd.read_sql(query, engine)

if not df_market.empty:
    # Display highlights with style
    st.dataframe(
        df_market.style.format(
            {"opening_price": "{:.2f}", "closing_price": "{:.2f}", "drift": "{:+.3f}"}
        ).background_gradient(subset=["drift"], cmap="RdYlGn"),
        width="stretch",
    )

    # 2. Drift Histogram
    st.markdown("### 📊 Price Drift Distribution")
    fig_drift = px.histogram(
        df_market,
        x="drift",
        nbins=20,
        marginal="box",
        color_discrete_sequence=["#00CC96"],
        title="Line Movement Distribution (Closing - Opening)",
        labels={"drift": "Price Change (Decimal Odds)"},
        template="plotly_dark",
    )
    st.plotly_chart(fig_drift, width="stretch")

else:
    st.info("No H2H market movement data found in `live_odds` yet.")
