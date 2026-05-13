import streamlit as st
import pandas as pd
from datetime import date
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM, LiveOddsORM
from algomlb.ml.monte_carlo.loader import MatchupLoader
from algomlb.ui import utils as ui_utils

def render_picks_view():
    st.title("🔮 Live Model Predictions")
    st.markdown("---")
    
    session_factory = get_session_factory()
    session = session_factory()
    
    try:
        # Default to the current day in our sim cycle
        target_date = date(2026, 5, 13)
        
        st.write(f"Scanned Slate for **{target_date}**")
        
        # 1. Fetch Today's Games
        games = (
            session.query(GameResultORM)
            .filter(GameResultORM.game_date == target_date)
            .all()
        )
        
        if not games:
            st.warning("No games scheduled for today.")
            return

        loader = MatchupLoader(session)
        picks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, game in enumerate(games):
            status_text.text(f"Analyzing {game.away_team} @ {game.home_team}...")
            progress_bar.progress((i + 1) / len(games))
            
            try:
                # Top-Down Prediction (Uranium)
                ctx = loader.load_matchup(int(game.game_id))
                if not ctx:
                    continue
                    
                model_prob, is_fallback = ui_utils.get_uranium_prediction(ctx)
                
                # Market Odds
                market_odds = (
                    session.query(LiveOddsORM)
                    .filter(LiveOddsORM.game_result_id == str(game.game_id))
                    .filter(LiveOddsORM.market_type.in_(['moneyline', 'h2h']))
                    .order_by(LiveOddsORM.timestamp.desc())
                    .first()
                )
                
                if market_odds:
                    implied_prob = 1.0 / market_odds.price if market_odds.price > 0 else 0.5
                    
                    # Logic to map implied_prob to the RIGHT team
                    # (Simplified: assume outcome is for the 'Selection' we calculate)
                    # For a robust version, we check the 'outcome' string vs team names
                    if market_odds.outcome == game.home_team:
                        h_implied = implied_prob
                    else:
                        h_implied = 1.0 - implied_prob
                    
                    edge = model_prob - h_implied
                    
                    picks.append({
                        "Matchup": f"{game.away_team} @ {game.home_team}",
                        "Model Prob": model_prob,
                        "Market Prob": h_implied,
                        "Edge %": edge,
                        "Selection": game.home_team if edge > 0 else game.away_team,
                        "EV %": abs(edge),
                        "Price": market_odds.price if market_odds.outcome == (game.home_team if edge > 0 else game.away_team) else (1 / (1 - implied_prob) if implied_prob < 1 else 0),
                        "Updated": market_odds.timestamp.strftime("%H:%M")
                    })
            except Exception as e:
                continue

        progress_bar.empty()
        status_text.empty()
        
        if not picks:
            st.info("No high-value edges detected in the current market.")
            return
            
        # 2. Display Picks Table
        picks_df = pd.DataFrame(picks)
        # Sort by absolute Edge to find the biggest discrepancies
        picks_df = picks_df.sort_values("Edge %", ascending=False)
        
        st.subheader("🔥 Top Market Inefficiencies")
        
        # Premium Styling
        def color_edge(val):
            color = '#2ecc71' if val > 0.05 else '#e74c3c' if val < -0.05 else '#95a5a6'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            picks_df.style.format({
                "Model Prob": "{:.1%}",
                "Market Prob": "{:.1%}",
                "Edge %": "{:+.1%}",
                "EV %": "{:.1%}",
                "Price": "{:.2f}"
            }).applymap(color_edge, subset=["Edge %"]),
            use_container_width=True
        )
        
        # 3. Featured Best Bet
        if not picks_df.empty:
            best_bet = picks_df.iloc[0]
            st.success(f"🎯 **Best Bet**: {best_bet['Selection']} ML ({best_bet['Price']:.2f}) — Edge: {best_bet['Edge %']:.1%}")

    finally:
        session.close()

if __name__ == "__main__":
    render_picks_view()
else:
    render_picks_view()
