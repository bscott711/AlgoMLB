import streamlit as st
import pandas as pd
from datetime import date, datetime
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM, LiveOddsORM
from algomlb.ml.monte_carlo.loader import MatchupLoader
from algomlb.ui import utils as ui_utils


@st.cache_data(ttl=1800, show_spinner="Running Uranium model across today's slate...")
def _compute_predictions(target_date_str: str) -> list[dict]:
    """Cached prediction engine. Re-runs only every 30 minutes or on manual refresh."""
    target_date = date.fromisoformat(target_date_str)
    session_factory = get_session_factory()
    session = session_factory()

    try:
        games = (
            session.query(GameResultORM)
            .filter(GameResultORM.game_date == target_date)
            .all()
        )
        if not games:
            return []

        loader = MatchupLoader(session)
        picks = []

        for game in games:
            try:
                ctx = loader.load_matchup(int(game.game_id))
                if not ctx:
                    continue

                model_prob, is_fallback = ui_utils.get_uranium_prediction(ctx)

                # Market Odds
                market_odds = (
                    session.query(LiveOddsORM)
                    .filter(LiveOddsORM.game_result_id == str(game.game_id))
                    .filter(LiveOddsORM.market_type.in_(["moneyline", "h2h"]))
                    .order_by(LiveOddsORM.timestamp.desc())
                    .first()
                )

                if market_odds:
                    implied_prob = (
                        1.0 / market_odds.price if market_odds.price > 0 else 0.5
                    )

                    if market_odds.outcome == game.home_team:
                        h_implied = implied_prob
                    else:
                        h_implied = 1.0 - implied_prob

                    edge = model_prob - h_implied

                    picks.append(
                        {
                            "Matchup": f"{game.away_team} @ {game.home_team}",
                            "Model Prob": model_prob,
                            "Market Prob": h_implied,
                            "Edge %": edge,
                            "Selection": game.home_team if edge > 0 else game.away_team,
                            "EV %": abs(edge),
                            "Price": market_odds.price
                            if market_odds.outcome
                            == (game.home_team if edge > 0 else game.away_team)
                            else (1 / (1 - implied_prob) if implied_prob < 1 else 0),
                            "Updated": market_odds.timestamp.strftime("%H:%M"),
                            "Fallback": is_fallback,
                        }
                    )
            except Exception:
                continue

        return picks
    finally:
        session.close()


def render_picks_view():
    st.title("🔮 Live Model Predictions")
    st.markdown("---")

    target_date = date.today()

    # Header row with date and refresh button
    col_date, col_refresh = st.columns([3, 1])
    with col_date:
        st.write(f"Scanned Slate for **{target_date}**")
    with col_refresh:
        if st.button("🔄 Refresh Predictions"):
            _compute_predictions.clear()
            st.rerun()

    # Run the cached prediction engine
    picks = _compute_predictions(target_date.isoformat())

    if not picks:
        st.info("No high-value edges detected in the current market.")
        return

    # Display Picks Table
    picks_df = pd.DataFrame(picks)
    picks_df = picks_df.sort_values("EV %", ascending=False)

    # Check for fallback warnings
    fallback_count = sum(1 for p in picks if p.get("Fallback", False))
    if fallback_count > 0:
        st.warning(
            f"⚠️ {fallback_count} game(s) used Elo fallback — no Uranium model found."
        )

    st.subheader("🔥 Top Market Inefficiencies")

    # Premium Styling
    def color_edge(val):
        color = "#2ecc71" if val > 0.05 else "#e74c3c" if val < -0.05 else "#95a5a6"
        return f"color: {color}; font-weight: bold"

    display_cols = [c for c in picks_df.columns if c != "Fallback"]
    st.dataframe(
        picks_df[display_cols]
        .style.format(
            {
                "Model Prob": "{:.1%}",
                "Market Prob": "{:.1%}",
                "Edge %": "{:+.1%}",
                "EV %": "{:.1%}",
                "Price": "{:.2f}",
            }
        )
        .map(color_edge, subset=["Edge %"]),
        use_container_width=True,
    )

    # Featured Best Bet
    if not picks_df.empty:
        best_bet = picks_df.iloc[0]
        st.success(
            f"🎯 **Best Bet**: {best_bet['Selection']} ({best_bet['Price']:.2f}) — EV: {best_bet['EV %']:.1%}"
        )

    # Cache freshness indicator
    st.caption(
        f"📊 Predictions cached at {datetime.now().strftime('%H:%M:%S')} · refreshes every 30 min"
    )


if __name__ == "__main__":
    render_picks_view()
else:
    render_picks_view()
