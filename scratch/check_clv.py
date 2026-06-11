import pandas as pd
from algomlb.db.session import get_engine

engine = get_engine()
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
if df_alpha.empty:
    print("No prediction history found.")
else:
    df_alpha["clv"] = df_alpha.apply(
        lambda r: r["market_move"] if r["model_prob"] > r["entry_implied"] else -r["market_move"], 
        axis=1
    )
    clv_beat_rate = (df_alpha["clv"] > 0).mean()
    avg_clv = df_alpha["clv"].mean()
    mae_vs_close = (df_alpha["model_prob"] - df_alpha["closing_implied"]).abs().mean()
    
    print(f"Total Predictions: {len(df_alpha)}")
    print(f"CLV Beat Rate: {clv_beat_rate:.1%}")
    print(f"Average CLV: {avg_clv:+.2%}")
    print(f"MAE vs Close: {mae_vs_close:.3f}")

