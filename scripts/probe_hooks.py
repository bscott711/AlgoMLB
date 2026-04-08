"""Quick probe of retrosheet pitcher change patterns for hook event derivation."""

from algomlb.db.session import get_session_factory
import pandas as pd

session_factory = get_session_factory()
engine = session_factory.kw["bind"]

# 1. Count of games with pitcher changes (2024)
q1 = """
SELECT COUNT(*) as games_with_hooks FROM (
    SELECT game_id, pit_team, COUNT(DISTINCT pitcher_id) as n_pitchers
    FROM retrosheet_events
    WHERE EXTRACT(YEAR FROM date) = 2024 AND pa_flag = 1
    GROUP BY game_id, pit_team
    HAVING COUNT(DISTINCT pitcher_id) > 1
) sub
"""
r1 = pd.read_sql(q1, engine)
print(f"Games-sides with pitcher changes (2024): {r1.iloc[0, 0]}")

# 2. Verify nump coverage
q2 = """
SELECT COUNT(*) as total, 
       SUM(CASE WHEN nump IS NULL THEN 1 ELSE 0 END) as null_nump
FROM retrosheet_events WHERE EXTRACT(YEAR FROM date) = 2024 AND pa_flag = 1
"""
r2 = pd.read_sql(q2, engine)
print(f"nump coverage: {r2.to_dict('records')}")

# 3. Sample pitcher transition in a single game
q3 = """
SELECT play_number, inning, top_bot, pitcher_id, pa_flag, 
       outs_pre, nump, lp, score_v, score_h,
       CASE WHEN br1_pre IS NOT NULL THEN 1 ELSE 0 END as r1,
       CASE WHEN br2_pre IS NOT NULL THEN 1 ELSE 0 END as r2,
       CASE WHEN br3_pre IS NOT NULL THEN 1 ELSE 0 END as r3
FROM retrosheet_events
WHERE game_id = (
    SELECT game_id FROM retrosheet_events 
    WHERE EXTRACT(YEAR FROM date) = 2024 
    AND pa_flag = 1
    LIMIT 1
)
AND top_bot = 0
ORDER BY play_number
"""
df3 = pd.read_sql(q3, engine)
print("\n=== Sample game (top half only) ===")
print(df3.to_string(index=False))

# 4. Statcast TTO distinct values
q4 = """
SELECT DISTINCT n_thruorder_pitcher::int as tto 
FROM statcast_raw WHERE game_year = 2024 
ORDER BY 1 LIMIT 10
"""
tto = pd.read_sql(q4, engine)
print(f"\nTTO values in statcast: {tto['tto'].tolist()}")

# 5. Team manager coverage check
q5 = "SELECT season, COUNT(*) as managers FROM team_managers GROUP BY season ORDER BY season"
mgr = pd.read_sql(q5, engine)
print(f"\nManager coverage:\n{mgr.to_string(index=False)}")
