import sys
from algomlb.db.session import get_session_factory
from sqlalchemy import text

db = get_session_factory()()
try:
    games = db.execute(text("SELECT game_id FROM game_results WHERE game_date = '2026-05-03' AND (home_team LIKE '%Athletics%' OR away_team LIKE '%Athletics%')")).fetchall()
    print("Games:", games)
    if games:
        game_pk = int(games[0][0])
        print(f"Game PK: {game_pk}")
        res2 = db.execute(text(f"SELECT COUNT(*) FROM uranium_simulated_player_props WHERE game_pk = {game_pk}")).fetchall()
        print("Props count for game:", res2)
        res3 = db.execute(text(f"SELECT stat_type, COUNT(*) FROM uranium_simulated_player_props WHERE game_pk = {game_pk} GROUP BY stat_type")).fetchall()
        print("Props by type for game:", res3)
except Exception as e:
    print(e)
finally:
    db.close()
