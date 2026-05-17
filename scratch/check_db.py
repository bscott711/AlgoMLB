from algomlb.db.session import get_session_factory
from sqlalchemy import text

db = get_session_factory()()
try:
    res = db.execute(text("SELECT stat_type, COUNT(*) FROM uranium_simulated_player_props GROUP BY stat_type")).fetchall()
    print("Props by type:", res)
    res2 = db.execute(text("SELECT game_pk, stat_type, player_id, mean FROM uranium_simulated_player_props WHERE stat_type = 'R' ORDER BY game_pk DESC LIMIT 10")).fetchall()
    print("Recent rows (R):", res2)
except Exception as e:
    print(e)
finally:
    db.close()
