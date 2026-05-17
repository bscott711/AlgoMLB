from algomlb.db.session import get_session_factory
from sqlalchemy import text

db = get_session_factory()()
try:
    res = db.execute(text("SELECT MAX(game_date) FROM team_elo_history")).fetchone()
    print("Max Elo date:", res)
    
    res2 = db.execute(text("SELECT MAX(game_date) FROM team_sabermetrics_history")).fetchone()
    print("Max Sabermetrics date:", res2)
    
    res3 = db.execute(text("SELECT COUNT(*) FROM team_elo_history WHERE game_date >= '2026-01-01'")).fetchone()
    print("2026 Elo count:", res3)
except Exception as e:
    print(e)
finally:
    db.close()
