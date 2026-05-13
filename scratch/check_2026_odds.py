import sys
from algomlb.db.session import get_session_factory
from sqlalchemy import text

db = get_session_factory()()
try:
    res = db.execute(text("SELECT COUNT(*) FROM live_odds WHERE game_date = '2026-05-13'")).fetchone()
    print("2026-05-13 Odds count:", res)
    
    if res[0] > 0:
        res2 = db.execute(text("SELECT * FROM live_odds WHERE game_date = '2026-05-13' LIMIT 1")).fetchone()
        print("Sample odds:", res2)
except Exception as e:
    print(e)
finally:
    db.close()
