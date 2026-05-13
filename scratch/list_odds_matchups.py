import sys
from algomlb.db.session import get_session_factory
from sqlalchemy import text

db = get_session_factory()()
try:
    res = db.execute(text("SELECT DISTINCT home_team, away_team FROM live_odds WHERE game_date = '2026-05-13'")).fetchall()
    for row in res:
        print(f"{row[1]} @ {row[0]}")
except Exception as e:
    print(e)
finally:
    db.close()
