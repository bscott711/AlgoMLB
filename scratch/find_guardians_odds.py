import sys
from algomlb.db.session import get_session_factory
from sqlalchemy import text

db = get_session_factory()()
try:
    print("Searching for Guardians or Angels odds on 2026-05-13...")
    res = db.execute(text("SELECT * FROM live_odds WHERE game_date = '2026-05-13' AND (home_team LIKE '%%Cleveland%%' OR away_team LIKE '%%Cleveland%%')")).fetchall()
    for row in res:
        print(row)
except Exception as e:
    print(e)
finally:
    db.close()
