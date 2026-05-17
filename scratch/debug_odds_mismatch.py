from algomlb.db.session import get_session_factory
from sqlalchemy import text

db = get_session_factory()()
try:
    print("Listing ALL raw team names in live_odds for 2026-05-13...")
    res = db.execute(text("SELECT DISTINCT home_team, away_team FROM live_odds WHERE game_date = '2026-05-13'")).fetchall()
    for row in res:
        print(f"RAW: {row[1]} @ {row[0]}")
        
    print("\nChecking for any record that contains 'Cleveland' or 'Angels' or 'Anaheim'...")
    res2 = db.execute(text("SELECT * FROM live_odds WHERE home_team LIKE '%%Cleveland%%' OR away_team LIKE '%%Cleveland%%' OR home_team LIKE '%%Angels%%' OR away_team LIKE '%%Angels%%'")).fetchall()
    for row in res2:
        print(row)
except Exception as e:
    print(e)
finally:
    db.close()
