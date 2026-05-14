import datetime
from algomlb.db.session import get_session_factory
from algomlb.strategy.betting_service import BettingService

session = get_session_factory()()
try:
    service = BettingService(session)
    # Use May 14th as target date
    target_date = datetime.date(2026, 5, 14)
    placed = service.place_daily_bets(target_date)
    print(f"Placed {placed} bets for {target_date}")
except Exception as e:
    print(f"Error: {e}")
finally:
    session.close()
