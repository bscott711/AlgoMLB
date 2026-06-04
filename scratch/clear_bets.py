import sys
sys.path.insert(0, 'src')
from algomlb.db.session import get_session_factory
from algomlb.db.models import BankrollLedgerORM
import datetime

s = get_session_factory()()
today = datetime.datetime.now(datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

try:
    bets = s.query(BankrollLedgerORM).filter(BankrollLedgerORM.timestamp >= today).all()
    print(f'Deleting {len(bets)} bets placed today...')
    for b in bets:
        s.delete(b)
    s.commit()
    print('Cleared bets.')
except Exception as e:
    print(e)
finally:
    s.close()
