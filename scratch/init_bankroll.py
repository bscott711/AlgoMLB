import uuid
import datetime
from algomlb.db.session import get_session_factory
from algomlb.db.models import BankrollLedgerORM
from algomlb.domain import TransactionStatus

session = get_session_factory()()
try:
    # Check if we already have an adjustment
    exists = (
        session.query(BankrollLedgerORM)
        .filter(BankrollLedgerORM.status == TransactionStatus.ADJUSTMENT)
        .first()
    )
    if not exists:
        initial = BankrollLedgerORM(
            transaction_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(datetime.UTC),
            stake=100.0,
            odds=1.0,
            selection="INITIAL_SEED",
            edge=0.0,
            status=TransactionStatus.ADJUSTMENT,
            pnl=100.0,
            game_id=None,
        )
        session.add(initial)
        session.commit()
        print("Initialized bankroll with $100.")
    else:
        print("Bankroll already initialized.")
except Exception as e:
    print(f"Error: {e}")
finally:
    session.close()
