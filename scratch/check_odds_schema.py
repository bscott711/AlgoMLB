from sqlalchemy import inspect
from algomlb.db.session import get_engine

engine = get_engine()
inspector = inspect(engine)
columns = inspector.get_columns("live_odds")
for col in columns:
    print(f"{col['name']}: {col['type']}")
