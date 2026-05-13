from sqlalchemy import inspect
from algomlb.db.session import get_engine

engine = get_engine()
inspector = inspect(engine)
tables = inspector.get_table_names()
print("Tables:", [t for t in tables if "odds" in t or "market" in t])
