from algomlb.db.session import get_session_factory
from algomlb.db.models import Base
from sqlalchemy import text
engine = get_session_factory().kw['bind']
with engine.begin() as conn:
    conn.execute(text('DROP TABLE IF EXISTS pitcher_daily_usage'))
Base.metadata.create_all(engine, tables=[Base.metadata.tables['pitcher_daily_usage']])
print('Table pitcher_daily_usage recreated successfully.')
