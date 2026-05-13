from algomlb.db.session import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE bankroll_ledger ADD COLUMN selection VARCHAR(100)"))
        print("Added 'selection' column.")
    except Exception as e:
        print(f"Column 'selection' probably exists: {e}")
        
    try:
        conn.execute(text("ALTER TABLE bankroll_ledger ADD COLUMN edge DOUBLE PRECISION"))
        print("Added 'edge' column.")
    except Exception as e:
        print(f"Column 'edge' probably exists: {e}")
    conn.commit()
