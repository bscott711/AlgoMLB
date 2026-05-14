from algomlb.db.session import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    try:
        # Drop the old one if it exists
        conn.execute(text("ALTER TABLE model_predictions DROP COLUMN IF EXISTS market_price_at_prediction"))
        print("Dropped old column.")
    except Exception as e:
        print(f"Note: {e}")
        
    try:
        # Add the new one if it doesn't exist
        conn.execute(text("ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS market_home_implied_at_prediction DOUBLE PRECISION"))
        print("Added new column.")
    except Exception as e:
        print(f"Note: {e}")
    
    conn.commit()
print("Migration Complete.")
