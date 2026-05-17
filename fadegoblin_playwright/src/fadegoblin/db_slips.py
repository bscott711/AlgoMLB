import json
from sqlalchemy import create_engine, text
from fadegoblin import config

def get_engine():
    return create_engine(config.DATABASE_URL)

def init_db():
    """Initializes the fadegoblin_slips database table if it does not exist."""
    engine = get_engine()
    query = """
    CREATE TABLE IF NOT EXISTS fadegoblin_slips (
        slip_id SERIAL PRIMARY KEY,
        slip_type VARCHAR(20) NOT NULL, -- 'degen' or 'potd'
        legs JSONB NOT NULL,            -- [{"game": "away @ home", "pick": "pick", "odds": -110, "game_id": "12345"}]
        final_odds VARCHAR(20) NOT NULL,
        stake NUMERIC(10, 2) NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'PENDING', -- 'PENDING', 'SETTLED'
        pnl NUMERIC(10, 2),
        bsky_uri VARCHAR(255),
        bsky_cid VARCHAR(255),
        twitter_tweet_id VARCHAR(255),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        settled_at TIMESTAMP WITH TIME ZONE
    );
    """
    with engine.begin() as conn:
        conn.execute(text(query))
    print("✅ initialized fadegoblin_slips table.")

def save_slip(
    slip_type: str,
    legs: list[dict],
    final_odds: str,
    stake: float,
    bsky_uri: str | None = None,
    bsky_cid: str | None = None,
    twitter_tweet_id: str | None = None
) -> int:
    """Inserts a new slip into the database."""
    engine = get_engine()
    query = """
    INSERT INTO fadegoblin_slips (slip_type, legs, final_odds, stake, status, bsky_uri, bsky_cid, twitter_tweet_id, created_at)
    VALUES (:slip_type, :legs, :final_odds, :stake, 'PENDING', :bsky_uri, :bsky_cid, :twitter_tweet_id, NOW())
    RETURNING slip_id;
    """
    with engine.begin() as conn:
        result = conn.execute(
            text(query),
            {
                "slip_type": slip_type,
                "legs": json.dumps(legs),
                "final_odds": final_odds,
                "stake": stake,
                "bsky_uri": bsky_uri,
                "bsky_cid": bsky_cid,
                "twitter_tweet_id": twitter_tweet_id,
            }
        )
        slip_id = result.scalar()
    print(f"💾 Saved {slip_type.upper()} slip #{slip_id} to DB.")
    return slip_id

def get_pending_slips() -> list[dict]:
    """Retrieves all pending slips from the database."""
    engine = get_engine()
    query = """
    SELECT slip_id, slip_type, legs, final_odds, stake, status, bsky_uri, bsky_cid, twitter_tweet_id, created_at
    FROM fadegoblin_slips
    WHERE status = 'PENDING'
    ORDER BY created_at ASC;
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        
    pending = []
    for r in rows:
        pending.append({
            "slip_id": r[0],
            "slip_type": r[1],
            "legs": r[2] if isinstance(r[2], list) else json.loads(r[2]),
            "final_odds": r[3],
            "stake": float(r[4]),
            "status": r[5],
            "bsky_uri": r[6],
            "bsky_cid": r[7],
            "twitter_tweet_id": r[8],
            "created_at": r[9]
        })
    return pending

def settle_slip(slip_id: int, pnl: float) -> None:
    """Updates slip status to SETTLED and records the final P&L."""
    engine = get_engine()
    query = """
    UPDATE fadegoblin_slips
    SET status = 'SETTLED', pnl = :pnl, settled_at = NOW()
    WHERE slip_id = :slip_id;
    """
    with engine.begin() as conn:
        conn.execute(text(query), {"slip_id": slip_id, "pnl": pnl})
    print(f"⚖️ Settled slip #{slip_id} with P&L: ${pnl:+.2f}")


def has_potd_been_saved(id_str: str) -> bool:
    """Checks if a POTD transaction ID is already saved in the database."""
    engine = get_engine()
    query = """
    SELECT 1 FROM fadegoblin_slips 
    WHERE slip_type = 'potd' 
    AND legs @> :id_json::jsonb 
    LIMIT 1;
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {"id_json": json.dumps([{"id": id_str}])})
        return result.scalar() is not None
