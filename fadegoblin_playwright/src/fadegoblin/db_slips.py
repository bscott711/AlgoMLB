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
        original_post_text TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        settled_at TIMESTAMP WITH TIME ZONE
    );
    """
    with engine.begin() as conn:
        conn.execute(text(query))
        try:
            conn.execute(
                text(
                    "ALTER TABLE fadegoblin_slips ADD COLUMN IF NOT EXISTS original_post_text TEXT;"
                )
            )
        except Exception as e:
            print(f"⚠️ Warning adding original_post_text column: {e}")
            
        # Create table for tracking posted green slips
        green_slips_query = """
        CREATE TABLE IF NOT EXISTS fadegoblin_posted_green_slips (
            slip_id VARCHAR(255) PRIMARY KEY,
            bookmaker VARCHAR(50) NOT NULL,
            pick_name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """
        conn.execute(text(green_slips_query))
    print("✅ initialized fadegoblin_slips and fadegoblin_posted_green_slips tables.")


def is_slip_posted(slip_id: str) -> bool:
    """Checks if a Fliff/Rebet slip ID has already been posted in a recap."""
    engine = get_engine()
    query = "SELECT 1 FROM fadegoblin_posted_green_slips WHERE slip_id = :slip_id LIMIT 1;"
    with engine.connect() as conn:
        result = conn.execute(text(query), {"slip_id": slip_id})
        return result.scalar() is not None


def record_posted_slip(slip_id: str, bookmaker: str, pick_name: str) -> None:
    """Records a green slip ID so it is never reused for future recaps."""
    engine = get_engine()
    query = """
    INSERT INTO fadegoblin_posted_green_slips (slip_id, bookmaker, pick_name, created_at)
    VALUES (:slip_id, :bookmaker, :pick_name, NOW())
    ON CONFLICT (slip_id) DO NOTHING;
    """
    with engine.begin() as conn:
        conn.execute(text(query), {"slip_id": slip_id, "bookmaker": bookmaker, "pick_name": pick_name})
    print(f"🔒 Recorded {bookmaker} green slip #{slip_id} to prevent reuse.")


def save_slip(
    slip_type: str,
    legs: list[dict],
    final_odds: str,
    stake: float,
    bsky_uri: str | None = None,
    bsky_cid: str | None = None,
    twitter_tweet_id: str | None = None,
    original_post_text: str | None = None,
) -> int:
    """Inserts a new slip into the database."""
    engine = get_engine()
    query = """
    INSERT INTO fadegoblin_slips (slip_type, legs, final_odds, stake, status, bsky_uri, bsky_cid, twitter_tweet_id, original_post_text, created_at)
    VALUES (:slip_type, :legs, :final_odds, :stake, 'PENDING', :bsky_uri, :bsky_cid, :twitter_tweet_id, :original_post_text, NOW())
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
                "original_post_text": original_post_text,
            },
        )
        slip_id = result.scalar()
    print(f"💾 Saved {slip_type.upper()} slip #{slip_id} to DB.")
    return slip_id


def get_pending_slips() -> list[dict]:
    """Retrieves all pending slips from the database."""
    engine = get_engine()
    query = """
    SELECT slip_id, slip_type, legs, final_odds, stake, status, bsky_uri, bsky_cid, twitter_tweet_id, created_at, original_post_text
    FROM fadegoblin_slips
    WHERE status = 'PENDING'
    ORDER BY created_at ASC;
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()

    pending = []
    for r in rows:
        pending.append(
            {
                "slip_id": r[0],
                "slip_type": r[1],
                "legs": r[2] if isinstance(r[2], list) else json.loads(r[2]),
                "final_odds": r[3],
                "stake": float(r[4]),
                "status": r[5],
                "bsky_uri": r[6],
                "bsky_cid": r[7],
                "twitter_tweet_id": r[8],
                "created_at": r[9],
                "original_post_text": r[10] if len(r) > 10 else None,
            }
        )
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
    AND legs @> CAST(:id_json AS jsonb) 
    LIMIT 1;
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {"id_json": json.dumps([{"id": id_str}])})
        return result.scalar() is not None
