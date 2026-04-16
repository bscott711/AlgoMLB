from sqlalchemy import text, select
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM
from algomlb.core.logger import logger

def safe_propagation():
    session = get_session_factory()()
    
    logger.info("Starting SAFE (Point-Update) game date propagation...")
    
    try:
        # 1. Get the mapping of already-fixed game dates
        # We only need to propagate for games where pitch_events or statcast_raw are out of sync.
        query = text("""
            SELECT game_id, game_date 
            FROM game_results 
            WHERE game_datetime IS NOT NULL
        """)
        mappings = session.execute(query).fetchall()
        logger.info(f"Found {len(mappings)} games in game_results to check for propagation.")

        count = 0
        for row in mappings:
            gid = row.game_id
            new_date = row.game_date
            
            # Update pitch_events (using Primary Key index)
            session.execute(text("""
                UPDATE pitch_events 
                SET game_date = :new_date 
                WHERE game_id = :gid 
                AND game_date != :new_date
            """), {"gid": gid, "new_date": new_date})
            
            # Update statcast_raw (using Primary Key index)
            # We use gid::bigint because statcast_raw.game_pk is BigInt
            try:
                gpk = int(gid)
                session.execute(text("""
                    UPDATE statcast_raw 
                    SET game_date = :new_date 
                    WHERE game_pk = :gpk 
                    AND game_date != :new_date
                """), {"gpk": gpk, "new_date": new_date})
            except ValueError:
                pass # Skip non-numeric game_ids
            
            count += 1
            if count % 50 == 0:
                session.commit()
                logger.info(f"Propagated {count} / {len(mappings)} games...")
        
        session.commit()
        logger.info("Safe propagation COMPLETE.")
    except Exception as e:
        session.rollback()
        logger.error(f"Safe propagation FAILED: {e}")
        raise

if __name__ == "__main__":
    safe_propagation()
