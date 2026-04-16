from sqlalchemy import text, select
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM
from algomlb.core.logger import logger

def batched_bulk_fix():
    session = get_session_factory()()
    
    logger.info("Starting BATCHED game date correction (Safe Mode)...")
    
    try:
        # 1. Update game_results (Atomic update for all affected games)
        logger.info("Updating game_results...")
        res = session.execute(text("""
            UPDATE game_results 
            SET game_date = (game_datetime AT TIME ZONE 'America/New_York')::date
            WHERE game_datetime IS NOT NULL 
            AND game_date != (game_datetime AT TIME ZONE 'America/New_York')::date;
        """))
        session.commit()
        logger.info(f"Updated and committed {res.rowcount} games in game_results.")

        # 2. Identify all games that were misdated (to use for batched cascades)
        # We look for games where game_date matches the localized datetime date
        # (Since we just fixed game_results, we can join on all games or just find the set of IDs)
        # Actually, it's easier to find games where pitch_events is still out of sync.
        
        # Get list of all game_ids in game_results (to chunk updates)
        game_ids = session.execute(select(GameResultORM.game_id)).scalars().all()
        logger.info(f"Processing cascades for {len(game_ids)} games in batches...")
        
        batch_size = 500
        for i in range(0, len(game_ids), batch_size):
            batch = game_ids[i : i + batch_size]
            
            # Update pitch_events for this batch of games
            session.execute(text("""
                UPDATE pitch_events pe
                SET game_date = gr.game_date
                FROM game_results gr
                WHERE pe.game_id = gr.game_id
                AND gr.game_id IN :batch_ids
                AND pe.game_date != gr.game_date;
            """), {"batch_ids": tuple(batch)})
            
            # Update statcast_raw for this batch of games
            session.execute(text("""
                UPDATE statcast_raw sr
                SET game_date = gr.game_date
                FROM game_results gr
                WHERE sr.game_pk = gr.game_id::bigint
                AND gr.game_id IN :batch_ids
                AND sr.game_date != gr.game_date;
            """), {"batch_ids": tuple(batch)})
            
            session.commit()
            if (i // batch_size) % 10 == 0:
                logger.info(f"Processed {i + len(batch)} / {len(game_ids)} games...")

        logger.info("Batch correction COMPLETE.")
    except Exception as e:
        session.rollback()
        logger.error(f"Batch correction FAILED: {e}")
        raise

if __name__ == "__main__":
    batched_bulk_fix()
