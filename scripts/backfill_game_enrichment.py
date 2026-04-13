import logging
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import select

from algomlb.db.models import (
    BallparkORM,
    GameResultORM,
    OpenMeteoWeatherProgressionORM,
)
from algomlb.db.repository import DatabaseRepository
from algomlb.db.session import get_session_factory
from algomlb.core.geography import haversine_distance

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def backfill_enrichment():
    """
    Optimized backfill script for enriching game_results.
    Uses an in-memory state tracker for fatigue to achieve O(N) performance.
    """
    log.info("Starting historical enrichment backfill...")

    factory = get_session_factory()
    with factory() as session:
        repo = DatabaseRepository(session)

        # 1. Fetch All Ballpark Coordinates (for cache)
        log.info("Caching ballpark coordinates...")
        bp_stmt = select(BallparkORM)
        ballparks = {bp.id: bp for bp in session.execute(bp_stmt).scalars().all()}

        # 2. Map Existing Weather Progression
        log.info("Mapping existing weather actuals (T0)...")
        w_stmt = select(
            OpenMeteoWeatherProgressionORM.game_id,
            OpenMeteoWeatherProgressionORM.temp_t0_f,
            OpenMeteoWeatherProgressionORM.wind_speed_t0,
            OpenMeteoWeatherProgressionORM.humidity_t0,
        )
        weather_results = session.execute(w_stmt).all()
        w_map = {row[0]: row for row in weather_results}

        # 3. Fetch All Games Chronologically
        log.info("Fetching game results spine...")
        g_stmt = select(GameResultORM).order_by(
            GameResultORM.game_date, GameResultORM.game_datetime
        )
        games = session.execute(g_stmt).scalars().all()

        # 4. Process Chronologically
        # team_state[team_name] = (last_game_date, last_ballpark_id)
        team_state: Dict[str, Tuple[Any, Optional[int]]] = {}
        updates = []
        chunk_size = 1000

        log.info(f"Processing {len(games)} games...")
        for i, g in enumerate(games):
            # -- Fatigue Logic (In-Memory Tracking) --

            # Home Team
            h_last_date, h_last_bp = team_state.get(g.home_team, (None, None))
            h_rest, h_dist = _compute_fatigue(g, h_last_date, h_last_bp, ballparks)
            team_state[g.home_team] = (g.game_date, g.ballpark_id)

            # Away Team
            a_last_date, a_last_bp = team_state.get(g.away_team, (None, None))
            a_rest, a_dist = _compute_fatigue(g, a_last_date, a_last_bp, ballparks)
            team_state[g.away_team] = (g.game_date, g.ballpark_id)

            # -- Weather Logic --
            w = w_map.get(g.game_id)
            temp = w[1] if w else None
            wind = w[2] if w else None
            hum = w[3] if w else None

            updates.append(
                {
                    "game_id": g.game_id,
                    "temperature": temp,
                    "wind_speed": wind,
                    "humidity": hum,
                    "home_rest_days": h_rest,
                    "home_travel_distance_km": h_dist,
                    "away_rest_days": a_rest,
                    "away_travel_distance_km": a_dist,
                }
            )

            if len(updates) >= chunk_size:
                processed = repo.update_game_enrichment(updates)
                log.info(
                    f"Progress: {i + 1}/{len(games)} (Last chunk updated {processed})"
                )
                updates = []

        if updates:
            repo.update_game_enrichment(updates)

    log.info("✅ Backfill enrichment complete!")


def _compute_fatigue(game, last_date, last_bp_id, ballparks_cache) -> Tuple[int, float]:
    """Helper for the optimized backfill loop."""
    if not last_date:
        return 0, 0.0

    # Rest days
    rest = (game.game_date - last_date).days
    if rest > 60:  # Season gap
        return 0, 0.0

    # Distance
    if not game.ballpark_id or not last_bp_id or game.ballpark_id == last_bp_id:
        return rest, 0.0

    bp1 = ballparks_cache.get(last_bp_id)
    bp2 = ballparks_cache.get(game.ballpark_id)

    if not bp1 or not bp2 or bp1.latitude is None or bp2.latitude is None:
        return rest, 0.0

    dist = haversine_distance(bp1.latitude, bp1.longitude, bp2.latitude, bp2.longitude)
    return rest, dist


if __name__ == "__main__":
    backfill_enrichment()
