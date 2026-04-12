from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sqlalchemy import and_, or_, select

from algomlb.db.models import BallparkORM, GameResultORM
from algomlb.execution.geography import haversine_distance

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class FatigueCalculator:
    """
    Computes team-level rest days and travel distance.
    Logic aligned with planning_docs/data_strategy.md Tier 3.
    """

    def __init__(self, session: Session):
        self.session = session

    def enrich_batch(self, game_ids: list[str]) -> int:
        """Enrich a batch of specific games (atomic persistence)."""
        count = 0
        for g_id in game_ids:
            if self.calculate_and_save_fatigue(g_id):
                count += 1
        return count

    def calculate_and_save_fatigue(self, game_id: str) -> bool:
        """Computes and persists rest/travel for a single game."""
        stmt = select(GameResultORM).where(GameResultORM.game_id == game_id)
        game = self.session.execute(stmt).scalar_one_or_none()
        if not game:
            return False

        # Home Team Fatigue
        h_rest, h_dist = self._get_fatigue_for_team(
            game.home_team, game.game_date, game.ballpark_id
        )

        # Away Team Fatigue
        a_rest, a_dist = self._get_fatigue_for_team(
            game.away_team, game.game_date, game.ballpark_id
        )

        game.home_rest_days = h_rest
        game.home_travel_distance_km = h_dist
        game.away_rest_days = a_rest
        game.away_travel_distance_km = a_dist

        self.session.commit()
        return True

    def _get_fatigue_for_team(
        self, team_name: str, game_date: any, current_ballpark_id: Optional[int]
    ) -> tuple[int, float]:
        """Finds prev game for team and computes rest/travel."""
        stmt = (
            select(GameResultORM)
            .where(
                and_(
                    or_(
                        GameResultORM.home_team == team_name,
                        GameResultORM.away_team == team_name,
                    ),
                    GameResultORM.game_date < game_date,
                )
            )
            .order_by(
                GameResultORM.game_date.desc(), GameResultORM.game_datetime.desc()
            )
            .limit(1)
        )

        prev_game = self.session.execute(stmt).scalar_one_or_none()

        if not prev_game:
            # First game of season or no history
            return 0, 0.0

        # Rest Days
        rest_days = (game_date - prev_game.game_date).days

        # Only reset on season boundary (~March/April) or major gaps
        # MLB offseason is > 100 days.
        if rest_days > 60:
            return 0, 0.0

        # Travel Distance
        if not current_ballpark_id or not prev_game.ballpark_id:
            return rest_days, 0.0

        if current_ballpark_id == prev_game.ballpark_id:
            return rest_days, 0.0

        # Fetch coords
        p1 = self.session.get(BallparkORM, prev_game.ballpark_id)
        p2 = self.session.get(BallparkORM, current_ballpark_id)

        if not p1 or not p2 or p1.latitude is None or p2.latitude is None:
            return rest_days, 0.0

        dist = haversine_distance(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
        return rest_days, dist
