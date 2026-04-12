from __future__ import annotations

import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from algomlb.core.logger import logger
from algomlb.db.models import (
    GameLineupORM,
    GameResultORM,
    PlayerRollingFeaturesORM,
)
from algomlb.domain import PlayerRole
from algomlb.ml.monte_carlo.state import (
    BatterSimState,
    PitcherSimState,
)


class MatchupContext(BaseModel):
    """Holds all features and pre-game states for a specific simulation matchup."""

    game_pk: int
    game_date: datetime.date
    home_lineup: List[BatterSimState]
    away_lineup: List[BatterSimState]
    home_starter: PitcherSimState
    away_starter: PitcherSimState

    # Feature vectors for the ML model (BatterID -> FeatureDict)
    batter_features: Dict[int, Dict[str, float]]
    pitcher_features: Dict[int, Dict[str, float]]

    # Context features (Stadium, Weather, etc.)
    game_context: Dict[str, float]


class MatchupLoader:
    """Orchestrates data retrieval from Gold/Silver layers for Monte Carlo simulation."""

    def __init__(self, session: Session):
        self.session = session

    def load_matchup(self, game_pk: int) -> Optional[MatchupContext]:
        """Fetch all necessary data for a specific game PK."""
        logger.info(f"Loading matchup data for game_pk={game_pk}...")

        # 1. Fetch Game Metadata
        game = self.session.execute(
            select(GameResultORM).where(GameResultORM.id == game_pk)
        ).scalar_one_or_none()
        if not game:
            # Try game_id if id fails (mappings can be messy)
            game = self.session.execute(
                select(GameResultORM).where(GameResultORM.game_id == str(game_pk))
            ).scalar_one_or_none()
            if not game:
                logger.error(f"Game {game_pk} not found in database.")
                return None

        # 2. Fetch Lineups
        lineup_rows = (
            self.session.execute(
                select(GameLineupORM)
                .where(GameLineupORM.game_pk == int(game.game_id))
                .order_by(GameLineupORM.batting_order)
            )
            .scalars()
            .all()
        )

        if not lineup_rows:
            logger.warning(
                f"SKIPPING: No starting lineups found for game {game.game_id} in 'game_lineups' table."
            )
            return None

        home_batters = [
            BatterSimState(player_id=r.player_id, player_name=r.player_name)
            for r in lineup_rows
            if r.team_side == "home"
        ]
        away_batters = [
            BatterSimState(player_id=r.player_id, player_name=r.player_name)
            for r in lineup_rows
            if r.team_side == "away"
        ]

        if not home_batters or not away_batters:
            logger.warning(
                f"SKIPPING: Incomplete rosters for game {game.game_id}. Home: {len(home_batters)}, Away: {len(away_batters)}"
            )
            return None

        # 3. Fetch Pitchers
        if not game.home_pitcher_id or not game.away_pitcher_id:
            logger.warning(
                f"SKIPPING: Starting pitchers missing for game {game.game_id} (Home ID: {game.home_pitcher_id}, Away ID: {game.away_pitcher_id})."
            )
            return None

        home_starter = PitcherSimState(
            pitcher_id=game.home_pitcher_id, player_name=game.home_pitcher
        )
        away_starter = PitcherSimState(
            pitcher_id=game.away_pitcher_id, player_name=game.away_pitcher
        )

        # 4. Fetch Rolling Features
        player_ids = [b.player_id for b in (home_batters + away_batters)]
        player_ids.extend([home_starter.pitcher_id, away_starter.pitcher_id])

        feature_rows = (
            self.session.execute(
                select(PlayerRollingFeaturesORM)
                .where(PlayerRollingFeaturesORM.player_id.in_(player_ids))
                .where(PlayerRollingFeaturesORM.game_date == game.game_date)
            )
            .scalars()
            .all()
        )

        if not feature_rows:
            logger.warning(
                f"SKIPPING: No rolling features found for any players in game {game.game_id} on {game.game_date}."
            )
            return None

        batter_features = {}
        pitcher_features = {}

        for row in feature_rows:
            # Convert ORM to dict of numeric features
            feat_dict = self._extract_features(row)
            if row.role == PlayerRole.BATTER:
                batter_features[row.player_id] = feat_dict
            else:
                pitcher_features[row.player_id] = feat_dict

        # 5. Build Context
        context = MatchupContext(
            game_pk=int(game.game_id),
            game_date=game.game_date,
            home_lineup=home_batters,
            away_lineup=away_batters,
            home_starter=home_starter,
            away_starter=away_starter,
            batter_features=batter_features,
            pitcher_features=pitcher_features,
            game_context={
                "temp": game.temperature or 70.0,
                "wind_speed": game.wind_speed or 5.0,
                "is_night": 1.0,  # Placeholder
            },
        )

        logger.success(
            f"Successfully loaded matchup for {game.away_team} @ {game.home_team}"
        )
        return context

    def _extract_features(self, row: PlayerRollingFeaturesORM) -> Dict[str, float]:
        """Extracts all numeric 'roll_' and 'ema_' features from the ORM row."""
        feats = {}
        for col in row.__table__.columns:
            if col.name.startswith(("roll_", "ema_", "std_", "seasonal_")):
                val = getattr(row, col.name)
                feats[col.name] = float(val) if val is not None else 0.0
        return feats
