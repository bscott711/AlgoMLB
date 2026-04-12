import datetime
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any, cast

from pydantic import BaseModel
from sqlalchemy import select, and_, or_, func, distinct, desc
from sqlalchemy.orm import Session

from algomlb.core.logger import logger
from algomlb.db.models import (
    GameLineupORM,
    GameResultORM,
    PlayerRollingFeaturesORM,
    TeamManagerORM,
    ManagerHookProfileORM,
    StatcastPlayerGameLog,
    StatcastRawORM,
)
from algomlb.domain import PlayerRole
from algomlb.ml.monte_carlo.state import (
    BatterSimState,
    PitcherSimState,
    ManagerHookProfile,
)
from algomlb.ingestion.lineup_ingester import LineupIngester


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

    # Bullpen & Reliever features
    home_relievers: List[PitcherSimState]
    away_relievers: List[PitcherSimState]

    # Manager Data
    home_manager_id: Optional[int] = None
    away_manager_id: Optional[int] = None
    manager_profiles: Dict[int, ManagerHookProfile] = {}

    # Context features (Stadium, Weather, etc.)
    game_context: Dict[str, float]


class MatchupLoader:
    """Orchestrates data retrieval from Gold/Silver layers for Monte Carlo simulation."""

    def __init__(self, session: Session):
        self.session = session

    def load_matchup(self, game_pk: int) -> MatchupContext:
        """Fetch all necessary data for a specific game PK. Raises ValueError on failure."""
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
                raise ValueError(f"Game {game_pk} not found in database.")

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
            # Fallback: Try to ingest lineups on-demand
            logger.info(f"Lineups missing for game_pk={game.game_id}. Attempting on-demand ingestion...")
            ingester = LineupIngester(self.session)
            ingester.ingest_game(int(game.game_id), game.game_date)
            
            # Retry fetching from DB
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
            # Fallback 2: Try to fetch projected lineups from historical games
            logger.warning(f"Lineups unavailable for game {game.game_id}. Using historical projections...")
            home_lineup_rows = self._fetch_projected_lineup(game.home_team_id, game.game_date)
            away_lineup_rows = self._fetch_projected_lineup(game.away_team_id, game.game_date)
            
            if not home_lineup_rows or not away_lineup_rows:
                raise ValueError(
                    f"No starting lineups (official or projected) found for game {game.game_id} across {game.away_team} @ {game.home_team}. "
                    "Lineups may not be announced yet or are unavailable."
                )
            
            # Combine them for processing (SimState factory expects a flat list)
            lineup_rows = home_lineup_rows + away_lineup_rows

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
            raise ValueError(f"Incomplete rosters for game {game.game_id}. Home: {len(home_batters)}, Away: {len(away_batters)}")

        # 3. Fetch Pitchers
        h_sp_id = game.home_pitcher_id
        h_sp_name = game.home_pitcher
        a_sp_id = game.away_pitcher_id
        a_sp_name = game.away_pitcher

        if not h_sp_id or not a_sp_id:
            logger.warning(f"Starting pitchers missing for game {game.game_id}. Attempting projections...")
            if not h_sp_id:
                proj_h = self._fetch_projected_pitcher(game.home_team_id, game.game_date)
                if proj_h:
                    h_sp_id, h_sp_name = proj_h
            if not a_sp_id:
                proj_a = self._fetch_projected_pitcher(game.away_team_id, game.game_date)
                if proj_a:
                    a_sp_id, a_sp_name = proj_a

        if not h_sp_id or not a_sp_id:
            raise ValueError(f"Starting pitchers missing for game {game.game_id} and no historical fallback available.")

        home_starter = PitcherSimState(
            pitcher_id=h_sp_id, player_name=h_sp_name
        )
        away_starter = PitcherSimState(
            pitcher_id=a_sp_id, player_name=a_sp_name
        )

        # 4. Fetch Rolling Features (Latest available up to game_date)
        player_ids = [b.player_id for b in (home_batters + away_batters)]
        player_ids.extend([h_sp_id, a_sp_id])

        # We fetch the latest feature record for each player that is <= game.game_date
        # This is more resilient than strictly matching the exact game_date.
        from sqlalchemy import and_, func

        subq = (
            select(
                PlayerRollingFeaturesORM.player_id,
                func.max(PlayerRollingFeaturesORM.game_date).label("latest_date"),
            )
            .where(PlayerRollingFeaturesORM.player_id.in_(player_ids))
            .where(PlayerRollingFeaturesORM.game_date <= game.game_date)
            .group_by(PlayerRollingFeaturesORM.player_id)
            .subquery()
        )

        stmt = select(PlayerRollingFeaturesORM).join(
            subq,
            and_(
                PlayerRollingFeaturesORM.player_id == subq.c.player_id,
                PlayerRollingFeaturesORM.game_date == subq.c.latest_date,
            ),
        )

        feature_rows = self.session.execute(stmt).scalars().all()

        if not feature_rows:
            raise ValueError(f"No rolling features found for any players in game {game.game_id} on or before {game.game_date}.")

        batter_features = {}
        pitcher_features = {}

        for row in feature_rows:
            # Convert ORM to dict of numeric features
            feat_dict = self._extract_features(row)
            if row.role == PlayerRole.BATTER:
                batter_features[row.player_id] = feat_dict
            else:
                pitcher_features[row.player_id] = feat_dict

        # 5. Fetch Managers & Hook Profiles
        h_mgr_id, a_mgr_id = self._fetch_manager_ids(game)
        mgr_profiles = self._fetch_hook_profiles([h_mgr_id, a_mgr_id], game.game_date.year)

        # 6. Fetch Bullpens (Targeting top 5 relievers by rolling workload/availability)
        home_relievers = self._fetch_bullpen(game.home_team_id, game.game_date, pitcher_features, game.home_pitcher_id, game.away_pitcher_id)
        away_relievers = self._fetch_bullpen(game.away_team_id, game.game_date, pitcher_features, game.home_pitcher_id, game.away_pitcher_id)

        # 7. Build Context
        context = MatchupContext(
            game_pk=int(game.game_id),
            game_date=game.game_date,
            home_lineup=home_batters,
            away_lineup=away_batters,
            home_starter=home_starter,
            away_starter=away_starter,
            batter_features=batter_features,
            pitcher_features=pitcher_features,
            home_relievers=home_relievers,
            away_relievers=away_relievers,
            home_manager_id=h_mgr_id,
            away_manager_id=a_mgr_id,
            manager_profiles=mgr_profiles,
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
            if col.name.startswith(("roll_", "ema_", "std_", "seasonal_", "fatigue_", "delta_")):
                val = getattr(row, col.name)
                feats[col.name] = float(val) if val is not None else 0.0
        return feats

    def _fetch_manager_ids(self, game: GameResultORM) -> Tuple[Optional[int], Optional[int]]:
        """Find the active manager ID for both teams from the lookup table."""
        h_mgr = self.session.execute(
            select(TeamManagerORM.manager_id).where(
                TeamManagerORM.team_id == game.home_team_id,
                TeamManagerORM.season == game.game_date.year
            )
        ).scalar()
        a_mgr = self.session.execute(
            select(TeamManagerORM.manager_id).where(
                TeamManagerORM.team_id == game.away_team_id,
                TeamManagerORM.season == game.game_date.year
            )
        ).scalar()
        return h_mgr, a_mgr

    def _fetch_hook_profiles(self, manager_ids: List[int], season: int) -> Dict[int, ManagerHookProfile]:
        """Load aggregated manager hook tendencies."""
        ids = [m for m in manager_ids if m is not None]
        if not ids:
            return {}
        
        rows = self.session.execute(
            select(ManagerHookProfileORM).where(
                ManagerHookProfileORM.manager_id.in_(ids)
            )
        ).scalars().all()

        profiles = {}
        for r in rows:
            profiles[r.manager_id] = ManagerHookProfile(
                manager_id=r.manager_id,
                manager_name=r.manager_name,
                avg_sp_pitch_count=float(r.avg_sp_pitch_count or 90.0),
                pull_before_3rd_tto_pct=float(r.pull_before_3rd_tto_pct or 0.1),
                pull_with_lead_pct=float(r.pull_with_lead_pct or 0.5),
                bullpen_protective_pct=float(r.bullpen_protective_pct or 0.2)
            )
        return profiles

    def _fetch_bullpen(self, team_id: int, game_date: date, existing_features: Dict[int, Dict[str, float]], h_starter_id: int, a_starter_id: int) -> List[PitcherSimState]:
        """
        Retrieves the top available relievers for a team by checking recent game logs.
        """
        # 1. Get the team abbreviation for the team_id
        team_meta = self.session.execute(
            select(TeamManagerORM.team_abbr).where(TeamManagerORM.team_id == team_id).limit(1)
        ).scalar()
        
        if not team_meta:
            return []

        # 2. Find all pitchers who played for this team recently in statcast_raw
        # This is a bit heavy but avoids the problematic string-casting join on game_pk
        p_stmt = select(distinct(StatcastRawORM.pitcher)).where(
            or_(StatcastRawORM.home_team == team_meta, StatcastRawORM.away_team == team_meta),
            StatcastRawORM.game_date >= game_date - timedelta(days=20),
            StatcastRawORM.game_date < game_date
        )
        
        p_ids = self.session.execute(p_stmt).scalars().all()
        
        # 3. Fetch latest rolling features for these potential relievers
        relievers = []
        for pid in p_ids:
            if len(relievers) >= 6:
                break
                
            # Exclude the starting pitcher
            if pid == h_starter_id or pid == a_starter_id:
                continue

            # Fetch the latest PITCHER features for this arm
            stmt = select(PlayerRollingFeaturesORM).where(
                PlayerRollingFeaturesORM.player_id == pid,
                PlayerRollingFeaturesORM.game_date <= game_date,
                PlayerRollingFeaturesORM.role == PlayerRole.PITCHER
            ).order_by(PlayerRollingFeaturesORM.game_date.desc()).limit(1)
            
            r = self.session.execute(stmt).scalar()
            if r:
                relievers.append(PitcherSimState(pitcher_id=r.player_id))
                if r.player_id not in existing_features:
                    existing_features[r.player_id] = self._extract_features(r)
                    
        return relievers

    def _fetch_projected_lineup(self, team_id: int, current_date: date) -> List[GameLineupORM]:
        """Find the most recent starting lineup for a team with at least 9 slots."""
        if team_id is None:
            return []

        # Find recent games for this team
        stmt = (
            select(GameResultORM)
            .where(or_(GameResultORM.home_team_id == team_id, GameResultORM.away_team_id == team_id))
            .where(GameResultORM.status == "COMPLETED")
            .where(GameResultORM.game_date < current_date)
            .order_by(desc(GameResultORM.game_date))
            .limit(10)  # Check last 10 games to find one with full lineup data
        )
        recent_games = self.session.execute(stmt).scalars().all()

        for g in recent_games:
            side = "home" if g.home_team_id == team_id else "away"
            lineup = self.session.execute(
                select(GameLineupORM)
                .where(GameLineupORM.game_pk == int(g.game_id))
                .where(GameLineupORM.team_side == side)
                .order_by(GameLineupORM.batting_order)
            ).scalars().all()

            if len(lineup) >= 9:
                return lineup
        return []

    def _fetch_projected_pitcher(self, team_id: int, current_date: date) -> Optional[Tuple[int, str]]:
        """Find the most recent starting pitcher for a team."""
        if team_id is None:
            return None

        stmt = (
            select(GameResultORM)
            .where(or_(GameResultORM.home_team_id == team_id, GameResultORM.away_team_id == team_id))
            .where(GameResultORM.status == "COMPLETED")
            .where(GameResultORM.game_date < current_date)
            .order_by(desc(GameResultORM.game_date))
            .limit(5)
        )
        recent_games = self.session.execute(stmt).scalars().all()

        for g in recent_games:
            # Check who started for the team in this historical game
            if g.home_team_id == team_id and g.home_pitcher_id:
                return g.home_pitcher_id, g.home_pitcher
            elif g.away_team_id == team_id and g.away_pitcher_id:
                return g.away_pitcher_id, g.away_pitcher
        
        return None


MatchupContext.model_rebuild()
