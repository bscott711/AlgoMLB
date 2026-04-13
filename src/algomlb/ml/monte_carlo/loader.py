import datetime
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

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
    StatcastRawORM,
    TeamEloHistoryORM,
    TeamSabermetricsHistoryORM,
)
from algomlb.domain import PlayerRole, GameStatus
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
    
    # Global Matchup Features (Elo, Pythag, etc.)
    matchup_features: Dict[str, float] = {}

    # Projection Metadata
    home_sp_projected: bool = False
    away_sp_projected: bool = False


class MatchupLoader:
    """Orchestrates data retrieval from Gold/Silver layers for Monte Carlo simulation."""

    def __init__(self, session: Session):
        self.session = session

    def load_matchup(self, game_pk: int) -> MatchupContext:
        """Fetch all necessary data for a specific game PK. Raises ValueError on failure."""
        logger.info(f"Loading matchup data for game_pk={game_pk}...")

        # 1. Fetch Game Metadata
        game = self._get_game_metadata(game_pk)

        # 2. Fetch Lineups
        home_batters, away_batters = self._prepare_lineups(game)

        # 3. Fetch Pitchers
        home_starter, away_starter, h_proj, a_proj = self._prepare_starting_pitchers(
            game
        )

        # 4. Fetch Rolling Features (Latest available up to game_date)
        batter_features, pitcher_features = self._fetch_all_rolling_features(
            game,
            home_batters,
            away_batters,
            home_starter.pitcher_id,
            away_starter.pitcher_id,
        )

        # 5. Fetch Global Matchup Features (Elo, Pythag)
        matchup_feats = self._fetch_global_features(game)

        # 6. Fetch Managers & Hook Profiles
        h_mgr_id, a_mgr_id = self._fetch_manager_ids(game)
        active_mgr_ids = [m for m in [h_mgr_id, a_mgr_id] if m is not None]
        mgr_profiles = self._fetch_hook_profiles(active_mgr_ids, game.game_date.year)

        # 7. Fetch Bullpens
        if game.home_team_id is None or game.away_team_id is None:
            raise ValueError(f"Game {game.game_id} missing team IDs for bullpen fetch.")

        home_relievers = self._fetch_bullpen(
            game.home_team_id,
            game.game_date,
            pitcher_features,
            home_starter.pitcher_id,
            away_starter.pitcher_id,
        )
        away_relievers = self._fetch_bullpen(
            game.away_team_id,
            game.game_date,
            pitcher_features,
            home_starter.pitcher_id,
            away_starter.pitcher_id,
        )

        # 8. Build Context
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
                "is_night": 1.0,
            },
            matchup_features=matchup_feats,
            home_sp_projected=h_proj,
            away_sp_projected=a_proj,
        )

        logger.success(
            f"Successfully loaded matchup for {game.away_team} @ {game.home_team}"
        )
        return context

    def _get_game_metadata(self, game_pk: int) -> GameResultORM:
        """Fetch basic game info, trying both ID and Game PK string."""
        game = self.session.execute(
            select(GameResultORM).where(GameResultORM.id == game_pk)
        ).scalar_one_or_none()
        if not game:
            game = self.session.execute(
                select(GameResultORM).where(GameResultORM.game_id == str(game_pk))
            ).scalar_one_or_none()
        if not game:
            raise ValueError(f"Game {game_pk} not found in database.")
        return game

    def _prepare_lineups(
        self, game: GameResultORM
    ) -> Tuple[List[BatterSimState], List[BatterSimState]]:
        """Fetch or project starting lineups for both teams."""
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
            logger.info(
                f"Lineups missing for game_pk={game.game_id}. Attempting on-demand ingestion..."
            )
            LineupIngester(self.session).ingest_game(int(game.game_id), game.game_date)
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
                f"Lineups unavailable for game {game.game_id}. Using historical projections..."
            )
            if game.home_team_id is None or game.away_team_id is None:
                raise ValueError(f"Game {game.game_id} is missing team IDs.")

            h_rows = self._fetch_projected_lineup(game.home_team_id, game.game_date)
            a_rows = self._fetch_projected_lineup(game.away_team_id, game.game_date)

            if not h_rows or not a_rows:
                h_name = game.home_team
                a_name = game.away_team
                missing = []
                if not h_rows:
                    missing.append(f"Home ({h_name})")
                if not a_rows:
                    missing.append(f"Away ({a_name})")
                raise ValueError(
                    f"Incomplete rosters for game {game.game_id}. Missing data for: {', '.join(missing)}. Try a different matchup or check database sync."
                )

            # Use the rows directly for each side, bypassing team_side filtering from historical records
            home_batters = [
                BatterSimState(player_id=r.player_id, player_name=r.player_name)
                for r in h_rows
            ]
            away_batters = [
                BatterSimState(player_id=r.player_id, player_name=r.player_name)
                for r in a_rows
            ]
            return home_batters, away_batters

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
            h_name = game.home_team
            a_name = game.away_team
            missing = []
            if not home_batters:
                missing.append(f"Home ({h_name})")
            if not away_batters:
                missing.append(f"Away ({a_name})")
            raise ValueError(
                f"Incomplete rosters for game {game.game_id}. Missing data for: {', '.join(missing)}. Try a different matchup or check database sync."
            )
        return home_batters, away_batters

    def _prepare_starting_pitchers(
        self, game: GameResultORM
    ) -> Tuple[PitcherSimState, PitcherSimState, bool, bool]:
        """Fetch or project starting pitchers for both teams."""
        h_sp_id = game.home_pitcher_id
        h_sp_name = game.home_pitcher
        a_sp_id = game.away_pitcher_id
        a_sp_name = game.away_pitcher
        h_projected = False
        a_projected = False

        if not h_sp_id or not a_sp_id:
            logger.warning(
                f"Starting pitchers missing for game {game.game_id}. Attempting projections..."
            )
            if not h_sp_id and game.home_team_id is not None:
                res = self._fetch_rotation_projection(game.home_team_id, game.game_date)
                if res:
                    h_sp_id, h_sp_name = res
                    h_projected = True
            if not a_sp_id and game.away_team_id is not None:
                res = self._fetch_rotation_projection(game.away_team_id, game.game_date)
                if res:
                    a_sp_id, a_sp_name = res
                    a_projected = True

        if not h_sp_id or not a_sp_id:
            raise ValueError(f"Starting pitchers missing for game {game.game_id}.")

        return (
            PitcherSimState(pitcher_id=h_sp_id, player_name=h_sp_name),
            PitcherSimState(pitcher_id=a_sp_id, player_name=a_sp_name),
            h_projected,
            a_projected,
        )

    def _fetch_all_rolling_features(
        self,
        game: GameResultORM,
        home_batters: List[BatterSimState],
        away_batters: List[BatterSimState],
        home_starter_id: int,
        away_starter_id: int,
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
        """Bulk fetch rolling features for all relevant players."""
        player_ids = [b.player_id for b in (home_batters + away_batters)]
        player_ids.extend([home_starter_id, away_starter_id])

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
            raise ValueError(f"No rolling features found for game {game.game_id}.")

        bat_feats, pit_feats = {}, {}
        for row in feature_rows:
            feats = self._extract_features(row)
            if row.role == PlayerRole.BATTER:
                bat_feats[row.player_id] = feats
            else:
                pit_feats[row.player_id] = feats
        return bat_feats, pit_feats

    def _extract_features(self, row: PlayerRollingFeaturesORM) -> Dict[str, float]:
        """Extracts all numeric 'roll_' and 'ema_' features from the ORM row."""
        feats = {}
        for col in row.__table__.columns:
            if col.name.startswith(
                ("roll_", "ema_", "std_", "seasonal_", "fatigue_", "delta_")
            ):
                val = getattr(row, col.name)
                feats[col.name] = float(val) if val is not None else 0.0
        return feats

    def _fetch_manager_ids(
        self, game: GameResultORM
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find the active manager ID for both teams from the lookup table."""
        h_mgr = self.session.execute(
            select(TeamManagerORM.manager_id).where(
                TeamManagerORM.team_id == game.home_team_id,
                TeamManagerORM.season == game.game_date.year,
            )
        ).scalar()
        a_mgr = self.session.execute(
            select(TeamManagerORM.manager_id).where(
                TeamManagerORM.team_id == game.away_team_id,
                TeamManagerORM.season == game.game_date.year,
            )
        ).scalar()
        return h_mgr, a_mgr

    def _fetch_hook_profiles(
        self, manager_ids: List[int], season: int
    ) -> Dict[int, ManagerHookProfile]:
        """Load aggregated manager hook tendencies."""
        ids = [m for m in manager_ids if m is not None]
        if not ids:
            return {}

        rows = (
            self.session.execute(
                select(ManagerHookProfileORM).where(
                    ManagerHookProfileORM.manager_id.in_(ids)
                )
            )
            .scalars()
            .all()
        )

        profiles = {}
        for r in rows:
            profiles[r.manager_id] = ManagerHookProfile(
                manager_id=r.manager_id,
                manager_name=r.manager_name,
                avg_sp_pitch_count=float(r.avg_sp_pitch_count or 90.0),
                pull_before_3rd_tto_pct=float(r.pull_before_3rd_tto_pct or 0.1),
                pull_with_lead_pct=float(r.pull_with_lead_pct or 0.5),
                bullpen_protective_pct=float(r.bullpen_protective_pct or 0.2),
            )
        return profiles

    def _fetch_bullpen(
        self,
        team_id: int,
        game_date: date,
        existing_features: Dict[int, Dict[str, float]],
        h_starter_id: int,
        a_starter_id: int,
    ) -> List[PitcherSimState]:
        """
        Retrieves the top available relievers for a team by checking recent game logs.
        """
        # 1. Get the team abbreviation for the team_id
        team_meta = self.session.execute(
            select(TeamManagerORM.team_abbr)
            .where(TeamManagerORM.team_id == team_id)
            .limit(1)
        ).scalar()

        if not team_meta:
            return []

        # 2. Find all pitchers who played for this team recently in statcast_raw
        # This is a bit heavy but avoids the problematic string-casting join on game_pk
        p_stmt = select(distinct(StatcastRawORM.pitcher)).where(
            or_(
                StatcastRawORM.home_team == team_meta,
                StatcastRawORM.away_team == team_meta,
            ),
            StatcastRawORM.game_date >= game_date - timedelta(days=20),
            StatcastRawORM.game_date < game_date,
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
            stmt = (
                select(PlayerRollingFeaturesORM)
                .where(
                    PlayerRollingFeaturesORM.player_id == pid,
                    PlayerRollingFeaturesORM.game_date <= game_date,
                    PlayerRollingFeaturesORM.role == PlayerRole.PITCHER,
                )
                .order_by(PlayerRollingFeaturesORM.game_date.desc())
                .limit(1)
            )

            r = self.session.execute(stmt).scalar()
            if r:
                relievers.append(PitcherSimState(pitcher_id=r.player_id))
                if r.player_id not in existing_features:
                    existing_features[r.player_id] = self._extract_features(r)

        return relievers

    def _fetch_global_features(self, game: GameResultORM) -> Dict[str, float]:
        """Fetch materialized team-level features (Elo, Pythag) up to the game date."""
        home_team = str(game.home_team)
        away_team = str(game.away_team)
        game_date = game.game_date

        feats: Dict[str, float] = {}

        # 1. Fetch Elo (pre-game)
        h_elo = (
            self.session.query(TeamEloHistoryORM)
            .filter(
                TeamEloHistoryORM.team_id == home_team,
                TeamEloHistoryORM.game_date < game_date,
            )
            .order_by(TeamEloHistoryORM.game_date.desc(), TeamEloHistoryORM.id.desc())
            .first()
        )
        a_elo = (
            self.session.query(TeamEloHistoryORM)
            .filter(
                TeamEloHistoryORM.team_id == away_team,
                TeamEloHistoryORM.game_date < game_date,
            )
            .order_by(TeamEloHistoryORM.game_date.desc(), TeamEloHistoryORM.id.desc())
            .first()
        )

        h_elo_val = h_elo.elo_post if h_elo else 1500.0
        a_elo_val = a_elo.elo_post if a_elo else 1500.0

        feats["home_team_elo_pre"] = h_elo_val
        feats["away_team_elo_pre"] = a_elo_val
        feats["elo_diff"] = h_elo_val - a_elo_val

        # 2. Fetch Sabermetrics (pre-game)
        h_saber = (
            self.session.query(TeamSabermetricsHistoryORM)
            .filter(
                TeamSabermetricsHistoryORM.team_id == home_team,
                TeamSabermetricsHistoryORM.game_date < game_date,
            )
            .order_by(
                TeamSabermetricsHistoryORM.game_date.desc(),
                TeamSabermetricsHistoryORM.id.desc(),
            )
            .first()
        )
        a_saber = (
            self.session.query(TeamSabermetricsHistoryORM)
            .filter(
                TeamSabermetricsHistoryORM.team_id == away_team,
                TeamSabermetricsHistoryORM.game_date < game_date,
            )
            .order_by(
                TeamSabermetricsHistoryORM.game_date.desc(),
                TeamSabermetricsHistoryORM.id.desc(),
            )
            .first()
        )

        # Home
        feats["h_pythag_win_pct"] = h_saber.pythag_win_pct if h_saber else 0.5
        feats["h_roll_run_diff"] = h_saber.roll_run_diff if h_saber else 0.0
        feats["h_roll_rs_per_game"] = h_saber.roll_rs_per_game if h_saber else 4.5
        feats["h_roll_ra_per_game"] = h_saber.roll_ra_per_game if h_saber else 4.5

        # Away
        feats["a_pythag_win_pct"] = a_saber.pythag_win_pct if a_saber else 0.5
        feats["a_roll_run_diff"] = a_saber.roll_run_diff if a_saber else 0.0
        feats["a_roll_rs_per_game"] = a_saber.roll_rs_per_game if a_saber else 4.5
        feats["a_roll_ra_per_game"] = a_saber.roll_ra_per_game if a_saber else 4.5

        # Derived
        feats["pythag_diff"] = feats["h_pythag_win_pct"] - feats["a_pythag_win_pct"]

        return feats

    def _fetch_rotation_projection(
        self, team_id: int, current_date: date
    ) -> Optional[Tuple[int, str]]:
        """
        Identifies the starting rotation sequence and projects the 'Next Man Up'.
        Returns (pitcher_id, pitcher_name).
        """
        if team_id is None:
            return None

        # 1. Fetch last 20 games chronologically to see the pattern
        stmt = (
            select(GameResultORM)
            .where(
                or_(
                    GameResultORM.home_team_id == team_id,
                    GameResultORM.away_team_id == team_id,
                )
            )
            .where(
                GameResultORM.status.in_([GameStatus.COMPLETED, GameStatus.IN_PROGRESS])
            )
            .where(GameResultORM.game_date < current_date)
            .order_by(desc(GameResultORM.game_date))
            .limit(50)  # Expanded window to 50 games
        )
        recent_games = self.session.execute(stmt).scalars().all()
        if not recent_games:
            return None

        # Extract [pitcher_id, name] in Chronological order (reverse the DESC result)
        rotation_history = []
        for g in reversed(recent_games):
            if g.home_team_id == team_id and g.home_pitcher_id:
                rotation_history.append((g.home_pitcher_id, g.home_pitcher))
            elif g.away_team_id == team_id and g.away_pitcher_id:
                rotation_history.append((g.away_pitcher_id, g.away_pitcher))

        if not rotation_history:
            return None

        # 2. Identify the unique rotation members (usually 5)
        unique_starters = []
        seen = set()
        for p_id, p_name in reversed(rotation_history):
            if p_id not in seen:
                unique_starters.append((p_id, p_name))
                seen.add(p_id)
            if len(unique_starters) >= 6:  # Most MLB teams use 5 or 6
                break

        # 3. Find the 'Last Starter' and project the 'Next'
        # unique_starters is in reverse chronological order: [Latest, Lat-1, Lat-2...]
        if len(unique_starters) < 2:
            return unique_starters[0]  # Just the same guy if only 1 found

        # Let's find the cycle.
        cycle = []
        seen_cycle = set()
        for i in range(len(rotation_history) - 1, -1, -1):
            pid = rotation_history[i][0]
            if pid not in seen_cycle:
                cycle.insert(0, rotation_history[i])
                seen_cycle.add(pid)
            else:
                break  # We've completed one full loop backward

        if not cycle:
            return rotation_history[-1]  # Fallback to most recent

        # Now cycle is [Candidate1, Candidate2, ..., LastStarter]
        # We want Candidate1.
        return cycle[0]

    def _fetch_projected_lineup(
        self, team_id: int, current_date: date
    ) -> List[GameLineupORM]:
        """Find the most recent starting lineup via batch fetching for performance."""
        if team_id is None:
            return []

        # 1. Find the 50 most recent games for this team
        recent_games = (
            self.session.execute(
                select(GameResultORM)
                .where(
                    or_(
                        GameResultORM.home_team_id == team_id,
                        GameResultORM.away_team_id == team_id,
                    )
                )
                .where(
                    GameResultORM.status.in_(
                        [GameStatus.COMPLETED, GameStatus.IN_PROGRESS]
                    )
                )
                .where(GameResultORM.game_date < current_date)
                .order_by(desc(GameResultORM.game_date))
                .limit(50)
            )
            .scalars()
            .all()
        )

        if not recent_games:
            return []

        # 2. Batch fetch ALL lineups for these 50 games in one query
        game_pks = [int(g.game_id) for g in recent_games if g.game_id.isdigit()]
        if not game_pks:
            return []

        all_slots = (
            self.session.execute(
                select(GameLineupORM)
                .where(GameLineupORM.game_pk.in_(game_pks))
                .order_by(desc(GameLineupORM.game_pk), GameLineupORM.batting_order)
            )
            .scalars()
            .all()
        )

        # 3. Process in Python to find the first complete 9-man lineup for our team
        # Group by game_pk
        slots_by_game: Dict[int, List[GameLineupORM]] = {}
        for s in all_slots:
            if s.game_pk not in slots_by_game:
                slots_by_game[s.game_pk] = []
            slots_by_game[s.game_pk].append(s)

        # Iterate through the games in chronological order (recent_games is already sorted)
        for g in recent_games:
            pk = int(g.game_id)
            if pk in slots_by_game:
                side = "home" if g.home_team_id == team_id else "away"
                lineup = [s for s in slots_by_game[pk] if s.team_side == side]
                if len(lineup) >= 9:
                    return lineup

        return []

    def _fetch_projected_pitcher(
        self, team_id: int, current_date: date
    ) -> Optional[Tuple[int, str]]:
        """Find the most recent starting pitcher for a team."""
        if team_id is None:
            return None

        stmt = (
            select(GameResultORM)
            .where(
                or_(
                    GameResultORM.home_team_id == team_id,
                    GameResultORM.away_team_id == team_id,
                )
            )
            .where(GameResultORM.status == "COMPLETED")
            .where(GameResultORM.game_date < current_date)
            .order_by(desc(GameResultORM.game_date))
            .limit(5)
        )
        recent_games = self.session.execute(stmt).scalars().all()

        for g in recent_games:
            # Check who started for the team in this historical game
            if g.home_team_id == team_id and g.home_pitcher_id and g.home_pitcher:
                return g.home_pitcher_id, g.home_pitcher
            elif g.away_team_id == team_id and g.away_pitcher_id and g.away_pitcher:
                return g.away_pitcher_id, g.away_pitcher

        return None


MatchupContext.model_rebuild()
