from datetime import date
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from algomlb.db.models import (
    BallparkORM,
    BankrollLedgerORM,
    GameResultORM,
    HistoricalOddsORM,
    LiveOddsORM,
    PitchEventORM,
    PlayerRollingFeaturesORM,
    PlayerTransactionORM,
    RetrosheetEventORM,
    StatcastRawORM,
    UmpireScorecardORM,
)
from algomlb.domain import BankrollTransaction, Game, Odds
from algomlb.domain.teams import TEAM_NAME_TO_ABB


class DatabaseRepository:
    """
    Repository for mediating between Domain models and SQLAlchemy ORM.
    Handles all database I/O while exposing only Pydantic Domain models.
    """

    def __init__(self, session: Session):
        self.session = session

    def save_live_odds(self, odds: Odds) -> Odds:
        """Add a new point-in-time odds snapshot."""
        from algomlb.db.models import GameResultORM

        game = (
            self.session.query(GameResultORM)
            .filter_by(
                home_team=odds.home_team,
                away_team=odds.away_team,
                game_date=odds.game_date,
            )
            .first()
        )

        orm = LiveOddsORM(
            odds_game_id=odds.odds_game_id,
            home_team=odds.home_team,
            away_team=odds.away_team,
            game_date=odds.game_date,
            game_result_id=game.game_id if game else None,
            sportsbook=odds.sportsbook,
            market_type=odds.market_type,
            outcome=odds.outcome,
            price=odds.price,
            timestamp=odds.timestamp,
        )
        self.session.add(orm)
        self.session.commit()
        return odds

    def get_live_odds(self, odds_game_id: str) -> List[Odds]:
        """Retrieve all odds snapshots for a specific game."""
        stmt = select(LiveOddsORM).where(LiveOddsORM.odds_game_id == odds_game_id)
        results = self.session.execute(stmt).scalars().all()
        return [Odds.model_validate(orm, from_attributes=True) for orm in results]

    def _resolve_ballpark_id(self, game: Game) -> Optional[int]:
        """Resolve the primary key of the ballpark for a given game."""
        ballpark_id = None
        if game.venue_name:
            v_name = game.venue_name.strip()
            bp = (
                self.session.query(BallparkORM)
                .filter(BallparkORM.ballpark.ilike(f"%{v_name}%"))
                .first()
            )
            if bp:
                ballpark_id = bp.id

        if not ballpark_id:
            h_team = game.home_team.strip()
            abb = TEAM_NAME_TO_ABB.get(h_team)
            if abb:
                bp = self.session.query(BallparkORM).filter_by(team_name=abb).first()
                if bp:
                    ballpark_id = bp.id

            # Final fallback: keyword match on team_name column
            if not ballpark_id:
                bp = (
                    self.session.query(BallparkORM)
                    .filter(BallparkORM.team_name.ilike(f"%{h_team}%"))
                    .first()
                )
                if bp:
                    ballpark_id = bp.id

        if not ballpark_id:
            from algomlb.core.logger import logger

            logger.warning(
                f"Could not resolve ballpark_id for game {game.game_id} (Home: {game.home_team}, Venue: {game.venue_name})"
            )
        return ballpark_id

    def save_game(self, game: Game) -> Game:
        """Save or update a game result record."""
        ballpark_id = self._resolve_ballpark_id(game)

        existing = (
            self.session.query(GameResultORM).filter_by(game_id=game.game_id).first()
        )
        if existing:
            existing.game_date = game.date
            existing.game_datetime = game.game_datetime
            existing.home_team = game.home_team
            existing.away_team = game.away_team
            existing.home_pitcher = game.home_pitcher
            existing.away_pitcher = game.away_pitcher
            existing.home_pitcher_id = game.home_pitcher_id
            existing.away_pitcher_id = game.away_pitcher_id
            existing.home_score = game.home_score
            existing.away_score = game.away_score
            existing.status = game.status
            existing.game_type = game.game_type
            existing.home_team_id = game.home_team_id
            existing.away_team_id = game.away_team_id
            existing.doubleheader_num = game.doubleheader_num
            if ballpark_id:
                existing.ballpark_id = ballpark_id
        else:
            orm = GameResultORM(
                game_id=game.game_id,
                game_date=game.date,
                game_datetime=game.game_datetime,
                home_team=game.home_team,
                away_team=game.away_team,
                home_team_id=game.home_team_id,
                away_team_id=game.away_team_id,
                doubleheader_num=game.doubleheader_num,
                home_pitcher=game.home_pitcher,
                away_pitcher=game.away_pitcher,
                home_pitcher_id=game.home_pitcher_id,
                away_pitcher_id=game.away_pitcher_id,
                home_score=game.home_score,
                away_score=game.away_score,
                status=game.status,
                game_type=game.game_type,
                ballpark_id=ballpark_id,
            )
            self.session.add(orm)
        self.session.commit()
        return game

    def get_game(self, game_id: str) -> Optional[Game]:
        """Retrieve a game by its MLB string ID."""
        stmt = select(GameResultORM).where(GameResultORM.game_id == game_id)
        orm = self.session.execute(stmt).scalar_one_or_none()
        if not orm:
            return None
        return Game.model_validate(orm, from_attributes=True)

    def save_transaction(self, tx: BankrollTransaction) -> BankrollTransaction:
        """Persist a bankroll transaction."""
        orm = BankrollLedgerORM(
            transaction_id=tx.transaction_id,
            timestamp=tx.timestamp,
            stake=tx.stake,
            odds=tx.odds,
            status=tx.status,
            pnl=tx.pnl,
            game_id=tx.game_id,
        )
        self.session.merge(orm)
        self.session.commit()
        return tx

    def save_pitch_events(self, events: List[PitchEventORM]) -> None:
        """Bulk save pitch events using PostgreSQL UPSERT with chunking."""
        if not events:
            return

        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from algomlb.db.models import PitchEventORM

        # Convert ORM objects to dicts for bulk UPSERT
        rows_as_dicts = [
            {k: v for k, v in evt.__dict__.items() if not k.startswith("_")}
            for evt in events
        ]

        # Chunk to stay under SQL variable limits (SQLite=999, Postgres=65535)
        # 50 rows * ~13 fields = 650 variables (safe default)
        chunk_size = 50
        for i in range(0, len(rows_as_dicts), chunk_size):
            chunk = rows_as_dicts[i : i + chunk_size]
            stmt = pg_insert(PitchEventORM).values(chunk)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["game_id", "at_bat_number", "pitch_number"]
            )
            self.session.execute(stmt)
        self.session.commit()


    def get_bankroll_balance(self) -> float:
        """Calculate the current cumulative PnL."""
        stmt = select(func.sum(BankrollLedgerORM.pnl))
        result = self.session.execute(stmt).scalar()
        return float(result) if result is not None else 0.0

    def save_historical_odds(self, odds: List[HistoricalOddsORM]) -> None:
        """Bulk save historical opening/closing odds snapshots."""
        self.session.add_all(odds)
        self.session.commit()

    def save_ballparks(self, ballparks: List[BallparkORM]) -> None:
        """Bulk save or merge ballpark data."""
        for bp in ballparks:
            self.session.merge(bp)
        self.session.commit()

    def save_umpire_scorecards(self, scorecards: List[UmpireScorecardORM]) -> None:
        """Bulk upsert umpire scorecard data via game_pk with chunking."""
        if not scorecards:
            return

        from sqlalchemy.dialects.postgresql import insert as pg_insert

        rows_as_dicts = [
            {k: v for k, v in sc.__dict__.items() if not k.startswith("_")}
            for sc in scorecards
        ]

        # Chunk to stay under SQL variable limits (SQLite=999, Postgres=65535)
        # 50 rows * ~44 fields = 2200 variables (safe for Postgres, barely over default SQLite)
        # Wait, if SQLite limit is 999, 50 * 44 = 2200 is too much for SQLite but fine for Postgres.
        # Let's use 20 for safer SQLite compatibility and still good Postgres performance.
        chunk_size = 20
        for i in range(0, len(rows_as_dicts), chunk_size):
            chunk = rows_as_dicts[i : i + chunk_size]
            stmt = pg_insert(UmpireScorecardORM).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["game_pk"],
                set_={c: stmt.excluded[c] for c in rows_as_dicts[0] if c != "id"},
            )
            self.session.execute(stmt)
        self.session.commit()

    def save_retrosheet_events(self, events: List[RetrosheetEventORM]) -> None:
        """Bulk merge retrosheet play-by-play events."""
        for event in events:
            self.session.merge(event)
        self.session.commit()

    def save_player_transactions(self, transactions: List[PlayerTransactionORM]) -> int:
        """Bulk upsert player transactions to handle updates/resolutions."""
        if not transactions:
            return 0

        from sqlalchemy.dialects.postgresql import insert as pg_insert

        # Deduplicate by transaction_id in the input list to prevent internal batch conflicts
        deduped = {}
        for tx in transactions:
            deduped[tx.transaction_id] = tx

        rows_as_dicts = [
            {k: v for k, v in tx.__dict__.items() if not k.startswith("_")}
            for tx in deduped.values()
        ]

        # Use 100 as chunk size for transactions (relatively light rows)
        chunk_size = 100
        total_upserted = 0
        for i in range(0, len(rows_as_dicts), chunk_size):
            chunk = rows_as_dicts[i : i + chunk_size]
            stmt = pg_insert(PlayerTransactionORM).values(chunk)
            # transaction_id is the primary key
            stmt = stmt.on_conflict_do_update(
                index_elements=["transaction_id"],
                set_={
                    c: stmt.excluded[c]
                    for c in rows_as_dicts[0]
                    if c != "transaction_id"
                },
            )
            self.session.execute(stmt)
            total_upserted += len(chunk)

        self.session.commit()
        return total_upserted

    def save_statcast_raw(self, rows: List[dict]) -> int:
        """Bulk upsert raw Statcast data into statcast_raw via PostgreSQL."""
        if not rows:
            return 0

        from sqlalchemy.dialects.postgresql import insert as pg_insert

        # Chunk to stay under SQL variable limits
        chunk_size = 500
        total_inserted = 0
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            stmt = pg_insert(StatcastRawORM).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["game_pk", "at_bat_number", "pitch_number"],
                set_={
                    c: stmt.excluded[c]
                    for c in chunk[0].keys()
                    if c not in ["game_pk", "at_bat_number", "pitch_number"]
                },
            )
            self.session.execute(stmt)
            total_inserted += len(chunk)
        self.session.commit()
        return total_inserted

    def get_season_start_date(self, year: int) -> date:
        """
        Dynamically find the first regular season game date for a given year.
        Uses game_results table which must be pre-populated.
        """
        from algomlb.domain import GameType

        stmt = (
            select(func.min(GameResultORM.game_date))
            .where(func.extract("year", GameResultORM.game_date) == year)
            .where(GameResultORM.game_type == GameType.REGULAR_SEASON)
        )
        res = self.session.execute(stmt).scalar()
        if not res:
            # Fallback for years without games in DB yet
            return date(year, 3, 20)
        return res

    def save_player_rolling_features_records(
        self, records: List[PlayerRollingFeaturesORM]
    ) -> int:
        """Bulk upsert rolling features with chunking."""
        if not records:
            return 0

        from sqlalchemy.dialects.postgresql import insert as pg_insert

        rows_as_dicts = [
            {k: v for k, v in r.__dict__.items() if not k.startswith("_")}
            for r in records
        ]

        chunk_size = 500
        total_upserted = 0
        for i in range(0, len(rows_as_dicts), chunk_size):
            chunk = rows_as_dicts[i : i + chunk_size]
            stmt = pg_insert(PlayerRollingFeaturesORM).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["player_id", "game_date", "role"],
                set_={
                    c: stmt.excluded[c]
                    for c in rows_as_dicts[0]
                    if c not in ["id", "player_id", "game_date", "role"]
                },
            )
            self.session.execute(stmt)
            total_upserted += len(chunk)

        self.session.commit()
        return total_upserted

    def update_game_enrichment(self, updates: List[dict]) -> int:
        """
        Bulk update environmental and fatigue columns in game_results.
        Each dict must contain 'game_id' and the fields to update.
        """
        from sqlalchemy import bindparam, update

        # Use the Core table object to bypass ORM bulk update requirements for primary keys
        table = GameResultORM.__table__
        stmt = (
            update(table)
            .where(table.c.game_id == bindparam("b_game_id"))
            .values(
                temperature=bindparam("temperature", None),
                wind_speed=bindparam("wind_speed", None),
                humidity=bindparam("humidity", None),
                home_rest_days=bindparam("home_rest_days", None),
                away_rest_days=bindparam("away_rest_days", None),
                home_travel_distance_km=bindparam("home_travel_distance_km", None),
                away_travel_distance_km=bindparam("away_travel_distance_km", None),
            )
        )

        params = []
        for d in updates:
            p = {
                "b_game_id": d["game_id"],
                "temperature": d.get("temperature"),
                "wind_speed": d.get("wind_speed"),
                "humidity": d.get("humidity"),
                "home_rest_days": d.get("home_rest_days"),
                "away_rest_days": d.get("away_rest_days"),
                "home_travel_distance_km": d.get("home_travel_distance_km"),
                "away_travel_distance_km": d.get("away_travel_distance_km"),
            }
            params.append(p)

        result = self.session.execute(stmt, params)
        self.session.commit()
        return result.rowcount
