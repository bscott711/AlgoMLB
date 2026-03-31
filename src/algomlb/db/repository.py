from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from algomlb.db.models import (
    BallparkORM,
    BankrollLedgerORM,
    GameResultORM,
    HistoricalDataORM,
    HistoricalOddsORM,
    LiveOddsORM,
    PitchEventORM,
    RetrosheetEventORM,
    UmpireScorecardORM,
)
from algomlb.domain import BankrollTransaction, Game, Odds


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

    def save_game(self, game: Game) -> Game:
        """Save or update a game result record."""
        existing = (
            self.session.query(GameResultORM).filter_by(game_id=game.game_id).first()
        )
        if existing:
            existing.game_date = game.date
            existing.home_team = game.home_team
            existing.away_team = game.away_team
            existing.home_pitcher = game.home_pitcher
            existing.away_pitcher = game.away_pitcher
            existing.home_pitcher_id = game.home_pitcher_id
            existing.away_pitcher_id = game.away_pitcher_id
            existing.home_score = game.home_score
            existing.away_score = game.away_score
            existing.status = game.status
        else:
            orm = GameResultORM(
                game_id=game.game_id,
                game_date=game.date,
                home_team=game.home_team,
                away_team=game.away_team,
                home_pitcher=game.home_pitcher,
                away_pitcher=game.away_pitcher,
                home_pitcher_id=game.home_pitcher_id,
                away_pitcher_id=game.away_pitcher_id,
                home_score=game.home_score,
                away_score=game.away_score,
                status=game.status,
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

    def save_historical_data(self, data: List[HistoricalDataORM]) -> None:
        """Bulk save historical data manually."""
        self.session.add_all(data)
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
