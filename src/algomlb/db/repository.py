from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from algomlb.db.models import (
    BankrollLedgerORM,
    GradedGamesORM,
    HistoricalDataORM,
    LiveOddsORM,
    PitchEventORM,
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
        orm = LiveOddsORM(
            game_id=odds.game_id,
            sportsbook=odds.sportsbook,
            market=odds.market,
            price=odds.price,
            timestamp=odds.timestamp,
        )
        self.session.add(orm)
        self.session.commit()
        return odds

    def get_live_odds(self, game_id: str) -> List[Odds]:
        """Retrieve all odds snapshots for a specific game."""
        stmt = select(LiveOddsORM).where(LiveOddsORM.game_id == game_id)
        results = self.session.execute(stmt).scalars().all()
        return [Odds.model_validate(orm, from_attributes=True) for orm in results]

    def save_game(self, game: Game) -> Game:
        """Save or update a graded game record."""
        orm = GradedGamesORM(
            game_id=game.game_id,
            date=game.date,
            home_team=game.home_team,
            away_team=game.away_team,
            home_pitcher=game.home_pitcher,
            away_pitcher=game.away_pitcher,
            home_score=game.home_score,
            away_score=game.away_score,
            status=game.status,
        )
        self.session.merge(orm)
        self.session.commit()
        return game

    def get_game(self, game_id: str) -> Optional[Game]:
        """Retrieve a game by its ID."""
        orm = self.session.get(GradedGamesORM, game_id)
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
        """Bulk save pitch events manually (simple version for now)."""
        self.session.add_all(events)
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
