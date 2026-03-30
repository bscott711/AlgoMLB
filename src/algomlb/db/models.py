import datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Enum, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from algomlb.db.session import Base
from algomlb.domain import GameStatus, TransactionStatus


class LiveOddsORM(Base):
    """Volatile time-series table for active games/odds."""

    __tablename__ = "live_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    sportsbook: Mapped[str] = mapped_column(String(50), nullable=False)
    market: Mapped[str] = mapped_column(String(50), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class GradedGamesORM(Base):
    """Storage for completed/settled games."""

    __tablename__ = "graded_games"

    game_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String(50), nullable=False)
    away_team: Mapped[str] = mapped_column(String(50), nullable=False)
    home_pitcher: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    away_pitcher: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[GameStatus] = mapped_column(
        Enum(GameStatus), nullable=False, default=GameStatus.SCHEDULED
    )


class BankrollLedgerORM(Base):
    """Persistent state for the paper bankroll."""

    __tablename__ = "bankroll_ledger"

    transaction_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    stake: Mapped[float] = mapped_column(Float, nullable=False)
    odds: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[TransactionStatus] = mapped_column(
        Enum(TransactionStatus), nullable=False, default=TransactionStatus.PENDING
    )
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    game_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, index=True
    )
