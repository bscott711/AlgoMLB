import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GameStatus(StrEnum):
    """Status of a MLB game."""

    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    POSTPONED = "postponed"


class Game(BaseModel):
    """Represents a scheduled or completed baseball game."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(..., description="Unique identifier for the game")
    date: datetime.date = Field(..., description="Date of the game")
    home_team: str = Field(..., min_length=2, max_length=50)
    away_team: str = Field(..., min_length=2, max_length=50)
    home_pitcher: Optional[str] = Field(default=None, min_length=2, max_length=100)
    away_pitcher: Optional[str] = Field(default=None, min_length=2, max_length=100)
    home_score: Optional[int] = Field(default=None, ge=0)
    away_score: Optional[int] = Field(default=None, ge=0)
    status: GameStatus = Field(default=GameStatus.SCHEDULED)


class Odds(BaseModel):
    """Represents a single point-in-time odds snapshot."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(..., description="Reference to the Game ID")
    sportsbook: str = Field(..., min_length=2, max_length=50)
    market: str = Field(
        ..., description="Market type (e.g., moneyline, runline, total)"
    )
    price: float = Field(
        ..., description="The odds price (e.g., decimal 1.91 or American -110)"
    )
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )


class TransactionStatus(StrEnum):
    """Status of a bankroll transaction."""

    PENDING = "pending"
    SETTLED = "settled"
    CANCELLED = "cancelled"
    ADJUSTMENT = "adjustment"


class BankrollTransaction(BaseModel):
    """Represents a pending bet, settlement, or manual adjustment."""

    model_config = ConfigDict(frozen=True)

    transaction_id: str = Field(
        ..., description="Unique identifier for the transaction"
    )
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
    stake: float = Field(..., gt=0, description="Amount wagered or adjusted")
    odds: float = Field(
        ..., gt=1.0, description="Decimal odds at which the bet was placed"
    )
    status: TransactionStatus = Field(default=TransactionStatus.PENDING)
    pnl: Optional[float] = Field(
        default=None, description="Profit and loss after settlement"
    )
    game_id: Optional[str] = Field(
        default=None, description="Reference to the Game ID if applicable"
    )
