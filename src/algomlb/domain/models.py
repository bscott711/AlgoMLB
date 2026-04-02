import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GameStatus(StrEnum):
    """Status of a MLB game."""

    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    POSTPONED = "POSTPONED"


class GameType(StrEnum):
    """MLB Game Type codes (R=Regular, P=Postseason, S=Spring, E=Exhibition, A=All-Star)."""

    REGULAR_SEASON = "R"
    POSTSEASON = "P"
    SPRING_TRAINING = "S"
    EXHIBITION = "E"
    ALL_STAR = "A"


class Game(BaseModel):
    """Represents a scheduled or completed baseball game."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    game_id: str = Field(..., description="Unique identifier for the game")
    game_type: GameType = Field(
        default=GameType.REGULAR_SEASON, description="R, P, S, etc."
    )
    date: datetime.date = Field(
        ..., description="Date of the game", validation_alias="game_date"
    )
    game_datetime: Optional[datetime.datetime] = Field(
        default=None, description="ISO datetime of the 1st pitch"
    )
    venue_name: Optional[str] = Field(default=None, description="Ballpark name")
    home_team: str = Field(..., min_length=2, max_length=50)
    away_team: str = Field(..., min_length=2, max_length=50)
    home_pitcher: Optional[str] = Field(default=None, min_length=2, max_length=100)
    away_pitcher: Optional[str] = Field(default=None, min_length=2, max_length=100)
    home_pitcher_id: Optional[int] = Field(default=None)
    away_pitcher_id: Optional[int] = Field(default=None)
    home_score: Optional[int] = Field(default=None, ge=0)
    away_score: Optional[int] = Field(default=None, ge=0)
    status: GameStatus = Field(default=GameStatus.SCHEDULED)


class Odds(BaseModel):
    """Represents a single point-in-time odds snapshot."""

    model_config = ConfigDict(frozen=True)

    odds_game_id: str = Field(..., description="The Odds API internal hash")
    home_team: str = Field(..., min_length=2, max_length=50)
    away_team: str = Field(..., min_length=2, max_length=50)
    game_date: datetime.date = Field(..., description="Date of the game")
    sportsbook: str = Field(..., min_length=2, max_length=50)
    market_type: str = Field(
        ..., description="Market type (e.g., moneyline, runline, total)"
    )
    outcome: str = Field(..., description="Outcome (e.g., Team Name, Over, Under)")
    price: float = Field(..., description="The odds price (e.g., decimal 1.91)")
    point: Optional[float] = Field(
        default=None, description="Spread or total point value (e.g., -1.5, 8.5)"
    )
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )

    @property
    def implied_probability(self) -> float:
        if self.price <= 1.0:
            return 1.0
        return 1.0 / self.price

    @property
    def american_odds(self) -> int:
        if self.price >= 2.0:
            return int(round((self.price - 1.0) * 100))
        if self.price <= 1.01:
            return -10000
        return int(round(-100.0 / (self.price - 1.0)))


class TransactionStatus(StrEnum):
    """Status of a bankroll transaction."""

    PENDING = "PENDING"
    SETTLED = "SETTLED"
    CANCELLED = "CANCELLED"
    ADJUSTMENT = "ADJUSTMENT"


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


class HistoricalOdds(BaseModel):
    """Represents historical opening or closing odds for a game."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(..., description="MLB Game ID reference")
    bookmaker: str = Field(..., max_length=50)
    market_type: str = Field(..., description="h2h, spreads, or totals")
    odds_type: str = Field(..., description="opening or closing")
    home_price: int = Field(..., description="American odds for home team")
    away_price: int = Field(..., description="American odds for away team")
    spread: Optional[float] = Field(default=None)
    total: Optional[float] = Field(default=None)
    snapshot_at: datetime.datetime = Field(..., description="Market snapshot timestamp")


class Ballpark(BaseModel):
    """Represents MLB ballpark physical and environmental characteristics."""

    model_config = ConfigDict(frozen=True)

    team_name: str = Field(..., max_length=50)
    ballpark: str = Field(..., max_length=100)
    left_field: Optional[int] = Field(default=None)
    center_field: Optional[int] = Field(default=None)
    right_field: Optional[int] = Field(default=None)
    min_wall_height: Optional[float] = Field(default=None)
    max_wall_height: Optional[float] = Field(default=None)
    hr_park_effects: Optional[float] = Field(default=None)
    extra_distance: Optional[float] = Field(default=None)
    avg_temp: Optional[float] = Field(default=None)
    elevation: Optional[int] = Field(default=None)
    roof: Optional[float] = Field(default=None)
    daytime: Optional[float] = Field(default=None)
