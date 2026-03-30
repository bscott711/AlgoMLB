from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from algomlb.domain import (
    BankrollTransaction,
    Game,
    GameStatus,
    Odds,
    TransactionStatus,
)


def test_game_creation() -> None:
    """Test valid Game creation."""
    game = Game(
        game_id="20260330NYYTOR",
        date=date(2026, 3, 30),
        home_team="Toronto Blue Jays",
        away_team="New York Yankees",
        home_pitcher="Gerrit Cole",
        away_pitcher="Kevin Gausman",
        home_score=0,
        away_score=2,
        status=GameStatus.COMPLETED,
    )
    assert isinstance(game.date, date)
    assert game.home_team == "Toronto Blue Jays"
    assert game.home_score == 0
    assert game.away_score == 2
    assert game.status == GameStatus.COMPLETED


def test_game_validation() -> None:
    """Test Game validation constraints."""
    # Negative score should fail
    with pytest.raises(ValidationError):
        Game(
            game_id="20260330NYYTOR",
            date=date(2026, 3, 30),
            home_team="Toronto Blue Jays",
            away_team="New York Yankees",
            home_score=-1,
        )

    # Team name too short
    with pytest.raises(ValidationError):
        Game(
            game_id="20260330NYYTOR",
            date=date(2026, 3, 30),
            home_team="T",
            away_team="New York Yankees",
        )


def test_odds_creation() -> None:
    """Test valid Odds creation."""
    now = datetime.now(UTC)
    odds = Odds(
        game_id="20260330NYYTOR",
        sportsbook="DraftKings",
        market="moneyline",
        price=1.91,
        timestamp=now,
    )
    assert odds.price == 1.91
    assert odds.timestamp == now
    assert odds.sportsbook == "DraftKings"


def test_bankroll_transaction_creation() -> None:
    """Test valid BankrollTransaction creation."""
    tx = BankrollTransaction(
        transaction_id="TX001",
        stake=100.0,
        odds=2.10,
        status=TransactionStatus.PENDING,
    )
    assert tx.stake == 100.0
    assert tx.odds == 2.10
    assert tx.status == TransactionStatus.PENDING
    assert tx.pnl is None


def test_bankroll_transaction_validation() -> None:
    """Test BankrollTransaction validation constraints."""
    # Zero stake should fail
    with pytest.raises(ValidationError):
        BankrollTransaction(
            transaction_id="TX001",
            stake=0.0,
            odds=2.10,
        )

    # Odds 1.0 (no profit) or less should fail
    with pytest.raises(ValidationError):
        BankrollTransaction(
            transaction_id="TX001",
            stake=100.0,
            odds=1.0,
        )

    with pytest.raises(ValidationError):
        BankrollTransaction(
            transaction_id="TX001",
            stake=100.0,
            odds=0.5,
        )


def test_models_frozen() -> None:
    """Test that models are immutable (frozen)."""
    game = Game(
        game_id="20260330NYYTOR",
        date=date(2026, 3, 30),
        home_team="Toronto Blue Jays",
        away_team="New York Yankees",
    )
    with pytest.raises(ValidationError):
        # Pydantic v2 raises ValidationError when trying to mutate a frozen model field.
        game.home_team = "LAD"  # type: ignore


def test_transaction_pnl_settlement() -> None:
    """Test PnL can be set for a transaction (though it's frozen)."""
    # Create initial pending
    tx = BankrollTransaction(
        transaction_id="TX001",
        stake=100.0,
        odds=2.10,
        status=TransactionStatus.PENDING,
    )

    # To 'settle', we create a new instance (since it's frozen)
    settled_tx = tx.model_copy(
        update={"status": TransactionStatus.SETTLED, "pnl": 110.0}
    )
    assert settled_tx.status == TransactionStatus.SETTLED
    assert settled_tx.pnl == 110.0
    assert settled_tx.transaction_id == tx.transaction_id
