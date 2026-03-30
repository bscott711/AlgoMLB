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
        odds_game_id="20260330NYYTOR",
        home_team="Team A",
        away_team="Team B",
        game_date=now.date(),
        sportsbook="DraftKings",
        market_type="moneyline",
        outcome="Team A",
        price=1.91,
        timestamp=now,
    )
    assert odds.price == 1.91
    assert odds.timestamp == now
    assert odds.sportsbook == "DraftKings"
    assert odds.implied_probability == pytest.approx(1 / 1.91)
    assert odds.american_odds == -110


def test_odds_american_various_prices() -> None:
    """Test American odds calculations for edge cases."""
    now = datetime.now(UTC)
    d = now.date()
    o1 = Odds(
        odds_game_id="1",
        home_team="Team A",
        away_team="Team B",
        game_date=d,
        sportsbook="SB",
        market_type="h2h",
        outcome="A",
        price=2.50,
    )
    assert o1.american_odds == 150

    o2 = Odds(
        odds_game_id="2",
        home_team="Team A",
        away_team="Team B",
        game_date=d,
        sportsbook="SB",
        market_type="h2h",
        outcome="A",
        price=1.0,
    )
    assert o2.implied_probability == 1.0
    assert o2.american_odds == -10000

    o3 = Odds(
        odds_game_id="3",
        home_team="Team A",
        away_team="Team B",
        game_date=d,
        sportsbook="SB",
        market_type="h2h",
        outcome="A",
        price=1.01,
    )
    assert o3.american_odds == -10000


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
