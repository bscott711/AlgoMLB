from datetime import UTC, date, datetime

import pytest
from sqlalchemy.orm import Session

from algomlb.db import Base, DatabaseRepository, create_db_engine, get_session_factory
from algomlb.domain import (
    BankrollTransaction,
    Game,
    GameStatus,
    Odds,
    TransactionStatus,
)


@pytest.fixture
def test_session():
    """Create an in-memory SQLite database and session for testing."""
    engine = create_db_engine("sqlite:///:memory:")
    # Create all tables defined in models.py which are registered with Base
    Base.metadata.create_all(engine)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        yield session
    Base.metadata.drop_all(engine)


def test_save_and_get_game(test_session: Session) -> None:
    """Test saving a Pydantic Game model to the DB and retrieving it back."""
    repo = DatabaseRepository(test_session)
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

    repo.save_game(game)
    retrieved = repo.get_game("20260330NYYTOR")

    assert retrieved is not None
    assert retrieved.game_id == game.game_id
    assert retrieved.home_team == game.home_team
    assert retrieved.home_score == 0
    assert retrieved.status == GameStatus.COMPLETED


def test_save_and_get_live_odds(test_session: Session) -> None:
    """Test saving Pydantic Odds models and retrieving them."""
    repo = DatabaseRepository(test_session)
    game_id = "test_game"
    now = datetime.now(UTC)

    odds1 = Odds(
        game_id=game_id,
        sportsbook="DraftKings",
        market="moneyline",
        price=1.91,
        timestamp=now,
    )
    odds2 = Odds(
        game_id=game_id,
        sportsbook="FanDuel",
        market="moneyline",
        price=1.95,
        timestamp=now,
    )

    repo.save_live_odds(odds1)
    repo.save_live_odds(odds2)

    retrieved_list = repo.get_live_odds(game_id)
    assert len(retrieved_list) == 2
    prices = [o.price for o in retrieved_list]
    assert 1.91 in prices
    assert 1.95 in prices


def test_bankroll_ledger_and_balance(test_session: Session) -> None:
    """Test saving transactions and calculating the overall balance."""
    repo = DatabaseRepository(test_session)

    tx1 = BankrollTransaction(
        transaction_id="TX001",
        stake=100.0,
        odds=2.10,
        status=TransactionStatus.SETTLED,
        pnl=110.0,
    )
    tx2 = BankrollTransaction(
        transaction_id="TX002",
        stake=100.0,
        odds=1.91,
        status=TransactionStatus.SETTLED,
        pnl=-100.0,
    )

    repo.save_transaction(tx1)
    repo.save_transaction(tx2)

    balance = repo.get_bankroll_balance()
    # 110.0 - 100.0 = 10.0
    assert balance == 10.0


def test_get_nonexistent_game(test_session: Session) -> None:
    """Test retrieving a game that doesn't exist."""
    repo = DatabaseRepository(test_session)
    retrieved = repo.get_game("NONEXISTENT")
    assert retrieved is None


def test_empty_balance(test_session: Session) -> None:
    """Test bankroll balance with no transactions."""
    repo = DatabaseRepository(test_session)
    assert repo.get_bankroll_balance() == 0.0
