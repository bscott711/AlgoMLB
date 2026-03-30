from datetime import UTC, date, datetime

import pytest
from sqlalchemy.orm import Session

from algomlb.db import (
    Base,
    DatabaseRepository,
    HistoricalDataORM,
    PitchEventORM,
    create_db_engine,
)
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
    from sqlalchemy.orm import sessionmaker

    session_factory = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
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


def test_update_existing_game(test_session: Session) -> None:
    """Test updating an existing game record in the DB."""
    repo = DatabaseRepository(test_session)
    game_id = "UPDATE_ME"
    g1 = Game(
        game_id=game_id,
        date=date(2023, 4, 1),
        home_team="Team A",
        away_team="Team B",
        status=GameStatus.SCHEDULED,
    )
    repo.save_game(g1)

    # Update
    g2 = Game(
        game_id=game_id,
        date=date(2023, 4, 1),
        home_team="Team A",
        away_team="Team B",
        status=GameStatus.COMPLETED,
        home_score=5,
        away_score=3,
    )
    repo.save_game(g2)

    retrieved = repo.get_game(game_id)
    assert retrieved is not None
    assert retrieved.status == GameStatus.COMPLETED
    assert retrieved.home_score == 5


def test_save_and_get_live_odds(test_session: Session) -> None:
    """Test saving Pydantic Odds models and retrieving them."""
    repo = DatabaseRepository(test_session)
    game_id = "test_game"
    now = datetime.now(UTC)

    odds1 = Odds(
        odds_game_id=game_id,
        home_team="Team A",
        away_team="Team B",
        game_date=now.date(),
        sportsbook="DraftKings",
        market_type="moneyline",
        outcome="Team A",
        price=1.91,
        timestamp=now,
    )
    odds2 = Odds(
        odds_game_id=game_id,
        home_team="Team A",
        away_team="Team B",
        game_date=now.date(),
        sportsbook="FanDuel",
        market_type="moneyline",
        outcome="Team A",
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


def test_save_pitch_events(test_session: Session) -> None:
    """Test bulk saving pitch events."""
    repo = DatabaseRepository(test_session)
    events = [
        PitchEventORM(
            game_id="G1",
            game_date=date(2023, 4, 1),
            pitcher_id=1,
            batter_id=2,
            release_speed=95.5,
        ),
        PitchEventORM(
            game_id="G1",
            game_date=date(2023, 4, 1),
            pitcher_id=1,
            batter_id=3,
            release_speed=94.2,
        ),
    ]
    repo.save_pitch_events(events)
    # Verify by querying directly from session
    from sqlalchemy import select

    retrieved = test_session.execute(select(PitchEventORM)).scalars().all()
    retrieved = test_session.execute(select(PitchEventORM)).scalars().all()
    assert len(retrieved) == 2


def test_save_pitch_events_empty(test_session: Session) -> None:
    """Test saving empty list of pitch events returns early."""
    repo = DatabaseRepository(test_session)
    repo.save_pitch_events([])


def test_save_historical_data(test_session: Session) -> None:
    """Test bulk saving historical data."""
    repo = DatabaseRepository(test_session)
    data = [
        HistoricalDataORM(
            player_id=1, date=date(2023, 4, 1), metric_name="era", metric_value=3.50
        ),
        HistoricalDataORM(
            player_id=2, date=date(2023, 4, 1), metric_name="woba", metric_value=0.350
        ),
    ]
    repo.save_historical_data(data)
    # Verify
    from sqlalchemy import select

    retrieved = test_session.execute(select(HistoricalDataORM)).scalars().all()
    assert len(retrieved) == 2


def test_create_db_engine_with_config():
    """Verify create_db_engine handles DatabaseConfig objects."""
    from algomlb.config.settings import DatabaseConfig
    from algomlb.db.session import create_db_engine
    from typing import cast
    from pydantic import PostgresDsn

    config = DatabaseConfig(
        url=cast(PostgresDsn, "postgresql://user:pass@localhost/db"),
        echo=True,
        pool_size=10,
    )
    # Don't actually call connect, just check engine url
    engine = create_db_engine(url=config)
    # Pydantic masks the password from the DSN with ***
    assert str(engine.url) == "postgresql://user:***@localhost/db"
    assert engine.echo is True


def test_save_ballparks(test_session: Session) -> None:
    """Test bulk saving/merging ballpark records."""
    repo = DatabaseRepository(test_session)
    from algomlb.db.models import BallparkORM

    ballparks = [
        BallparkORM(team_name="NYY", ballpark="Yankee Stadium"),
        BallparkORM(team_name="TOR", ballpark="Rogers Centre"),
    ]
    repo.save_ballparks(ballparks)

    from sqlalchemy import select

    retrieved = test_session.execute(select(BallparkORM)).scalars().all()
    assert len(retrieved) == 2
