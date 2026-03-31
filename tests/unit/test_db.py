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


def test_save_historical_odds(test_session: Session) -> None:
    """Test bulk saving historical odds records."""
    repo = DatabaseRepository(test_session)
    from algomlb.db.models import HistoricalOddsORM

    odds = [
        HistoricalOddsORM(
            game_id=1,
            bookmaker="test_bk",
            market_type="h2h",
            odds_type="opening",
            home_price=100,
            away_price=-110,
            snapshot_at=datetime.now(UTC),
        )
    ]
    repo.save_historical_odds(odds)

    from sqlalchemy import select

    retrieved = test_session.execute(select(HistoricalOddsORM)).scalars().all()
    assert len(retrieved) == 1


def test_save_umpire_scorecards(test_session: Session) -> None:
    """Test bulk upsert umpire scorecard records."""
    from algomlb.db.models import UmpireScorecardORM

    scorecards = [
        UmpireScorecardORM(
            game_pk=718760,
            game_date=date(2023, 4, 1),
            umpire_name="Ump1",
            home_team="HOU",
            away_team="CWS",
            accuracy=95.0,
            consistency=95.0,
            favoritism_home=0.0,
            expected_runs=1.0,
            actual_runs=1.0,
        )
    ]
    repo = DatabaseRepository(test_session)
    # Call the repository method
    repo.save_umpire_scorecards(scorecards)

    from sqlalchemy import select

    retrieved = test_session.execute(select(UmpireScorecardORM)).scalars().all()
    assert len(retrieved) == 1
    assert retrieved[0].accuracy == 95.0


def test_save_umpire_scorecards_update(test_session: Session) -> None:
    """Test bulk upsert update on conflict."""
    repo = DatabaseRepository(test_session)
    from algomlb.db.models import UmpireScorecardORM

    pk = 12345
    sc1 = UmpireScorecardORM(
        game_pk=pk,
        game_date=date(2023, 4, 1),
        umpire_name="Test Ump",
        home_team="HOU",
        away_team="CWS",
        accuracy=90.0,
        consistency=90.0,
        favoritism_home=0.1,
        expected_runs=1.0,
        actual_runs=1.0,
    )
    repo.save_umpire_scorecards([sc1])

    # Now update
    sc2 = UmpireScorecardORM(
        game_pk=pk,
        game_date=date(2023, 4, 1),
        umpire_name="Test Ump",
        home_team="HOU",
        away_team="CWS",
        accuracy=99.0,
        consistency=99.0,
        favoritism_home=0.2,
        expected_runs=1.0,
        actual_runs=1.0,
    )
    repo.save_umpire_scorecards([sc2])

    from sqlalchemy import select

    retrieved = test_session.execute(select(UmpireScorecardORM)).scalars().all()
    assert len(retrieved) == 1
    assert retrieved[0].accuracy == 99.0


def test_save_umpire_scorecards_empty(test_session: Session) -> None:
    """Test saving empty list return early."""
    repo = DatabaseRepository(test_session)
    repo.save_umpire_scorecards([])


def test_save_retrosheet_events(test_session: Session) -> None:
    """Test bulk merging retrosheet event records."""
    repo = DatabaseRepository(test_session)
    from algomlb.db.models import RetrosheetEventORM

    events = [
        RetrosheetEventORM(
            game_id="G1",
            play_number=1,
            event_text="K",
            inning=1,
            top_bot=0,
            vis_home=0,
            site="TEST",
            bat_team="ATL",
            pit_team="NYY",
            batter_id="b1",
            pitcher_id="p1",
            lp=1,
            bat_f=2,
            date=date(2023, 4, 1),
        )
    ]
    repo.save_retrosheet_events(events)

    from sqlalchemy import select

    retrieved = test_session.execute(select(RetrosheetEventORM)).scalars().all()
    assert len(retrieved) == 1


def test_save_umpire_scorecards_large_batch(test_session: Session) -> None:
    """Test chunking with 50 rows (each with 44 fields = 2200 variables)."""
    repo = DatabaseRepository(test_session)
    from algomlb.db.models import UmpireScorecardORM
    from datetime import date

    scorecards = []
    for i in range(50):
        sc = UmpireScorecardORM(
            game_pk=1000 + i,
            game_date=date(2025, 3, 27),
            umpire_name="Test Umpire",
            home_team="NYY",
            away_team="MIL",
            accuracy=95.0,
            consistency=94.0,
            favoritism_home=0.5,
            expected_runs=1.0,
            actual_runs=2.0,
        )
        scorecards.append(sc)

    repo.save_umpire_scorecards(scorecards)

    # Verify all 50 rows were saved
    count = test_session.query(UmpireScorecardORM).count()
    assert count == 50
