from datetime import UTC, date, datetime

import pytest
from sqlalchemy.orm import Session

from algomlb.db import (
    Base,
    DatabaseRepository,
    HistoricalDataORM,
    PitchEventORM,
    PlayerRollingFeaturesORM,
    GameResultORM,
    create_db_engine,
)
from algomlb.domain import (
    BankrollTransaction,
    Game,
    GameStatus,
    Odds,
    TransactionStatus,
    PlayerRole,
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


def test_save_player_transactions(test_session: Session) -> None:
    """Test bulk upserting player transaction records."""
    from algomlb.db.models import PlayerTransactionORM

    repo = DatabaseRepository(test_session)
    transactions = [
        PlayerTransactionORM(
            transaction_id="TX1",
            player_id=1,
            team_id=10,
            transaction_date=date(2024, 3, 1),
            type_desc="Placed on 10-Day IL",
            raw_description="Hamstring strain",
        ),
        PlayerTransactionORM(
            transaction_id="TX2",
            player_id=2,
            team_id=10,
            transaction_date=date(2024, 3, 1),
            type_desc="Activated",
            raw_description="Activated from IL",
        ),
    ]

    count = repo.save_player_transactions(transactions)
    assert count == 2

    # Check deduplication and update
    transactions_update = [
        PlayerTransactionORM(
            transaction_id="TX1",
            player_id=1,
            team_id=10,
            transaction_date=date(2024, 3, 1),
            type_desc="Placed on 60-Day IL",  # Update
            raw_description="Hamstring strain",
        ),
        PlayerTransactionORM(
            transaction_id="TX1",  # Duplicate in same batch
            player_id=1,
            team_id=10,
            transaction_date=date(2024, 3, 1),
            type_desc="Placed on 60-Day IL",
            raw_description="Hamstring strain",
        ),
    ]

    count2 = repo.save_player_transactions(transactions_update)
    assert count2 == 1  # Deduped to 1

    from sqlalchemy import select

    retrieved = test_session.execute(select(PlayerTransactionORM)).scalars().all()
    # TX1 and TX2 still exist
    assert len(retrieved) == 2
    tx1 = test_session.get(PlayerTransactionORM, "TX1")
    assert tx1 is not None
    assert tx1.type_desc == "Placed on 60-Day IL"


def test_save_player_transactions_empty(test_session: Session) -> None:
    """Test saving empty list return early."""
    repo = DatabaseRepository(test_session)
    assert repo.save_player_transactions([]) == 0


def test_save_statcast_raw(test_session: Session) -> None:
    """Test bulk upserting raw Statcast data."""
    repo = DatabaseRepository(test_session)
    rows = [
        {
            "game_pk": 1,
            "at_bat_number": 1,
            "pitch_number": 1,
            "game_date": date(2024, 4, 1),
            "home_team": "NYY",
            "away_team": "BOS",
            "batter": 101,
            "pitcher": 201,
            "release_speed": 95.0,
        },
        {
            "game_pk": 1,
            "at_bat_number": 1,
            "pitch_number": 2,
            "game_date": date(2024, 4, 1),
            "home_team": "NYY",
            "away_team": "BOS",
            "batter": 101,
            "pitcher": 201,
            "release_speed": 96.0,
        },
    ]

    count = repo.save_statcast_raw(rows)
    assert count == 2

    # Update test
    rows_update = [
        {
            "game_pk": 1,
            "at_bat_number": 1,
            "pitch_number": 1,
            "game_date": date(2024, 4, 1),
            "home_team": "NYY",
            "away_team": "BOS",
            "batter": 101,
            "pitcher": 201,
            "release_speed": 99.0,  # Updated
        }
    ]
    count2 = repo.save_statcast_raw(rows_update)
    assert count2 == 1

    from algomlb.db.models import StatcastRawORM

    res = (
        test_session.query(StatcastRawORM)
        .filter_by(game_pk=1, at_bat_number=1, pitch_number=1)
        .one()
    )
    assert res.release_speed == 99.0


def test_save_statcast_raw_empty(test_session: Session) -> None:
    """Test saving empty list return early."""
    repo = DatabaseRepository(test_session)
    assert repo.save_statcast_raw([]) == 0


def test_get_season_start_date(test_session: Session) -> None:
    """Test getting the earliest regular season game date."""
    from algomlb.domain import GameType

    repo = DatabaseRepository(test_session)
    g1 = GameResultORM(
        game_id="G1",
        game_date=date(2024, 3, 28),
        home_team="NYY",
        away_team="HOU",
        home_score=0,
        away_score=0,
        game_type=GameType.REGULAR_SEASON,
    )
    test_session.add(g1)
    test_session.commit()

    res = repo.get_season_start_date(2024)
    assert res == date(2024, 3, 28)


def test_get_season_start_date_fallback(test_session: Session) -> None:
    """Test fallback date when no games are in DB."""
    repo = DatabaseRepository(test_session)
    res = repo.get_season_start_date(1999)
    assert res == date(1999, 3, 20)


def test_save_player_rolling_features_records(test_session: Session) -> None:
    """Test bulk upserting rolling features."""
    repo = DatabaseRepository(test_session)
    from algomlb.domain import BaselineQuality

    records = [
        PlayerRollingFeaturesORM(
            player_id=1,
            game_date=date(2024, 4, 1),
            season=2024,
            role=PlayerRole.PITCHER,
            window_games=5,
            baseline_quality=BaselineQuality.FULL,
            roll_avg_pitcher_xwoba=0.300,
        )
    ]
    count = repo.save_player_rolling_features_records(records)
    assert count == 1

    # Update
    records[0].roll_avg_pitcher_xwoba = 0.250
    count2 = repo.save_player_rolling_features_records(records)
    assert count2 == 1

    retrieved = test_session.query(PlayerRollingFeaturesORM).one()
    assert retrieved.roll_avg_pitcher_xwoba == 0.250


def test_save_player_rolling_features_records_empty(test_session: Session) -> None:
    """Test empty list returns 0."""
    repo = DatabaseRepository(test_session)
    assert repo.save_player_rolling_features_records([]) == 0
