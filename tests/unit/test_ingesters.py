import datetime
from unittest.mock import MagicMock
import pandas as pd
import pytest
from algomlb.db import Base, create_db_engine
from algomlb.domain import Odds
from algomlb.ingestion.ballpark_ingester import BallparkIngester
from algomlb.ingestion.historical_odds import HistoricalOddsIngester
from algomlb.db.models import GameResultORM, BallparkORM, HistoricalOddsORM
from sqlalchemy.orm import sessionmaker
from algomlb.db.repository import DatabaseRepository


@pytest.fixture
def test_session():
    """Create an in-memory SQLite database and session for testing."""
    engine = create_db_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        yield session
    Base.metadata.drop_all(engine)


def test_ballpark_ingester_basic(test_session, tmp_path):
    """Test full ingestion of ballpark data from a CSV file."""
    csv_file = tmp_path / "ballparks.csv"
    data = [
        {
            "team_name": "NYY",
            "ballpark": "Yankee Stadium",
            "left_field": 318,
            "center_field": 408,
            "right_field": 314,
            "min_wall_height": 8.0,
            "max_wall_height": 15.0,
            "hr_park_effects": 1.1,
            "extra_distance": 5.0,
            "avg_temp": 70.0,
            "elevation": 50,
            "roof": 0.0,
            "daytime": 1.0,
        },
        {
            "team_name": "BOS",
            "ballpark": "Fenway Park",
            "left_field": 310,
            "center_field": 390,
            "right_field": 302,
            "min_wall_height": None,
            "max_wall_height": None,
            "hr_park_effects": None,
            "extra_distance": None,
            "avg_temp": None,
            "elevation": None,
            "roof": None,
            "daytime": None,
        },
    ]
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    ingester = BallparkIngester(test_session)
    ingester.ingest_from_csv(str(csv_file))

    nyy = test_session.query(BallparkORM).filter_by(team_name="NYY").first()
    assert nyy is not None
    assert nyy.ballpark == "Yankee Stadium"
    assert nyy.left_field == 318

    bos = test_session.query(BallparkORM).filter_by(team_name="BOS").first()
    assert bos.left_field == 310
    assert bos.min_wall_height is None


def test_historical_odds_ingester_full(test_session):
    """Test processing historical odds snapshots and mapping to ORMs."""
    game_date = datetime.date(2023, 4, 1)
    game = GameResultORM(
        game_id="20230401NYYTOR",
        game_date=game_date,
        home_team="Toronto Blue Jays",
        away_team="New York Yankees",
    )
    test_session.add(game)
    test_session.commit()

    mock_client = MagicMock()

    # Snapshot timestamp in Odds response
    ts = datetime.datetime(2023, 4, 1, 10, 0, tzinfo=datetime.UTC)

    mock_odds_list = [
        Odds(
            odds_game_id="h1",
            home_team="Toronto Blue Jays",
            away_team="New York Yankees",
            game_date=game_date,
            sportsbook="DraftKings",
            market_type="h2h",
            outcome="Toronto Blue Jays",
            price=-110.0,
            timestamp=ts,
        ),
        Odds(
            odds_game_id="h1",
            home_team="Toronto Blue Jays",
            away_team="New York Yankees",
            game_date=game_date,
            sportsbook="DraftKings",
            market_type="h2h",
            outcome="New York Yankees",
            price=-110.0,
            timestamp=ts,
        ),
    ]
    mock_client.fetch_historical_odds.return_value = mock_odds_list

    repo = DatabaseRepository(test_session)
    ingester = HistoricalOddsIngester(repo, client=mock_client)

    ingester.ingest_day_snapshots(game_date)

    results = test_session.query(HistoricalOddsORM).all()
    # ingest_day_snapshots calls it twice (opening/closing). Both return 1 record in this mock.
    assert len(results) == 2
    record = results[0]
    assert record.game_id == game.id
    assert record.home_price == -110
    assert record.snapshot_at == ts.replace(tzinfo=None)


def test_historical_odds_ingester_run_backfill_loop(test_session):
    """Test run_backfill loop when games exist in DB."""
    game = GameResultORM(
        game_id="GBF1",
        game_date=datetime.date(2023, 4, 1),
        home_team="A",
        away_team="B",
    )
    test_session.add(game)
    test_session.commit()

    repo = DatabaseRepository(test_session)
    ingester = HistoricalOddsIngester(repo)
    # This will hit the 'pass' or loop through games
    ingester.run_backfill(datetime.date(2023, 4, 1), datetime.date(2023, 4, 1))


def test_historical_odds_ingester_create_orm_invalid(test_session):
    """Test _create_orm returning None when no prices found."""
    repo = DatabaseRepository(test_session)
    ingester = HistoricalOddsIngester(repo)
    data = {"outcomes": {}, "snapshot_at": datetime.datetime.now()}
    # H2H with no matching teams in outcomes
    orm = ingester._create_orm(1, "Home", "Away", "Bookie", "h2h", "opening", data)
    assert orm is None


def test_historical_odds_ingester_no_odds(test_session):
    mock_client = MagicMock()
    mock_client.fetch_historical_odds.return_value = []
    repo = DatabaseRepository(test_session)
    ingester = HistoricalOddsIngester(repo, client=mock_client)
    ingester._process_snapshot("fake_ts", "opening")
    assert test_session.query(HistoricalOddsORM).count() == 0


def test_historical_odds_ingester_no_game(test_session):
    mock_client = MagicMock()
    ts = datetime.datetime(2023, 4, 1, 10, 0, tzinfo=datetime.UTC)
    mock_client.fetch_historical_odds.return_value = [
        Odds(
            odds_game_id="h1",
            home_team="NonExistent",
            away_team="TeamB",
            game_date=datetime.date(2023, 4, 1),
            sportsbook="DraftKings",
            market_type="h2h",
            outcome="NonExistent",
            price=-110.0,
            timestamp=ts,
        )
    ]
    repo = DatabaseRepository(test_session)
    ingester = HistoricalOddsIngester(repo, client=mock_client)
    ingester._process_snapshot("fake_ts", "opening")
    assert test_session.query(HistoricalOddsORM).count() == 0


def test_historical_odds_ingester_error(test_session):
    mock_client = MagicMock()
    mock_client.fetch_historical_odds.side_effect = Exception("API Fail")
    repo = DatabaseRepository(test_session)
    ingester = HistoricalOddsIngester(repo, client=mock_client)
    # Should not raise exception
    ingester._process_snapshot("fake_ts", "opening")


def test_historical_odds_ingester_run_backfill(test_session):
    repo = DatabaseRepository(test_session)
    ingester = HistoricalOddsIngester(repo)
    # Just verify it doesn't crash (logic is mostly 'pass' or simple loops)
    ingester.run_backfill(datetime.date(2023, 4, 1), datetime.date(2023, 4, 2))
