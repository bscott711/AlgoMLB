import pytest
from datetime import date
from sqlalchemy.orm import Session
from algomlb.db import Base, DatabaseRepository, create_db_engine
from algomlb.db.models import BallparkORM, GameResultORM
from algomlb.domain import Game, GameStatus


@pytest.fixture
def test_session():
    """Create an in-memory SQLite database and session for testing."""
    engine = create_db_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import sessionmaker

    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with session_factory() as session:
        yield session
    Base.metadata.drop_all(engine)


def test_resolve_ballpark_id_venue_match(test_session: Session):
    repo = DatabaseRepository(test_session)
    bp = BallparkORM(id=10, team_name="ARI", ballpark="Chase Field")
    test_session.add(bp)
    test_session.commit()

    game = Game(
        game_id="g1",
        date=date(2024, 4, 1),
        home_team="Arizona Diamondbacks",
        away_team="LA Dodgers",
        venue_name="Chase Field",  # Exact match ilike
        status=GameStatus.SCHEDULED,
    )
    bid = repo._resolve_ballpark_id(game)
    assert bid == 10


def test_resolve_ballpark_id_abb_match(test_session: Session):
    repo = DatabaseRepository(test_session)
    bp = BallparkORM(id=20, team_name="NYY", ballpark="Yankee Stadium")
    test_session.add(bp)
    test_session.commit()

    game = Game(
        game_id="g2",
        date=date(2024, 4, 1),
        home_team="New York Yankees",
        away_team="Boston Red Sox",
        venue_name="Unknown Venue",
        status=GameStatus.SCHEDULED,
    )
    # NYY is in TEAM_NAME_TO_ABB
    bid = repo._resolve_ballpark_id(game)
    assert bid == 20


def test_resolve_ballpark_id_ilike_team_match(test_session: Session):
    repo = DatabaseRepository(test_session)
    bp = BallparkORM(id=30, team_name="Strange Team", ballpark="Strange Park")
    test_session.add(bp)
    test_session.commit()

    game = Game(
        game_id="g3",
        date=date(2024, 4, 1),
        home_team="Strange Team",
        away_team="Visitor",
        venue_name="Unknown",
        status=GameStatus.SCHEDULED,
    )
    # Falling back to ilike team_name
    bid = repo._resolve_ballpark_id(game)
    assert bid == 30


def test_save_game_update_ballpark(test_session: Session):
    repo = DatabaseRepository(test_session)
    # 1. Save game without ballpark
    game = Game(
        game_id="g4",
        date=date(2024, 4, 1),
        home_team="Team X",
        away_team="Team Y",
        status=GameStatus.SCHEDULED,
    )
    repo.save_game(game)

    orm = test_session.query(GameResultORM).filter_by(game_id="g4").one()
    assert orm.ballpark_id is None

    # 2. Add ballpark and update game
    bp = BallparkORM(id=40, team_name="TX", ballpark="X Park")
    test_session.add(bp)
    test_session.commit()

    # Use model_copy because Game model is frozen
    game = game.model_copy(update={"home_team": "TX"})
    repo.save_game(game)

    test_session.refresh(orm)
    assert orm.ballpark_id == 40
