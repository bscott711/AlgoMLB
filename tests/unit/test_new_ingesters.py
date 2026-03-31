import pytest
import datetime
import respx
import httpx
import json
from typing import Any
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from algomlb.db.models import (
    Base,
    GameResultORM,
)
from algomlb.ingestion.umpire_ingester import UmpireScorecardIngester


@pytest.fixture
def test_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _make_api_row(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid API row with sensible defaults."""
    base: dict[str, Any] = {
        "game_pk": 718760,
        "failed": False,
        "has_basic_game_data": True,
        "has_detailed_game_data": True,
        "fully_valid": True,
        "num_pitches_no_data": 0,
        "below_missing_cutoff": True,
        "asterisk": False,
        "ND": False,
        "date": "2023-04-01",
        "type": "R",
        "umpire": "Chad Whitson",
        "home_team": "HOU",
        "away_team": "CWS",
        "home_score": 6,
        "away_score": 4,
        "called_pitches": 175,
        "called_correct": 169,
        "called_wrong": 6,
        "overall_accuracy": 96.57,
        "baseline_x_correct_calls": 167.6,
        "x_correct_calls": 168.1,
        "correct_calls_above_x": 0.92,
        "x_overall_accuracy": 96.05,
        "accuracy_above_x": 0.52,
        "consistency": 94.86,
        "favor": -0.28,
        "home_batter_impact": -0.93,
        "home_pitcher_impact": 0.65,
        "away_batter_impact": -0.65,
        "away_pitcher_impact": 0.93,
        "total_run_impact": 1.58,
        "n_overturned": None,
        "n_challenged": None,
        "challenge_success_rate": None,
        "n_overturned_home": None,
        "n_challenged_home": None,
        "n_overturned_away": None,
        "n_challenged_away": None,
        "abs_away_A": None,
        "abs_away_B": None,
        "abs_away_C": None,
        "abs_away_D": None,
        "abs_home_A": None,
        "abs_home_B": None,
        "abs_home_C": None,
        "abs_home_D": None,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------
# UmpireScorecardIngester._api_row_to_orm
# ------------------------------------------------------------------


def test_api_row_to_orm_maps_all_fields(test_session):
    ingester = UmpireScorecardIngester(test_session)
    row = _make_api_row()
    orm = ingester._api_row_to_orm(row)

    assert orm is not None
    assert orm.game_pk == 718760
    assert orm.umpire_name == "Chad Whitson"
    assert orm.home_team == "HOU"
    assert orm.away_team == "CWS"
    assert orm.game_date == datetime.date(2023, 4, 1)
    assert orm.game_type == "R"
    assert orm.home_score == 6
    assert orm.away_score == 4
    assert orm.called_pitches == 175
    assert orm.called_correct == 169
    assert orm.called_wrong == 6
    assert orm.accuracy == pytest.approx(96.57)
    assert orm.consistency == pytest.approx(94.86)
    assert orm.favoritism_home == pytest.approx(-0.28)
    assert orm.home_batter_impact == pytest.approx(-0.93)
    assert orm.total_run_impact == pytest.approx(1.58)
    assert orm.actual_runs == pytest.approx(10.0)  # 6 + 4
    # Nullable fields
    assert orm.n_overturned is None
    assert orm.abs_away_a is None


def test_api_row_to_orm_skips_no_game_pk(test_session):
    ingester = UmpireScorecardIngester(test_session)
    row = _make_api_row(game_pk=None)
    assert ingester._api_row_to_orm(row) is None


def test_api_row_to_orm_skips_old_date(test_session):
    ingester = UmpireScorecardIngester(test_session, since_year=2023)
    row = _make_api_row(date="2022-09-15")
    assert ingester._api_row_to_orm(row) is None


def test_api_row_to_orm_resolves_game_id(test_session):
    """When a matching GameResultORM exists, game_id should be populated."""
    game = GameResultORM(
        game_id="718760",
        game_date=datetime.date(2023, 4, 1),
        home_team="Houston Astros",
        away_team="Chicago White Sox",
    )
    test_session.add(game)
    test_session.commit()

    ingester = UmpireScorecardIngester(test_session)
    orm = ingester._api_row_to_orm(_make_api_row())
    assert orm is not None
    assert orm.game_id == "718760"


def test_api_row_to_orm_no_game_id_when_no_match(test_session):
    """When no GameResultORM matches, game_id should be None."""
    ingester = UmpireScorecardIngester(test_session)
    orm = ingester._api_row_to_orm(_make_api_row())
    assert orm is not None
    assert orm.game_id is None


def test_api_row_to_orm_with_abs_data(test_session):
    """Test ABS zone fields are captured when present."""
    ingester = UmpireScorecardIngester(test_session)
    row = _make_api_row(abs_away_A=12.5, abs_away_B=3.2, abs_home_C=7.1, abs_home_D=1.0)
    orm = ingester._api_row_to_orm(row)
    assert orm is not None
    assert orm.abs_away_a == pytest.approx(12.5)
    assert orm.abs_away_b == pytest.approx(3.2)
    assert orm.abs_away_c is None
    assert orm.abs_home_c == pytest.approx(7.1)
    assert orm.abs_home_d == pytest.approx(1.0)


# ------------------------------------------------------------------
# UmpireScorecardIngester.ingest_from_api
# ------------------------------------------------------------------


@respx.mock
def test_ingest_from_api_success(test_session):
    ingester = UmpireScorecardIngester(test_session, since_year=2023)

    row1 = _make_api_row(game_pk=100001, date="2023-06-01")
    row2 = _make_api_row(game_pk=100002, date="2023-06-02")
    response_body = json.dumps({"rows": [row1, row2]})

    respx.get(
        "https://umpscorecards.us/api/games",
        params={
            "startDate": "2023-01-01",
            "endDate": "2023-12-31",
            "seasonType": "R",
        },
    ).mock(return_value=httpx.Response(200, content=response_body))

    # SQLite doesn't support pg_insert, so we mock save_umpire_scorecards
    with patch.object(ingester.repo, "save_umpire_scorecards") as mock_save:
        count = ingester.ingest_from_api(seasons=[2023])

    assert count == 2
    mock_save.assert_called_once()
    saved_scorecards = mock_save.call_args[0][0]
    assert len(saved_scorecards) == 2
    assert saved_scorecards[0].game_pk == 100001
    assert saved_scorecards[1].game_pk == 100002


@respx.mock
def test_ingest_from_api_filters_failed_rows(test_session):
    ingester = UmpireScorecardIngester(test_session, since_year=2023)

    good = _make_api_row(game_pk=200001)
    failed = _make_api_row(game_pk=200002, failed=True)
    response_body = json.dumps({"rows": [good, failed]})

    respx.get(
        "https://umpscorecards.us/api/games",
        params={
            "startDate": "2023-01-01",
            "endDate": "2023-12-31",
            "seasonType": "R",
        },
    ).mock(return_value=httpx.Response(200, content=response_body))

    with patch.object(ingester.repo, "save_umpire_scorecards") as mock_save:
        count = ingester.ingest_from_api(seasons=[2023])

    assert count == 1
    saved = mock_save.call_args[0][0]
    assert len(saved) == 1
    assert saved[0].game_pk == 200001


@respx.mock
def test_ingest_from_api_empty_response(test_session):
    ingester = UmpireScorecardIngester(test_session, since_year=2023)

    respx.get(
        "https://umpscorecards.us/api/games",
        params={
            "startDate": "2023-01-01",
            "endDate": "2023-12-31",
            "seasonType": "R",
        },
    ).mock(return_value=httpx.Response(200, content='{"rows": []}'))

    with patch.object(ingester.repo, "save_umpire_scorecards") as mock_save:
        count = ingester.ingest_from_api(seasons=[2023])

    assert count == 0
    mock_save.assert_not_called()


@respx.mock
def test_ingest_from_api_default_seasons(test_session):
    """When no seasons passed, defaults to since_year through current year."""
    ingester = UmpireScorecardIngester(test_session, since_year=2025)

    # Mock the current year (2025+2026 if current year is 2026)
    current_year = datetime.datetime.now().year
    expected_seasons = list(range(2025, current_year + 1))

    for year in expected_seasons:
        respx.get(
            "https://umpscorecards.us/api/games",
            params={
                "startDate": f"{year}-01-01",
                "endDate": f"{year}-12-31",
                "seasonType": "R",
            },
        ).mock(return_value=httpx.Response(200, content='{"rows": []}'))

    with patch.object(ingester.repo, "save_umpire_scorecards"):
        count = ingester.ingest_from_api()

    assert count == 0


@respx.mock
def test_ingest_from_api_multiple_seasons(test_session):
    """Test that multiple seasons are fetched with rate limiting."""
    ingester = UmpireScorecardIngester(test_session, since_year=2023)

    for year in [2023, 2024]:
        row = _make_api_row(game_pk=300000 + year, date=f"{year}-06-15")
        respx.get(
            "https://umpscorecards.us/api/games",
            params={
                "startDate": f"{year}-01-01",
                "endDate": f"{year}-12-31",
                "seasonType": "R",
            },
        ).mock(return_value=httpx.Response(200, content=json.dumps({"rows": [row]})))

    with patch.object(ingester.repo, "save_umpire_scorecards") as mock_save:
        with patch("algomlb.ingestion.umpire_ingester.time.sleep") as mock_sleep:
            count = ingester.ingest_from_api(seasons=[2023, 2024])

    assert count == 2
    assert mock_save.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


# ------------------------------------------------------------------
# Static helpers
# ------------------------------------------------------------------


def test_safe_float():
    assert UmpireScorecardIngester._safe_float(None) == 0.0
    assert UmpireScorecardIngester._safe_float("ND") == 0.0
    assert UmpireScorecardIngester._safe_float("invalid") == 0.0
    assert UmpireScorecardIngester._safe_float(42.5) == 42.5
    assert UmpireScorecardIngester._safe_float("3.14") == pytest.approx(3.14)


def test_safe_float_or_none():
    assert UmpireScorecardIngester._safe_float_or_none(None) is None
    assert UmpireScorecardIngester._safe_float_or_none("ND") is None
    assert UmpireScorecardIngester._safe_float_or_none("bad") is None
    assert UmpireScorecardIngester._safe_float_or_none(1.5) == 1.5


def test_safe_int():
    assert UmpireScorecardIngester._safe_int(None) is None
    assert UmpireScorecardIngester._safe_int("ND") is None
    assert UmpireScorecardIngester._safe_int("bad") is None
    assert UmpireScorecardIngester._safe_int(42) == 42
    assert UmpireScorecardIngester._safe_int(3.9) == 3
