import datetime
import io
import zipfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from algomlb.db import Base, create_db_engine
from algomlb.ingestion.retrosheet_ingester import RetrosheetIngester


@pytest.fixture
def test_session():
    """Create an in-memory SQLite database and session for testing."""
    engine = create_db_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import sessionmaker

    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        yield session
    Base.metadata.drop_all(engine)


@pytest.fixture
def ingester(test_session):
    return RetrosheetIngester(test_session, since_year=2020)


def test_handle_row_filter_year(ingester):
    # Skip if year < since_year
    row_old = pd.Series({"date": 20190401})
    assert ingester._handle_row(row_old) is None

    # Process if year >= since_year
    row_new = pd.Series(
        {
            "gid": "G1",
            "pn": 1,
            "event": "K",
            "inning": 1,
            "top_bot": 0,
            "vis_home": 0,
            "site": "TEST",
            "batteam": "ATL",
            "pitteam": "NYY",
            "batter": "b1",
            "pitcher": "p1",
            "lp": 1,
            "bat_f": 2,
            "date": 20200401,
        }
    )
    event = ingester._handle_row(row_new)
    assert event is not None
    assert event.date == datetime.date(2020, 4, 1)


def test_ingest_from_csv(ingester):
    csv_content = (
        "gid,pn,event,inning,top_bot,vis_home,site,batteam,pitteam,batter,pitcher,lp,bat_f,date\n"
        "G1,1,K,1,0,0,TEST,ATL,NYY,b1,p1,1,2,20200401\n"
    )
    df = pd.read_csv(io.StringIO(csv_content))
    with patch("pandas.read_csv") as mock_read:
        mock_read.return_value = [df]  # chunked

        with patch.object(ingester.repo, "save_retrosheet_events") as mock_save:
            ingester.ingest_from_csv("dummy.csv")
            assert mock_save.called
            events = mock_save.call_args[0][0]
            assert len(events) == 1
            assert events[0].game_id == "G1"


@patch("httpx.get")
@patch("algomlb.ingestion.retrosheet_ingester.logger")
def test_ingest_from_url(mock_logger, mock_get, ingester):
    # Mock ZIP content
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.writestr(
            "test.csv",
            "gid,pn,event,inning,top_bot,vis_home,site,batteam,pitteam,batter,pitcher,lp,bat_f,date\n"
            "G1,1,K,1,0,0,TEST,ATL,NYY,b1,p1,1,2,20200401\n",
        )

    mock_response = MagicMock()
    mock_response.content = zip_buffer.getvalue()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    with patch.object(ingester, "ingest_from_csv") as mock_ingest:
        ingester.ingest_from_url("http://example.com/data.zip")
        assert mock_ingest.called


def test_row_to_orm_mapping(ingester):
    row = pd.Series(
        {
            "gid": "G2",
            "pn": 5,
            "event": "HR",
            "inning": 9,
            "top_bot": 1,
            "vis_home": 1,
            "site": "NYY01",
            "batteam": "NYY",
            "pitteam": "BOS",
            "batter": "judga001",
            "pitcher": "salec001",
            "lp": 3,
            "bat_f": 9,
            "date": "20230704",
        }
    )
    orm = ingester._row_to_orm(row)
    assert orm.game_id == "G2"
    assert orm.play_number == 5
    assert orm.event_text == "HR"
    assert orm.date == datetime.date(2023, 7, 4)


def test_handle_row_exception(ingester):
    # Test exception handling (e.g., ValueError in _row_to_orm)
    # We need to pass the year filter first
    row = pd.Series({"date": 20230101, "pn": "invalid_int"})
    with patch("algomlb.ingestion.retrosheet_ingester.logger") as mock_logger:
        result = ingester._handle_row(row)
        assert result is None
        # Verify logger.debug was called because _row_to_orm raised an exception
        assert mock_logger.debug.called


def test_row_to_orm_missing_date(ingester):
    # Line 104: Test when date is null
    row = pd.Series({"gid": "G3", "pn": 1, "date": None})
    orm = ingester._row_to_orm(row)
    assert orm.date == datetime.date(1900, 1, 1)
