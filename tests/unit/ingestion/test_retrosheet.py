import pandas as pd
import pytest
import datetime
import io
import zipfile
from unittest.mock import MagicMock, patch
from algomlb.ingestion.retrosheet_ingester import RetrosheetIngester
from algomlb.db.models import RetrosheetEventORM


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def ingester(mock_session):
    return RetrosheetIngester(mock_session, since_year=2023)


def test_extract_helpers(ingester):
    row = {
        "int_col": 5,
        "nan_col": float("nan"),
        "str_col": "test",
        "date_col": "20230401",
        "game_post": "postseason",
    }

    assert ingester._extract_int(row, "int_col") == 5
    assert ingester._extract_int(row, "nan_col", default=1) == 1
    assert ingester._extract_opt_int(row, "nan_col") is None

    assert ingester._extract_str(row, "str_col") == "test"
    assert ingester._extract_str(row, "none_col", "def") == "def"

    assert ingester._extract_date(row, "date_col") == datetime.date(2023, 4, 1)
    assert ingester._extract_date(row, "none_col") == datetime.date(1900, 1, 1)

    assert ingester._extract_gametype(row, "game_post") == "P"
    assert ingester._extract_gametype(row, "none_col") is None


def test_handle_row_filtering(ingester):
    # Old date
    row_old = {"date": "20200101"}
    assert ingester._handle_row(row_old) is None

    # Valid date
    row_valid = {"date": "20230401", "gid": "ANA202304010", "pn": 1}
    with patch.object(ingester, "_row_to_orm") as mock_to_orm:
        mock_event = MagicMock()
        mock_event.date = datetime.date(2023, 4, 1)
        mock_to_orm.return_value = mock_event
        assert ingester._handle_row(row_valid) == mock_event


def test_ingest_from_csv(ingester):
    csv_path = "dummy.csv"
    mock_df = pd.DataFrame([{"date": "20230401", "gid": "G1", "pn": 1}])

    with (
        patch("pandas.read_csv", return_value=[mock_df]),
        patch.object(ingester.repo, "save_retrosheet_events") as mock_save,
    ):
        ingester.ingest_from_csv(csv_path)
        assert mock_save.called


def test_ingest_from_url(ingester):
    url = "http://test.com/data.zip"

    # Mock ZIP content
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.writestr("event2023.csv", "gid,date,pn\nG1,20230401,1")

    mock_resp = MagicMock()
    mock_resp.content = zip_buffer.getvalue()

    with (
        patch("httpx.get", return_value=mock_resp),
        patch("zipfile.ZipFile", return_value=zipfile.ZipFile(zip_buffer)),
        patch.object(ingester, "ingest_from_csv") as mock_ingest_csv,
    ):
        # We need to mock extractall to not actually write to disk
        with patch("tempfile.TemporaryDirectory") as mock_tmp:
            mock_tmp.return_value.__enter__.return_value = "/tmp/fake"
            with patch("os.walk", return_value=[("/tmp/fake", [], ["event2023.csv"])]):
                ingester.ingest_from_url(url)
                assert mock_ingest_csv.called


def test_row_to_orm_mapping(ingester):
    row = {
        "gid": "G1",
        "pn": 1,
        "event": "S8/L.2-H",
        "inning": 1,
        "top_bot": 0,
        "date": "20230401",
        "gametype": "regular",
        "single": 1,
        "hr": 0,
        "outs_pre": 0,
        "outs_post": 0,
        "runs": 1,
        "pitcher": "pitcher_id",
        "br1_pre": "runner_id",
    }
    event = ingester._row_to_orm(row)
    assert isinstance(event, RetrosheetEventORM)
    assert event.game_id == "G1"
    assert event.single == 1
    assert event.br1_pre == "runner_id"
    assert event.pitcher_id == "pitcher_id"
