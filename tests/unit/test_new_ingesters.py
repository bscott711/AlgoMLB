import pytest
import pandas as pd
from typing import Any
from sqlalchemy.orm import sessionmaker
from algomlb.db.models import (
    Base,
    UmpireScorecardORM,
    RetrosheetEventORM,
    GameResultORM,
)
from algomlb.ingestion.umpire_ingester import UmpireScorecardIngester
from algomlb.ingestion.retrosheet_ingester import RetrosheetIngester
import datetime
import respx
import httpx
import zipfile
import io
from unittest.mock import patch


@pytest.fixture
def test_session():
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_umpire_scorecard_ingester_no_game(test_session, tmp_path):
    # Setup - CSV data with no corresponding game in DB
    csv_file = tmp_path / "umpire_no_game.csv"
    df = pd.DataFrame(
        [
            {
                "date": "2023-04-01",
                "home_team": "NO_GAME",
                "away_team": "TOR",
                "umpire_name": "Pat Hoberg",
                "accuracy": 98.5,
                "consistency": 99.0,
                "favoritism_home": 0.5,
                "expected_runs": 8.0,
                "actual_runs": 8.0,
            }
        ]
    )
    df.to_csv(csv_file, index=False)

    ingester = UmpireScorecardIngester(test_session)
    ingester.ingest_from_csv(str(csv_file))

    assert test_session.query(UmpireScorecardORM).count() == 0


def test_umpire_scorecard_ingester(test_session, tmp_path):
    # Setup - Need a game to map to
    game = GameResultORM(
        game_id="20230401NYYTOR",
        game_date=datetime.date(2023, 4, 1),
        home_team="New York Yankees",
        away_team="Toronto Blue Jays",
    )
    test_session.add(game)
    test_session.commit()

    csv_file = tmp_path / "umpire.csv"
    df = pd.DataFrame(
        [
            {
                "date": "2023-04-01",
                "home_team": "NYY",
                "away_team": "TOR",
                "umpire_name": "Pat Hoberg",
                "accuracy": 98.5,
                "consistency": 99.0,
                "favoritism_home": 0.5,
                "expected_runs": 8.0,
                "actual_runs": 8.0,
            }
        ]
    )
    df.to_csv(csv_file, index=False)

    ingester = UmpireScorecardIngester(test_session)
    ingester.ingest_from_csv(str(csv_file))

    sc = test_session.query(UmpireScorecardORM).first()
    assert sc is not None
    assert sc.umpire_name == "Pat Hoberg"
    assert sc.game_id == "20230401NYYTOR"


def test_retrosheet_ingester_full(test_session, tmp_path):
    csv_file = tmp_path / "retro_full.csv"
    # Create a minimal row that satisfies all columns expected by the ingester
    cols = [
        "gid",
        "pn",
        "event",
        "inning",
        "top_bot",
        "vis_home",
        "site",
        "batteam",
        "pitteam",
        "score_v",
        "score_h",
        "batter",
        "pitcher",
        "lp",
        "bat_f",
        "bathand",
        "pithand",
        "balls",
        "strikes",
        "count",
        "pitches",
        "nump",
        "pa",
        "ab",
        "single",
        "double",
        "triple",
        "hr",
        "sh",
        "sf",
        "hbp",
        "walk",
        "k",
        "xi",
        "roe",
        "fc",
        "othout",
        "noout",
        "bip",
        "bunt",
        "ground",
        "fly",
        "line",
        "iw",
        "gdp",
        "othdp",
        "tp",
        "fle",
        "wp",
        "pb",
        "bk",
        "oa",
        "di",
        "sb2",
        "sb3",
        "sbh",
        "cs2",
        "cs3",
        "csh",
        "pko1",
        "pko2",
        "pko3",
        "k_safe",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "e6",
        "e7",
        "e8",
        "e9",
        "outs_pre",
        "outs_post",
        "br1_pre",
        "br2_pre",
        "br3_pre",
        "br1_post",
        "br2_post",
        "br3_post",
        "lob_id1",
        "lob_id2",
        "lob_id3",
        "pr1_pre",
        "pr2_pre",
        "pr3_pre",
        "pr1_post",
        "pr2_post",
        "pr3_post",
        "run_b",
        "run1",
        "run2",
        "run3",
        "prun_b",
        "prun1",
        "prun2",
        "prun3",
        "ur_b",
        "ur1",
        "ur2",
        "ur3",
        "rbi_b",
        "rbi1",
        "rbi2",
        "rbi3",
        "runs",
        "rbi",
        "er",
        "tur",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "po1",
        "po2",
        "po3",
        "po4",
        "po5",
        "po6",
        "po7",
        "po8",
        "po9",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
        "a9",
        "fseq",
        "firstf",
        "loc",
        "hittype",
        "dpopp",
        "pivot",
        "umphome",
        "ump1b",
        "ump2b",
        "ump3b",
        "umplf",
        "umprf",
        "date",
        "gametype",
        "pbp",
    ]
    data1: dict[str, Any] = {c: 0 for c in cols}
    data1.update(
        {
            "gid": "ATL202304010",
            "pn": 1,
            "date": "2023-04-01",
            "site": "ATL03",
            "event": "K",
            "inning": 1,
            "batter": "b1",
            "pitcher": "p1",
            "count": "11",
            "fseq": "2",
        }
    )
    data2 = data1.copy()
    data2.update({"pn": 2, "event": "S"})
    data3 = data1.copy()
    data3.update({"pn": 3, "event": "W"})

    df = pd.DataFrame([data1, data2, data3])
    df.to_csv(csv_file, index=False)

    ingester = RetrosheetIngester(test_session, chunk_size=2)
    ingester.ingest_from_csv(str(csv_file))

    assert test_session.query(RetrosheetEventORM).count() == 3


def test_retrosheet_ingester_error_handling(test_session, tmp_path):
    csv_file = tmp_path / "retro_error.csv"
    # Row with missing required field 'gid' to trigger exception
    df = pd.DataFrame([{"pn": 1, "date": "2023-04-01"}])
    df.to_csv(csv_file, index=False)

    ingester = RetrosheetIngester(test_session)
    # Should not raise exception but log error
    ingester.ingest_from_csv(str(csv_file))
    assert test_session.query(RetrosheetEventORM).count() == 0


def test_retrosheet_ingester_missing_date(test_session, tmp_path):
    csv_file = tmp_path / "retro_missing_date.csv"
    # Use a minimal set of all expected columns to avoid KeyErrors
    cols = [
        "gid",
        "pn",
        "event",
        "inning",
        "top_bot",
        "vis_home",
        "site",
        "batteam",
        "pitteam",
        "score_v",
        "score_h",
        "batter",
        "pitcher",
        "lp",
        "bat_f",
        "bathand",
        "pithand",
        "balls",
        "strikes",
        "count",
        "pitches",
        "nump",
        "pa",
        "ab",
        "single",
        "double",
        "triple",
        "hr",
        "sh",
        "sf",
        "hbp",
        "walk",
        "k",
        "xi",
        "roe",
        "fc",
        "othout",
        "noout",
        "bip",
        "bunt",
        "ground",
        "fly",
        "line",
        "iw",
        "gdp",
        "othdp",
        "tp",
        "fle",
        "wp",
        "pb",
        "bk",
        "oa",
        "di",
        "sb2",
        "sb3",
        "sbh",
        "cs2",
        "cs3",
        "csh",
        "pko1",
        "pko2",
        "pko3",
        "k_safe",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "e6",
        "e7",
        "e8",
        "e9",
        "outs_pre",
        "outs_post",
        "br1_pre",
        "br2_pre",
        "br3_pre",
        "br1_post",
        "br2_post",
        "br3_post",
        "lob_id1",
        "lob_id2",
        "lob_id3",
        "pr1_pre",
        "pr2_pre",
        "pr3_pre",
        "pr1_post",
        "pr2_post",
        "pr3_post",
        "run_b",
        "run1",
        "run2",
        "run3",
        "prun_b",
        "prun1",
        "prun2",
        "prun3",
        "ur_b",
        "ur1",
        "ur2",
        "ur3",
        "rbi_b",
        "rbi1",
        "rbi2",
        "rbi3",
        "runs",
        "rbi",
        "er",
        "tur",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "po1",
        "po2",
        "po3",
        "po4",
        "po5",
        "po6",
        "po7",
        "po8",
        "po9",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
        "a9",
        "fseq",
        "firstf",
        "loc",
        "hittype",
        "dpopp",
        "pivot",
        "umphome",
        "ump1b",
        "ump2b",
        "ump3b",
        "umplf",
        "umprf",
        "date",
        "gametype",
        "pbp",
    ]
    data: dict[str, Any] = {c: 0 for c in cols}
    data.update({"gid": "T1", "pn": 1, "date": None})
    df = pd.DataFrame([data])
    df.to_csv(csv_file, index=False)

    ingester = RetrosheetIngester(test_session, since_year=0)
    ingester.ingest_from_csv(str(csv_file))

    event = test_session.query(RetrosheetEventORM).first()
    assert event is not None
    assert event.date == datetime.date(1900, 1, 1)


@respx.mock
def test_retrosheet_ingester_url(test_session):
    ingester = RetrosheetIngester(test_session)
    url = "https://example.com/plays.zip"

    # Create a mock ZIP with a CSV
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        # Re-use the columns list logic or just a simple mock CSV
        z.writestr(
            "test.csv", "gid,pn,event,inning,top_bot,date\nT1,1,K,1,0,2023-04-01\n"
        )

    respx.get(url).mock(return_value=httpx.Response(200, content=buf.getvalue()))

    # We need to mock _row_to_orm or provide all columns to avoid KeyError
    # Let's mock _row_to_orm to simplify since we tested mapping already
    with patch.object(ingester, "_row_to_orm") as mock_mapping:
        mock_mapping.return_value = RetrosheetEventORM(
            game_id="T1",
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
            date=datetime.date(2023, 4, 1),
        )
        ingester.ingest_from_url(url)

    assert test_session.query(RetrosheetEventORM).count() > 0


@respx.mock
def test_umpire_scorecard_ingester_url(test_session):
    game = GameResultORM(
        game_id="20230401NYYTOR",
        game_date=datetime.date(2023, 4, 1),
        home_team="New York Yankees",
        away_team="Toronto Blue Jays",
    )
    test_session.add(game)
    test_session.commit()

    ingester = UmpireScorecardIngester(test_session)
    url = "https://example.com/umpire.csv"
    csv_content = "date,home_team,away_team,umpire_name,accuracy,consistency,favoritism_home,expected_runs,actual_runs\n2023-04-01,NYY,TOR,Pat Hoberg,98.5,99.0,0.5,8.0,8.0\n"

    respx.get(url).mock(return_value=httpx.Response(200, content=csv_content.encode()))

    ingester.ingest_from_url(url)
    assert test_session.query(UmpireScorecardORM).count() == 1


def test_umpire_scorecard_ingester_kaggle(test_session, tmp_path):
    game = GameResultORM(
        game_id="20230401NYYTOR",
        game_date=datetime.date(2023, 4, 1),
        home_team="New York Yankees",
        away_team="Toronto Blue Jays",
    )
    test_session.add(game)
    test_session.commit()

    ingester = UmpireScorecardIngester(test_session)

    # Create dummy kaggle download path
    kaggle_dir = tmp_path / "kaggle_data"
    kaggle_dir.mkdir()
    csv_file = kaggle_dir / "data.csv"
    csv_file.write_text(
        "date,home_team,away_team,umpire_name,accuracy,consistency,favoritism_home,expected_runs,actual_runs\n2023-04-01,NYY,TOR,Pat Hoberg,98.5,99.0,0.5,8.0,8.0\n"
    )

    with patch("kagglehub.dataset_download") as mock_download:
        mock_download.return_value = str(kaggle_dir)
        ingester.ingest_from_kaggle()

    assert test_session.query(UmpireScorecardORM).count() == 1


def test_umpire_scorecard_ingester_kaggle_no_csv(test_session, tmp_path):
    ingester = UmpireScorecardIngester(test_session)
    kaggle_dir = tmp_path / "empty_kaggle_data"
    kaggle_dir.mkdir()

    with patch("kagglehub.dataset_download") as mock_download:
        mock_download.return_value = str(kaggle_dir)
        # Should log error and return
        ingester.ingest_from_kaggle()

    assert test_session.query(UmpireScorecardORM).count() == 0


def test_umpire_scorecard_ingester_filters_old_data(test_session, tmp_path):
    ingester = UmpireScorecardIngester(test_session, since_year=2020)  # Filter out 2019
    csv_file = tmp_path / "old_umpire.csv"
    csv_file.write_text(
        "date,home_team,away_team,umpire_name,accuracy,consistency,favoritism_home,expected_runs,actual_runs\n2019-04-01,NYY,TOR,Pat Hoberg,98.5,99.0,0.5,8.0,8.0\n"
    )

    ingester.ingest_from_csv(str(csv_file))
    assert test_session.query(UmpireScorecardORM).count() == 0


def test_retrosheet_ingester_filters_old_data(test_session, tmp_path):
    ingester = RetrosheetIngester(test_session, since_year=2020)
    csv_file = tmp_path / "old_retro.csv"
    csv_file.write_text(
        "date,gid,pn,event,inning,top_bot\n20190401,ATL201904010,1,K,1,0\n"
    )

    ingester.ingest_from_csv(str(csv_file))
    assert test_session.query(RetrosheetEventORM).count() == 0


def test_umpire_scorecard_ingester_missing_team(test_session, tmp_path):
    ingester = UmpireScorecardIngester(test_session)
    csv_file = tmp_path / "missing_team.csv"
    csv_file.write_text(
        "date,home_team,away_team,umpire_name,accuracy,consistency\n2022-04-01,,TOR,Ump1,95.0,95.0\n"
    )

    ingester.ingest_from_csv(str(csv_file))
    assert test_session.query(UmpireScorecardORM).count() == 0


def test_umpire_scorecard_ingester_safe_float_and_runs(test_session, tmp_path):
    """Test _safe_float with ND and non-numeric values, and test home/away_team_runs mapping."""
    # Setup - Need a game to map to
    game = GameResultORM(
        game_id="G1",
        game_date=datetime.date(2023, 4, 1),
        home_team="New York Yankees",
        away_team="Toronto Blue Jays",
    )
    test_session.add(game)
    test_session.commit()

    ingester = UmpireScorecardIngester(test_session)

    # Directly test _safe_float for coverage
    assert ingester._safe_float("ND") == 0.0
    assert ingester._safe_float("invalid") == 0.0
    assert ingester._safe_float(None) == 0.0

    # Test CSV with home_team_runs/away_team_runs and invalid numeric formats
    csv_file = tmp_path / "umpire_runs.csv"
    csv_file.write_text(
        "date,home_team,away_team,umpire_name,accuracy,consistency,favoritism_home,expected_runs,home_team_runs,away_team_runs\n"
        "2023-04-01,NYY,TOR,Pat Hoberg,98.5,ND,0.5,8.0,4,4\n"
        "2023-04-01,NYY,TOR,Missing Team,,98.5,0.5,8.0,4,4\n"
    )
    # The second row is fine but would have been flagged if missing teams, but we use NYY/TOR there.
    # Let's add a row with missing teams to specifically hit 61-62
    with open(csv_file, "a") as f:
        f.write("2023-04-01,,TOR,No Home,95,95,0,0,0,0\n")

    ingester.ingest_from_csv(str(csv_file))
    # Should have 2 valid rows (the ones with teams)
    assert test_session.query(UmpireScorecardORM).count() == 2
