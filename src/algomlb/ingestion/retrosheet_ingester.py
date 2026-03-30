import pandas as pd
from typing import Any, Optional
import datetime
import httpx
import zipfile
import io
import tempfile
import os
from sqlalchemy.orm import Session
from algomlb.db.models import RetrosheetEventORM
from algomlb.db.repository import DatabaseRepository
from algomlb.core.logger import logger


class RetrosheetIngester:
    """Ingests play-by-play events from parsed Retrosheet CSV files."""

    def __init__(self, session: Session, chunk_size: int = 1000):
        self.session = session
        self.repo = DatabaseRepository(session)
        self.chunk_size = chunk_size

    def ingest_from_csv(self, csv_path: str):
        """
        Ingest a Retrosheet CSV file.
        Expects ALL columns from the Retrosheet play-by-play event specification.
        """
        logger.info(f"Ingesting full Retrosheet events from {csv_path}...")
        df = pd.read_csv(csv_path)

        events = []
        for i, row in df.iterrows():
            try:
                event = self._row_to_orm(row)
                events.append(event)
            except Exception as e:
                logger.error(f"Error parsing Retrosheet row {i}: {e}")
                continue

            # Commit in chunks to avoid memory pressure or session bloat
            if len(events) >= self.chunk_size:
                self.repo.save_retrosheet_events(events)
                events = []

        if events:
            self.repo.save_retrosheet_events(events)

        logger.success("Successfully processed Retrosheet file.")

    def ingest_from_url(self, url: str):
        """Download a ZIP file from Retrosheet, unzip, and ingest found CSVs."""
        logger.info(f"Downloading Retrosheet data from {url}...")
        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with tempfile.TemporaryDirectory() as tmpdir:
                z.extractall(tmpdir)
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith(".csv"):
                            self.ingest_from_csv(os.path.join(root, file))

    def _row_to_orm(self, row: Any) -> RetrosheetEventORM:
        """Helper to convert a pandas row to a RetrosheetEventORM object."""

        # Type-safe helpers for extracting row values
        def get_int(col: str) -> int:
            val = row[col]
            return int(val) if pd.notnull(val) else 0

        def get_str(col: str) -> str:
            val = row[col]
            return str(val) if pd.notnull(val) else ""

        def get_opt_int(col: str) -> Optional[int]:
            val = row[col]
            return int(val) if pd.notnull(val) else None

        def get_opt_str(col: str) -> Optional[str]:
            val = row[col]
            return str(val) if pd.notnull(val) else None

        def get_date(col: str) -> datetime.date:
            val = row[col]
            if pd.isnull(val):
                return datetime.date(1900, 1, 1)
            return pd.to_datetime(val).date()

        # Mapping dictionary to handle the high volume of fields
        data: dict[str, Any] = {
            "game_id": get_str("gid"),
            "play_number": get_int("pn"),
            "event_text": get_str("event"),
            "inning": get_int("inning"),
            "top_bot": get_int("top_bot"),
            "vis_home": get_int("vis_home"),
            "site": get_str("site"),
            "bat_team": get_str("batteam"),
            "pit_team": get_str("pitteam"),
            "score_v": get_int("score_v"),
            "score_h": get_int("score_h"),
            "batter_id": get_str("batter"),
            "pitcher_id": get_str("pitcher"),
            "lp": get_int("lp"),
            "bat_f": get_int("bat_f"),
            "batter_hand": get_opt_str("bathand"),
            "pitcher_hand": get_opt_str("pithand"),
            "balls": get_int("balls"),
            "strikes": get_int("strikes"),
            "count_text": get_opt_str("count"),
            "pitches": get_opt_str("pitches"),
            "nump": get_opt_int("nump"),
            "pa_flag": get_int("pa"),
            "ab_flag": get_int("ab"),
            "single": get_int("single"),
            "double_flag": get_int("double"),
            "triple": get_int("triple"),
            "hr": get_int("hr"),
            "sh": get_int("sh"),
            "sf": get_int("sf"),
            "hbp": get_int("hbp"),
            "walk": get_int("walk"),
            "k": get_int("k"),
            "xi": get_int("xi"),
            "roe": get_int("roe"),
            "fc": get_int("fc"),
            "othout": get_int("othout"),
            "noout": get_int("noout"),
            "bip": get_int("bip"),
            "bunt": get_int("bunt"),
            "ground": get_int("ground"),
            "fly": get_int("fly"),
            "line_flag": get_int("line"),
            "iw": get_int("iw"),
            "gdp": get_int("gdp"),
            "othdp": get_int("othdp"),
            "tp": get_int("tp"),
            "fle": get_int("fle"),
            "wp": get_int("wp"),
            "pb": get_int("pb"),
            "bk": get_int("bk"),
            "oa": get_int("oa"),
            "di": get_int("di"),
            "sb2": get_int("sb2"),
            "sb3": get_int("sb3"),
            "sbh": get_int("sbh"),
            "cs2": get_int("cs2"),
            "cs3": get_int("cs3"),
            "csh": get_int("csh"),
            "pko1": get_int("pko1"),
            "pko2": get_int("pko2"),
            "pko3": get_int("pko3"),
            "k_safe": get_int("k_safe"),
            "e1": get_int("e1"),
            "e2": get_int("e2"),
            "e3": get_int("e3"),
            "e4": get_int("e4"),
            "e5": get_int("e5"),
            "e6": get_int("e6"),
            "e7": get_int("e7"),
            "e8": get_int("e8"),
            "e9": get_int("e9"),
            "outs_pre": get_int("outs_pre"),
            "outs_post": get_int("outs_post"),
            "br1_pre": get_opt_str("br1_pre"),
            "br2_pre": get_opt_str("br2_pre"),
            "br3_pre": get_opt_str("br3_pre"),
            "br1_post": get_opt_str("br1_post"),
            "br2_post": get_opt_str("br2_post"),
            "br3_post": get_opt_str("br3_post"),
            "lob_id1": get_opt_str("lob_id1"),
            "lob_id2": get_opt_str("lob_id2"),
            "lob_id3": get_opt_str("lob_id3"),
            "pr1_pre": get_opt_str("pr1_pre"),
            "pr2_pre": get_opt_str("pr2_pre"),
            "pr3_pre": get_opt_str("pr3_pre"),
            "pr1_post": get_opt_str("pr1_post"),
            "pr2_post": get_opt_str("pr2_post"),
            "pr3_post": get_opt_str("pr3_post"),
            "run_b": get_opt_str("run_b"),
            "run1": get_opt_str("run1"),
            "run2": get_opt_str("run2"),
            "run3": get_opt_str("run3"),
            "prun_b": get_opt_str("prun_b"),
            "prun1": get_opt_str("prun1"),
            "prun2": get_opt_str("prun2"),
            "prun3": get_opt_str("prun3"),
            "ur_b": get_int("ur_b"),
            "ur1": get_int("ur1"),
            "ur2": get_int("ur2"),
            "ur3": get_int("ur3"),
            "rbi_b": get_int("rbi_b"),
            "rbi1": get_int("rbi1"),
            "rbi2": get_int("rbi2"),
            "rbi3": get_int("rbi3"),
            "runs": get_int("runs"),
            "rbi": get_int("rbi"),
            "er": get_int("er"),
            "tur": get_int("tur"),
            "f2": get_opt_str("f2"),
            "f3": get_opt_str("f3"),
            "f4": get_opt_str("f4"),
            "f5": get_opt_str("f5"),
            "f6": get_opt_str("f6"),
            "f7": get_opt_str("f7"),
            "f8": get_opt_str("f8"),
            "f9": get_opt_str("f9"),
            "po1": get_int("po1"),
            "po2": get_int("po2"),
            "po3": get_int("po3"),
            "po4": get_int("po4"),
            "po5": get_int("po5"),
            "po6": get_int("po6"),
            "po7": get_int("po7"),
            "po8": get_int("po8"),
            "po9": get_int("po9"),
            "a1": get_int("a1"),
            "a2": get_int("a2"),
            "a3": get_int("a3"),
            "a4": get_int("a4"),
            "a5": get_int("a5"),
            "a6": get_int("a6"),
            "a7": get_int("a7"),
            "a8": get_int("a8"),
            "a9": get_int("a9"),
            "fseq": get_opt_str("fseq"),
            "firstf": get_int("firstf"),
            "loc": get_opt_str("loc"),
            "hittype": get_opt_str("hittype"),
            "dpopp": get_int("dpopp"),
            "pivot": get_int("pivot"),
            "umphome": get_opt_str("umphome"),
            "ump1b": get_opt_str("ump1b"),
            "ump2b": get_opt_str("ump2b"),
            "ump3b": get_opt_str("ump3b"),
            "umplf": get_opt_str("umplf"),
            "umprf": get_opt_str("umprf"),
            "date": get_date("date"),
            "gametype": get_opt_str("gametype"),
            "pbp": get_opt_str("pbp"),
        }
        return RetrosheetEventORM(**data)
