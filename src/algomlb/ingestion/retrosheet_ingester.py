import pandas as pd
from typing import Any, Mapping, Optional, cast
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

    def __init__(
        self, session: Session, chunk_size: int = 1000, since_year: int = 2019
    ):
        self.session = session
        self.repo = DatabaseRepository(session)
        self.chunk_size = chunk_size
        self.since_year = since_year

    def ingest_from_csv(self, csv_path: str):
        """
        Ingest a Retrosheet CSV file.
        Expects ALL columns from the Retrosheet play-by-play event specification.
        """
        logger.info(
            f"Ingesting Retrosheet events from {csv_path} (since {self.since_year})..."
        )
        total_ingested = 0
        # Use chunking to avoid memory issues with the massive Retrosheet file
        for chunk in pd.read_csv(csv_path, chunksize=self.chunk_size):
            events = []
            for _, row in chunk.iterrows():
                event = self._handle_row(row)
                if event:
                    events.append(event)

            if events:
                self.repo.save_retrosheet_events(events)
                total_ingested += len(events)

        logger.success(
            f"Successfully processed Retrosheet file: {total_ingested} events ingested."
        )

    def _handle_row(self, row: Mapping[str, object]) -> Optional[RetrosheetEventORM]:
        """Process a single row with filtering and ORM conversion."""
        try:
            date_val = str(row.get("date", ""))
            year_pref = int(date_val[:4]) if date_val and date_val[:4].isdigit() else 0
            if year_pref < self.since_year:
                return None

            event = self._row_to_orm(row)
            if event and event.date and event.date.year >= self.since_year:
                return event
        except Exception as e:
            logger.debug(f"Row filtered/skipped: {e}")

        return None

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

    def _extract_int(
        self, row: Mapping[str, object], col: str, default: int = 0
    ) -> int:
        val = row.get(col)
        return (
            int(float(cast(Any, val)))
            if val is not None and not pd.isna(cast(Any, val))
            else default
        )

    def _extract_str(
        self, row: Mapping[str, object], col: str, default: str = ""
    ) -> str:
        val = row.get(col)
        return str(val) if val is not None and not pd.isna(cast(Any, val)) else default

    def _extract_opt_int(self, row: Mapping[str, object], col: str) -> Optional[int]:
        val = row.get(col)
        return (
            int(float(cast(Any, val)))
            if val is not None and not pd.isna(cast(Any, val))
            else None
        )

    def _extract_opt_str(self, row: Mapping[str, object], col: str) -> Optional[str]:
        val = row.get(col)
        return str(val) if val is not None and not pd.isna(cast(Any, val)) else None

    def _extract_date(self, row: Mapping[str, object], col: str) -> datetime.date:
        val = row.get(col)
        if val is None or pd.isna(cast(Any, val)):
            return datetime.date(1900, 1, 1)
        return pd.to_datetime(str(val)).date()

    def _extract_gametype(self, row: Mapping[str, object], col: str) -> Optional[str]:
        val = row.get(col)
        if val is None or pd.isna(cast(Any, val)):
            return None
        val_str = str(val).lower()
        return "P" if "post" in val_str else "R"

    def _row_to_orm(self, row: Mapping[str, object]) -> RetrosheetEventORM:
        """Helper to convert a pandas row to a RetrosheetEventORM object."""
        # Split mapping into logical segments to keep complexity low
        identity = {
            "game_id": self._extract_str(row, "gid"),
            "play_number": self._extract_int(row, "pn"),
            "event_text": self._extract_str(row, "event"),
            "inning": self._extract_int(row, "inning"),
            "top_bot": self._extract_int(row, "top_bot"),
            "vis_home": self._extract_int(row, "vis_home"),
            "bat_team": self._extract_str(row, "batteam"),
            "pit_team": self._extract_str(row, "pitteam"),
            "batter_id": self._extract_str(row, "batter"),
            "pitcher_id": self._extract_str(row, "pitcher"),
            "date": self._extract_date(row, "date"),
            "gametype": self._extract_gametype(row, "gametype"),
        }

        outcomes = {
            "single": self._extract_int(row, "single"),
            "double_flag": self._extract_int(row, "double"),
            "triple": self._extract_int(row, "triple"),
            "hr": self._extract_int(row, "hr"),
            "walk": self._extract_int(row, "walk"),
            "hbp": self._extract_int(row, "hbp"),
            "k": self._extract_int(row, "k"),
            "k_safe": self._extract_int(row, "k_safe"),
            "error_flag": any(
                self._extract_int(row, f"e{i}") > 0 for i in range(1, 10)
            ),
            "outs_pre": self._extract_int(row, "outs_pre"),
            "outs_post": self._extract_int(row, "outs_post"),
            "runs": self._extract_int(row, "runs"),
            "rbi": self._extract_int(row, "rbi"),
        }

        # Mapping the remainder including those with direct name mappings
        remainder = {
            "site": self._extract_str(row, "site"),
            "score_v": self._extract_int(row, "score_v"),
            "score_h": self._extract_int(row, "score_h"),
            "lp": self._extract_int(row, "lp"),
            "bat_f": self._extract_int(row, "bat_f"),
            "batter_hand": self._extract_opt_str(row, "bathand"),
            "pitcher_hand": self._extract_opt_str(row, "pithand"),
            "balls": self._extract_int(row, "balls"),
            "strikes": self._extract_int(row, "strikes"),
            "count_text": self._extract_opt_str(row, "count"),
            "pitches": self._extract_opt_str(row, "pitches"),
            "nump": self._extract_opt_int(row, "nump"),
            "pa_flag": self._extract_int(row, "pa"),
            "ab_flag": self._extract_int(row, "ab"),
        }

        # Handle defensive and other columns (po, a, e, f, br, pr, etc.)
        for prefix in ["e", "po", "a", "f"]:
            for i in range(1, 10):
                field = f"{prefix}{i}"
                if prefix == "f":
                    identity[field] = self._extract_opt_str(row, field)
                else:
                    identity[field] = self._extract_int(row, field)

        # Runner states
        for base in [1, 2, 3]:
            for event in ["_pre", "_post"]:
                field = f"br{base}{event}"
                identity[field] = self._extract_opt_str(row, field)
                field_p = f"pr{base}{event}"
                identity[field_p] = self._extract_opt_str(row, field_p)

        data = {**identity, **outcomes, **remainder}
        # Final pass for specific optional strings not caught in loops
        data.update(
            {
                "fseq": self._extract_opt_str(row, "fseq"),
                "loc": self._extract_opt_str(row, "loc"),
                "hittype": self._extract_opt_str(row, "hittype"),
                "umphome": self._extract_opt_str(row, "umphome"),
                "ump1b": self._extract_opt_str(row, "ump1b"),
                "ump2b": self._extract_opt_str(row, "ump2b"),
                "ump3b": self._extract_opt_str(row, "ump3b"),
                "pbp": self._extract_opt_str(row, "pbp"),
            }
        )

        return RetrosheetEventORM(
            **{k: v for k, v in data.items() if hasattr(RetrosheetEventORM, k)}
        )
