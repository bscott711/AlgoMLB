import pandas as pd
from typing import Optional
import datetime
import httpx
import tempfile
import os
import kagglehub
from sqlalchemy.orm import Session
from algomlb.db.models import UmpireScorecardORM, GameResultORM
from algomlb.db.repository import DatabaseRepository
from algomlb.core.logger import logger


class UmpireScorecardIngester:
    """Ingests umpire efficiency data from CSV or API sources."""

    def __init__(self, session: Session):
        self.session = session
        self.repo = DatabaseRepository(session)

    def ingest_from_csv(self, csv_path: str):
        """
        Ingest umpire data from a CSV file.
        Expected columns: date, home_team, away_team, umpire_name, accuracy, consistency,
        favoritism_home, expected_runs, actual_runs
        """
        logger.info(f"Ingesting umpire scorecards from {csv_path}...")
        df = pd.read_csv(csv_path)

        scorecards = []
        for _, row in df.iterrows():
            game_date = pd.to_datetime(row["date"]).date()
            home = row["home_team"]
            away = row["away_team"]

            # Find Game ID
            game_id = self._find_game_id(game_date, home, away)
            if not game_id:
                logger.warning(
                    f"Could not find Game ID for {home} vs {away} on {game_date}"
                )
                continue

            orm = UmpireScorecardORM(
                game_id=game_id,
                umpire_name=row["umpire_name"],
                accuracy=float(row["accuracy"]),
                consistency=float(row["consistency"]),
                favoritism_home=float(row["favoritism_home"]),
                expected_runs=float(row["expected_runs"]),
                actual_runs=float(row["actual_runs"]),
            )
            scorecards.append(orm)

        if scorecards:
            self.repo.save_umpire_scorecards(scorecards)
            logger.success(
                f"Successfully ingested {len(scorecards)} umpire scorecards."
            )

    def ingest_from_url(self, url: str):
        """Download and ingest umpire CSV from a direct URL."""
        logger.info(f"Downloading umpire scorecard data from {url}...")
        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        try:
            self.ingest_from_csv(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def ingest_from_kaggle(self):
        """Automatically fetch and ingest the umpire scorecard dataset from Kaggle."""
        logger.info("Downloading historical umpire scorecards from Kaggle...")
        path = kagglehub.dataset_download(
            "mattop/mlb-baseball-umpire-scorecards-2015-2022"
        )
        # Find the CSV in the downloaded path
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    self.ingest_from_csv(os.path.join(root, file))
                    return
        logger.error("No CSV found in Kaggle dataset.")

    def _find_game_id(
        self, game_date: datetime.date, home: str, away: str
    ) -> Optional[str]:
        """Lookup MLB Game ID from team names and date."""
        # Simple lookup in GameResultORM
        game = (
            self.session.query(GameResultORM)
            .filter_by(game_date=game_date, home_team=home)
            .first()
        )
        return game.game_id if game else None
