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

    def __init__(self, session: Session, since_year: int = 2019):
        self.session = session
        self.repo = DatabaseRepository(session)
        self.since_year = since_year

    def _safe_float(self, value, default=0.0):
        """Safely convert a value to float, handling 'ND' or other non-numeric strings."""
        if pd.isna(value) or value == "ND":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def ingest_from_csv(self, csv_path: str):
        """
        Ingest umpire data from a CSV file.
        Expected columns: date, home_team, away_team, umpire_name, accuracy, consistency,
        favoritism_home, expected_runs, actual_runs
        """
        logger.info(
            f"Ingesting umpire scorecards from {csv_path} (since {self.since_year})..."
        )
        df = pd.read_csv(csv_path)

        # Standardize column names if we detect the Kaggle format
        column_map = {
            "home": "home_team",
            "away": "away_team",
            "umpire": "umpire_name",
            "favor_home": "favoritism_home",
            "total_run_impact": "expected_runs",
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        scorecards = []
        for _, row in df.iterrows():
            game_date = pd.to_datetime(row["date"]).date()
            if game_date.year < self.since_year:
                continue

            home = row.get("home_team")
            away = row.get("away_team")
            if pd.isna(home) or pd.isna(away) or not home or not away:
                logger.warning(f"Skipping row on {game_date}: Missing team names.")
                continue

            # Find Game ID
            game_id = self._find_game_id(game_date, str(home), str(away))
            if not game_id:
                logger.warning(
                    f"Could not find Game ID for {home} vs {away} on {game_date}"
                )
                continue

            # Map runs if available
            actual_runs = 0.0
            if "home_team_runs" in row and "away_team_runs" in row:
                actual_runs = self._safe_float(
                    row["home_team_runs"]
                ) + self._safe_float(row["away_team_runs"])
            elif "actual_runs" in row:
                actual_runs = self._safe_float(row["actual_runs"])

            ump_name = str(row.get("umpire_name", "Unknown"))[:100]

            orm = UmpireScorecardORM(
                game_id=game_id,
                umpire_name=ump_name,
                accuracy=self._safe_float(row.get("accuracy", 0)),
                consistency=self._safe_float(row.get("consistency", 0)),
                favoritism_home=self._safe_float(row.get("favoritism_home", 0)),
                expected_runs=self._safe_float(row.get("expected_runs", 0)),
                actual_runs=actual_runs,
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
        # Team Mapping: Abbreviation -> Full Name (as stored by StatsAPI)
        team_map = {
            "ARI": "Arizona Diamondbacks",
            "ATL": "Atlanta Braves",
            "BAL": "Baltimore Orioles",
            "BOS": "Boston Red Sox",
            "CHC": "Chicago Cubs",
            "CWS": "Chicago White Sox",
            "CIN": "Cincinnati Reds",
            "CLE": "Cleveland Guardians",
            "COL": "Colorado Rockies",
            "DET": "Detroit Tigers",
            "HOU": "Houston Astros",
            "KC": "Kansas City Royals",
            "LAA": "Los Angeles Angels",
            "LAD": "Los Angeles Dodgers",
            "MIA": "Miami Marlins",
            "MIL": "Milwaukee Brewers",
            "MIN": "Minnesota Twins",
            "NYM": "New York Mets",
            "NYY": "New York Yankees",
            "OAK": "Oakland Athletics",
            "PHI": "Philadelphia Phillies",
            "PIT": "Pittsburgh Pirates",
            "SD": "San Diego Padres",
            "SF": "San Francisco Giants",
            "SEA": "Seattle Mariners",
            "STL": "St. Louis Cardinals",
            "TB": "Tampa Bay Rays",
            "TEX": "Texas Rangers",
            "TOR": "Toronto Blue Jays",
            "WSH": "Washington Nationals",
        }

        # Resolve full name if it's an abbreviation
        full_home_name = team_map.get(home, home)

        # Simple lookup in GameResultORM
        game = (
            self.session.query(GameResultORM)
            .filter_by(game_date=game_date, home_team=full_home_name)
            .first()
        )
        return game.game_id if game else None
