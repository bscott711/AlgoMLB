import time
import datetime
from typing import Mapping, Optional, TypeAlias, cast

import httpx
from sqlalchemy.orm import Session

from algomlb.core.logger import logger
from algomlb.db.models import GameResultORM, UmpireScorecardORM
from algomlb.db.repository import DatabaseRepository

# Base URL for the undocumented umpscorecards.us JSON API
_API_BASE = "https://umpscorecards.us/api/games"


class UmpireScorecardIngester:
    """Ingests umpire accuracy data from the umpscorecards.us API."""

    def __init__(self, session: Session, since_year: int = 2019):
        self.session = session
        self.repo = DatabaseRepository(session)
        self.since_year = since_year

    # ------------------------------------------------------------------
    # Public ingestion entry points
    # ------------------------------------------------------------------

    def ingest_from_api(self, seasons: list[int] | None = None) -> int:
        """Scrape umpire scorecard data from the umpscorecards.us JSON API.

        Fetches one full season per HTTP request (no pagination needed).
        Uses a 1-second sleep between season requests for polite throttling.
        """
        if seasons is None:
            current_year = datetime.datetime.now().year
            seasons = list(range(self.since_year, current_year + 1))

        total = 0
        for year in seasons:
            start = f"{year}-01-01"
            end = f"{year}-12-31"
            url = f"{_API_BASE}?startDate={start}&endDate={end}&seasonType=R"

            logger.info(f"Fetching UmpScorecards for {year} season...")
            response = httpx.get(url, timeout=60.0, follow_redirects=True)
            response.raise_for_status()

            data = response.json()
            rows = data.get("rows", [])

            scorecards: list[UmpireScorecardORM] = []
            for row in rows:
                if row.get("failed"):
                    continue
                sc = self._api_row_to_orm(row)
                if sc is not None:
                    scorecards.append(sc)

            if scorecards:
                self.repo.save_umpire_scorecards(scorecards)
                total += len(scorecards)
                logger.info(f"  → Ingested {len(scorecards)} scorecards for {year}.")

            # Polite rate-limiting between seasons
            if year != seasons[-1]:
                time.sleep(1.0)

        logger.success(f"Total UmpScorecard API ingestion: {total} records.")
        return total

    # ------------------------------------------------------------------
    # Row mapping
    # ------------------------------------------------------------------

    def _api_row_to_orm(self, row: Mapping[str, object]) -> Optional[UmpireScorecardORM]:
        """Convert a single API JSON row to an ORM object."""
        game_pk = row.get("game_pk")
        if not game_pk:
            return None

        game_date = datetime.datetime.strptime(cast(str, row["date"]), "%Y-%m-%d").date()
        if game_date.year < self.since_year:
            return None

        # Lookup the game_results FK if the game exists in our DB
        game_id = self._find_game_id(cast(int | str, game_pk))

        return UmpireScorecardORM(
            game_pk=int(cast(int, game_pk)),
            game_id=game_id,
            game_date=game_date,
            game_type=row.get("type"),
            umpire_name=str(row.get("umpire", "Unknown"))[:100],
            home_team=str(row.get("home_team", ""))[:5],
            away_team=str(row.get("away_team", ""))[:5],
            home_score=self._safe_int(row.get("home_score")),
            away_score=self._safe_int(row.get("away_score")),
            # Accuracy
            called_pitches=self._safe_int(row.get("called_pitches")),
            called_correct=self._safe_int(row.get("called_correct")),
            called_wrong=self._safe_int(row.get("called_wrong")),
            accuracy=self._safe_float(row.get("overall_accuracy", 0)),
            x_overall_accuracy=self._safe_float_or_none(row.get("x_overall_accuracy")),
            accuracy_above_x=self._safe_float_or_none(row.get("accuracy_above_x")),
            baseline_x_correct_calls=self._safe_float_or_none(
                row.get("baseline_x_correct_calls")
            ),
            x_correct_calls=self._safe_float_or_none(row.get("x_correct_calls")),
            correct_calls_above_x=self._safe_float_or_none(
                row.get("correct_calls_above_x")
            ),
            # Consistency & Favor
            consistency=self._safe_float(row.get("consistency", 0)),
            favoritism_home=self._safe_float(row.get("favor", 0)),
            # Impact
            home_batter_impact=self._safe_float_or_none(row.get("home_batter_impact")),
            home_pitcher_impact=self._safe_float_or_none(
                row.get("home_pitcher_impact")
            ),
            away_batter_impact=self._safe_float_or_none(row.get("away_batter_impact")),
            away_pitcher_impact=self._safe_float_or_none(
                row.get("away_pitcher_impact")
            ),
            total_run_impact=self._safe_float_or_none(row.get("total_run_impact")),
            expected_runs=self._safe_float(row.get("total_run_impact", 0)),
            actual_runs=self._safe_float(
                (cast(float, row.get("home_score")) or 0.0) + (cast(float, row.get("away_score")) or 0.0)
            ),
            # Challenges
            n_overturned=self._safe_int(row.get("n_overturned")),
            n_challenged=self._safe_int(row.get("n_challenged")),
            challenge_success_rate=self._safe_float_or_none(
                row.get("challenge_success_rate")
            ),
            n_overturned_home=self._safe_int(row.get("n_overturned_home")),
            n_challenged_home=self._safe_int(row.get("n_challenged_home")),
            n_overturned_away=self._safe_int(row.get("n_overturned_away")),
            n_challenged_away=self._safe_int(row.get("n_challenged_away")),
            # ABS Zones
            abs_away_a=self._safe_float_or_none(row.get("abs_away_A")),
            abs_away_b=self._safe_float_or_none(row.get("abs_away_B")),
            abs_away_c=self._safe_float_or_none(row.get("abs_away_C")),
            abs_away_d=self._safe_float_or_none(row.get("abs_away_D")),
            abs_home_a=self._safe_float_or_none(row.get("abs_home_A")),
            abs_home_b=self._safe_float_or_none(row.get("abs_home_B")),
            abs_home_c=self._safe_float_or_none(row.get("abs_home_C")),
            abs_home_d=self._safe_float_or_none(row.get("abs_home_D")),
            # Metadata
            fully_valid=cast(bool, row.get("fully_valid")),
            num_pitches_no_data=self._safe_int(row.get("num_pitches_no_data")),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_game_id(self, game_pk: int | str) -> Optional[str]:
        """Lookup MLB game_id from GameResultORM by matching game_pk."""
        game = self.session.query(GameResultORM).filter_by(game_id=str(game_pk)).first()
        return game.game_id if game else None

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        """Safely convert a value to float."""
        if value is None or value == "ND":
            return default
        try:
            return float(value)  # type: ignore
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float_or_none(value: object) -> Optional[float]:
        """Safely convert a value to float, returning None for missing data."""
        if value is None or value == "ND":
            return None
        try:
            return float(value)  # type: ignore
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(value: object) -> Optional[int]:
        """Safely convert a value to int, returning None for missing data."""
        if value is None or value == "ND":
            return None
        try:
            return int(value)  # type: ignore
        except (ValueError, TypeError):
            return None
