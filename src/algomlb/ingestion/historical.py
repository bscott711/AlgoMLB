import datetime
import warnings
from pathlib import Path

import pandas as pd
import pybaseball  # type: ignore
from loguru import logger

from algomlb.db.repository import DatabaseRepository
from algomlb.db.models import HistoricalDataORM, PitchEventORM

# Suppress pybaseball pandas datetime FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module=".*pybaseball.*")


class HistoricalDataLoader:
    """Fetches, cleans, and caches historical MLB data using pybaseball and persists to DB."""

    def __init__(self, repo: DatabaseRepository, cache_dir: Path = Path(".data/cache")):
        self.repo = repo
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        return df

    def _validate_completeness(
        self, df: pd.DataFrame, required_metrics: list[str]
    ) -> None:
        """
        Verify that the DataFrame isn't poisoned with excessive NaNs (Completeness Check).
        If more than 20% of required metrics are missing, raise a ValueError.
        """
        if df.empty:
            return

        for metric in required_metrics:
            col_name = str(metric).lower().replace(" ", "_")
            if col_name in df.columns:
                nan_ratio = df[col_name].isna().mean()
                if nan_ratio > 0.2:
                    logger.error(
                        f"Data completeness failure: {col_name} has {nan_ratio:.1%} NaNs. "
                        "Source might be malformed or blocked."
                    )
                    raise ValueError(f"Excessive NaNs detected in {col_name}")

    def _persist_stats(self, df: pd.DataFrame, metric_subset: list[str]) -> None:
        """Convert DF rows to HistoricalDataORM and save."""
        import datetime

        today = datetime.date.today()
        orms = []

        # Use index as a player identifier proxy if 'mlb_id' or 'id' not present
        pid_col = "id" if "id" in df.columns else "playerid"

        for _, row in df.iterrows():
            if pid_col not in row or pd.isna(row[pid_col]):
                continue
            player_id = int(row[pid_col])
            for metric in metric_subset:
                if metric in row and not pd.isna(row[metric]):
                    orms.append(
                        HistoricalDataORM(
                            player_id=player_id,
                            date=today,
                            metric_name=metric,
                            metric_value=float(row[metric]),
                        )
                    )
        if orms:
            self.repo.save_historical_data(orms)

    def fetch_pitching_stats(
        self, start_year: int, end_year: int, persist: bool = True
    ) -> pd.DataFrame:
        """Fetch pitching stats for a year range, using Parquet cache if available."""
        cache_path = self.cache_dir / f"pitching_{start_year}_{end_year}.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
        else:
            df = pybaseball.pitching_stats(start_year, end_year)
            df = self._clean_columns(df)
            df.to_parquet(cache_path)

        if persist:
            # Validate completeness of key ERA estimators to detect poisoned FanGraphs scrapes
            self._validate_completeness(df, ["era", "fip"])
            # Persist key metrics identified in Tier 1 (ERA estimators)
            self._persist_stats(df, ["era", "fip", "xfip", "siera"])

        return df

    def fetch_team_batting(
        self, start_year: int, end_year: int, persist: bool = True
    ) -> pd.DataFrame:
        """Fetch team batting stats for a year range, using Parquet cache if available."""
        cache_path = self.cache_dir / f"team_batting_{start_year}_{end_year}.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
        else:
            df = pybaseball.team_batting(start_year, end_year)
            df = self._clean_columns(df)
            df.to_parquet(cache_path)

        if persist:
            # Validate completeness for hitting metrics
            self._validate_completeness(df, ["woba", "wrc+"])
            # Persist key metrics for hitting (wOBA, wRC+)
            self._persist_stats(df, ["woba", "wrc+"])

        return df

    def _row_to_pitch_event(
        self, row: pd.Series, game_date: datetime.date
    ) -> PitchEventORM:
        """Map a single Statcast row to a PitchEventORM object."""
        return PitchEventORM(
            game_id=str(row.get("game_pk", "")),
            game_date=game_date,
            pitcher_id=int(row.get("pitcher", 0)),
            batter_id=int(row.get("batter", 0)),
            release_speed=float(row.get("release_speed"))  # type: ignore
            if not pd.isna(row.get("release_speed"))
            else None,
            release_spin_rate=float(row.get("release_spin_rate"))  # type: ignore
            if not pd.isna(row.get("release_spin_rate"))
            else None,
            pfx_x=float(row.get("pfx_x"))  # type: ignore
            if not pd.isna(row.get("pfx_x"))
            else None,
            pfx_z=float(row.get("pfx_z"))  # type: ignore
            if not pd.isna(row.get("pfx_z"))
            else None,
            plate_x=float(row.get("plate_x"))  # type: ignore
            if not pd.isna(row.get("plate_x"))
            else None,
            plate_z=float(row.get("plate_z"))  # type: ignore
            if not pd.isna(row.get("plate_z"))
            else None,
            launch_speed=float(row.get("launch_speed"))  # type: ignore
            if not pd.isna(row.get("launch_speed"))
            else None,
            launch_angle=float(row.get("launch_angle"))  # type: ignore
            if not pd.isna(row.get("launch_angle"))
            else None,
            pitch_type=str(row.get("pitch_type"))  # type: ignore
            if not pd.isna(row.get("pitch_type"))
            else None,
            stand=str(row.get("stand"))  # type: ignore
            if not pd.isna(row.get("stand"))
            else None,
            p_throws=str(row.get("p_throws"))  # type: ignore
            if not pd.isna(row.get("p_throws"))
            else None,
            description=str(row.get("description"))  # type: ignore
            if not pd.isna(row.get("description"))
            else None,
            events=str(row.get("events"))  # type: ignore
            if not pd.isna(row.get("events"))
            else None,
            release_extension=float(row.get("release_extension"))  # type: ignore
            if not pd.isna(row.get("release_extension"))
            else None,
            effective_speed=float(row.get("effective_speed"))  # type: ignore
            if not pd.isna(row.get("effective_speed"))
            else None,
            pitch_number=int(row.get("pitch_number", 0))
            if not pd.isna(row.get("pitch_number"))
            else None,
            at_bat_number=int(row.get("at_bat_number", 0))
            if not pd.isna(row.get("at_bat_number"))
            else None,
            inning=int(row.get("inning", 0))  # type: ignore
            if not pd.isna(row.get("inning"))
            else None,
            zone=int(row.get("zone"))  # type: ignore
            if not pd.isna(row.get("zone"))
            else None,
            bb_type=str(row.get("bb_type"))  # type: ignore
            if not pd.isna(row.get("bb_type"))
            else None,
        )

    def _fetch_statcast_df(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch Statcast data from primary or fallback API endpoints."""
        # Optimized Statcast scraper
        df = pybaseball.statcast(start_date, end_date)
        return self._clean_columns(df)

    def _persist_pitch_events(self, df: pd.DataFrame) -> None:
        """Parse rows and persist pitch events to the database."""
        orms = []
        for _, row in df.iterrows():
            try:
                raw_date = row.get("game_date")
                if isinstance(raw_date, (datetime.date, datetime.datetime)):
                    game_date = (
                        raw_date
                        if isinstance(raw_date, datetime.date)
                        else raw_date.date()
                    )
                else:
                    game_date = datetime.datetime.strptime(
                        str(raw_date), "%Y-%m-%d"
                    ).date()

                orms.append(self._row_to_pitch_event(row, game_date))
            except Exception as e:
                if not orms:
                    logger.debug(f"Row iteration failed: {e}")
                continue

        if orms:
            logger.info(f"Saving {len(orms)} pitch events to DB...")
            self.repo.save_pitch_events(orms)
        else:
            logger.warning("No valid PitchEventORM records created from Statcast data.")

    def fetch_statcast(
        self, start_date: str, end_date: str, persist: bool = True
    ) -> pd.DataFrame:
        """Fetch pitch-level Statcast data for a date range and persist to DB."""
        cache_path = self.cache_dir / f"statcast_{start_date}_{end_date}.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
        else:
            df = self._fetch_statcast_df(start_date, end_date)
            df.to_parquet(cache_path)

        # Pybaseball cache artifacts can bleed dates. Apply strict window filtering.
        if "game_date" in df.columns and not df.empty:
            start = pd.to_datetime(start_date).date()
            end = pd.to_datetime(end_date).date()
            # Coerce errors to handle malformed dates in test data, specify format for performance/warning fix
            date_series = pd.to_datetime(
                df["game_date"], errors="coerce", format="%Y-%m-%d"
            ).dt.date
            df = df[(date_series >= start) & (date_series <= end)]

        if persist:
            self._persist_pitch_events(df)

        return df
