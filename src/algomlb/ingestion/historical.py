from pathlib import Path

import pandas as pd
import pybaseball  # type: ignore


from algomlb.db.repository import DatabaseRepository
from algomlb.db.models import HistoricalDataORM, PitchEventORM


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

    def _persist_stats(self, df: pd.DataFrame, metric_subset: list[str]) -> None:
        """Convert DF rows to HistoricalDataORM and save."""
        # Simple implementation: use a default date (today) if not present
        # This is high-level; production ETL would handle dates per row
        import datetime

        today = datetime.date.today()
        orms = []

        # Use index as a player identifier proxy if 'mlb_id' or 'id' not present
        pid_col = "id" if "id" in df.columns else "playerid"

        for _, row in df.iterrows():
            if pid_col not in row:
                continue
            player_id = int(row[pid_col])
            for metric in metric_subset:
                if metric in row:
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
            # Persist key metrics for hitting (wOBA, wRC+)
            self._persist_stats(df, ["woba", "wrc+"])

        return df

    def fetch_statcast(
        self, start_date: str, end_date: str, persist: bool = True
    ) -> pd.DataFrame:
        """Fetch pitch-level Statcast data for a date range and persist to DB."""
        cache_path = self.cache_dir / f"statcast_{start_date}_{end_date}.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
        else:
            df = pybaseball.statcast(start_date, end_date)
            df = self._clean_columns(df)
            df.to_parquet(cache_path)

        if persist:
            import datetime

            orms = []
            for _, row in df.iterrows():
                try:
                    game_date = datetime.datetime.strptime(
                        str(row["game_date"]), "%Y-%m-%d"
                    ).date()
                    orms.append(
                        PitchEventORM(
                            game_id=str(row.get("game_pk", "")),
                            game_date=game_date,
                            pitcher_id=int(row.get("pitcher", 0)),
                            batter_id=int(row.get("batter", 0)),
                            release_speed=float(row.get("release_speed", 0.0))
                            if not pd.isna(row.get("release_speed"))
                            else None,
                            release_spin_rate=float(row.get("release_spin_rate", 0.0))
                            if not pd.isna(row.get("release_spin_rate"))
                            else None,
                            pfx_x=float(row.get("pfx_x", 0.0))
                            if not pd.isna(row.get("pfx_x"))
                            else None,
                            pfx_z=float(row.get("pfx_z", 0.0))
                            if not pd.isna(row.get("pfx_z"))
                            else None,
                            plate_x=float(row.get("plate_x", 0.0))
                            if not pd.isna(row.get("plate_x"))
                            else None,
                            plate_z=float(row.get("plate_z", 0.0))
                            if not pd.isna(row.get("plate_z"))
                            else None,
                            launch_speed=float(row.get("launch_speed", 0.0))
                            if not pd.isna(row.get("launch_speed"))
                            else None,
                            launch_angle=float(row.get("launch_angle", 0.0))
                            if not pd.isna(row.get("launch_angle"))
                            else None,
                        )
                    )
                except Exception:
                    continue
            if orms:
                self.repo.save_pitch_events(orms)

        return df
