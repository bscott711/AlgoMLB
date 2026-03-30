from pathlib import Path

import pandas as pd
import pybaseball  # type: ignore


class HistoricalDataLoader:
    """Fetches, cleans, and caches historical MLB data using pybaseball."""

    def __init__(self, cache_dir: Path = Path(".data/cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df

    def fetch_pitching_stats(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch pitching stats for a year range, using Parquet cache if available."""
        cache_path = self.cache_dir / f"pitching_{start_year}_{end_year}.parquet"

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        # pybaseball format: year range
        df = pybaseball.pitching_stats(start_year, end_year)
        df = self._clean_columns(df)
        df.to_parquet(cache_path)
        return df

    def fetch_team_batting(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch team batting stats for a year range, using Parquet cache if available."""
        cache_path = self.cache_dir / f"team_batting_{start_year}_{end_year}.parquet"

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        df = pybaseball.team_batting(start_year, end_year)
        df = self._clean_columns(df)
        df.to_parquet(cache_path)
        return df
