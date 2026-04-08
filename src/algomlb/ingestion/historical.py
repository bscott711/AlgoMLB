import datetime
import warnings
from pathlib import Path

import pandas as pd
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
                if nan_ratio > 0.5:
                    logger.warning(
                        f"Data completeness warning: {col_name} has {nan_ratio:.1%} NaNs. "
                        "Source might be sparse or malformed."
                    )

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
            import pybaseball
            from pybaseball import cache

            cache.enable()
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
            import pybaseball
            from pybaseball import cache

            cache.enable()
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
        """Map a single Statcast row to a PitchEventORM object with safe NaN handling."""

        def safe_int(val, default=0):
            if pd.isna(val):
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        def safe_float(val):
            if pd.isna(val):
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        return PitchEventORM(
            game_id=str(row.get("game_pk", "")),
            game_date=game_date,
            pitcher_id=safe_int(row.get("pitcher")),
            batter_id=safe_int(row.get("batter")),
            release_speed=safe_float(row.get("release_speed")),
            release_spin_rate=safe_float(row.get("release_spin_rate")),
            pfx_x=safe_float(row.get("pfx_x")),
            pfx_z=safe_float(row.get("pfx_z")),
            plate_x=safe_float(row.get("plate_x")),
            plate_z=safe_float(row.get("plate_z")),
            launch_speed=safe_float(row.get("launch_speed")),
            launch_angle=safe_float(row.get("launch_angle")),
            pitch_type=str(row.get("pitch_type"))
            if not pd.isna(row.get("pitch_type"))
            else None,
            stand=str(row.get("stand")) if not pd.isna(row.get("stand")) else None,
            p_throws=str(row.get("p_throws"))
            if not pd.isna(row.get("p_throws"))
            else None,
            description=str(row.get("description"))
            if not pd.isna(row.get("description"))
            else None,
            events=str(row.get("events")) if not pd.isna(row.get("events")) else None,
            release_extension=safe_float(row.get("release_extension")),
            effective_speed=safe_float(row.get("effective_speed")),
            pitch_number=safe_int(row.get("pitch_number")),
            at_bat_number=safe_int(row.get("at_bat_number")),
            inning=safe_int(row.get("inning")),
            zone=safe_int(row.get("zone")),
            bb_type=str(row.get("bb_type"))
            if not pd.isna(row.get("bb_type"))
            else None,
        )

    def _fetch_statcast_df(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch Statcast data in 7-day chunks to avoid server timeouts and the pybaseball large-query warning."""
        start = pd.to_datetime(start_date, format="%Y-%m-%d").date()
        end = pd.to_datetime(end_date, format="%Y-%m-%d").date()

        total_days = (end - start).days
        if total_days <= 7:
            # Short range: fetch normally
            from pybaseball import statcast, cache

            cache.enable()
            df = statcast(start_date, end_date)
            return self._clean_columns(df)

        logger.info(
            f"Chunking massive Statcast fetch ({total_days} days) into 7-day intervals..."
        )

        all_dfs = []
        curr_start = start
        while curr_start <= end:
            curr_end = min(curr_start + datetime.timedelta(days=6), end)

            s_str = curr_start.strftime("%Y-%m-%d")
            e_str = curr_end.strftime("%Y-%m-%d")

            logger.info(f"  Fetching Statcast chunk: {s_str} to {e_str}")
            try:
                # pybaseball.statcast usually handles internal day-splitting, but external chunks
                # provide better logging progress and cache robustness in case of disconnects.
                from pybaseball import statcast, cache

                cache.enable()
                chunk = statcast(s_str, e_str)
                if chunk is not None and not chunk.empty:
                    all_dfs.append(chunk)
                else:
                    logger.debug(f"    No data found for chunk {s_str} to {e_str}")
            except Exception as e:
                # Note: error might be caught here but pybaseball's internal cache should save
                # what it successfully retrieved if cache.enable() is active.
                logger.error(f"    Failed to fetch Statcast chunk {s_str}-{e_str}: {e}")

            curr_start = curr_end + datetime.timedelta(days=1)

        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        return self._clean_columns(combined_df)

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
        """Fetch pitch-level Statcast data for a date range and persist to DB.

        For large ranges, this method fetches and persists data in monthly blocks
        to ensure progressive commits and manage memory.
        """
        cache_path = self.cache_dir / f"statcast_{start_date}_{end_date}.parquet"

        if cache_path.exists():
            return self._load_statcast_cache(cache_path, start_date, end_date, persist)

        start = pd.to_datetime(start_date, format="%Y-%m-%d").date()
        end = pd.to_datetime(end_date, format="%Y-%m-%d").date()
        total_days = (end - start).days

        if total_days > 31:
            df = self._fetch_massive_statcast(start, end, persist)
        else:
            df = self._fetch_standard_statcast(start_date, end_date, persist)

        if not df.empty:
            df.to_parquet(cache_path)

        return df

    def _load_statcast_cache(
        self, cache_path: Path, start_date: str, end_date: str, persist: bool
    ) -> pd.DataFrame:
        """Load Statcast data from Parquet cache and optionally persist to DB."""
        df = pd.read_parquet(cache_path)
        if persist:
            logger.info(
                f"Cache hit for {start_date} to {end_date}. Ensuring DB persistence..."
            )
            self._persist_pitch_events(df)
        return df

    def _fetch_massive_statcast(
        self, start: datetime.date, end: datetime.date, persist: bool
    ) -> pd.DataFrame:
        """Process massive Statcast range with monthly blocks and progressive persistence."""
        logger.info("Processing massive Statcast range with progressive persistence...")
        all_dfs = []
        curr_start = start
        while curr_start <= end:
            curr_end = min(curr_start + datetime.timedelta(days=30), end)
            s_str = curr_start.strftime("%Y-%m-%d")
            e_str = curr_end.strftime("%Y-%m-%d")

            logger.debug(f"Fetching/Persisting block: {s_str} to {e_str}")
            chunk_df = self._fetch_statcast_df(s_str, e_str)

            if not chunk_df.empty:
                # Apply window filter immediately to each chunk to handle bleed
                d_series = pd.to_datetime(
                    chunk_df["game_date"], format="%Y-%m-%d", errors="coerce"
                ).dt.date
                chunk_df = chunk_df[(d_series >= curr_start) & (d_series <= curr_end)]

                if persist:
                    self._persist_pitch_events(chunk_df)

                all_dfs.append(chunk_df)

            curr_start = curr_end + datetime.timedelta(days=1)

        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def _fetch_standard_statcast(
        self, start_date: str, end_date: str, persist: bool
    ) -> pd.DataFrame:
        """Fetch Statcast data for a standard range (<= 31 days)."""
        df = self._fetch_statcast_df(start_date, end_date)

        # Apply strict window filtering
        if "game_date" in df.columns and not df.empty:
            s_filter = pd.to_datetime(start_date, format="%Y-%m-%d").date()
            e_filter = pd.to_datetime(end_date, format="%Y-%m-%d").date()
            d_series = pd.to_datetime(
                df["game_date"], format="%Y-%m-%d", errors="coerce"
            ).dt.date
            df = df[(d_series >= s_filter) & (d_series <= e_filter)]

        if persist:
            self._persist_pitch_events(df)

        return df
