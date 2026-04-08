import datetime
from datetime import date
from typing import Optional

from algomlb.db.repository import DatabaseRepository
from algomlb.ingestion.mlb_stats import MLBStatsAPIClient
from algomlb.ingestion.odds_api import OddsAPIClient
from algomlb.ingestion.historical import HistoricalDataLoader
from algomlb.ingestion.transactions_ingester import PlayerTransactionsIngester
from algomlb.ingestion.openmeteo_ingester import OpenMeteoIngester
from algomlb.ingestion.statcast_ingester import StatcastIngester
from algomlb.ingestion.umpire_ingester import UmpireScorecardIngester
from algomlb.ingestion.gumbo_ingester import GumboIngester


class IngestionOrchestrator:
    """Orchestrates data flow from external APIs to the domestic database."""

    def __init__(
        self,
        repo: DatabaseRepository,
        odds_client: OddsAPIClient,
        stats_client: MLBStatsAPIClient,
        historical_loader: HistoricalDataLoader,
        transactions_ingester: PlayerTransactionsIngester,
        openmeteo_ingester: OpenMeteoIngester,
        statcast_ingester: StatcastIngester,
        umpire_ingester: UmpireScorecardIngester,
        gumbo_ingester: GumboIngester | None = None,
    ):
        self.repo = repo
        self.odds_client = odds_client
        self.stats_client = stats_client
        self.historical_loader = historical_loader
        self.transactions_ingester = transactions_ingester
        self.openmeteo_ingester = openmeteo_ingester
        self.statcast_ingester = statcast_ingester
        self.umpire_ingester = umpire_ingester
        self.gumbo_ingester = gumbo_ingester

    def run_historical_ingestion(self, start_year: int, end_year: int) -> int:
        """Fetch and persist historical pitching and batting stats."""
        # This will trigger the persistence logic in the loader
        p_df = self.historical_loader.fetch_pitching_stats(start_year, end_year)
        b_df = self.historical_loader.fetch_team_batting(start_year, end_year)
        # For simplicity, returning the total count of rows processed
        return len(p_df) + len(b_df)

    def run_odds_ingestion(self) -> int:
        """Fetch live odds and persist them to the database."""
        odds_list = self.odds_client.fetch_live_odds()
        for odds in odds_list:
            self.repo.save_live_odds(odds)
        return len(odds_list)

    def run_schedule_ingestion(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> int:
        """Fetch daily game schedule/results and chunk by year for API reliability."""
        if not start_date:
            start_date = date.today()
        if not end_date:
            end_date = start_date

        current_start = start_date
        total_ingested = 0

        while current_start <= end_date:
            # Chunk by year to avoid API truncation
            year_end = date(current_start.year, 12, 31)
            current_end = min(year_end, end_date)

            games = self.stats_client.fetch_daily_schedule(
                start_date=current_start, end_date=current_end
            )
            for game in games:
                self.repo.save_game(game)

            total_ingested += len(games)
            # Move to start of next year or next day after end
            current_start = current_end + datetime.timedelta(days=1)

        return total_ingested

    def run_transaction_ingestion(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> int:
        """Fetch and persist player transactions for a given date range."""

        today = datetime.date.today()
        if start_date is None:
            start_date = today - datetime.timedelta(days=7)
        if end_date is None:
            end_date = today

        return self.transactions_ingester.ingest_range(start_date, end_date)

    def run_weather_ingestion(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> None:
        """Fetch and persist Open-Meteo weather progression for a date range."""
        today = datetime.date.today()
        if start_date is None:
            start_date = today - datetime.timedelta(days=7)
        if end_date is None:
            end_date = today

        self.openmeteo_ingester.ingest_range(start_date, end_date)

    def run_statcast_ingestion(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> int:
        """Fetch and persist raw Statcast pitch data."""
        if start_date is None:
            start_date = date.today() - datetime.timedelta(days=1)
        if end_date is None:
            end_date = start_date

        return self.statcast_ingester.ingest_range(start_date, end_date)

    def run_umpire_ingestion(self, seasons: list[int] | None = None) -> int:
        """Fetch and persist umpire scorecards."""
        return self.umpire_ingester.ingest_from_api(seasons=seasons)

    def run_gumbo_ingestion(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> int:
        """Fetch and persist GUMBO pitch timestamps for scheduled games."""
        from algomlb.db.models import GameResultORM
        from sqlalchemy import select

        if self.gumbo_ingester is None:
            return 0

        today = datetime.date.today()
        if start_date is None:
            start_date = today - datetime.timedelta(days=1)
        if end_date is None:
            end_date = start_date

        stmt = select(GameResultORM.game_id).where(
            (GameResultORM.game_date >= start_date)
            & (GameResultORM.game_date <= end_date)
        )
        game_ids = self.repo.session.execute(stmt).scalars().all()
        game_pks = []
        for gid in game_ids:
            try:
                game_pks.append(int(gid))
            except (ValueError, TypeError):
                pass

        if not game_pks:
            return 0

        return self.gumbo_ingester.ingest_games(game_pks)
