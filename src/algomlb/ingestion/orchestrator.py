from algomlb.db.repository import DatabaseRepository
from algomlb.ingestion.mlb_stats import MLBStatsAPIClient
from algomlb.ingestion.odds_api import OddsAPIClient
from algomlb.ingestion.historical import HistoricalDataLoader


class IngestionOrchestrator:
    """Orchestrates data flow from external APIs to the domestic database."""

    def __init__(
        self,
        repo: DatabaseRepository,
        odds_client: OddsAPIClient,
        stats_client: MLBStatsAPIClient,
        historical_loader: HistoricalDataLoader,
    ):
        self.repo = repo
        self.odds_client = odds_client
        self.stats_client = stats_client
        self.historical_loader = historical_loader

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
        self, start_date=None, end_date=None
    ) -> int:
        """Fetch daily game schedule/results and persist games to the database."""
        games = self.stats_client.fetch_daily_schedule(start_date=start_date, end_date=end_date)
        for game in games:
            self.repo.save_game(game)
        return len(games)
