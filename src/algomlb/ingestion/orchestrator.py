from algomlb.db.repository import DatabaseRepository
from algomlb.ingestion.mlb_stats import MLBStatsAPIClient
from algomlb.ingestion.odds_api import OddsAPIClient


class IngestionOrchestrator:
    """Orchestrates data flow from external APIs to the domestic database."""

    def __init__(
        self,
        repo: DatabaseRepository,
        odds_client: OddsAPIClient,
        stats_client: MLBStatsAPIClient,
    ):
        self.repo = repo
        self.odds_client = odds_client
        self.stats_client = stats_client

    def run_odds_ingestion(self) -> int:
        """Fetch live odds and persist them to the database."""
        odds_list = self.odds_client.fetch_live_odds()
        for odds in odds_list:
            self.repo.save_live_odds(odds)
        return len(odds_list)

    def run_schedule_ingestion(self) -> int:
        """Fetch daily game schedule and persist games to the database."""
        games = self.stats_client.fetch_daily_schedule()
        for game in games:
            self.repo.save_game(game)
        return len(games)
