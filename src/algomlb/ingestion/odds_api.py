from typing import List

from algomlb.config import get_settings
from algomlb.domain import Odds
from algomlb.ingestion.http_client import BaseAPIClient


class OddsAPIClient(BaseAPIClient):
    """Client for fetching live MLB odds from The-Odds-API."""

    def __init__(
        self, base_url: str = "https://api.the-odds-api.com", timeout: float = 30.0
    ):
        super().__init__(base_url=base_url, timeout=timeout)
        key = get_settings().api.odds_api_key
        if key is None:
            raise RuntimeError("The Odds API key is not configured.")
        self.api_key = key.get_secret_value()

    def fetch_live_odds(
        self, sport: str = "baseball_mlb", regions: str = "us", markets: str = "h2h"
    ) -> List[Odds]:
        """
        Fetch live odds for a specific sport and parse into Domain Odds models.
        """
        path = f"/v4/sports/{sport}/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
        }

        response = self._request("GET", path, params=params)
        data = response.json()

        odds_list: List[Odds] = []
        for game_entry in data:
            game_id = game_entry.get("id", "unknown")
            bookmakers = game_entry.get("bookmakers", [])

            for bookmaker in bookmakers:
                book_key = bookmaker.get("title") or bookmaker.get("key", "unknown")
                book_markets = bookmaker.get("markets", [])

                for market_entry in book_markets:
                    market_key = market_entry.get("key", "unknown")
                    outcomes = market_entry.get("outcomes", [])

                    for outcome in outcomes:
                        # Incorporate outcome name into market field if we want to distinguish
                        # For now, following requested Odds domain model
                        # We'll use "market:outcome" to ensure it's usable
                        outcome_name = outcome.get("name", "unknown")
                        odds_list.append(
                            Odds(
                                game_id=game_id,
                                sportsbook=book_key,
                                market=f"{market_key}:{outcome_name}",
                                price=float(outcome.get("price", 0.0)),
                                # timestamp is handled by default_factory in Odds model
                            )
                        )
        return odds_list
