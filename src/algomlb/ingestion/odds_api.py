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
        import datetime

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
            home_team = game_entry.get("home_team", "unknown")
            away_team = game_entry.get("away_team", "unknown")
            commence_time_str = game_entry.get("commence_time")
            game_date = datetime.datetime.now(datetime.UTC).date()
            if commence_time_str:
                try:
                    ct = commence_time_str.replace("Z", "+00:00")
                    game_date = datetime.datetime.fromisoformat(ct).date()
                except Exception:
                    pass

            bookmakers = game_entry.get("bookmakers", [])
            for bookmaker in bookmakers:
                book_key = bookmaker.get("title") or bookmaker.get("key", "unknown")
                book_markets = bookmaker.get("markets", [])

                for market_entry in book_markets:
                    market_key = market_entry.get("key", "unknown")
                    outcomes = market_entry.get("outcomes", [])

                    for outcome in outcomes:
                        outcome_name = outcome.get("name", "unknown")
                        odds_list.append(
                            Odds(
                                odds_game_id=game_id,
                                home_team=home_team,
                                away_team=away_team,
                                game_date=game_date,
                                sportsbook=book_key,
                                market_type=market_key,
                                outcome=outcome_name,
                                price=float(outcome.get("price", 0.0)),
                            )
                        )
        return odds_list

    def fetch_historical_odds(
        self,
        date_snapshot: str,
        sport: str = "baseball_mlb",
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
    ) -> List[Odds]:
        """
        Fetch historical odds snapshots for a specific point in time.
        date_snapshot: ISO8601 string (e.g. 2023-05-20T12:00:00Z)
        """
        import datetime

        path = f"/v4/sports/{sport}/odds-history/"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",  # User requested American odds
            "date": date_snapshot,
        }

        response = self._request("GET", path, params=params)
        data = response.json()

        # The historical endpoint returns a 'data' list inside the response
        events = data.get("data", [])
        timestamp_str = data.get("timestamp")

        odds_list: List[Odds] = []
        for game_entry in events:
            game_id = game_entry.get("id", "unknown")
            home_team = game_entry.get("home_team", "unknown")
            away_team = game_entry.get("away_team", "unknown")
            commence_time_str = game_entry.get("commence_time")
            game_date = datetime.date.today()
            if commence_time_str:
                try:
                    ct = commence_time_str.replace("Z", "+00:00")
                    game_date = datetime.datetime.fromisoformat(ct).date()
                except Exception:
                    pass

            bookmakers = game_entry.get("bookmakers", [])
            for bookmaker in bookmakers:
                book_key = bookmaker.get("title") or bookmaker.get("key", "unknown")
                book_markets = bookmaker.get("markets", [])

                for market_entry in book_markets:
                    market_key = market_entry.get("key", "unknown")
                    outcomes = market_entry.get("outcomes", [])

                    for outcome in outcomes:
                        outcome_name = outcome.get("name", "unknown")
                        price = float(outcome.get("price", 0.0))
                        # Note: 'price' will be American format since oddsFormat=american
                        # domain.Odds model might need adjustment if it expects decimal
                        odds_list.append(
                            Odds(
                                odds_game_id=game_id,
                                home_team=home_team,
                                away_team=away_team,
                                game_date=game_date,
                                sportsbook=book_key,
                                market_type=market_key,
                                outcome=outcome_name,
                                price=price,
                                timestamp=datetime.datetime.fromisoformat(
                                    timestamp_str.replace("Z", "+00:00")
                                )
                                if timestamp_str
                                else datetime.datetime.now(datetime.UTC),
                            )
                        )
        return odds_list
