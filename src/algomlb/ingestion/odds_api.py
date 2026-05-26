import os
import json
import datetime
import httpx
from typing import List

from algomlb.config import get_settings
from algomlb.core.logger import logger
from algomlb.domain import Odds
from algomlb.ingestion.http_client import BaseAPIClient

STATUS_FILE = "/home/opc/AlgoMLB/.odds_api_status.json"


def load_status() -> dict:
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_status(status_dict: dict) -> None:
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(status_dict, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving key status: {e}")


def is_key_exhausted(api_key: str) -> bool:
    if not api_key:
        return False
    status_dict = load_status()
    key_info = status_dict.get(api_key)
    if not key_info:
        return False

    if key_info.get("status") == "exhausted":
        reset_at_str = key_info.get("reset_at")
        if reset_at_str:
            try:
                reset_at = datetime.datetime.fromisoformat(reset_at_str)
                if datetime.datetime.now(datetime.timezone.utc) >= reset_at:
                    key_info["status"] = "active"
                    save_status(status_dict)
                    return False
                return True
            except Exception:
                pass
    return False


def mark_key_exhausted(api_key: str) -> None:
    if not api_key:
        return
    status_dict = load_status()

    now = datetime.datetime.now(datetime.timezone.utc)
    if now.month == 12:
        next_month = 1
        next_year = now.year + 1
    else:
        next_month = now.month + 1
        next_year = now.year

    reset_at = datetime.datetime(next_year, next_month, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

    status_dict[api_key] = {
        "status": "exhausted",
        "exhausted_at": now.isoformat(),
        "reset_at": reset_at.isoformat()
    }
    save_status(status_dict)


class OddsAPIClient(BaseAPIClient):
    """Client for fetching live MLB odds from The-Odds-API."""

    def __init__(
        self, base_url: str = "https://api.the-odds-api.com", timeout: float = 30.0
    ):
        super().__init__(base_url=base_url, timeout=timeout)
        self.api_keys = []
        settings = get_settings()
        if settings.api.odds_api_key:
            self.api_keys.append(settings.api.odds_api_key.get_secret_value())
        if settings.api.odds_api_key_secondary:
            self.api_keys.append(settings.api.odds_api_key_secondary.get_secret_value())

        if not self.api_keys:
            raise RuntimeError("The Odds API key is not configured.")

        self.api_key = self.api_keys[0]
        self._active_key_index = 0

    def _request_with_rotation(self, method: str, path: str, params: dict) -> httpx.Response:
        """
        Execute request and dynamically rotate keys if one is exhausted (401 OUT_OF_USAGE_CREDITS or 429).
        """
        # Filter available keys using is_key_exhausted
        available_indices = [
            idx for idx, key in enumerate(self.api_keys)
            if not is_key_exhausted(key)
        ]

        if not available_indices:
            available_indices = list(range(len(self.api_keys)))

        # Find the first available index >= self._active_key_index
        active_indices = [i for i in available_indices if i >= self._active_key_index]
        if active_indices:
            self._active_key_index = active_indices[0]

        last_exc = None
        while self._active_key_index < len(self.api_keys):
            active_key = self.api_keys[self._active_key_index]
            params["apiKey"] = active_key

            try:
                response = self._request(method, path, params=params)
                return response
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                is_exhausted = False
                if exc.response is not None:
                    if exc.response.status_code == 401:
                        try:
                            err_data = exc.response.json()
                            if err_data.get("error_code") == "OUT_OF_USAGE_CREDITS":
                                is_exhausted = True
                        except Exception:
                            pass
                    elif exc.response.status_code == 429:
                        is_exhausted = True

                if is_exhausted:
                    logger.warning(
                        f"Odds API key at index {self._active_key_index} ({active_key[:6]}...) exhausted. "
                        "Swapping to next key."
                    )
                    mark_key_exhausted(active_key)
                    self._active_key_index += 1
                    continue
                else:
                    raise
            except Exception:
                raise

        if last_exc:
            raise last_exc
        raise RuntimeError("All configured Odds API keys are exhausted.")

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

        response = self._request_with_rotation("GET", path, params=params)
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
                        point = outcome.get("point")
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
                                point=float(point) if point is not None else None,
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

        # Correct v4 historical path: /v4/historical/sports/{sport}/odds
        path = f"/v4/historical/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",  # User requested American odds
            "date": date_snapshot,
        }

        response = self._request_with_rotation("GET", path, params=params)
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
                        point = outcome.get("point")
                        # Note: 'price' will be American format since oddsFormat=american
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
                                point=float(point) if point is not None else None,
                                timestamp=datetime.datetime.fromisoformat(
                                    timestamp_str.replace("Z", "+00:00")
                                )
                                if timestamp_str
                                else datetime.datetime.now(datetime.UTC),
                            )
                        )
        return odds_list
