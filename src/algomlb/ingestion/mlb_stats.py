import datetime
from typing import List, Optional

from algomlb.config import get_settings
from algomlb.domain import Game, GameStatus
from algomlb.ingestion.http_client import BaseAPIClient


class MLBStatsAPIClient(BaseAPIClient):
    """Client for fetching MLB daily schedules and game data from MLB Stats API."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        url = base_url or get_settings().api.mlb_stats_url
        super().__init__(base_url=url, timeout=timeout)

    def fetch_daily_schedule(
        self, target_date: Optional[datetime.date] = None
    ) -> List[Game]:
        """
        Fetch MLB games for a specific date and parse into Domain Game models.
        """
        date_str = (target_date or datetime.date.today()).strftime("%Y-%m-%d")
        # sportId=1 is the identifier for MLB
        path = "/schedule"
        params = {"sportId": 1, "date": date_str, "hydrate": "probablePitcher"}

        response = self._request("GET", path, params=params)
        data = response.json()

        games_list: List[Game] = []
        for date_entry in data.get("dates", []):
            for game_data in date_entry.get("games", []):
                game_id = str(game_data.get("gamePk", "unknown"))
                teams = game_data.get("teams", {})

                # Get teams and pitchers
                home = teams.get("home", {})
                away = teams.get("away", {})

                home_team = home.get("team", {}).get("name", "unknown")
                away_team = away.get("team", {}).get("name", "unknown")

                # Extract probable pitchers if available via hydrate
                home_pitcher = home.get("probablePitcher", {}).get("fullName")
                away_pitcher = away.get("probablePitcher", {}).get("fullName")

                # Map MLB status to GameStatus
                detailed_state = (
                    game_data.get("status", {}).get("detailedState", "").lower()
                )
                status = GameStatus.SCHEDULED
                if "final" in detailed_state or "completed" in detailed_state:
                    status = GameStatus.COMPLETED
                elif "in progress" in detailed_state or "live" in detailed_state:
                    status = GameStatus.IN_PROGRESS
                elif "cancelled" in detailed_state:
                    status = GameStatus.CANCELLED
                elif "postponed" in detailed_state:
                    status = GameStatus.POSTPONED

                games_list.append(
                    Game(
                        game_id=game_id,
                        date=game_data.get("gameDate", date_str)[
                            :10
                        ],  # Truncate ISO time
                        home_team=home_team,
                        away_team=away_team,
                        home_pitcher=home_pitcher,
                        away_pitcher=away_pitcher,
                        home_score=home.get("score"),
                        away_score=away.get("score"),
                        status=status,
                    )
                )
        return games_list
