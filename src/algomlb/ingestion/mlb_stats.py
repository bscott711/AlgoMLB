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
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        game_types: Optional[List[str]] = None,
    ) -> List[Game]:
        """
        Fetch MLB games for a specific date range and parse into Domain Game models.
        """
        s_str = (start_date or datetime.date.today()).strftime("%Y-%m-%d")
        e_str = (end_date or start_date or datetime.date.today()).strftime("%Y-%m-%d")

        # sportId=1 is the identifier for MLB
        # Default to Regular Season (R) and all Postseason types (F, D, L, W)
        g_types = game_types or ["R", "P", "F", "D", "L", "W"]
        path = "/schedule"
        params = {
            "sportId": 1,
            "startDate": s_str,
            "endDate": e_str,
            "hydrate": "probablePitcher,venue",
            "gameType": g_types,
        }

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
                home_team_id = home.get("team", {}).get("id")
                away_team_id = away.get("team", {}).get("id")

                # Extract probable pitchers and IDs if available via hydrate
                h_p_data = home.get("probablePitcher", {})
                a_p_data = away.get("probablePitcher", {})
                home_pitcher = h_p_data.get("fullName")
                away_pitcher = a_p_data.get("fullName")
                home_pitcher_id = h_p_data.get("id")
                away_pitcher_id = a_p_data.get("id")

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

                # Doubleheader identification
                # Retrosheet uses 0 for single games, 1/2 for DH.
                # MLB uses gameNumber (1, 2) and doubleHeader (Y/N/S).
                is_dh = game_data.get("doubleHeader") in ("Y", "S")
                game_num = game_data.get("gameNumber", 1)
                dh_num = 0 if not is_dh else game_num

                # Extract venue name
                venue_name = game_data.get("venue", {}).get("name")

                # Extract canonical date from the date_entry (preferred over UTC gameDate rollover)
                official_date = date_entry.get("date", s_str)

                games_list.append(
                    Game(
                        game_id=game_id,
                        date=official_date,
                        game_datetime=game_data.get("gameDate"),
                        venue_name=venue_name,
                        home_team=home_team,
                        away_team=away_team,
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        doubleheader_num=dh_num,
                        home_pitcher=home_pitcher,
                        away_pitcher=away_pitcher,
                        home_pitcher_id=home_pitcher_id,
                        away_pitcher_id=away_pitcher_id,
                        home_score=home.get("score"),
                        away_score=away.get("score"),
                        status=status,
                    )
                )
        return games_list
