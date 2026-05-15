import random
from typing import Any

import requests

from fadegoblin import config

ACTIVE_LEAGUES = [
    "soccer_epl",
    "soccer_uefa_champs_league",
    "soccer_italy_serie_a",
    "soccer_spain_la_liga",
    "soccer_usa_mls",
    "basketball_nba",
    "icehockey_nhl",
]


def format_odds(odds: int | float | None) -> str:
    """Format odds to American string format (+150 or -150)."""
    if odds is None:
        return "N/A"
    return f"+{int(odds)}" if odds > 0 else str(int(odds))


def get_live_games(max_games: int = 15) -> list[dict[str, Any]]:
    """Fetch upcoming games across multiple random active leagues."""
    random.shuffle(ACTIVE_LEAGUES)

    parsed_games = []
    leagues_queried = 0

    for league in ACTIVE_LEAGUES:
        if leagues_queried >= 3 or len(parsed_games) >= max_games:
            break

        print(f"🎲 Checking odds for {league}...")
        url = f"https://api.the-odds-api.com/v4/sports/{league}/odds"
        params = {
            "api_key": config.ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
            "bookmakers": "draftkings",
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            leagues_queried += 1

            data = response.json()
            if not data:
                continue

            for game in data:
                if not game.get("bookmakers"):
                    continue
                market = game["bookmakers"][0].get("markets", [])
                if not market:
                    continue

                outcomes = market[0].get("outcomes", [])
                home_odds = away_odds = draw_odds = None

                for outcome in outcomes:
                    if outcome["name"] == game["home_team"]:
                        home_odds = outcome["price"]
                    elif outcome["name"] == game["away_team"]:
                        away_odds = outcome["price"]
                    elif outcome["name"].lower() == "draw":
                        draw_odds = outcome["price"]

                game_info = {
                    "sport": league,
                    "home": game["home_team"],
                    "away": game["away_team"],
                    "home_odds": format_odds(home_odds),
                    "away_odds": format_odds(away_odds),
                }

                if draw_odds:
                    game_info["draw_odds"] = format_odds(draw_odds)

                parsed_games.append(game_info)

                if len(parsed_games) >= max_games:
                    break

        except Exception as e:
            print(f"⚠️ Odds API Error on {league}: {e}")
            continue

    return parsed_games
