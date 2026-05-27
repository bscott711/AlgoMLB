import os
import json
import datetime
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
        print(f"⚠️ Error saving key status: {e}")


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
                # If now is past reset_at, it's no longer exhausted!
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

    # Calculate first day of next month in UTC
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


def format_odds(odds: int | float | None) -> str:
    """Format odds to American string format (+150 or -150)."""
    if odds is None:
        return "N/A"
    return f"+{int(odds)}" if odds > 0 else str(int(odds))


def get_fliff_mlb_odds() -> dict[str, dict[str, str]]:
    """Fetches upcoming MLB games from OddsAPI for Fliff.
    Returns a dict mapping 'AWAY @ HOME' -> {'home_odds': '+120', 'away_odds': '-140', 'home': 'HOME', 'away': 'AWAY'}
    """
    all_keys = []
    if getattr(config, "ODDS_API_KEY", None):
        all_keys.append(config.ODDS_API_KEY)
    if getattr(config, "ODDS_API_KEY_SECONDARY", None):
        all_keys.append(config.ODDS_API_KEY_SECONDARY)

    keys = [k for k in all_keys if not is_key_exhausted(k)]
    if not keys and all_keys:
        keys = all_keys

    if not keys:
        print("⚠️ No Odds API keys configured.")
        return {}

    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    
    # We will import abbreviate_team locally to avoid circular imports if any,
    # or just rely on ev_logic's abbreviate_team
    from fadegoblin.ev_logic import abbreviate_team
    
    fliff_odds = {}
    key_index = 0
    while key_index < len(keys):
        active_key = keys[key_index]
        params = {
            "api_key": active_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
            "bookmakers": "fliff",
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            is_exhausted = False
            if response.status_code == 401:
                try:
                    err_data = response.json()
                    if err_data.get("error_code") == "OUT_OF_USAGE_CREDITS":
                        is_exhausted = True
                except Exception:
                    pass
            elif response.status_code == 429:
                is_exhausted = True

            if is_exhausted:
                print(f"⚠️ Odds API key {active_key[:6]}... exhausted. Swapping to next key.")
                mark_key_exhausted(active_key)
                key_index += 1
                continue

            response.raise_for_status()
            data = response.json()
            
            for game in data:
                if not game.get("bookmakers"):
                    continue
                market = game["bookmakers"][0].get("markets", [])
                if not market:
                    continue

                outcomes = market[0].get("outcomes", [])
                home_odds = away_odds = None

                for outcome in outcomes:
                    if outcome["name"] == game["home_team"]:
                        home_odds = outcome["price"]
                    elif outcome["name"] == game["away_team"]:
                        away_odds = outcome["price"]

                home_abbr = abbreviate_team(game["home_team"])
                away_abbr = abbreviate_team(game["away_team"])
                game_str = f"{away_abbr} @ {home_abbr}"
                
                fliff_odds[game_str] = {
                    "home": home_abbr,
                    "away": away_abbr,
                    "home_odds": format_odds(home_odds),
                    "away_odds": format_odds(away_odds),
                }
            break

        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                status = e.response.status_code
                if status in (401, 429):
                    mark_key_exhausted(active_key)
                    key_index += 1
                    continue
            print(f"⚠️ Odds API Error on Fliff fetch with key {active_key[:6]}...: {e}")
            break

    return fliff_odds


def get_live_games(max_games: int = 15) -> list[dict[str, Any]]:
    """Fetch upcoming games across multiple random active leagues."""
    random.shuffle(ACTIVE_LEAGUES)

    parsed_games = []
    leagues_queried = 0

    all_keys = []
    if getattr(config, "ODDS_API_KEY", None):
        all_keys.append(config.ODDS_API_KEY)
    if getattr(config, "ODDS_API_KEY_SECONDARY", None):
        all_keys.append(config.ODDS_API_KEY_SECONDARY)

    # Filter out keys known to be exhausted
    keys = [k for k in all_keys if not is_key_exhausted(k)]

    # Fallback to all keys if all configured keys are marked exhausted
    if not keys and all_keys:
        print("⚠️ All configured keys are marked as exhausted. Trying them anyway as a fallback.")
        keys = all_keys

    if not keys:
        print("⚠️ No Odds API keys configured.")
        return parsed_games

    key_index = 0

    for league in ACTIVE_LEAGUES:
        if leagues_queried >= 3 or len(parsed_games) >= max_games:
            break

        print(f"🎲 Checking odds for {league}...")
        url = f"https://api.the-odds-api.com/v4/sports/{league}/odds"

        while key_index < len(keys):
            active_key = keys[key_index]
            params = {
                "api_key": active_key,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
                "bookmakers": "draftkings",
            }

            try:
                response = requests.get(url, params=params, timeout=10)

                # Check for quota exhaustion specifically
                is_exhausted = False
                if response.status_code == 401:
                    try:
                        err_data = response.json()
                        if err_data.get("error_code") == "OUT_OF_USAGE_CREDITS":
                            is_exhausted = True
                    except Exception:
                        pass
                elif response.status_code == 429:
                    is_exhausted = True

                if is_exhausted:
                    print(f"⚠️ Odds API key {active_key[:6]}... exhausted. Swapping to next key.")
                    mark_key_exhausted(active_key)
                    key_index += 1
                    continue

                response.raise_for_status()
                leagues_queried += 1

                data = response.json()
                if not data:
                    break

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

                break  # Successful check completed

            except Exception as e:
                if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                    status = e.response.status_code
                    if status in (401, 429):
                        print(f"⚠️ Odds API key {active_key[:6]}... failed with status {status}. Swapping to next key.")
                        mark_key_exhausted(active_key)
                        key_index += 1
                        continue
                print(f"⚠️ Odds API Error on {league} with key {active_key[:6]}...: {e}")
                break

    return parsed_games
