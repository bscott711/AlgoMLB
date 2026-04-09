import requests
from sqlalchemy import select
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM
from algomlb.domain import GameType
from loguru import logger


def populate_game_types(years=[2019, 2020, 2021, 2022, 2023, 2024, 2025]):
    """
    Update game_results table with correct gameType (R, P, S, E, A) from MLB Stats API
    for all available years in the database.
    """
    session_factory = get_session_factory()

    for year in years:
        type_map = _fetch_master_schedule(year)
        if not type_map:
            continue

        with session_factory() as session:
            _update_year_games(session, year, type_map)


def _fetch_master_schedule(year: int) -> dict[str, str]:
    """Helper to fetch and parse the master schedule for a given year."""
    api_url = (
        f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&"
        f"startDate={year}-01-01&endDate={year}-12-31&"
        f"gameType=R&gameType=P&gameType=S&gameType=A&gameType=E"
    )

    logger.info(f"📡 Fetching master schedule for {year}...")
    try:
        resp = requests.get(api_url, timeout=30)
        data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch {year}: {e}")
        return {}

    type_map = {}
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            type_map[str(g.get("gamePk"))] = g.get("gameType")
    return type_map


def _update_year_games(session, year: int, type_map: dict[str, str]):
    """Helper to update a specific year's games in the DB."""
    stmt = select(GameResultORM).where(
        GameResultORM.game_date >= f"{year}-01-01",
        GameResultORM.game_date <= f"{year}-12-31",
    )
    games = session.execute(stmt).scalars().all()

    updated = 0
    found_none = 0

    for game in games:
        g_type_code = type_map.get(game.game_id)
        if g_type_code:
            _apply_type_label(game, g_type_code)
            updated += 1
        else:
            found_none += 1

    session.commit()
    logger.info(
        f"{year}: Updated {updated} labels. {found_none} remained unmapped (likely noise/minor leagues)."
    )


def _apply_type_label(game, g_type_code: str):
    """Encapsulates the mapping logic for game type codes."""
    try:
        game.game_type = GameType(g_type_code)
    except ValueError:
        # Extra postseason types (F, D, L, W)
        if g_type_code in ["F", "D", "L", "W"]:
            game.game_type = GameType.POSTSEASON
        else:
            logger.warning(f"Unknown gameType code: {g_type_code}")


if __name__ == "__main__":
    populate_game_types()
