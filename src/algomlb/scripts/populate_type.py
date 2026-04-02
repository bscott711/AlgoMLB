
import requests
from sqlalchemy import select, update
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM
from algomlb.domain import GameType

def populate_game_types(year=2025):
    """
    Update game_results table with correct gameType (R, P, etc.) from MLB Stats API.
    """
    session_factory = get_session_factory()
    with session_factory() as session:
        # Fetch all games for the year
        stmt = select(GameResultORM).where(
            GameResultORM.game_date >= f"{year}-01-01",
            GameResultORM.game_date <= f"{year}-12-31"
        )
        games = session.execute(stmt).scalars().all()
        print(f"Found {len(games)} games in DB for {year}")

        # Fetch full schedule from MLB API to get types
        api_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={year}-01-01&endDate={year}-12-31&gameType=R&gameType=P&gameType=S&gameType=A&gameType=E"
        resp = requests.get(api_url)
        data = resp.json()

        type_map = {}
        for date_entry in data.get("dates", []):
            for g in date_entry.get("games", []):
                type_map[str(g.get("gamePk"))] = g.get("gameType")

        updated = 0
        deleted = 0
        for game in games:
            g_type = type_map.get(game.game_id)
            if g_type:
                game.game_type = GameType(g_type)
                updated += 1
            else:
                # If it's not even in the master schedule for R, P, S, A, E, might be extra noise
                pass

        session.commit()
        print(f"Updated {updated} games with game_type labels.")

if __name__ == "__main__":
    populate_game_types(2025)
