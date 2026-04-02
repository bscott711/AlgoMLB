from sqlalchemy import select
from loguru import logger
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM, BallparkORM


# Map Full/Partial names to DB abbreviations used in BallparkORM
TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "AZ",
    "ARI": "AZ",
    "AZ": "AZ",
    "ARZ": "AZ",
    "Arizona": "AZ",
    "Atlanta Braves": "ATL",
    "ATL": "ATL",
    "Baltimore Orioles": "BAL",
    "BAL": "BAL",
    "Boston Red Sox": "BOS",
    "BOS": "BOS",
    "Chicago Cubs": "CHC",
    "CHC": "CHC",
    "Chicago White Sox": "CWS",
    "CWS": "CWS",
    "Cincinnati Reds": "CIN",
    "CIN": "CIN",
    "Cleveland Guardians": "CLE",
    "CLE": "CLE",
    "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL",
    "COL": "COL",
    "Detroit Tigers": "DET",
    "DET": "DET",
    "Houston Astros": "HOU",
    "HOU": "HOU",
    "Kansas City Royals": "KC",
    "KC": "KC",
    "Los Angeles Angels": "LAA",
    "LAA": "LAA",
    "Los Angeles Dodgers": "LAD",
    "LAD": "LAD",
    "Miami Marlins": "MIA",
    "MIA": "MIA",
    "Milwaukee Brewers": "MIL",
    "MIL": "MIL",
    "Minnesota Twins": "MIN",
    "MIN": "MIN",
    "New York Mets": "NYM",
    "NYM": "NYM",
    "New York Yankees": "NYY",
    "NYY": "NYY",
    "Oakland Athletics": "OAK",
    "OAK": "OAK",
    "Athletics": "OAK",
    "Oakland": "OAK",
    "Philadelphia Phillies": "PHI",
    "PHI": "PHI",
    "Pittsburgh Pirates": "PIT",
    "PIT": "PIT",
    "San Diego Padres": "SD",
    "SD": "SD",
    "San Francisco Giants": "SF",
    "SF": "SF",
    "Seattle Mariners": "SEA",
    "SEA": "SEA",
    "St. Louis Cardinals": "STL",
    "STL": "STL",
    "Tampa Bay Rays": "TB",
    "TB": "TB",
    "Texas Rangers": "TEX",
    "TEX": "TEX",
    "Toronto Blue Jays": "TOR",
    "TOR": "TOR",
    "Washington Nationals": "WSH",
    "WSH": "WSH",
    "Wats": "WSH",
}


def repair_ballparks():
    session_factory = get_session_factory()
    with session_factory() as session:
        # 1. Fetch all ballparks for reverse mapping
        ballparks = session.execute(select(BallparkORM)).scalars().all()
        # Map shorthand (ARI) -> Ballpark ID
        abbrev_to_id = {bp.team_name: bp.id for bp in ballparks}

        # 2. Fetch all games with missing ballpark_id
        stmt = select(GameResultORM).where(GameResultORM.ballpark_id.is_(None))
        games = session.execute(stmt).scalars().all()
        if not games:
            logger.info("✅ No games with missing ballpark_id found.")
            return

        logger.info(f"🛠️ Found {len(games)} games to repair. Identifying ballparks...")

        repaired = 0
        for game in games:
            target_abbrev = TEAM_NAME_MAP.get(game.home_team)
            if not target_abbrev:
                # Try case-insensitive or partial
                target_abbrev = next(
                    (
                        v
                        for k, v in TEAM_NAME_MAP.items()
                        if k.lower() in game.home_team.lower()
                    ),
                    None,
                )

            if target_abbrev and target_abbrev in abbrev_to_id:
                game.ballpark_id = abbrev_to_id[target_abbrev]
                repaired += 1
            else:
                logger.warning(
                    f"❌ Could not match ballpark for home team: {game.home_team} (Game {game.game_id})"
                )

        if repaired > 0:
            session.commit()
            logger.success(
                f"✅ Successfully repaired ballpark_id for {repaired} games."
            )
        else:
            logger.info("No games could be repaired with current mapping.")


if __name__ == "__main__":
    repair_ballparks()
