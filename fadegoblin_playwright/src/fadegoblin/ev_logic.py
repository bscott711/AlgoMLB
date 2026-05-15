import pandas as pd
from sqlalchemy import create_engine, text

from fadegoblin import config


def decimal_to_american(decimal_odds: float) -> str:
    """Goblins don't read decimal odds. Convert to American."""
    if decimal_odds >= 2.0:
        american = int(round((decimal_odds - 1.0) * 100.0))
        return f"+{american}"
    else:
        american = int(round(-100.0 / (decimal_odds - 1.0)))
        return str(american)


MLB_ABBREVIATIONS = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
    "Athletics": "OAK",
    "Guardians": "CLE",
}


def abbreviate_team(name: str) -> str:
    """Uses MLB mapping for known teams, falls back to uppers/prefix."""
    if name in MLB_ABBREVIATIONS:
        return MLB_ABBREVIATIONS[name]

    # Fallback for non-MLB or new names
    uppers = [char for char in name if char.isupper()]
    if len(uppers) > 1:
        return "".join(uppers)
    return name[:3].upper()


def get_sniper_bets() -> tuple[list[dict], list[str]]:
    """Fetches the SINGLE best PENDING bet from AlgoMLB DB that hasn't started yet."""
    if not config.DATABASE_URL:
        print("⚠️ DATABASE_URL not set. Cannot run EV Sniper.")
        return [], []

    engine = create_engine(config.DATABASE_URL)

    # Adapted to AlgoMLB's bankroll_ledger and game_results schema
    query = text("""
        SELECT b.transaction_id as id, b.game_id as match_id, b.selection as outcome,
               'ML' as market_type, b.odds as dec_odds, b.edge as ev,
               g.home_team, g.away_team, g.game_datetime as date_time_utc
        FROM bankroll_ledger b
        JOIN game_results g ON b.game_id = g.game_id
        WHERE b.status = 'PENDING'
        AND g.game_datetime > NOW()
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return [], []

    # Return ALL +EV bets sorted by edge, not just one
    df = df.sort_values(by="ev", ascending=False)

    formatted_legs = []
    db_ids_to_update = []

    for _, row in df.iterrows():
        # Create compact abbreviations for Twitter
        home = abbreviate_team(row["home_team"])
        away = abbreviate_team(row["away_team"])

        pick_name = str(row["outcome"])
        if pick_name == row["home_team"]:
            pick_name = home
        elif pick_name == row["away_team"]:
            pick_name = away

        formatted_legs.append(
            {
                "game": f"{away} @ {home}",
                "pick": pick_name,
                "odds": decimal_to_american(row["dec_odds"]),
                "edge": round(float(row["ev"]) * 100, 1),
            }
        )
        db_ids_to_update.append(str(row["id"]))

    return formatted_legs, db_ids_to_update


def mark_bets_placed(pick_ids: list[str]) -> None:
    """Updates the AlgoMLB ledger so we don't tweet the same bet twice."""
    if not pick_ids or not config.DATABASE_URL:
        return

    engine = create_engine(config.DATABASE_URL)
    with engine.connect() as conn:
        for pid in pick_ids:
            # Updates AlgoMLB's bankroll_ledger status
            conn.execute(
                text(
                    "UPDATE bankroll_ledger SET status = 'PLACED' WHERE transaction_id = :pid"
                ),
                {"pid": pid},
            )
        conn.commit()
