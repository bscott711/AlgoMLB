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


# Maximum plays per card and edge sanity cap
MAX_CARD_PLAYS = 5
MAX_EDGE_PCT = 15.0  # discard edges above this — likely stale/garbage data


def edge_to_goblins(edge_pct: float) -> str:
    """Converts edge % to a star-based confidence rating for display on the card.

    1 star  = edge 1–3%   (marginal edge)
    2 stars = edge 3–6%   (solid edge)
    3 stars = edge 6%+    (extreme edge)
    """
    if edge_pct >= 6.0:
        return "★★★"
    elif edge_pct >= 3.0:
        return "★★"
    else:
        return "★"


def get_sniper_bets() -> tuple[list[dict], list[str]]:
    """Fetches PENDING and PLACED bets from AlgoMLB DB for upcoming games.

    Includes PLACED bets so the morning card can render the full card of 5 plays
    including the POTD previewed the night before. Returns only PENDING bet IDs
    in db_ids_to_update so only new bets are transitioned to PLACED status.
    """
    if not config.DATABASE_URL:
        print("⚠️ DATABASE_URL not set. Cannot run EV Sniper.")
        return [], []

    engine = create_engine(config.DATABASE_URL)

    query = text("""
        SELECT
            b.transaction_id as id, b.game_id as match_id, b.selection as outcome,
            b.odds as dec_odds, b.edge as ev, b.status as status,
            g.home_team, g.away_team, g.game_datetime as date_time_utc,
            m.home_win_prob, m.market_home_implied_at_prediction as opening_implied
        FROM bankroll_ledger b
        JOIN game_results g ON b.game_id = g.game_id
        LEFT JOIN (
            SELECT DISTINCT ON (game_id) game_id, home_win_prob, market_home_implied_at_prediction
            FROM model_predictions
            ORDER BY game_id, timestamp DESC
        ) m ON b.game_id = m.game_id
        WHERE b.status IN ('PENDING', 'PLACED')
        AND g.game_datetime > NOW()
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return [], []

    # Sort by EV descending to identify POTD
    df = df.sort_values(by="ev", ascending=False)

    # ── Sanity filter: drop suspiciously high edges ─────────────────
    original_count = len(df)
    df = df[df["ev"] * 100 <= MAX_EDGE_PCT]
    if len(df) < original_count:
        print(f"⚠️  Filtered {original_count - len(df)} legs with edge > {MAX_EDGE_PCT}% (likely stale data).")

    # ── Cap to MAX_CARD_PLAYS best plays ────────────────────────────
    df = df.head(MAX_CARD_PLAYS)

    formatted_legs = []
    db_ids_to_update = []

    for i, (_, row) in enumerate(df.iterrows()):
        home = abbreviate_team(row["home_team"])
        away = abbreviate_team(row["away_team"])

        is_home = row["outcome"] == row["home_team"]

        if row["home_win_prob"] is not None:
            model_prob = float(row["home_win_prob"]) if is_home else (1.0 - float(row["home_win_prob"]))
        else:
            model_prob = (1.0 / float(row["dec_odds"])) + float(row["ev"])


        closing_prob = 1.0 / float(row["dec_odds"])
        implied_pct = round(closing_prob * 100, 1)  # market implied probability %

        open_prob = None
        if row["opening_implied"] is not None:
            open_prob = float(row["opening_implied"]) if is_home else (1.0 - float(row["opening_implied"]))

        market_move = (closing_prob - open_prob) if open_prob is not None else 0

        edge_pct = round(float(row["ev"]) * 100, 1)
        goblins = edge_to_goblins(edge_pct)

        badges = []
        if i == 0:
            badges.append("🎯 POTD")
        if model_prob > 0.60:
            badges.append("💎 HIGH CONFIDENCE")
        if market_move > 0.03:
            badges.append("🕵️‍♂️ SHARP MOVE")

        pick_name = home if is_home else away

        formatted_legs.append(
            {
                "id": str(row["id"]),
                "game_id": str(row["match_id"]),
                "game": f"{away} @ {home}",
                "pick": pick_name,
                "odds": decimal_to_american(row["dec_odds"]),
                "edge": edge_pct,
                "implied": implied_pct,
                "goblins": goblins,
                "badges": badges,
                "model_prob": round(model_prob * 100, 1),
            }
        )
        if row["status"] == "PENDING":
            db_ids_to_update.append(str(row["id"]))

    print(f"📋 Card locked: {len(formatted_legs)} plays (max {MAX_CARD_PLAYS}, edge cap {MAX_EDGE_PCT}%).")
    return formatted_legs, db_ids_to_update



def get_preview_potd() -> dict | None:
    """Fetches the single best PENDING pick for an upcoming game (tomorrow's slate).

    Used by the 8 PM MT night preview post. Reads from tomorrow's newly synced
    PENDING bets to hype the upcoming Play of the Day early. This pick will
    then be marked as PLACED.
    """
    if not config.DATABASE_URL:
        return None

    engine = create_engine(config.DATABASE_URL)

    query = text("""
        SELECT
            b.transaction_id as id, b.game_id as game_id, b.selection as outcome,
            b.odds as dec_odds, b.edge as ev,
            g.home_team, g.away_team, g.game_datetime,
            m.home_win_prob
        FROM bankroll_ledger b
        JOIN game_results g ON b.game_id = g.game_id
        LEFT JOIN (
            SELECT DISTINCT ON (game_id) game_id, home_win_prob
            FROM model_predictions
            ORDER BY game_id, timestamp DESC
        ) m ON b.game_id = m.game_id
        WHERE b.status = 'PENDING'
        AND g.game_datetime > NOW()
        AND b.edge <= :edge_cap
        ORDER BY b.edge DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"edge_cap": MAX_EDGE_PCT / 100.0})

    if df.empty:
        return None

    row = df.iloc[0]
    home = abbreviate_team(row["home_team"])
    away = abbreviate_team(row["away_team"])
    is_home = row["outcome"] == row["home_team"]
    pick_name = home if is_home else away

    if row["home_win_prob"] is not None:
        model_prob = float(row["home_win_prob"]) if is_home else (1.0 - float(row["home_win_prob"]))
    else:
        model_prob = (1.0 / float(row["dec_odds"])) + float(row["ev"])

    edge_pct = round(float(row["ev"]) * 100, 1)

    badges = ["⭐ POTD"]
    if model_prob > 0.60:
        badges.append("💎 HIGH CONFIDENCE")

    return {
        "id": str(row["id"]),
        "game_id": str(row["game_id"]),
        "game": f"{away} @ {home}",
        "pick": pick_name,
        "odds": decimal_to_american(float(row["dec_odds"])),
        "edge": edge_pct,
        "goblins": edge_to_goblins(edge_pct),
        "badges": badges,
        "model_prob": round(model_prob * 100, 1),
    }


def get_recap_stats(target_date_str: str | None = None) -> dict:
    """Returns W/L/Push record and net PnL for yesterday's PLACED bets.

    Determines wins/losses by comparing the selection against game_results
    scores, since pnl may not be settled yet. If target_date_str is None,
    defaults to yesterday in ET.

    Returns a dict with keys:
        date, wins, losses, pushes, total, net_pnl (or None), picks (list of dicts)
    """
    from datetime import date, timedelta, datetime
    from zoneinfo import ZoneInfo

    if not config.DATABASE_URL:
        return {}

    et = ZoneInfo("America/New_York")
    if target_date_str:
        target_date = date.fromisoformat(target_date_str)
    else:
        target_date = (datetime.now(tz=et) - timedelta(days=1)).date()

    engine = create_engine(config.DATABASE_URL)

    query = text("""
        SELECT
            b.selection, b.odds as dec_odds, b.edge as ev, b.pnl, b.stake,
            g.home_team, g.away_team, g.home_score, g.away_score, g.game_datetime
        FROM bankroll_ledger b
        JOIN game_results g ON b.game_id = g.game_id
        WHERE b.status IN ('PLACED', 'SETTLED')
        AND DATE(g.game_datetime AT TIME ZONE 'America/New_York') = :target_date
        ORDER BY g.game_datetime
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"target_date": str(target_date)})

    if df.empty:
        return {"date": str(target_date), "wins": 0, "losses": 0, "pushes": 0, "total": 0, "net_pnl": None, "picks": []}

    wins = losses = pushes = 0
    net_pnl = 0.0
    has_pnl = False
    picks = []

    for _, row in df.iterrows():
        home_score = row["home_score"]
        away_score = row["away_score"]
        selection = row["selection"]

        # Determine result from scores if available
        result = "?"
        if home_score is not None and away_score is not None:
            winning_team = row["home_team"] if home_score > away_score else row["away_team"]
            if home_score == away_score:
                result = "PUSH"
                pushes += 1
            elif selection == winning_team:
                result = "WIN"
                wins += 1
            else:
                result = "LOSS"
                losses += 1
        elif row["pnl"] is not None:
            # Fall back to pnl if scores not available
            if float(row["pnl"]) > 0:
                result = "WIN"
                wins += 1
            elif float(row["pnl"]) == 0:
                result = "PUSH"
                pushes += 1
            else:
                result = "LOSS"
                losses += 1

        if row["pnl"] is not None:
            net_pnl += float(row["pnl"])
            has_pnl = True

        home = abbreviate_team(row["home_team"])
        away = abbreviate_team(row["away_team"])
        pick_abbr = abbreviate_team(selection)

        picks.append({
            "matchup": f"{away} @ {home}",
            "pick": pick_abbr,
            "odds": decimal_to_american(float(row["dec_odds"])),
            "edge": round(float(row["ev"]) * 100, 1),
            "result": result,
        })

    return {
        "date": str(target_date),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total": len(df),
        "net_pnl": round(net_pnl, 2) if has_pnl else None,
        "picks": picks,
    }


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
