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
MAX_EDGE_PCT = 15.0  # If any edge is above this, data is likely stale. Abort.


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

    # ── Sanity filter: Abort on suspiciously high edges ─────────────
    suspicious = df[df["ev"] * 100 > MAX_EDGE_PCT]
    if not suspicious.empty:
        raise RuntimeError(
            f"🚨 CRITICAL ALERT: Found {len(suspicious)} legs with edge > {MAX_EDGE_PCT}%! "
            f"Max edge is {suspicious['ev'].max() * 100:.1f}%. "
            "This indicates stale or missing data. Aborting card generation."
        )

    # ── Cap to MAX_CARD_PLAYS best plays ────────────────────────────
    df = df.head(MAX_CARD_PLAYS)

    formatted_legs = []
    db_ids_to_update = []

    for i, (_, row) in enumerate(df.iterrows()):
        home = abbreviate_team(row["home_team"])
        away = abbreviate_team(row["away_team"])

        is_home = row["outcome"] == row["home_team"]

        if pd.notna(row["home_win_prob"]):
            model_prob = (
                float(row["home_win_prob"])
                if is_home
                else (1.0 - float(row["home_win_prob"]))
            )
        else:
            model_prob = (1.0 / float(row["dec_odds"])) + float(row["ev"])

        closing_prob = 1.0 / float(row["dec_odds"])
        implied_pct = round(closing_prob * 100, 1)  # market implied probability %

        open_prob = None
        if pd.notna(row["opening_implied"]):
            open_prob = (
                float(row["opening_implied"])
                if is_home
                else (1.0 - float(row["opening_implied"]))
            )

        market_move = (closing_prob - open_prob) if open_prob is not None else 0

        edge_pct = round(float(row["ev"]) * 100, 1)
        goblins = edge_to_goblins(edge_pct)

        badges = []
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
                "status": str(row["status"]),
            }
        )
        if row["status"] == "PENDING":
            db_ids_to_update.append(str(row["id"]))

    print(
        f"📋 Card locked: {len(formatted_legs)} plays (max {MAX_CARD_PLAYS}, edge cap {MAX_EDGE_PCT}%)."
    )
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
        ORDER BY b.edge DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return None
        
    if df.iloc[0]["ev"] * 100 > MAX_EDGE_PCT:
        raise RuntimeError(
            f"🚨 CRITICAL ALERT: POTD preview edge is suspiciously high "
            f"({df.iloc[0]['ev'] * 100:.1f}% > {MAX_EDGE_PCT}%). Aborting preview post."
        )

    row = df.iloc[0]
    home = abbreviate_team(row["home_team"])
    away = abbreviate_team(row["away_team"])
    is_home = row["outcome"] == row["home_team"]
    pick_name = home if is_home else away

    if pd.notna(row["home_win_prob"]):
        model_prob = (
            float(row["home_win_prob"])
            if is_home
            else (1.0 - float(row["home_win_prob"]))
        )
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

    Determines wins/losses from game_results scores, but ONLY once MLB has
    actually called the game final. The schedule API reports score=0/0 for a
    game that hasn't started yet (or is mid-inning and genuinely tied) just
    as readily as for a real final -- trusting "both scores present" alone
    used to misread an unresolved game as a 0-0 push. Falls back to `pnl`
    (set by settle_bets, which applies the same COMPLETED-only rule) when a
    bet's already been graded through that path. Anything still unresolved
    comes back as result="?" and its game_id in `pending_game_ids`, instead
    of being guessed at.

    If target_date_str is None, defaults to yesterday in ET.

    Returns a dict with keys:
        date, wins, losses, pushes, total, net_pnl (or None), picks (list of
        dicts), pending_game_ids (list of str)
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
            b.game_id, b.selection, b.odds as dec_odds, b.edge as ev, b.pnl, b.stake,
            g.home_team, g.away_team, g.home_score, g.away_score, g.game_datetime,
            g.status as game_status
        FROM bankroll_ledger b
        JOIN game_results g ON b.game_id = g.game_id
        WHERE b.status IN ('PLACED', 'SETTLED')
        AND DATE(g.game_datetime AT TIME ZONE 'America/New_York') = :target_date
        ORDER BY g.game_datetime
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"target_date": str(target_date)})

    if df.empty:
        return {
            "date": str(target_date),
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "total": 0,
            "net_pnl": None,
            "picks": [],
            "pending_game_ids": [],
        }

    wins = losses = pushes = 0
    net_pnl = 0.0
    has_pnl = False
    picks = []
    pending_game_ids = []

    for _, row in df.iterrows():
        home_score = row["home_score"]
        away_score = row["away_score"]
        selection = row["selection"]
        is_final = row["game_status"] == "COMPLETED"
        # pd.read_sql surfaces a SQL NULL in a numeric column as float NaN,
        # not None -- `x is not None` silently passes for an unresolved row,
        # so every None/null check on a DB-sourced numeric column below uses
        # pd.notna() instead.
        has_scores = pd.notna(home_score) and pd.notna(away_score)
        has_pnl_value = pd.notna(row["pnl"])

        result = "?"
        if is_final and has_scores:
            winning_team = (
                row["home_team"] if home_score > away_score else row["away_team"]
            )
            if home_score == away_score:
                result = "PUSH"
                pushes += 1
            elif selection == winning_team:
                result = "WIN"
                wins += 1
            else:
                result = "LOSS"
                losses += 1
        elif has_pnl_value:
            # Already graded by settle_bets -- trust it even if game_status
            # hasn't caught up to COMPLETED for some reason.
            if float(row["pnl"]) > 0:
                result = "WIN"
                wins += 1
            elif float(row["pnl"]) == 0:
                result = "PUSH"
                pushes += 1
            else:
                result = "LOSS"
                losses += 1
        else:
            pending_game_ids.append(str(row["game_id"]))

        if has_pnl_value:
            net_pnl += float(row["pnl"])
            has_pnl = True

        home = abbreviate_team(row["home_team"])
        away = abbreviate_team(row["away_team"])
        pick_abbr = abbreviate_team(selection)

        picks.append(
            {
                "matchup": f"{away} @ {home}",
                "pick": pick_abbr,
                "odds": decimal_to_american(float(row["dec_odds"])),
                "edge": round(float(row["ev"]) * 100, 1),
                "result": result,
            }
        )

    return {
        "date": str(target_date),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total": len(df),
        "net_pnl": round(net_pnl, 2) if has_pnl else None,
        "picks": picks,
        "pending_game_ids": pending_game_ids,
    }


def refresh_pending_games(game_ids: list[str]) -> dict:
    """Actively checks the live MLB Stats API for specific games: settles any
    that have gone Final immediately, and reports live state (current inning,
    top/bottom, delayed/postponed) for whatever's still going.

    The nightly algomlb-sync job only ingests scores once a day, so a game
    that's still being played (or just finished) when recap runs would
    otherwise sit unresolved for up to ~24h. This does a narrow, targeted
    refresh -- just these game_ids -- instead of waiting on the next full
    sync. The per-game state is what lets the caller wait an amount of time
    proportional to how much of the game is actually left, rather than
    polling on a fixed timer regardless of whether it's the 1st or the 9th.

    Returns {"resolved": [game_id, ...], "pending": {game_id: {...}}}.
    """
    result: dict = {"resolved": [], "pending": {}}
    if not game_ids or not config.DATABASE_URL:
        return result

    import requests

    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={
                "sportId": 1,
                "gamePk": ",".join(game_ids),
                "hydrate": "linescore",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"⚠️ Failed to refresh pending games from MLB Stats API: {e}")
        return result

    wanted = set(game_ids)
    engine = create_engine(config.DATABASE_URL)

    with engine.begin() as conn:
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                game_id = str(game.get("gamePk", ""))
                if game_id not in wanted:
                    continue

                detailed_state = game.get("status", {}).get("detailedState", "")
                lowered = detailed_state.lower()
                is_final = (
                    "final" in lowered
                    or "completed" in lowered
                    or "game over" in lowered
                )

                teams = game.get("teams", {})
                home_score = teams.get("home", {}).get("score")
                away_score = teams.get("away", {}).get("score")

                if not is_final or home_score is None or away_score is None:
                    linescore = game.get("linescore", {}) or {}
                    result["pending"][game_id] = {
                        "detailed_state": detailed_state or "Unknown",
                        # POSTPONED/CANCELLED will never produce a result
                        # under this game_id -- no point waiting on it.
                        "abandoned": "postponed" in lowered or "cancelled" in lowered,
                        "inning": linescore.get("currentInning"),
                        "inning_state": linescore.get("inningState"),
                    }
                    continue

                conn.execute(
                    text("""
                        UPDATE game_results
                        SET status = 'COMPLETED', home_score = :hs, away_score = :as_
                        WHERE game_id = :gid
                    """),
                    {"hs": home_score, "as_": away_score, "gid": game_id},
                )

                game_row = conn.execute(
                    text(
                        "SELECT home_team, away_team FROM game_results WHERE game_id = :gid"
                    ),
                    {"gid": game_id},
                ).first()
                if not game_row:
                    continue

                winner = None
                if home_score > away_score:
                    winner = game_row.home_team
                elif away_score > home_score:
                    winner = game_row.away_team

                bets = conn.execute(
                    text("""
                        SELECT transaction_id, selection, odds, stake
                        FROM bankroll_ledger
                        WHERE game_id = :gid AND status IN ('PENDING', 'PLACED')
                    """),
                    {"gid": game_id},
                ).fetchall()

                for bet in bets:
                    pnl = 0.0
                    if winner:
                        pnl = (
                            bet.stake * (bet.odds - 1)
                            if bet.selection == winner
                            else -bet.stake
                        )
                    conn.execute(
                        text("""
                            UPDATE bankroll_ledger
                            SET status = 'SETTLED', pnl = :pnl
                            WHERE transaction_id = :tid
                        """),
                        {"pnl": pnl, "tid": bet.transaction_id},
                    )

                result["resolved"].append(game_id)

    if result["resolved"]:
        print(
            f"✅ Refreshed {len(result['resolved'])} game(s) to Final via "
            "targeted MLB Stats API check."
        )
    return result


# Modern pace-of-play rules put a 9-inning game around 2h35-2h45 (~18
# min/inning), but that's an average, not a floor -- pitching changes, replay
# review, and extra-frills games run longer, so we pad it a bit rather than
# risk checking back too early.
_MINUTES_PER_INNING = 20
_NOT_STARTED_WAIT_MINUTES = 30  # scheduled but hasn't thrown a pitch yet
_DELAYED_WAIT_MINUTES = 35  # rain delay etc. -- duration is unknowable, so just poll periodically
_MIN_WAIT_MINUTES = 10
_MAX_WAIT_MINUTES = 45
# A game going final isn't the same instant it's gradeable: MLB takes a few
# minutes to post the official final, and even then our own check has to land
# after that. Padding every estimate by this avoids polling right as the game
# is wrapping up, before there's anything to actually grade yet.
_GRADING_BUFFER_MINUTES = 15


def estimate_wait_seconds(pending: dict) -> int:
    """How long to wait before checking again, sized to the *slowest-to-finish*
    pending game's actual progress.

    We want a single, complete recap post -- not one that goes out as soon as
    the fastest pick clears -- so the wait is driven by whichever pending game
    has the most game left, not the least. A game in the 1st gets a much
    longer gap between checks than one in the 9th. Returns 0 when there's
    nothing worth waiting on (all pending games are postponed/cancelled).
    """
    waits = []
    for state in pending.values():
        if state.get("abandoned"):
            continue

        inning = state.get("inning")
        detailed = (state.get("detailed_state") or "").lower()

        if inning is None:
            minutes = (
                _DELAYED_WAIT_MINUTES if "delay" in detailed else _NOT_STARTED_WAIT_MINUTES
            )
        else:
            remaining_innings = max(9 - inning, 0) + (
                0.5 if state.get("inning_state") in ("Bottom", "End") else 1.0
            )
            minutes = remaining_innings * _MINUTES_PER_INNING

        minutes += _GRADING_BUFFER_MINUTES
        waits.append(max(_MIN_WAIT_MINUTES, min(_MAX_WAIT_MINUTES, minutes)))

    if not waits:
        return 0

    return int(max(waits) * 60)


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


def get_weekly_recap_stats() -> dict:
    """Returns W/L/Push record and net PnL for the past 7 days of PLACED bets.
    
    This only calculates the aggregated stats over the past 7 days.
    """
    from datetime import date, timedelta, datetime
    from zoneinfo import ZoneInfo

    if not config.DATABASE_URL:
        return {}

    et = ZoneInfo("America/New_York")
    end_date = (datetime.now(tz=et) - timedelta(days=1)).date()
    start_date = end_date - timedelta(days=6)

    engine = create_engine(config.DATABASE_URL)

    query = text("""
        SELECT
            b.selection, b.pnl,
            g.home_team, g.away_team, g.home_score, g.away_score, g.status as game_status
        FROM bankroll_ledger b
        JOIN game_results g ON b.game_id = g.game_id
        WHERE b.status IN ('PLACED', 'SETTLED')
        AND DATE(g.game_datetime AT TIME ZONE 'America/New_York') >= :start_date
        AND DATE(g.game_datetime AT TIME ZONE 'America/New_York') <= :end_date
    """)

    with engine.connect() as conn:
        df = pd.read_sql(
            query, conn, params={"start_date": str(start_date), "end_date": str(end_date)}
        )

    if df.empty:
        return {
            "date": f"{start_date} to {end_date}",
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "total": 0,
            "net_pnl": None,
            "picks": [],
        }

    wins = losses = pushes = 0
    net_pnl = 0.0
    has_pnl = False

    for _, row in df.iterrows():
        home_score = row["home_score"]
        away_score = row["away_score"]
        selection = row["selection"]
        is_final = row["game_status"] == "COMPLETED"
        has_scores = pd.notna(home_score) and pd.notna(away_score)
        has_pnl_value = pd.notna(row["pnl"])

        if is_final and has_scores:
            winning_team = (
                row["home_team"] if home_score > away_score else row["away_team"]
            )
            if home_score == away_score:
                pushes += 1
            elif selection == winning_team:
                wins += 1
            else:
                losses += 1
        elif has_pnl_value:
            if float(row["pnl"]) > 0:
                wins += 1
            elif float(row["pnl"]) == 0:
                pushes += 1
            else:
                losses += 1

        if has_pnl_value:
            net_pnl += float(row["pnl"])
            has_pnl = True

    return {
        "date": f"{start_date} to {end_date}",
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total": len(df),
        "net_pnl": round(net_pnl, 2) if has_pnl else None,
        "picks": [],
    }
