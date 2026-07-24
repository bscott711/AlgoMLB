"""
Closing Line Value (CLV): the model's de-vigged entry probability vs. a
de-vigged multi-book consensus taken from the latest live_odds snapshot
before first pitch.

Replaces scratch/check_clv.py, which compared a de-vigged entry probability
to a *single* raw (still-vigged) book's price, and used whatever live_odds
snapshot happened to exist (often the same ~01:00 UTC morning fetch used for
entry, making "market_move" mostly noise). This version:
  - de-vigs every book individually before averaging (comparing like to like)
  - averages across all books with a snapshot at/near first pitch (consensus,
    not a single book)
  - persists results to clv_results instead of printing to stdout
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from algomlb.core.logger import logger
from algomlb.db.models import ClvResultORM
from algomlb.db.session import get_engine

_CLOSING_QUERY = """
    WITH latest_predictions AS (
        SELECT DISTINCT ON (game_id, model_version)
            game_id, model_version, home_win_prob,
            market_home_implied_at_prediction AS entry_implied,
            timestamp AS pred_time
        FROM model_predictions
        ORDER BY game_id, model_version, timestamp DESC
    ),
    games AS (
        SELECT game_id, home_team, away_team, game_date, game_datetime
        FROM game_results
        WHERE game_date >= :start_date AND game_date <= :end_date
    ),
    closing_books AS (
        -- Latest snapshot per (game, book, outcome) at/before first pitch —
        -- the closest thing to a "closing" price this data source offers.
        SELECT DISTINCT ON (lo.game_result_id, lo.sportsbook, lo.outcome)
            lo.game_result_id AS game_id,
            lo.sportsbook,
            lo.outcome,
            lo.price,
            lo.timestamp
        FROM live_odds lo
        JOIN games g ON lo.game_result_id = g.game_id
        WHERE lo.market_type = 'h2h'
          AND lo.timestamp <= g.game_datetime
        ORDER BY lo.game_result_id, lo.sportsbook, lo.outcome, lo.timestamp DESC
    )
    SELECT
        lp.game_id, lp.model_version, lp.home_win_prob, lp.entry_implied,
        g.home_team, g.away_team, g.game_date,
        cb.sportsbook, cb.outcome, cb.price, cb.timestamp
    FROM latest_predictions lp
    JOIN games g ON lp.game_id = g.game_id
    JOIN closing_books cb ON lp.game_id = cb.game_id
"""


def _devig_two_way(price_home: float, price_away: float) -> float | None:
    """De-vig a two-way decimal-odds market. Returns the fair home-win probability."""
    if not price_home or not price_away or price_home <= 1 or price_away <= 1:
        return None
    raw_home = 1.0 / price_home
    raw_away = 1.0 / price_away
    total = raw_home + raw_away
    if total <= 0:
        return None
    return raw_home / total


def compute_clv_for_range(start_date: date, end_date: date) -> int:
    """
    Compute and store CLV for every model prediction with closing odds
    available in [start_date, end_date]. Idempotent — upserts on
    (game_id, model_version), safe to re-run as more closing snapshots land.
    """
    engine = get_engine()
    df = pd.read_sql(
        text(_CLOSING_QUERY),
        engine,
        params={"start_date": start_date, "end_date": end_date},
    )
    if df.empty:
        logger.info(f"[CLV] No predictions with closing odds for {start_date}..{end_date}.")
        return 0

    records = []
    for (game_id, model_version), grp in df.groupby(["game_id", "model_version"]):
        home_by_book = grp[grp["outcome"] == grp["home_team"]].set_index("sportsbook")["price"]
        away_by_book = grp[grp["outcome"] == grp["away_team"]].set_index("sportsbook")["price"]
        common_books = home_by_book.index.intersection(away_by_book.index)

        fair_home_probs = []
        for book in common_books:
            fair_home = _devig_two_way(home_by_book[book], away_by_book[book])
            if fair_home is not None:
                fair_home_probs.append(fair_home)
        if not fair_home_probs:
            continue

        closing_implied = float(sum(fair_home_probs) / len(fair_home_probs))
        row0 = grp.iloc[0]
        model_prob = float(row0["home_win_prob"])
        entry_implied = float(row0["entry_implied"])

        # Positive CLV = the closing line moved toward the side the model liked
        # at entry — the standard signal that the entry price captured value
        # the market later agreed with.
        market_move = closing_implied - entry_implied
        clv = market_move if model_prob > entry_implied else -market_move

        records.append(
            {
                "game_id": game_id,
                "game_date": row0["game_date"],
                "model_version": model_version,
                "model_prob": model_prob,
                "entry_implied": entry_implied,
                "closing_implied": closing_implied,
                "num_books_at_close": int(len(fair_home_probs)),
                "closing_snapshot_at": grp["timestamp"].max().to_pydatetime(),
                "clv": float(clv),
                "closing_edge": float(model_prob - closing_implied),
            }
        )

    if not records:
        logger.info(
            f"[CLV] No two-sided closing lines to de-vig for {start_date}..{end_date}."
        )
        return 0

    _upsert_clv_results(records)
    logger.success(
        f"[CLV] Computed/updated CLV for {len(records)} picks ({start_date}..{end_date})."
    )
    return len(records)


def _upsert_clv_results(records: list[dict]) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        for rec in records:
            stmt = pg_insert(ClvResultORM).values(**rec)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_clv_results_game_model",
                set_={
                    k: stmt.excluded[k]
                    for k in rec
                    if k not in ("game_id", "model_version")
                },
            )
            conn.execute(stmt)


def summarize_clv(start_date: date | None = None, end_date: date | None = None) -> dict:
    """Headline CLV stats (beat rate, avg CLV, avg closing edge) for reporting."""
    engine = get_engine()
    query = "SELECT clv, closing_edge FROM clv_results WHERE 1=1"
    params: dict = {}
    if start_date:
        query += " AND game_date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND game_date <= :end_date"
        params["end_date"] = end_date

    df = pd.read_sql(text(query), engine, params=params)
    if df.empty:
        return {"n": 0, "clv_beat_rate": None, "avg_clv": None, "avg_closing_edge": None}

    return {
        "n": len(df),
        "clv_beat_rate": float((df["clv"] > 0).mean()),
        "avg_clv": float(df["clv"].mean()),
        "avg_closing_edge": float(df["closing_edge"].mean()),
    }
