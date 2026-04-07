# src/algomlb/ml/elo.py
"""
Team-level Elo Rating Engine for AlgoMLB.

Computes slow-moving franchise strength ratings from game results only.
No market odds are used — this is a pure baseball outcomes prior.

References:
    - FiveThirtyEight MLB Elo methodology
    - Logistic Elo expectation with explicit home-field advantage
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from algomlb.core.logger import logger
from algomlb.db.session import get_engine


@dataclass
class EloConfig:
    """Configuration for the Elo rating system."""

    base_rating: float = 1500.0
    k: float = 12.0  # small K for noisy baseball results
    hfa: float = 30.0  # home-field bonus in Elo points


def _expected_home_win(elo_home: float, elo_away: float, cfg: EloConfig) -> float:
    """
    Logistic Elo expectation for home team, including home-field advantage.

    P(home_win) = 1 / (1 + 10^(-(R_home + HFA - R_away) / 400))
    """
    diff = (elo_home + cfg.hfa) - elo_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def _update_pair(
    elo_home: float, elo_away: float, y_home: int, cfg: EloConfig
) -> Tuple[float, float]:
    """
    Single-game Elo update.
    y_home: 1 if home team won, 0 if away team won.
    """
    exp_home = _expected_home_win(elo_home, elo_away, cfg)
    change = cfg.k * (y_home - exp_home)
    return elo_home + change, elo_away - change


def run_elo_offline(
    games: pd.DataFrame,
    cfg: EloConfig | None = None,
) -> pd.DataFrame:
    """
    Pure-Python Elo engine on a games DataFrame.

    games columns required:
        game_pk, game_date, home_team, away_team, home_score, away_score

    Returns:
        DataFrame with one row per (game_pk, team_id, is_home)
        and elo_pre / elo_post.
    """
    if cfg is None:
        cfg = EloConfig()

    games = games.copy()
    games = games.sort_values("game_date")

    ratings: Dict[str, float] = {}
    rows: list[dict] = []

    for _, g in games.iterrows():
        game_pk = int(g["game_pk"])
        game_date = g["game_date"]
        home = str(g["home_team"])
        away = str(g["away_team"])
        y_home = 1 if g["home_score"] > g["away_score"] else 0

        elo_home = ratings.get(home, cfg.base_rating)
        elo_away = ratings.get(away, cfg.base_rating)

        # Record pre-game ratings
        rows.append(
            dict(
                game_pk=game_pk,
                game_date=game_date,
                team_id=home,
                is_home=True,
                elo_pre=elo_home,
                elo_post=0.0,  # placeholder
            )
        )
        rows.append(
            dict(
                game_pk=game_pk,
                game_date=game_date,
                team_id=away,
                is_home=False,
                elo_pre=elo_away,
                elo_post=0.0,  # placeholder
            )
        )

        # Update ratings
        new_home, new_away = _update_pair(elo_home, elo_away, y_home, cfg)
        ratings[home] = new_home
        ratings[away] = new_away

        # Fill post values for the two rows we just added
        rows[-2]["elo_post"] = new_home
        rows[-1]["elo_post"] = new_away

    return pd.DataFrame(rows)


def backfill_team_elo_history(engine: Engine | None = None) -> None:
    """
    Offline backfill for team_elo_history from game_results.
    Intended for initial population; incremental job can reuse the same logic
    but filter to games after the latest stored game_date.
    """
    eng = engine or get_engine()

    query = text("""
        SELECT game_pk, game_date, home_team, away_team,
               home_score, away_score
        FROM game_results
        WHERE status = 'COMPLETED'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY game_date, game_pk
    """)
    games = pd.read_sql(query, eng)
    if games.empty:
        logger.warning("No completed games in game_results; Elo not updated.")
        return

    logger.info(f"Computing Elo ratings for {len(games)} games...")
    elo_df = run_elo_offline(games)

    # Write to DB via bulk upsert
    from algomlb.db.models import TeamEloHistoryORM

    records = elo_df.to_dict(orient="records")

    # Batch insert with conflict resolution (PostgreSQL)
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    batch_size = 500
    with eng.begin() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            stmt = pg_insert(TeamEloHistoryORM.__table__).values(batch)
            upsert_stmt = stmt.on_conflict_do_update(
                index_elements=["game_pk", "team_id", "is_home"],
                set_={
                    "elo_pre": stmt.excluded.elo_pre,
                    "elo_post": stmt.excluded.elo_post,
                    "game_date": stmt.excluded.game_date,
                },
            )
            conn.execute(upsert_stmt)

    logger.success(
        f"Backfilled/updated {len(records)} Elo rows into team_elo_history."
    )
