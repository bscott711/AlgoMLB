from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import Engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from algomlb.config.settings import get_settings
from algomlb.db.models import StatcastQuantFeatures
from algomlb.db.session import get_engine
from algomlb.ml.quant_service import build_quant_features

logger = logging.getLogger(__name__)
SETTINGS = get_settings()


def _fetch_raw_batch(engine: Engine, game_date: date) -> pd.DataFrame:
    """Fetch statcast_raw rows for a single game_date."""
    query = text("SELECT * FROM statcast_raw WHERE game_date = :gd")
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"gd": game_date})


def _fetch_baseline(engine: Engine, as_of: date, window_days: int) -> pd.DataFrame:
    """Fetch rolling baseline — strictly before as_of, within window_days."""
    start = as_of - timedelta(days=window_days)
    query = text(
        "SELECT * FROM statcast_raw WHERE game_date >= :start AND game_date < :end"
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"start": start, "end": as_of})


def _upsert_quant(engine: Engine, df: pd.DataFrame) -> int:
    """
    Upsert rows into statcast_quant_features.
    On conflict (composite PK), update all calibrated columns.
    Returns the number of rows processed.
    """
    if df.empty:
        return 0

    records = df.to_dict(orient="records")
    # Clean up records for NaN handling (Postgres preferred None over NaN)
    for rec in records:
        for k, v in rec.items():
            if pd.isna(v):
                rec[k] = None

    stmt = pg_insert(StatcastQuantFeatures).values(records)
    update_cols = {
        c.name: c
        for c in stmt.excluded
        if c.name not in ("game_pk", "at_bat_number", "pitch_number")
    }
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["game_pk", "at_bat_number", "pitch_number"],
        set_=update_cols,
    )
    with engine.begin() as conn:
        conn.execute(upsert_stmt)
    return len(records)


def process_quant_for_date(
    game_date: date,
    engine: Engine | None = None,
    dry_run: bool = False,
    baseline_window_days: Optional[int] = None,
) -> int:
    """
    Build and persist quant features for all events on game_date.
    Returns number of rows written (0 if dry_run).
    """
    eng = engine or get_engine()
    window = baseline_window_days or SETTINGS.ml.quant_baseline_window

    raw = _fetch_raw_batch(eng, game_date)

    if raw.empty:
        logger.warning(f"No statcast_raw rows for {game_date} — skipping.")
        return 0

    baseline = _fetch_baseline(eng, as_of=game_date, window_days=window)

    if baseline.empty:
        logger.warning(
            f"No baseline data before {game_date} (window={window} days). "
            "Z-scores will be limited for this batch."
        )

    quant_df = build_quant_features(
        raw=raw,
        baseline=baseline,
        as_of=game_date,
        baseline_window_days=window,
    )

    if dry_run:
        logger.info(
            f"[dry-run] Would upsert {len(quant_df)} quant rows for {game_date}."
        )
        return 0

    count = _upsert_quant(eng, quant_df)
    logger.info(f"Upserted {count} quant feature rows for {game_date}.")
    return count


def process_quant_for_game(
    game_pk: int,
    engine: Engine | None = None,
    dry_run: bool = False,
) -> int:
    """Process a single game by game_pk. Looks up game_date from statcast_raw."""
    eng = engine or get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT DISTINCT game_date FROM statcast_raw WHERE game_pk = :pk"),
            {"pk": game_pk},
        ).fetchone()
    if row is None:
        logger.error(f"game_pk {game_pk} not found in statcast_raw.")
        return 0

    game_date = row[0]
    if isinstance(game_date, str):
        game_date = date.fromisoformat(game_date)

    return process_quant_for_date(game_date, engine=eng, dry_run=dry_run)
