"""
Derive manager_hook_events and manager_hook_profiles from retrosheet_events.

Hook detection logic:
  For each game-side (game_id × pit_team), order PAs by play_number.
  When pitcher_id changes between consecutive PAs, the outgoing pitcher
  has been "hooked". The game state at the hook is captured from the
  LAST PA completed by that pitcher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from algomlb.core.logger import logger
from algomlb.db.models import (
    ManagerHookEventORM,
    ManagerHookProfileORM,
    GameResultORM,
)
from sqlalchemy import select


# ── Hook event extraction ────────────────────────────────────────────────


def extract_hook_events(engine: Engine, season: int) -> pd.DataFrame:
    """
    Extract all pitcher removals for a given season from retrosheet_events.
    """
    logger.info(f"Extracting hook events for {season}...")

    query = f"""
        SELECT game_id, date, play_number, inning, top_bot,
               pitcher_id, pit_team, bat_team, pa_flag, nump, lp,
               outs_pre, outs_post, score_v, score_h,
               CASE WHEN br1_post IS NOT NULL THEN 1 ELSE 0 END as r1,
               CASE WHEN br2_post IS NOT NULL THEN 1 ELSE 0 END as r2,
               CASE WHEN br3_post IS NOT NULL THEN 1 ELSE 0 END as r3,
               single, double_flag, triple, hr, walk, hbp, k, runs
        FROM retrosheet_events
        WHERE EXTRACT(YEAR FROM date) = {season} AND pa_flag = 1
        ORDER BY game_id, play_number
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        logger.warning(f"No retrosheet PAs found for {season}")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    mgr_map = _get_manager_map(engine, season)
    hook_rows: list[dict] = []

    for (game_id, pit_team), group in df.groupby(["game_id", "pit_team"]):
        group = group.sort_values("play_number").reset_index(drop=True)
        game_date = group["date"].iloc[0]
        is_home = bool(group["top_bot"].iloc[0] == 0)

        mgr_info = mgr_map.get(pit_team, {})
        manager_id = mgr_info.get("manager_id")
        manager_name = mgr_info.get("manager_name")

        pitchers_seen: list[str] = []
        stint_start_idx = 0

        for i in range(len(group)):
            current_pitcher = group.loc[i, "pitcher_id"]
            is_last_pa = i == len(group) - 1
            next_pitcher = group.loc[i + 1, "pitcher_id"] if not is_last_pa else None
            pitcher_changes = (
                next_pitcher is not None and next_pitcher != current_pitcher
            )

            if pitcher_changes or is_last_pa:
                stint = group.iloc[stint_start_idx : i + 1]
                is_starter = len(pitchers_seen) == 0

                hook_rows.append(
                    _process_pitcher_stint(
                        stint,
                        str(game_id),
                        game_date,
                        season,
                        str(pit_team),
                        manager_id,
                        manager_name,
                        str(current_pitcher),
                        is_starter,
                        is_home,
                        pitcher_changes,
                    )
                )

                pitchers_seen.append(str(current_pitcher))
                stint_start_idx = i + 1

    result = pd.DataFrame(hook_rows)
    logger.info(f"  Extracted {len(result)} hook events for {season}")
    return result


def _get_manager_map(engine: Engine, season: int) -> dict:
    """Load manager lookup for the season."""
    mgr_df = pd.read_sql(
        f"SELECT team_abbr, manager_id, manager_name FROM team_managers WHERE season = {season}",
        engine,
    )
    return dict(
        zip(
            mgr_df["team_abbr"],
            mgr_df[["manager_id", "manager_name"]].to_dict("records"),
        )
    )


def _calculate_outs_recorded(stint: pd.DataFrame) -> int:
    """Calculate internal outs recorded by a pitcher during a stint."""
    outs = 0
    for _, pa in stint.iterrows():
        if pa["outs_post"] >= pa["outs_pre"]:
            outs += pa["outs_post"] - pa["outs_pre"]
        else:
            outs += 3 - pa["outs_pre"]  # Inning turnover
    return outs


def _process_pitcher_stint(
    stint: pd.DataFrame,
    game_id: str,
    game_date: Any,
    season: int,
    pit_team: str,
    manager_id: Any,
    manager_name: str | None,
    pitcher_id: str,
    is_starter: bool,
    is_home: bool,
    is_removed: bool,
) -> dict:
    """Calculate all metrics for a single pitcher stint and return a hook record."""
    last_pa = stint.iloc[-1]
    lp_vals = stint["lp"].tolist()

    # Metrics
    tto = max(1, (len(set(lp_vals)) + 8) // 9)
    hits = int(stint[["single", "double_flag", "triple", "hr"]].sum().sum())
    walks = int(stint[["walk", "hbp"]].sum().sum())
    score_diff = (
        int(last_pa["score_h"] - last_pa["score_v"])
        if is_home
        else int(last_pa["score_v"] - last_pa["score_h"])
    )
    runners_on = (
        int(last_pa["r1"]) | (int(last_pa["r2"]) << 1) | (int(last_pa["r3"]) << 2)
    )

    outs_recorded = _calculate_outs_recorded(stint)
    was_qs = (
        is_starter and (outs_recorded / 3.0 >= 6.0) and (int(stint["runs"].sum()) <= 3)
    )

    return {
        "game_id": game_id,
        "game_date": game_date,
        "season": season,
        "team_abbr": pit_team,
        "manager_id": int(manager_id)
        if manager_id is not None
        and not (isinstance(manager_id, float) and np.isnan(manager_id))
        else None,
        "manager_name": manager_name,
        "pitcher_id": pitcher_id,
        "is_starter": is_starter,
        "inning": int(last_pa["inning"]),
        "outs_at_hook": int(last_pa["outs_post"]),
        "pitch_count": int(stint["nump"].sum()),
        "pa_count": len(stint),
        "tto": tto,
        "score_diff": score_diff,
        "is_home": is_home,
        "runners_on": runners_on,
        "inherited_runners": (
            int(last_pa["r1"]) + int(last_pa["r2"]) + int(last_pa["r3"])
        )
        if is_removed
        else 0,
        "hits_allowed": hits,
        "walks_allowed": walks,
        "strikeouts": int(stint["k"].sum()),
        "runs_allowed": int(stint["runs"].sum()),
        "hr_allowed": int(stint["hr"].sum()),
        "was_quality_start": was_qs,
        "hook_before_3rd_tto": bool(tto < 3 and is_starter),
    }


# ── Hook profile aggregation ─────────────────────────────────────────────


def compute_hook_profiles(hook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hook events into per-manager/season profiles.

    Returns a DataFrame with one row per (manager_id, season).
    """
    if hook_df.empty:
        return pd.DataFrame()

    # Filter to starter hooks only for profile computation
    sp_hooks = hook_df[hook_df["is_starter"]].copy()

    if sp_hooks.empty:
        return pd.DataFrame()

    profiles: list[dict] = []

    for (mgr_id, season), group in sp_hooks.groupby(["manager_id", "season"]):
        # Cast Hashable to float/int for numeric operations
        m_id = float(str(mgr_id))
        s_sn = int(float(str(season)))
        if pd.isna(m_id):
            continue

        mgr_name = group["manager_name"].iloc[0]
        n_starts = len(group)

        avg_pc = group["pitches_thrown"].mean()
        avg_inn = group["inning"].mean()

        before_3rd = (
            group["pull_before_3rd_tto_pct"].mean()
            if "pull_before_3rd_tto_pct" in group.columns
            else (group["tto_at_hook"] < 3).mean()
        )
        with_lead = (group["score_diff_at_hook"] > 0).mean()
        over_90 = (group["pitches_thrown"] > 90).mean()

        profiles.append(
            {
                "manager_id": int(m_id),
                "manager_name": str(mgr_name),
                "season": s_sn,
                "total_hooks": n_starts,
                "total_sp_starts": n_starts,
                "avg_sp_pitch_count": round(float(avg_pc), 1),
                "avg_ip_per_start": round(float(avg_inn), 2),
                "avg_hook_inning": round(float(avg_inn), 1),
                "pull_before_3rd_tto_pct": round(float(before_3rd), 4),
                "pull_with_lead_pct": round(float(with_lead), 4),
                "pull_when_over_90_pitches_pct": round(float(over_90), 4),
                "quick_hook_high_leverage_pct": 0.0,
                "bullpen_protective_pct": 0.0,
            }
        )

    return pd.DataFrame(profiles)


# ── Persistence ──────────────────────────────────────────────────────────


def persist_hook_events(engine: Engine, hook_df: pd.DataFrame) -> pd.DataFrame:
    """Upsert hook events into manager_hook_events. Returns the transformed DataFrame."""
    if hook_df.empty:
        return pd.DataFrame()

    retro_to_mlb = {
        "ANA": 108,
        "ARI": 109,
        "ATL": 144,
        "BAL": 110,
        "BOS": 111,
        "CHA": 145,
        "CHN": 112,
        "CIN": 113,
        "CLE": 114,
        "COL": 115,
        "DET": 116,
        "HOU": 117,
        "KCA": 118,
        "LAN": 119,
        "MIA": 146,
        "MIL": 158,
        "MIN": 142,
        "NYA": 147,
        "NYN": 121,
        "OAK": 133,
        "PHI": 143,
        "PIT": 134,
        "SDN": 135,
        "SEA": 136,
        "SFN": 137,
        "SLN": 138,
        "TBA": 139,
        "TEX": 140,
        "TOR": 141,
        "WAS": 120,
    }

    game_map = _get_persist_game_map(engine, hook_df)
    orms_data = []

    for _, row in hook_df.iterrows():
        transformed = _transform_hook_row(row, game_map, retro_to_mlb)
        if transformed:
            orms_data.append(transformed)

    if not orms_data:
        logger.warning("No valid ORM mappings found for hook events.")
        return pd.DataFrame()

    total = _bulk_upsert_hooks(engine, orms_data)
    logger.info(f"  Successfully persisted {total} hook events to the database.")
    return pd.DataFrame(orms_data)


def _get_persist_game_map(engine: Engine, hook_df: pd.DataFrame) -> dict:
    """Build a map of (date, home_team_id, doublet_num) -> (game_pk, away_team_id)."""
    with Session(engine) as session:
        games = (
            session.execute(
                select(GameResultORM).where(
                    GameResultORM.game_date.between(
                        hook_df["game_date"].min(), hook_df["game_date"].max()
                    )
                )
            )
            .scalars()
            .all()
        )

        game_map = {
            (g.game_date, g.home_team_id, g.doubleheader_num): (g.id, g.away_team_id)
            for g in games
        }
        # Handle doubleheader=0 vs 1 heuristic
        for g in games:
            if g.doubleheader_num == 0:
                game_map[(g.game_date, g.home_team_id, 1)] = (g.id, g.away_team_id)
    return game_map


def _transform_hook_row(row, game_map, retro_to_mlb) -> dict | None:
    """Map a single retrosheet hook row to an MLB-compatible ORM dictionary."""
    try:
        game_pk, pit_team_id, opp_id = _resolve_hook_ids(row, game_map, retro_to_mlb)
        if game_pk is None or pit_team_id is None or opp_id is None:
            return None

        return {
            "game_pk": game_pk,
            "game_date": row["game_date"],
            "season": int(row["season"]),
            "team_id": pit_team_id,
            "opponent_id": opp_id,
            "manager_id": int(row["manager_id"])
            if row["manager_id"] is not None
            else 0,
            "manager_name": str(row["manager_name"]),
            "pitcher_id": str(row["pitcher_id"]),
            "is_starter": bool(row["is_starter"]),
            "inning": int(row["inning"]),
            "outs_at_hook": int(row["outs_at_hook"]),
            "pitches_thrown": int(row["pitch_count"]),
            "tto_at_hook": int(row["tto"]),
            "score_diff_at_hook": int(row["score_diff"]),
            "base_state_at_hook": int(row["runners_on"]),
            "leverage_index_at_hook": 1.0,
            "manager_tenure_day": 0,
            "days_since_manager_change": 0,
            "hook_reason": "Stat-Based" if row["is_starter"] else "Relief",
            "removed_before_next_batter": True,
        }
    except Exception:
        return None


def _resolve_hook_ids(
    row, game_map, retro_to_mlb
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Helper to resolve game_pk, pitcher team_id, and opponent_id."""
    h_team_id = retro_to_mlb.get(row["game_id"][:3])
    if not h_team_id:
        return None, None, None

    gnum = int(row["game_id"][11:])
    ginfo = game_map.get((row["game_date"], h_team_id, gnum))
    if not ginfo and gnum == 1:
        ginfo = game_map.get((row["game_date"], h_team_id, 0))
    if not ginfo:
        return None, None, None

    game_pk, opp_id_fallback = ginfo
    pit_team_id = retro_to_mlb.get(row["team_abbr"])
    if not pit_team_id:
        return None, None, None

    opp_id = opp_id_fallback if pit_team_id == h_team_id else h_team_id
    return game_pk, pit_team_id, opp_id


def _bulk_upsert_hooks(engine: Engine, orms_data: list[dict]) -> int:
    """Bulk upsert transformed hook records into the database."""
    orm_cols = {c.name for c in ManagerHookEventORM.__table__.columns}
    total, chunk_size = 0, 1000
    with engine.begin() as conn:
        for i in range(0, len(orms_data), chunk_size):
            chunk = orms_data[i : i + chunk_size]
            db_chunk = [{k: v for k, v in r.items() if k in orm_cols} for r in chunk]
            stmt = pg_insert(ManagerHookEventORM).values(db_chunk)
            upsert = stmt.on_conflict_do_update(
                index_elements=["game_pk", "pitcher_id"],
                set_={
                    col: stmt.excluded[col]
                    for col in db_chunk[0].keys()
                    if col not in ("game_pk", "pitcher_id")
                },
            )
            conn.execute(upsert)
            total += len(chunk)
    return total


def persist_hook_profiles(engine: Engine, profile_df: pd.DataFrame) -> int:
    """Upsert hook profiles into manager_hook_profiles. Returns count."""
    if profile_df.empty:
        return 0

    rows = profile_df.to_dict("records")
    with engine.begin() as conn:
        stmt = pg_insert(ManagerHookProfileORM).values(rows)
        upsert = stmt.on_conflict_do_update(
            index_elements=["manager_id", "season"],
            set_={
                col: stmt.excluded[col]
                for col in profile_df.columns
                if col not in ("manager_id", "season")
            },
        )
        conn.execute(upsert)
    return len(rows)


# ── Full backfill ────────────────────────────────────────────────────────


def backfill_hook_events(
    engine: Engine,
    start_year: int = 2019,
    end_year: int = 2025,
) -> None:
    """Full pipeline: extract hooks → persist events → compute profiles → persist profiles."""
    all_hooks = []

    for season in range(start_year, end_year + 1):
        if season == 2020:
            continue  # Skip COVID season

        hook_df = extract_hook_events(engine, season)
        if hook_df.empty:
            continue

        transformed_df = persist_hook_events(engine, hook_df)
        if not transformed_df.empty:
            all_hooks.append(transformed_df)

    if all_hooks:
        all_df = pd.concat(all_hooks, ignore_index=True)
        profiles = compute_hook_profiles(all_df)
        n_profiles = persist_hook_profiles(engine, profiles)
        logger.success(
            f"Persisted {n_profiles} manager hook profiles across all seasons."
        )
    else:
        logger.warning("No hook events extracted — no profiles generated.")
