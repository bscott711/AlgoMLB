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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from algomlb.core.logger import logger
from algomlb.db.models import ManagerHookEventORM, ManagerHookProfileORM


# ── Hook event extraction ────────────────────────────────────────────────


def extract_hook_events(engine: Engine, season: int) -> pd.DataFrame:
    """
    Extract all pitcher removals for a given season from retrosheet_events.

    Returns a DataFrame with one row per hook, containing all columns
    needed for ManagerHookEventORM.
    """
    logger.info(f"Extracting hook events for {season}...")

    # Fetch all PAs for the season, ordered by game and play number
    query = f"""
        SELECT game_id, date, play_number, inning, top_bot,
               pitcher_id, pit_team, bat_team,
               pa_flag, nump, lp,
               outs_pre, outs_post,
               score_v, score_h,
               CASE WHEN br1_post IS NOT NULL THEN 1 ELSE 0 END as r1,
               CASE WHEN br2_post IS NOT NULL THEN 1 ELSE 0 END as r2,
               CASE WHEN br3_post IS NOT NULL THEN 1 ELSE 0 END as r3,
               single, double_flag, triple, hr, walk, hbp, k, runs
        FROM retrosheet_events
        WHERE EXTRACT(YEAR FROM date) = {season}
        AND pa_flag = 1
        ORDER BY game_id, play_number
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        logger.warning(f"No retrosheet PAs found for {season}")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Load manager lookup for this season
    mgr_df = pd.read_sql(
        f"SELECT team_abbr, manager_id, manager_name FROM team_managers WHERE season = {season}",
        engine,
    )
    mgr_map = dict(
        zip(
            mgr_df["team_abbr"],
            mgr_df[["manager_id", "manager_name"]].to_dict("records"),
        )
    )

    hook_rows: list[dict] = []

    # Process each game-side (one pitching team per half-inning direction)
    for (game_id, pit_team), group in df.groupby(["game_id", "pit_team"]):
        group = group.sort_values("play_number").reset_index(drop=True)
        game_date = group["date"].iloc[0]

        # Determine if this team is home (pitching in top half = home team)
        is_home = bool(group["top_bot"].iloc[0] == 0)

        # Manager lookup
        mgr_info = mgr_map.get(pit_team, {})
        manager_id = mgr_info.get("manager_id")
        manager_name = mgr_info.get("manager_name")

        # Track pitcher stints
        pitchers_seen: list[str] = []
        stint_start_idx = 0

        for i in range(len(group)):
            current_pitcher = group.loc[i, "pitcher_id"]

            # Detect pitcher change or end of game
            is_last_pa = i == len(group) - 1
            next_pitcher = group.loc[i + 1, "pitcher_id"] if not is_last_pa else None
            pitcher_changes = (
                next_pitcher is not None and next_pitcher != current_pitcher
            )

            if pitcher_changes or is_last_pa:
                # This pitcher's stint is rows[stint_start_idx : i+1]
                stint = group.iloc[stint_start_idx : i + 1]

                # Only record if pitcher is actually removed (not end-of-game)
                # End-of-game starters who finish are still interesting for the profile
                is_starter = len(pitchers_seen) == 0

                # Accumulate stats for this stint
                pitch_count = int(stint["nump"].sum())
                pa_count = len(stint)
                hits_allowed = int(
                    stint["single"].sum()
                    + stint["double_flag"].sum()
                    + stint["triple"].sum()
                    + stint["hr"].sum()
                )
                walks_allowed = int(stint["walk"].sum() + stint["hbp"].sum())
                strikeouts = int(stint["k"].sum())
                runs_allowed = int(stint["runs"].sum())
                hr_allowed = int(stint["hr"].sum())

                # TTO: count how many times lineup position 1 reappears
                lp_vals = stint["lp"].tolist()
                tto = 1
                seen_positions: set[int] = set()
                for lp_val in lp_vals:
                    if lp_val in seen_positions and lp_val == lp_vals[0]:
                        tto += 1
                    seen_positions.add(lp_val)
                # Simpler: count distinct lineup positions divided by 9
                n_distinct_lp = len(set(lp_vals))
                tto = max(1, (n_distinct_lp + 8) // 9)  # ceiling division

                # Game state at the hook (last PA by this pitcher)
                last_pa = stint.iloc[-1]
                hook_inning = int(last_pa["inning"])
                outs_at_hook = int(last_pa["outs_post"])

                # Score diff from pitcher's team perspective
                if is_home:
                    score_diff = int(last_pa["score_h"]) - int(last_pa["score_v"])
                else:
                    score_diff = int(last_pa["score_v"]) - int(last_pa["score_h"])

                # Runners on base after last PA (bitmask)
                runners_on = (
                    int(last_pa["r1"])
                    | (int(last_pa["r2"]) << 1)
                    | (int(last_pa["r3"]) << 2)
                )

                # Inherited runners for the NEXT pitcher
                inherited = int(last_pa["r1"]) + int(last_pa["r2"]) + int(last_pa["r3"])

                # Quality start: >= 6 IP and <= 3 ER (only for starters)
                # Better: sum (outs_post - outs_pre) for each PA, accounting for inning resets
                outs_by_pitcher = 0
                for _, pa in stint.iterrows():
                    if pa["outs_post"] >= pa["outs_pre"]:
                        outs_by_pitcher += pa["outs_post"] - pa["outs_pre"]
                    else:
                        # Inning rolled over (outs reset from 3 to 0)
                        outs_by_pitcher += 3 - pa["outs_pre"]

                ip_approx = outs_by_pitcher / 3.0
                was_qs = is_starter and ip_approx >= 6.0 and runs_allowed <= 3

                hook_before_3rd = tto < 3

                hook_rows.append(
                    {
                        "game_id": game_id,
                        "game_date": game_date,
                        "season": season,
                        "team_abbr": pit_team,
                        "manager_id": int(manager_id)
                        if manager_id is not None
                        and not (isinstance(manager_id, float) and np.isnan(manager_id))
                        else None,
                        "manager_name": manager_name,
                        "pitcher_id": current_pitcher,
                        "is_starter": is_starter,
                        "inning": hook_inning,
                        "outs_at_hook": outs_at_hook,
                        "pitch_count": pitch_count,
                        "pa_count": pa_count,
                        "tto": tto,
                        "score_diff": score_diff,
                        "is_home": is_home,
                        "runners_on": runners_on,
                        "inherited_runners": inherited if pitcher_changes else 0,
                        "hits_allowed": hits_allowed,
                        "walks_allowed": walks_allowed,
                        "strikeouts": strikeouts,
                        "runs_allowed": runs_allowed,
                        "hr_allowed": hr_allowed,
                        "was_quality_start": was_qs,
                        "hook_before_3rd_tto": hook_before_3rd and is_starter,
                    }
                )

                pitchers_seen.append(str(current_pitcher))
                stint_start_idx = i + 1

    result = pd.DataFrame(hook_rows)
    logger.info(f"  Extracted {len(result)} hook events for {season}")
    return result


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

        avg_pc = group["pitch_count"].mean()
        med_pc = group["pitch_count"].median()

        # Approx innings from outs (inning * 3 + outs is an overcount, use PA-based)
        avg_inn = group["inning"].mean()  # crude but useful

        before_3rd = group["hook_before_3rd_tto"].sum() / n_starts
        with_lead = (group["score_diff"] > 0).sum() / n_starts
        with_deficit = (group["score_diff"] < 0).sum() / n_starts
        quick_hook = (group["pitch_count"] < 80).sum() / n_starts
        long_hook = (group["pitch_count"] >= 100).sum() / n_starts

        profiles.append(
            {
                "manager_id": int(m_id),
                "manager_name": str(mgr_name),
                "season": s_sn,
                "total_hooks": n_starts,
                "total_sp_starts": n_starts,
                "avg_sp_pitch_count": round(float(avg_pc), 1),
                "median_sp_pitch_count": round(float(med_pc), 1),
                "avg_sp_innings": round(float(avg_inn), 2),
                "pull_before_3rd_tto_pct": round(float(before_3rd), 4),
                "pull_with_lead_pct": round(float(with_lead), 4),
                "pull_with_deficit_pct": round(float(with_deficit), 4),
                "quick_hook_under_80_pitches_pct": round(float(quick_hook), 4),
                "long_hook_over_100_pitches_pct": round(float(long_hook), 4),
            }
        )

    return pd.DataFrame(profiles)


# ── Persistence ──────────────────────────────────────────────────────────


def persist_hook_events(engine: Engine, hook_df: pd.DataFrame) -> int:
    """Upsert hook events into manager_hook_events. Returns count."""
    if hook_df.empty:
        return 0

    # To ensure compatibility with PostgreSQL nullable integer columns,
    # we convert to object and explicitly set NaPs to None.
    hook_df["manager_id"] = hook_df["manager_id"].astype(object)
    hook_df.loc[hook_df["manager_id"].isna(), "manager_id"] = None
    rows = hook_df.to_dict("records")
    chunk_size = 2000
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            stmt = pg_insert(ManagerHookEventORM).values(chunk)
            upsert = stmt.on_conflict_do_update(
                index_elements=["game_id", "pitcher_id"],
                set_={
                    col: stmt.excluded[col]
                    for col in hook_df.columns
                    if col not in ("game_id", "pitcher_id")
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

        n_events = persist_hook_events(engine, hook_df)
        logger.info(f"  Persisted {n_events} hook events for {season}")
        all_hooks.append(hook_df)

    if all_hooks:
        all_df = pd.concat(all_hooks, ignore_index=True)
        profiles = compute_hook_profiles(all_df)
        n_profiles = persist_hook_profiles(engine, profiles)
        logger.success(
            f"Persisted {n_profiles} manager hook profiles across all seasons."
        )
    else:
        logger.warning("No hook events extracted — no profiles generated.")
