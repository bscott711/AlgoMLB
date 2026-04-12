"""Backfill team_managers table from MLB StatsAPI coaching roster endpoint."""

from __future__ import annotations

import time

import httpx
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from algomlb.core.logger import logger
from algomlb.db.models import TeamManagerORM
from algomlb.db.session import get_engine

TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
ROSTER_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"


def _fetch_teams(season: int) -> list[dict]:
    """Return list of active MLB teams for a given season."""
    resp = httpx.get(
        TEAMS_URL,
        params={"sportId": 1, "season": season},
        timeout=15,
    )
    resp.raise_for_status()
    teams = resp.json().get("teams", [])
    return [
        {
            "id": t["id"],
            "abbreviation": t.get("abbreviation", ""),
            "name": t.get("name", ""),
        }
        for t in teams
        if t.get("sport", {}).get("id") == 1  # MLB only
    ]


def _fetch_manager(team_id: int, season: int) -> list[dict]:
    """Return manager(s) from the coaching roster for a team-season."""
    resp = httpx.get(
        ROSTER_URL.format(team_id=team_id),
        params={"rosterType": "coach", "season": season},
        timeout=15,
    )
    resp.raise_for_status()
    roster = resp.json().get("roster", [])
    managers = []
    for entry in roster:
        if entry.get("jobId") == "MNGR":
            person = entry.get("person", {})
            managers.append(
                {
                    "manager_id": person.get("id"),
                    "manager_name": person.get("fullName", ""),
                    "jersey_number": entry.get("jerseyNumber"),
                }
            )
    return managers


def backfill_team_managers(
    start_year: int = 2019,
    end_year: int | None = None,
    engine: Engine | None = None,
) -> None:
    """
    Fetch and upsert team managers for each season from StatsAPI.

    Parameters
    ----------
    start_year : int
        First season to backfill (default 2019).
    end_year : int | None
        Last season (inclusive). Defaults to current year.
    engine : Engine | None
        SQLAlchemy engine; uses default if None.
    """
    import datetime

    eng = engine or get_engine()
    end = end_year or datetime.date.today().year

    total_inserted = 0

    for season in range(start_year, end + 1):
        logger.info(f"Fetching teams for {season}...")
        teams = _fetch_teams(season)
        logger.info(f"  Found {len(teams)} teams")

        for team in teams:
            team_id = team["id"]
            team_abbr = team["abbreviation"]
            team_name = team["name"]

            try:
                managers = _fetch_manager(team_id, season)
            except Exception as e:
                logger.warning(
                    f"  ⚠️  Failed to fetch managers for {team_abbr} {season}: {e}"
                )
                continue

            if not managers:
                logger.warning(f"  ⚠️  No manager found for {team_abbr} {season}")
                continue

            for mgr in managers:
                row = {
                    "team_id": team_id,
                    "team_abbr": team_abbr,
                    "team_name": team_name,
                    "season": season,
                    "manager_id": mgr["manager_id"],
                    "manager_name": mgr["manager_name"],
                    "jersey_number": mgr["jersey_number"],
                    "effective_start_date": datetime.date(
                        season, 3, 1
                    ),  # Default to March 1st
                }

                with eng.begin() as conn:
                    stmt = pg_insert(TeamManagerORM).values([row])
                    upsert = stmt.on_conflict_do_update(
                        index_elements=["team_id", "season", "manager_id"],
                        set_={
                            "team_abbr": stmt.excluded.team_abbr,
                            "team_name": stmt.excluded.team_name,
                            "manager_name": stmt.excluded.manager_name,
                            "jersey_number": stmt.excluded.jersey_number,
                        },
                    )
                    conn.execute(upsert)
                    total_inserted += 1

            mgr_names = ", ".join(m["manager_name"] for m in managers)
            logger.info(f"  {team_abbr}: {mgr_names}")

            # Be polite to the API
            time.sleep(0.1)

        logger.success(f"Season {season} complete.")

    logger.success(
        f"Team managers backfill complete: {total_inserted} records upserted."
    )
