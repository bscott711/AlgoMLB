"""Ingest starting lineups from the MLB Stats API boxscore endpoint."""

from __future__ import annotations

import datetime
import time
from typing import Optional

import httpx
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from algomlb.core.logger import logger
from algomlb.db.models import GameLineupORM


class LineupIngester:
    """Fetches starting lineups (9 batters per side) from MLB Stats API boxscores."""

    BASE_URL = "https://statsapi.mlb.com/api/v1"

    def __init__(self, session: Session, timeout: float = 30.0):
        self.session = session
        self.timeout = timeout

    def _fetch_boxscore(self, game_pk: int) -> Optional[dict]:
        """Fetch boxscore JSON for a single game."""
        url = f"{self.BASE_URL}/game/{game_pk}/boxscore"
        try:
            resp = httpx.get(url, timeout=self.timeout, follow_redirects=True)
            if resp.status_code != 200:
                logger.warning(f"Boxscore {game_pk}: HTTP {resp.status_code}")
                return None
            return resp.json()
        except Exception as e:
            logger.error(f"Boxscore {game_pk} fetch error: {e}")
            return None

    def _parse_starters(
        self, boxscore: dict, game_pk: int, game_date: datetime.date
    ) -> list[dict]:
        """Extract starting 9 batters for home and away from boxscore JSON."""
        records = []

        for side in ("home", "away"):
            team_data = boxscore.get("teams", {}).get(side, {})
            players = team_data.get("players", {})

            for _pid_key, pdata in players.items():
                batting_order_raw = pdata.get("battingOrder")
                if batting_order_raw is None:
                    continue

                # battingOrder: 100=1st, 200=2nd, ..., 900=9th.
                # Substitutes are 101, 201, etc. We only want starters.
                try:
                    bo_int = int(batting_order_raw)
                except (ValueError, TypeError):
                    continue

                if bo_int % 100 != 0:
                    continue  # Skip substitutes

                slot = bo_int // 100
                if slot < 1 or slot > 9:
                    continue

                person = pdata.get("person", {})
                position = pdata.get("position", {})

                records.append(
                    {
                        "game_pk": game_pk,
                        "game_date": game_date,
                        "team_side": side,
                        "batting_order": slot,
                        "player_id": person.get("id"),
                        "player_name": person.get("fullName"),
                        "position": position.get("abbreviation"),
                    }
                )

        return records

    def ingest_game(self, game_pk: int, game_date: datetime.date) -> int:
        """Fetch and upsert starting lineup for a single game."""
        boxscore = self._fetch_boxscore(game_pk)
        if boxscore is None:
            return 0

        records = self._parse_starters(boxscore, game_pk, game_date)
        if not records:
            return 0

        stmt = insert(GameLineupORM).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_game_lineup_slot",
            set_={
                "player_id": stmt.excluded.player_id,
                "player_name": stmt.excluded.player_name,
                "position": stmt.excluded.position,
            },
        )
        self.session.execute(stmt)
        self.session.commit()
        return len(records)

    def backfill_range(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        throttle_ms: int = 200,
    ) -> int:
        """Backfill lineups for all completed games in a date range.

        Reads game_pks from game_results, skips games that already have lineups.
        """
        from sqlalchemy import text

        # Find all completed game_pks in the range that don't have lineups yet
        query = text("""
            SELECT CAST(gr.game_id AS INTEGER) as game_pk, gr.game_date
            FROM game_results gr
            WHERE gr.game_date BETWEEN :start AND :end
            AND gr.status = 'COMPLETED'
            AND NOT EXISTS (
                SELECT 1 FROM game_lineups gl WHERE gl.game_pk = CAST(gr.game_id AS INTEGER)
            )
            ORDER BY gr.game_date
        """)
        rows = self.session.execute(
            query, {"start": start_date, "end": end_date}
        ).fetchall()

        logger.info(
            f"Lineup backfill: {len(rows)} games to process ({start_date} to {end_date})"
        )

        total = 0
        for i, row in enumerate(rows):
            game_pk, game_date = row[0], row[1]
            n = self.ingest_game(game_pk, game_date)
            total += n

            if (i + 1) % 50 == 0:
                logger.info(
                    f"  Progress: {i + 1}/{len(rows)} games, {total} lineup slots"
                )

            # Throttle to respect MLB API
            if throttle_ms > 0:
                time.sleep(throttle_ms / 1000.0)

        logger.success(
            f"Lineup backfill complete: {total} lineup slots across {len(rows)} games"
        )
        return total
