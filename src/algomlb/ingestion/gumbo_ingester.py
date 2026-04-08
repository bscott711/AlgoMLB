import datetime
import httpx
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from algomlb.core.logger import logger
from algomlb.db.models import GumboPitchORM


class GumboIngester:
    """Fetches and persists the live game feed to extract precise pitch timestamps."""

    def __init__(self, session: Session):
        self.session = session

    def ingest_game(self, game_pk: int) -> int:
        """Fetch the GUMBO live feed for a single game and insert pitch events."""
        data = self._fetch_gumbo_json(game_pk)
        if not data:
            return 0

        all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])
        pitch_records = self._parse_all_plays(game_pk, all_plays)

        if not pitch_records:
            return 0

        self._upsert_pitches(pitch_records)
        return len(pitch_records)

    def _fetch_gumbo_json(self, game_pk: int) -> dict | None:
        """Perform HTTP fetch of GUMBO live feed."""
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(
                f"Failed to fetch GUMBO for {game_pk}: status {resp.status_code}"
            )
        except Exception as e:
            logger.error(f"Error fetching GUMBO for {game_pk}: {e}")
        return None

    def _parse_iso_time(self, time_str: str | None) -> datetime.datetime | None:
        """Sanitizer for ISO format timestamps."""
        if not time_str:
            return None
        try:
            return datetime.datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def _parse_all_plays(self, game_pk: int, all_plays: list) -> list[dict]:
        """Extract pitch-level records from the raw play data."""
        records = []
        for play in all_plays:
            at_bat_idx = play.get("about", {}).get("atBatIndex")
            at_bat_num = at_bat_idx + 1 if at_bat_idx is not None else -1

            for event in play.get("playEvents", []):
                if not event.get("isPitch"):
                    continue

                pitch_num = event.get("pitchNumber")
                if pitch_num is None:
                    continue

                records.append(
                    {
                        "game_pk": game_pk,
                        "at_bat_number": at_bat_num,
                        "pitch_number": pitch_num,
                        "play_id": event.get("playId"),
                        "start_time": self._parse_iso_time(event.get("startTime")),
                        "end_time": self._parse_iso_time(event.get("endTime")),
                    }
                )
        return records

    def _upsert_pitches(self, pitch_records: list[dict]):
        """Perform database upsert of pitch records."""
        stmt = insert(GumboPitchORM).values(pitch_records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["game_pk", "at_bat_number", "pitch_number"],
            set_={
                "play_id": stmt.excluded.play_id,
                "start_time": stmt.excluded.start_time,
                "end_time": stmt.excluded.end_time,
            },
        )
        self.session.execute(stmt)
        self.session.commit()
        return len(pitch_records)

    def ingest_games(self, game_pks: List[int]) -> int:
        """Fetch and inject GUMBO for a list of games."""
        total = 0
        for pk in game_pks:
            total += self.ingest_game(pk)
        return total
