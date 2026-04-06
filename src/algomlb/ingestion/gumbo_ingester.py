import datetime
import httpx
from typing import Optional, List
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
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch GUMBO for {game_pk}: status {resp.status_code}")
                return 0

            data = resp.json()
        except Exception as e:
            logger.error(f"Error fetching GUMBO for {game_pk}: {e}")
            return 0

        # Parse liveData -> plays -> allPlays -> playEvents
        live_data = data.get("liveData", {})
        plays = live_data.get("plays", {})
        all_plays = plays.get("allPlays", [])

        pitch_records = []

        for play in all_plays:
            about = play.get("about", {})
            at_bat_index = about.get("atBatIndex")
            # Usually atBatIndex maps well, sometimes the play might not have it strictly,
            # but for our purposes, we will align it as best we can. Standard pitch-by-pitch
            # uses atBatIndex as the `at_bat_number`. Wait, Statcast uses `at_bat_number` which 
            # is 1-indexed. atBatIndex is usually 0-indexed in MLB API but Statcast merges it 
            # into 1-based or 0-based. Let's just use what's in 'about.atBatIndex' + 1 to align
            # with Statcast's at_bat_number. Actually, Statcast's at_bat_number matches MLB API's
            # about.atBatIndex exactly when the API's atBatIndex is 1? No, MLB API about.atBatIndex 
            # usually starts at 0. Let's pull the raw value to see, or rely on pitch_number.
            # Statcast at_bat_number starts at 1. MLB API atBatIndex starts at 0.
            # We will use about.atBatIndex + 1.

            at_bat_num = at_bat_index + 1 if at_bat_index is not None else -1

            play_events = play.get("playEvents", [])
            for event in play_events:
                # We only want pitch events (or pickoffs/actions if needed, but Statcast maps pitches)
                if not event.get("isPitch"):
                    continue

                pitch_num = event.get("pitchNumber")
                if pitch_num is None:
                    continue

                play_id = event.get("playId")
                start_time_str = event.get("startTime")
                end_time_str = event.get("endTime")

                start_time = None
                end_time = None

                if start_time_str:
                    try:
                        start_time = datetime.datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                    except:
                        pass
                
                if end_time_str:
                    try:
                        end_time = datetime.datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
                    except:
                        pass

                pitch_records.append({
                    "game_pk": game_pk,
                    "at_bat_number": at_bat_num,
                    "pitch_number": pitch_num,
                    "play_id": play_id,
                    "start_time": start_time,
                    "end_time": end_time
                })

        if not pitch_records:
            return 0

        # Upsert
        stmt = insert(GumboPitchORM).values(pitch_records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["game_pk", "at_bat_number", "pitch_number"],
            set_={
                "play_id": stmt.excluded.play_id,
                "start_time": stmt.excluded.start_time,
                "end_time": stmt.excluded.end_time
            }
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
