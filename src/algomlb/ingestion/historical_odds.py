import datetime
import time
from typing import Optional

from loguru import logger

from algomlb.db.models import GameResultORM, HistoricalOddsORM
from algomlb.db.repository import DatabaseRepository
from algomlb.ingestion.odds_api import OddsAPIClient


class HistoricalOddsIngester:
    """
    Ingests opening and closing odds for historical games using The Odds API.
    Points to two snapshots per game: ~24h before (Opening) and ~5m before (Closing).
    """

    # Maps Odds API names to our database team names (standard MLB full names)
    TEAM_NAME_MAPPING = {
        "Oakland A's": "Oakland Athletics",
        "Cleveland Indians": "Cleveland Guardians",
        "Chicago White Sox": "Chicago White Sox",
        # Add others if discovered during backfill
    }

    def __init__(
        self, repo: DatabaseRepository, client: Optional[OddsAPIClient] = None
    ):
        self.repo = repo
        self.client = client or OddsAPIClient()

    def run_backfill(
        self, start_date: datetime.date, end_date: datetime.date, reverse: bool = True
    ):
        """
        Iterate through days and fetch opening/closing snapshots.
        """
        logger.info(
            f"Starting historical odds backfill: {start_date} to {end_date} (reverse={reverse})"
        )
        current = end_date if reverse else start_date
        step = datetime.timedelta(days=-1) if reverse else datetime.timedelta(days=1)

        total_days = (end_date - start_date).days + 1
        days_processed = 0

        while (reverse and current >= start_date) or (
            not reverse and current <= end_date
        ):
            try:
                self.ingest_day_snapshots(current)
                days_processed += 1
                logger.info(f"Progress: {days_processed}/{total_days} days.")
            except Exception as e:
                logger.error(f"Failed to process {current}: {e}")

            current += step
            # Small sleep to be respectful to API and logs
            time.sleep(0.5)

    def ingest_day_snapshots(self, date: datetime.date):
        """
        Fetch snapshots for a specific day.
        Opening: Snapshot at 10:00 AM UTC on 'date'.
        Closing: Snapshot at 11:50 PM UTC on 'date' (to capture late games).
        """
        # Opening snapshot timestamp
        opening_ts = datetime.datetime.combine(date, datetime.time(10, 0)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        # Closing snapshot timestamp
        closing_ts = datetime.datetime.combine(date, datetime.time(23, 50)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        logger.info(f"Ingesting odds snapshots for {date}...")

        self._process_snapshot(opening_ts, "opening")
        # Sleep to avoid rate limits if needed
        time.sleep(1)
        self._process_snapshot(closing_ts, "closing")

    def _process_snapshot(self, timestamp: str, odds_type: str):
        """Fetch a snapshot from Odds API and save to DB."""
        try:
            raw_odds = self.client.fetch_historical_odds(timestamp)
            if not raw_odds:
                logger.warning(f"No odds found for snapshot {timestamp}")
                return

            # Group outcomes by game and market
            processed = self._group_raw_odds(raw_odds)

            # Map to ORMs
            orms = self._build_orms(processed, odds_type)

            if orms:
                logger.info(
                    f"Saving {len(orms)} historical {odds_type} odds records..."
                )
                self.repo.save_historical_odds(orms)

        except Exception as e:
            logger.error(f"Error processing snapshot {timestamp}: {e}")

    def _group_raw_odds(self, raw_odds):
        """
        Groups raw outcomes by (home, away, date), then bookmaker, then market.
        Returns: { (home, away, date): { sportsbook: { market: { "outcomes": {name: price}, "snapshot_at": ts } } } }
        """
        processed = {}
        for o in raw_odds:
            key = (o.home_team, o.away_team, o.game_date)
            if key not in processed:
                processed[key] = {}

            if o.sportsbook not in processed[key]:
                processed[key][o.sportsbook] = {}

            if o.market_type not in processed[key][o.sportsbook]:
                processed[key][o.sportsbook][o.market_type] = {
                    "outcomes": {},
                    "snapshot_at": o.timestamp,
                }

            processed[key][o.sportsbook][o.market_type]["outcomes"][o.outcome] = {
                "price": o.price,
                "point": o.point,
            }
        return processed

    def _build_orms(self, processed, odds_type):
        """Iterates through processed data and builds ORM objects."""
        orms = []
        for (home, away, g_date), books in processed.items():
            game_orm = self._find_game(home, away, g_date)
            if not game_orm:
                continue

            for book, markets in books.items():
                for market_type, data in markets.items():
                    orm = self._create_orm(
                        game_orm.id, home, away, book, market_type, odds_type, data
                    )
                    if orm:
                        orms.append(orm)
        return orms

    def _find_game(self, home, away, g_date):
        """Finds a matching game in the database using name mapping."""
        # Normalize names
        home_norm = self.TEAM_NAME_MAPPING.get(home, home)
        away_norm = self.TEAM_NAME_MAPPING.get(away, away)

        return (
            self.repo.session.query(GameResultORM)
            .filter_by(home_team=home_norm, away_team=away_norm, game_date=g_date)
            .first()
        )

    def _create_orm(self, game_id, home, away, book, market_type, odds_type, data):
        """Creates a single HistoricalOddsORM record with spread/total support."""
        outcomes = data["outcomes"]
        home_price, away_price = 0, 0
        spread, total = None, None

        if market_type == "h2h":
            home_price = int(outcomes.get(home, {}).get("price", 0))
            away_price = int(outcomes.get(away, {}).get("price", 0))
        elif market_type == "spreads":
            # For spreads, we need the point and the price
            home_data = outcomes.get(home, {})
            away_data = outcomes.get(away, {})
            home_price = int(home_data.get("price", 0))
            away_price = int(away_data.get("price", 0))
            spread = home_data.get("point")
        elif market_type == "totals":
            over_data = outcomes.get("Over", {})
            under_data = outcomes.get("Under", {})
            home_price = int(over_data.get("price", 0))  # home_price stores Over
            away_price = int(under_data.get("price", 0))  # away_price stores Under
            total = over_data.get("point")

        if home_price == 0 and away_price == 0:
            return None

        return HistoricalOddsORM(
            game_id=game_id,
            bookmaker=book,
            market_type=market_type,
            odds_type=odds_type,
            home_price=home_price,
            away_price=away_price,
            spread=spread,
            total=total,
            snapshot_at=data["snapshot_at"],
            fetched_at=datetime.datetime.now(datetime.UTC),
        )
