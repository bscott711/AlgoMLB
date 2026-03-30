import datetime
import time
from typing import Optional

from loguru import logger
from sqlalchemy import select

from algomlb.db.models import GameResultORM, HistoricalOddsORM
from algomlb.db.repository import DatabaseRepository
from algomlb.ingestion.odds_api import OddsAPIClient


class HistoricalOddsIngester:
    """
    Ingests opening and closing odds for historical games using The Odds API.
    Points to two snapshots per game: ~24h before (Opening) and ~5m before (Closing).
    """

    def __init__(
        self, repo: DatabaseRepository, client: Optional[OddsAPIClient] = None
    ):
        self.repo = repo
        self.client = client or OddsAPIClient()

    def run_backfill(self, start_date: datetime.date, end_date: datetime.date):
        """
        Iterate through games in the date range and fetch opening/closing odds.
        Note: The Odds API historical endpoint is credit-intensive.
        """
        # 1. Get all games in the range from our DB
        with self.repo.session.begin_nested():
            stmt = select(GameResultORM).where(
                GameResultORM.game_date >= start_date,
                GameResultORM.game_date <= end_date,
            )
            games = self.repo.session.execute(stmt).scalars().all()

        if not games:
            logger.warning(f"No games found in DB for range {start_date} to {end_date}")
            return

        logger.info(f"Starting historical odds backfill for {len(games)} games...")

        for game in games:
            # We need a 'commence_time'. In our DB we only have 'game_date'.
            # For historical backfill, we might need to fetch the schedule again
            # or assume a default time if not stored.
            # However, Statcast results don't have start time in seconds.
            # Let's use 12:00 PM local or similar if we don't have it.
            # BETTER: Fetch a daily snapshot at 10:00 AM UTC for 'Opening'
            # and 11:55 PM UTC for 'Closing' for that day's games.
            pass

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

            processed[key][o.sportsbook][o.market_type]["outcomes"][o.outcome] = o.price
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
        """Finds a matching game in the database."""
        return (
            self.repo.session.query(GameResultORM)
            .filter_by(home_team=home, away_team=away, game_date=g_date)
            .first()
        )

    def _create_orm(self, game_id, home, away, book, market_type, odds_type, data):
        """Creates a single HistoricalOddsORM record if valid prices exist."""
        outcomes = data["outcomes"]
        home_price, away_price = 0, 0

        # Logic for H2H markets
        if market_type == "h2h":
            home_price = int(outcomes.get(home, 0))
            away_price = int(outcomes.get(away, 0))

        # TODO: Add logic for spreads/totals (spread, total fields in ORM)

        if home_price == 0 and away_price == 0:
            return None

        return HistoricalOddsORM(
            game_id=game_id,
            bookmaker=book,
            market_type=market_type,
            odds_type=odds_type,
            home_price=home_price,
            away_price=away_price,
            snapshot_at=data["snapshot_at"],
            fetched_at=datetime.datetime.now(datetime.UTC),
        )
