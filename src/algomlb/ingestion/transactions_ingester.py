from datetime import date, timedelta
from typing import Iterator, List, Optional

import httpx
from algomlb.db.models import PlayerTransactionORM
from algomlb.db.repository import DatabaseRepository
from loguru import logger

BODY_PARTS: tuple[str, ...] = (
    "hamstring",
    "oblique",
    "elbow",
    "shoulder",
    "knee",
    "forearm",
    "back",
    "hip",
    "quadriceps",
    "calf",
    "wrist",
    "ankle",
    "thumb",
    "groin",
    "finger",
    "neck",
    "rib",
    "lat",
    "tricep",
    "bicep",
)
DESCRIPTORS: tuple[str, ...] = (
    "strain",
    "inflammation",
    "fracture",
    "surgery",
    "tightness",
    "soreness",
    "fatigue",
    "nerve",
    "tendinitis",
    "sprain",
    "bruise",
    "contusion",
    "tear",
    "stress reaction",
)
IL_PLACEMENT_TYPES: frozenset[str] = frozenset(
    {
        "Placed on 10-Day IL",
        "Placed on 60-Day IL",
        "Placed on Restricted List",
    }
)


def parse_injury(description: str) -> tuple[str, str]:
    """Extract body part and descriptor from a raw transaction description."""
    desc = description.lower()
    part = next((p for p in BODY_PARTS if p in desc), "unknown")
    kind = next((d for d in DESCRIPTORS if d in desc), "unknown")
    return part, kind


def monthly_date_chunks(
    start: date,
    end: date,
) -> Iterator[tuple[date, date]]:
    """Generate (chunk_start, chunk_end) pairs in ~monthly windows."""
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=30), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


class PlayerTransactionsIngester:
    """Ingests player transactions from MLB StatsAPI and legacy BAM endpoints."""

    def __init__(self, repo: DatabaseRepository):
        self.repo = repo

    def fetch_transactions(self, start_date: date, end_date: date) -> List[dict]:
        """Fetch transactions from StatsAPI."""
        try:
            response = httpx.get(
                "https://statsapi.mlb.com/api/v1/transactions",
                params={
                    "startDate": start_date.isoformat(),
                    "endDate": end_date.isoformat(),
                    "sportId": 1,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("transactions", [])
        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")
            return []

    def fetch_legacy_transactions(self, start_date: date, end_date: date) -> List[dict]:
        """
        Fetch legacy transactions from BAM endpoint.
        Note: Implementation placeholder for details of BAM endpoint.
        """
        # For now, we use the same monthly chunking logic but different endpoint
        # legacy-lookup-service-prod.mlb.com/json/named.transaction_all.bam
        # This is a placeholder for actual legacy fetching logic
        return []

    def _map_stats_api_to_orm(self, tx: dict) -> Optional[PlayerTransactionORM]:
        """Map a single StatsAPI transaction dictionary to an ORM object."""
        player_id = tx.get("person", {}).get("id")
        team_id = tx.get("toTeam", {}).get("id")
        if player_id is None or team_id is None:
            return None

        type_desc = tx.get("typeDesc", "unknown")
        description = tx.get("description", "")

        il_type = None
        if "10-Day IL" in type_desc:
            il_type = "10day"
        elif "60-Day IL" in type_desc:
            il_type = "60day"

        part, kind = parse_injury(description)

        try:
            trans_date_raw = tx.get("date")
            if not trans_date_raw or not isinstance(trans_date_raw, str):
                logger.warning(
                    f"Skipping transaction {tx.get('id')} without valid date"
                )
                return None

            trans_date = date.fromisoformat(trans_date_raw)

            res_date_raw = tx.get("resolutionDate")
            eff_date_raw = tx.get("effectiveDate")

            res_date = (
                date.fromisoformat(res_date_raw)
                if isinstance(res_date_raw, str)
                else None
            )
            eff_date = (
                date.fromisoformat(eff_date_raw)
                if isinstance(eff_date_raw, str)
                else None
            )

            return PlayerTransactionORM(
                transaction_id=str(tx.get("id")),
                player_id=int(player_id),
                team_id=int(team_id),
                transaction_date=trans_date,
                effective_date=eff_date,
                resolution_date=res_date,
                type_desc=type_desc,
                il_type=il_type,
                injury_body_part=part if il_type else None,
                injury_descriptor=kind if il_type else None,
                raw_description=description,
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing date in transaction {tx.get('id')}: {e}")
            return None

    def ingest_range(self, start_date: date, end_date: date):
        """Ingest transactions for a given date range using monthly chunks."""
        total_ingested = 0
        for chunk_start, chunk_end in monthly_date_chunks(start_date, end_date):
            logger.info(f"Ingesting transactions from {chunk_start} to {chunk_end}")

            # Use appropriate endpoint based on date
            if chunk_start.year >= 2019:
                txs = self.fetch_transactions(chunk_start, chunk_end)
            else:
                txs = self.fetch_legacy_transactions(chunk_start, chunk_end)

            if not txs:
                continue

            orms = []
            for tx in txs:
                orm = self._map_stats_api_to_orm(tx)
                if orm:
                    orms.append(orm)

            if orms:
                self.repo.save_player_transactions(orms)
                total_ingested += len(orms)

        return total_ingested
