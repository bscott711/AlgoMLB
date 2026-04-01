import re
from datetime import date, timedelta
from typing import Any, Iterator, List, Optional

import httpx
from algomlb.db.models import PlayerTransactionORM
from algomlb.db.repository import DatabaseRepository
from loguru import logger


def parse_injury(description: str) -> tuple[str, str]:
    """Extract body part and descriptor from a raw transaction description using regex."""
    desc = description.lower()

    # Body Part Regex: include lateralization (optional) and common terms
    # We still use regex for body part to handle the (left|right) prefix cleanly
    parts_pattern = r"(left|right|bilateral)?\s*(hamstring|oblique|elbow|shoulder|knee|forearm|back|hip|quadriceps|quad|calf|wrist|ankle|thumb|groin|finger|neck|rib|lat|tricep|bicep|hand|toe|foot|ucl|achilles|head)"
    part_match = re.search(parts_pattern, desc)
    part = part_match.group(0).strip() if part_match else "unknown"

    # Descriptor: Priority-based keyword matching (matches list order, not string order)
    medical_terms = [
        "tommy john",
        "tj",
        "fracture",
        "strain",
        "inflammation",
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
        "concussion",
        "illness",
        "flu",
        "virus",
    ]
    status_terms = [
        "paternity",
        "bereavement",
        "restricted",
        "reinstated",
        "activated",
        "personal",
        "placed",
    ]

    kind = next((t for t in medical_terms if t in desc), None)
    if not kind:
        kind = next((t for t in status_terms if t in desc), "unknown")

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

    def _detect_il_type(self, type_desc: str, description: str) -> Optional[str]:
        """Detect IL type from type description or raw text."""
        type_lower = type_desc.lower()
        desc_lower = description.lower()
        if "10-day" in type_lower or "10-day injured list" in desc_lower:
            return "10day"
        if "60-day" in type_lower or "60-day injured list" in desc_lower:
            return "60day"
        if "15-day" in type_lower or "15-day injured list" in desc_lower:
            return "15day"
        if "7-day" in type_lower or "7-day injured list" in desc_lower:
            return "7day"
        return None

    def _parse_iso_date(self, date_str: Any) -> Optional[date]:
        """Safely parse an ISO date string."""
        if not isinstance(date_str, str):
            return None
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            return None

    def _map_stats_api_to_orm(self, tx: dict) -> Optional[PlayerTransactionORM]:
        """Map a single StatsAPI transaction dictionary to an ORM object."""
        person = tx.get("person", {})
        player_id = person.get("id")
        player_name = person.get("fullName")
        team_id = tx.get("toTeam", {}).get("id")
        if player_id is None or team_id is None:
            return None

        type_desc = tx.get("typeDesc", "unknown")
        description = tx.get("description", "")

        il_type = self._detect_il_type(type_desc, description)
        part, kind = parse_injury(description)

        trans_date = self._parse_iso_date(tx.get("date"))
        if not trans_date:
            logger.warning(f"Skipping transaction {tx.get('id')} without valid date")
            return None

        return PlayerTransactionORM(
            transaction_id=str(tx.get("id")),
            player_id=int(player_id),
            player_name=player_name,
            team_id=int(team_id),
            transaction_date=trans_date,
            effective_date=self._parse_iso_date(tx.get("effectiveDate")),
            resolution_date=self._parse_iso_date(tx.get("resolutionDate")),
            type_desc=type_desc,
            il_type=il_type,
            injury_body_part=part if part != "unknown" else None,
            injury_descriptor=kind if kind != "unknown" else None,
            raw_description=description,
        )

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
