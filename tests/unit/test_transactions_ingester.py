from unittest.mock import MagicMock
from datetime import date
from algomlb.ingestion.transactions_ingester import (
    parse_injury,
    monthly_date_chunks,
    PlayerTransactionsIngester,
)


def test_parse_injury_known():
    # Test known body part and descriptor
    part, kind = parse_injury("placed on 10-day IL with a left hamstring strain")
    assert part == "left hamstring"
    assert kind == "strain"


def test_parse_injury_various_cases():
    # Test case insensitivity and multiple keywords
    part, kind = parse_injury("Right Elbow Inflammation")
    assert part == "right elbow"
    assert kind == "inflammation"


def test_parse_injury_unknown():
    # Test unknown description but known paternity
    part, kind = parse_injury("placed on the paternity list")
    assert part == "unknown"
    assert kind == "paternity"


def test_parse_injury_complex():
    # Test complex string
    part, kind = parse_injury(
        "reinstated from 60-day IL (recovering from Tommy John surgery on right elbow)"
    )
    assert part == "right elbow"
    assert kind == "tommy john"  # Now we recognize Tommy John!


def test_monthly_date_chunks_single_month():
    start = date(2024, 1, 1)
    end = date(2024, 1, 15)
    chunks = list(monthly_date_chunks(start, end))
    assert len(chunks) == 1
    assert chunks[0] == (start, end)


def test_monthly_date_chunks_multiple_months():
    start = date(2024, 1, 1)
    end = date(2024, 3, 10)
    chunks = list(monthly_date_chunks(start, end))
    # Jan 1 - Jan 31 (31 days)
    # Feb 1 - Mar 2 (30 days)
    # Mar 3 - Mar 10 (8 days)
    # Actually the logic is cursor + 30 days.
    # Chunk 1: 2024-01-01 to 2024-01-31
    # Chunk 2: 2024-02-01 to 2024-03-02
    # Chunk 3: 2024-03-03 to 2024-03-10
    assert len(chunks) == 3
    assert chunks[0][0] == start
    assert chunks[-1][1] == end


def test_monthly_date_chunks_one_day():
    start = date(2024, 1, 1)
    end = date(2024, 1, 1)
    chunks = list(monthly_date_chunks(start, end))
    assert len(chunks) == 1
    assert chunks[0] == (start, end)


def test_ingest_statsapi(respx_mock):
    repo = MagicMock()
    ingester = PlayerTransactionsIngester(repo)

    # Mock API call
    respx_mock.get("https://statsapi.mlb.com/api/v1/transactions").respond(
        json={
            "transactions": [
                {
                    "id": "123",
                    "person": {"id": 100},
                    "toTeam": {"id": 200},
                    "date": "2024-03-31",
                    "typeDesc": "Placed on 10-Day IL",
                    "description": "Hamstring strain",
                }
            ]
        }
    )

    count = ingester.ingest_range(date(2024, 3, 31), date(2024, 3, 31))

    assert count == 1
    assert repo.save_player_transactions.called
    orms = repo.save_player_transactions.call_args[0][0]
    assert len(orms) == 1
    assert orms[0].transaction_id == "123"
    assert orms[0].il_type == "10day"
    assert orms[0].injury_body_part == "hamstring"


def test_ingest_no_results(respx_mock):
    repo = MagicMock()
    ingester = PlayerTransactionsIngester(repo)

    respx_mock.get("https://statsapi.mlb.com/api/v1/transactions").respond(
        json={"transactions": []}
    )

    count = ingester.ingest_range(date(2024, 3, 31), date(2024, 3, 31))
    assert count == 0
    assert not repo.save_player_transactions.called


def test_ingest_legacy(respx_mock):
    repo = MagicMock()
    ingester = PlayerTransactionsIngester(repo)
    # 2018 uses legacy
    count = ingester.ingest_range(date(2018, 3, 31), date(2018, 3, 31))
    # It returns empty list for now as per implementation
    assert count == 0


def test_map_stats_api_to_orm_missing_date():
    repo = MagicMock()
    ingester = PlayerTransactionsIngester(repo)
    tx = {
        "id": "123",
        "person": {"id": 100},
        "toTeam": {"id": 200},
        # No date
    }
    orm = ingester._map_stats_api_to_orm(tx)
    assert orm is None


def test_ingest_date_error(respx_mock):
    repo = MagicMock()
    ingester = PlayerTransactionsIngester(repo)

    # Invalid date string
    respx_mock.get("https://statsapi.mlb.com/api/v1/transactions").respond(
        json={
            "transactions": [
                {
                    "id": "123",
                    "person": {"id": 100},
                    "toTeam": {"id": 200},
                    "date": "INVALID-DATE",
                }
            ]
        }
    )

    count = ingester.ingest_range(date(2024, 3, 31), date(2024, 3, 31))
    assert count == 0


def test_ingest_missing_person(respx_mock):
    repo = MagicMock()
    ingester = PlayerTransactionsIngester(repo)

    # Missing person.id
    respx_mock.get("https://statsapi.mlb.com/api/v1/transactions").respond(
        json={
            "transactions": [
                {
                    "id": "123",
                    # No person key
                    "toTeam": {"id": 200},
                    "date": "2024-03-31",
                }
            ]
        }
    )

    count = ingester.ingest_range(date(2024, 3, 31), date(2024, 3, 31))
    assert count == 0


def test_fetch_transactions_error(respx_mock):
    repo = MagicMock()
    ingester = PlayerTransactionsIngester(repo)

    respx_mock.get("https://statsapi.mlb.com/api/v1/transactions").respond(
        status_code=500
    )

    txs = ingester.fetch_transactions(date(2024, 3, 31), date(2024, 3, 31))
    assert txs == []


def test_map_stats_api_to_orm_60day():
    """Test mapping a 60-Day IL transaction from StatsAPI."""
    repo = MagicMock()
    ingester = PlayerTransactionsIngester(repo)
    tx = {
        "id": "123",
        "person": {"id": 1, "fullName": "Test Player"},
        "toTeam": {"id": 10, "teamName": "Test Team"},
        "date": "2024-03-01",
        "typeDesc": "Placed on 60-Day IL",
        "description": "Elbow surgery",
    }
    orm = ingester._map_stats_api_to_orm(tx)

    assert orm is not None
    assert orm.il_type == "60day"
    assert orm.injury_body_part == "elbow"
    assert orm.injury_descriptor == "surgery"


def test_parse_injury_mookie_betts():
    # Example 1: Left hand fracture
    part, kind = parse_injury(
        "Los Angeles Dodgers placed RF Mookie Betts on the 10-day injured list. Left hand fracture."
    )
    assert part == "left hand"
    assert kind == "fracture"

    # Example 2: Right rib fracture
    part, kind = parse_injury(
        "Los Angeles Dodgers placed RF Mookie Betts on the 10-day injured list. Right rib fracture."
    )
    assert part == "right rib"
    assert kind == "fracture"

    # Example 3: Right hip inflammation
    part, kind = parse_injury(
        "Los Angeles Dodgers placed RF Mookie Betts on the 10-day injured list. Right hip inflammation."
    )
    assert part == "right hip"
    assert kind == "inflammation"

    # Example 4: Activated (should still find descriptor if present, though il_type might be None in ingester logic)
    part, kind = parse_injury("Los Angeles Dodgers activated RF Mookie Betts.")
    assert part == "unknown"
    assert kind == "activated"
