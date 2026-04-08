from datetime import date
from unittest.mock import MagicMock, patch
from algomlb.ingestion.transactions_ingester import (
    parse_injury,
    monthly_date_chunks,
    PlayerTransactionsIngester,
)
from algomlb.db.models import PlayerTransactionORM


def test_parse_injury():
    # Body part + Medical
    part, kind = parse_injury("Left hamstring strain")
    assert part == "left hamstring"
    assert kind == "strain"

    # Keyword priority (Tommy John)
    part, kind = parse_injury("Right elbow surgery, tommy john procedure")
    assert kind == "tommy john"

    # Status terms
    part, kind = parse_injury("Placed on the paternity list")
    assert kind == "paternity"

    # Unknowns
    part, kind = parse_injury("Some weird reason")
    assert part == "unknown"
    assert kind == "unknown"


def test_monthly_date_chunks():
    start = date(2023, 1, 1)
    end = date(2023, 2, 15)
    chunks = list(monthly_date_chunks(start, end))
    assert len(chunks) == 2
    assert chunks[0][0] == start
    assert chunks[1][1] == end


def test_detect_il_type():
    ingester = PlayerTransactionsIngester(MagicMock())
    assert ingester._detect_il_type("10-day IL", "") == "10day"
    assert ingester._detect_il_type("", "60-day injured list") == "60day"
    assert ingester._detect_il_type("15-day IL", "") == "15day"
    assert ingester._detect_il_type("", "7-day injured list") == "7day"
    assert ingester._detect_il_type("unknown", "none") is None


def test_fetch_transactions_error_handling():
    ingester = PlayerTransactionsIngester(MagicMock())
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = Exception("API Down")
        assert ingester.fetch_transactions(date.today(), date.today()) == []


def test_map_stats_api_to_orm():
    ingester = PlayerTransactionsIngester(MagicMock())
    tx = {
        "id": 123,
        "person": {"id": 1, "fullName": "Test Player"},
        "toTeam": {"id": 108},
        "typeDesc": "Injured List",
        "description": "Left elbow strain",
        "date": "2023-04-01",
        "effectiveDate": "2023-04-01",
        "resolutionDate": "2023-04-15",
    }

    orm = ingester._map_stats_api_to_orm(tx)
    assert isinstance(orm, PlayerTransactionORM)
    assert orm.transaction_id == "123"
    assert orm.injury_body_part == "left elbow"

    # Missing person
    tx_bad = {"id": 124, "toTeam": {"id": 108}}
    assert ingester._map_stats_api_to_orm(tx_bad) is None

    # Bad date
    tx_no_date = {"id": 125, "person": {"id": 1}, "toTeam": {"id": 101}, "date": "bad"}
    assert ingester._map_stats_api_to_orm(tx_no_date) is None


def test_ingest_range_routing():
    ingester = PlayerTransactionsIngester(MagicMock())
    with (
        patch.object(ingester, "fetch_transactions", return_value=[]) as mock_mod,
        patch.object(
            ingester, "fetch_legacy_transactions", return_value=[]
        ) as mock_leg,
    ):
        # Modern
        ingester.ingest_range(date(2023, 1, 1), date(2023, 1, 1))
        assert mock_mod.called
        assert not mock_leg.called

        # Legacy
        ingester.ingest_range(date(2010, 1, 1), date(2010, 1, 1))
        assert mock_leg.called
