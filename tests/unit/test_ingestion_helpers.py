import pandas as pd
import pytest
from unittest.mock import MagicMock
from algomlb.ingestion.historical import HistoricalDataLoader
from algomlb.db.models import PitchEventORM
from algomlb.db.repository import DatabaseRepository


@pytest.fixture
def loader():
    mock_repo = MagicMock(spec=DatabaseRepository)
    return HistoricalDataLoader(repo=mock_repo)


def test_row_to_pitch_event(loader):
    """Test mapping a Statcast row to ORM object with NaN handling."""
    row = pd.Series(
        {
            "game_date": "2024-04-01",
            "release_speed": 95.5,
            "launch_speed": float("nan"),  # Should become None
            "batter": 123,
            "pitcher": 456,
            "events": "strikeout",
            "description": "called_strike",
        }
    )
    import datetime

    game_date = datetime.date(2024, 4, 1)

    event = loader._row_to_pitch_event(row, game_date)

    assert isinstance(event, PitchEventORM)
    assert event.game_date == game_date
    assert event.release_speed == 95.5
    assert event.launch_speed is None


def test_persist_pitch_events(loader):
    """Test bulk persistence logic."""
    df = pd.DataFrame(
        [{"game_date": "2024-04-01", "pitcher": 1, "batter": 2, "release_speed": 90.0}]
    )

    loader._persist_pitch_events(df)

    # Verify repo.save_pitch_events was called
    assert loader.repo.save_pitch_events.called
    args, _ = loader.repo.save_pitch_events.call_args
    assert len(args[0]) == 1
    assert isinstance(args[0][0], PitchEventORM)
