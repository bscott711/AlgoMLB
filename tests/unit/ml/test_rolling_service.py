import pytest
from unittest.mock import MagicMock, patch
from datetime import date
import pandas as pd
from algomlb.ml.rolling_service import RollingService
from algomlb.domain import PlayerRole, BaselineQuality
from algomlb.ml.rolling_processor import PlayerRollingRecord
from algomlb.db.models import PlayerRollingFeaturesORM


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_season_start_date.return_value = date(2024, 3, 20)
    db.session.execute.return_value.fetchall.return_value = [
        (123, "PITCHER"),
        (456, "BATTER"),
    ]
    return db


@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.compute_for_player.return_value = MagicMock(
        player_id=123,
        game_date=date(2024, 4, 1),
        season=2024,
        role=PlayerRole.PITCHER,
        window_games=5,
        n_games_used=5,
        days_since_last_game=5,
        baseline_quality=BaselineQuality.FULL,
        shrinkage_applied=False,
        roll_pitches=100.0,
        roll_strikes_pct=0.6,
        roll_whiff_pct=0.1,
        roll_k_pct=0.08,
        roll_bb_pct=0.02,
        roll_avg_release_speed=95.0,
        roll_avg_pfx_x=-0.5,
        roll_avg_pfx_z=1.0,
        roll_avg_pitcher_xwoba=0.300,
        roll_pitcher_xwoba_shrunk=0.309,
        # Batter fields None
        roll_pas=None,
        roll_hits_per_pa=None,
        roll_k_pct_batter=None,
        roll_bb_pct_batter=None,
        roll_barrel_pct=None,
        roll_avg_launch_speed=None,
        roll_avg_launch_angle=None,
        roll_avg_batter_xwoba=None,
        roll_batter_xwoba_shrunk=None,
    )
    return processor


@patch("algomlb.ml.rolling_service.pd.read_sql")
def test_process_single_date(mock_read_sql, mock_db, mock_processor):
    mock_read_sql.return_value = pd.DataFrame(
        [
            {
                "player_id": 123,
                "role": "PITCHER",
                "game_date": date(2024, 3, 25),
                "pitches": 100,
            }
        ]
    )

    service = RollingService(mock_db, mock_processor)
    res = service.process_single_date(date(2024, 4, 1))

    assert mock_db.save_player_rolling_features_records.called
    assert res == mock_db.save_player_rolling_features_records.return_value


@patch("algomlb.ml.rolling_service.pd.read_sql")
def test_process_date_range(mock_read_sql, mock_db, mock_processor):
    mock_read_sql.return_value = pd.DataFrame([])
    service = RollingService(mock_db, mock_processor)

    with patch.object(service, "process_single_date", return_value=1) as mock_single:
        res = service.process_date_range(date(2024, 4, 1), date(2024, 4, 3))
        assert mock_single.call_count == 3
        assert res == 3


def test_dry_run(mock_db, mock_processor):
    with patch(
        "algomlb.ml.rolling_service.pd.read_sql",
        return_value=pd.DataFrame(columns=["player_id", "role", "game_date"]),
    ):
        service = RollingService(mock_db, mock_processor)
        res = service.process_single_date(date(2024, 4, 1), dry_run=True)
        assert not mock_db.save_player_rolling_features_records.called
        assert res == 2  # mock_db.session.execute returns 2 pairs


def test_map_features_to_orm_nans(mock_db, mock_processor):
    service = RollingService(mock_db, mock_processor)
    orm = PlayerRollingFeaturesORM()
    data = PlayerRollingRecord(
        player_id=123,
        game_date=date(2024, 4, 1),
        season=2024,
        role=PlayerRole.PITCHER,
        window_games=5,
        n_games_used=0,
        days_since_last_game=None,
        baseline_quality=BaselineQuality.COLD_START,
        shrinkage_applied=False,
        roll_avg_release_speed=float("nan"),  # Test NaN handling
    )
    service._map_features_to_orm(orm, data)
    assert orm.roll_avg_release_speed is None


def test_process_single_date_no_players(mock_db, mock_processor):
    mock_db.session.execute.return_value.fetchall.return_value = []
    service = RollingService(mock_db, mock_processor)
    res = service.process_single_date(date(2024, 4, 1))
    assert res == 0
