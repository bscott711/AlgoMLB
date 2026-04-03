import pandas as pd
import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch
from algomlb.ml.silver_processor import summarize_to_silver, apply_bayesian_shrinkage


class TestSilverProcessor:
    def _mock_df(self) -> pd.DataFrame:
        # Columns required by summarize_to_silver logic:
        # launch_speed_angle (for barrels), description (for whiffs),
        # pitcher, batter, game_pk, game_date, pitch_number, at_bat_number
        return pd.DataFrame(
            [
                {
                    "game_pk": 1,
                    "pitcher": 101,
                    "batter": 501,
                    "game_date": date(2025, 4, 1),
                    "at_bat_number": 1,
                    "pitch_number": 1,
                    "description": "swinging_strike",
                    "events": "strikeout",
                    "release_speed": 95.0,
                    "pfx_x": 1.0,
                    "pfx_z": 1.0,
                    "estimated_woba_using_speedangle": 0.3,
                    "launch_speed": 95.0,
                    "launch_angle": 10.0,
                    "launch_speed_angle": 1,  # Not a barrel
                    "ingested_at": datetime(2025, 4, 1, 10, 0),
                }
            ]
        )

    def test_summarize_logic(self):
        df = self._mock_df()
        res = summarize_to_silver(df)
        assert not res.empty
        assert "player_id" in res.columns
        assert "role" in res.columns
        # Verify the whiffs flag worked
        assert res[res["role"] == "PITCHER"]["whiffs"].iloc[0] == 1

    def test_bayesian_shrinkage(self):
        val = apply_bayesian_shrinkage(0.3, 10, 0.5, 5)
        assert val == pytest.approx(0.3666, rel=1e-3)
        assert apply_bayesian_shrinkage(0.3, 10, None, 5) == 0.3

    @patch("algomlb.ml.silver_processor.get_engine")
    def test_upsert_silver(self, mock_get_engine):
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        df = pd.DataFrame(
            [{"player_id": 1, "game_pk": 1, "role": "PITCHER", "val": 0.5}]
        )
        from algomlb.ml.silver_processor import _upsert_silver

        _upsert_silver(df)
        assert mock_conn.execute.called

    @patch("algomlb.ml.silver_processor.get_engine")
    @patch("algomlb.ml.silver_processor.pd.read_sql")
    @patch("algomlb.ml.silver_processor._upsert_silver")
    def test_process_silver_incremental_success(
        self, mock_upsert, mock_read_sql, mock_get_engine
    ):
        mock_engine = mock_get_engine.return_value
        mock_conn = mock_engine.connect.return_value.__enter__.return_value

        # 1. Mock Checkpoint
        mock_conn.execute.return_value.fetchone.return_value = [datetime(2025, 3, 31)]

        # 2. Mock finding raw pks
        mock_conn.execute.return_value.fetchall.return_value = [
            (1, date(2025, 4, 1), datetime(2025, 4, 1, 12, 0))
        ]

        # 3. Mock reading detail
        # process_silver_incremental calls read_sql for detail first, then fetch_prior_year_stats calls it for rollup
        mock_read_sql.side_effect = [
            self._mock_df(),  # game detail (first)
            pd.DataFrame(
                [{"player_id": 101, "role": "PITCHER", "avg_pitcher_xwoba": 0.4}]
            ),  # prior stats (second)
        ]

        from algomlb.ml.silver_processor import process_silver_incremental

        process_silver_incremental()
        assert mock_upsert.called

    @patch("algomlb.ml.silver_processor.get_engine")
    def test_fetch_prior_year_stats(self, mock_get_engine):
        from algomlb.ml.silver_processor import fetch_prior_year_stats

        with patch("pandas.read_sql", return_value=pd.DataFrame()):
            res = fetch_prior_year_stats(2024)
            assert res.empty

    def test_summarize_to_silver_empty(self):
        """Test summarize_to_silver with empty dataframe."""
        res = summarize_to_silver(pd.DataFrame())
        assert res.empty

    def test_summarize_to_silver_batter_shrinkage(self):
        """Test summarize_to_silver with batter shrinkage logic."""
        df = self._mock_df()
        # Add a batter row with xwOBA
        prior_stats = pd.DataFrame(
            [{"player_id": 501, "role": "BATTER", "avg_batter_xwoba": 0.4}]
        )
        res = summarize_to_silver(df, prior_year_stats=prior_stats)
        # Verify batter xwOBA was calculated/shrunk
        # Calculation: (n * curr + k * prior) / (n+k)
        # n=1 (pa), k=SETTINGS.ml.quant_batter_shrinkage_k (likely 50 or 500)
        # Setting values in mock or just verifying it's not the original 0.3
        batter_xwoba = res[res["role"] == "BATTER"]["avg_batter_xwoba"].iloc[0]
        assert batter_xwoba != 0.3

    @patch("algomlb.ml.silver_processor.get_engine")
    def test_fetch_prior_year_stats_error(self, mock_get_engine):
        """Test fetch_prior_year_stats exception handling."""
        from algomlb.ml.silver_processor import fetch_prior_year_stats

        with patch("pandas.read_sql", side_effect=Exception("Read error")):
            res = fetch_prior_year_stats(2024)
            assert res.empty

    @patch("algomlb.ml.silver_processor.get_engine")
    def test_process_silver_incremental_no_data(self, mock_get_engine):
        """Test incremental processor when no new raw data is found."""
        mock_engine = mock_get_engine.return_value
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        # Mock Checkpoint found
        mock_conn.execute.return_value.fetchone.return_value = [datetime(2025, 3, 31)]
        # Mock no new games found
        mock_conn.execute.return_value.fetchall.return_value = []

        from algomlb.ml.silver_processor import process_silver_incremental

        process_silver_incremental()
        # Should return early before reading sql for details
        assert mock_conn.execute.call_count == 2
