"""
Contract tests for count-state conditioning in the PA outcome pipeline.

Validates:
1. FeaturePipeline.build_pa_matrix() produces cnt_* columns when balls/strikes present
2. GameState count tracking and count_state property
3. SimulationEngine._simulate_count Markov chain termination
4. SimulationEngine._sample_pa graceful fallback for flat/3D cache
5. SimulationEngine._precompute_matchups 3D expansion
6. SimulationEngine._model_has_count_features detection
"""

import numpy as np
import pandas as pd
import pytest

from algomlb.ml.features import FeaturePipeline
from algomlb.ml.monte_carlo.state import GameState
from algomlb.ml.monte_carlo.engine import SimulationEngine, COUNT_STATES


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_pas_df():
    """Retrosheet-like PA dataframe WITH balls/strikes columns."""
    return pd.DataFrame({
        "game_id": ["G1"] * 4,
        "game_date": pd.Timestamp("2024-06-01"),
        "batter_id": [100, 101, 100, 101],
        "pitcher_id": [200, 200, 201, 201],
        "balls": [0, 3, 1, 0],
        "strikes": [2, 1, 0, 0],
        "pa_outcome": ["strikeout", "walk", "single", "out_in_play"],
    })


@pytest.fixture
def minimal_pas_df_no_count():
    """Retrosheet-like PA dataframe WITHOUT balls/strikes (legacy)."""
    return pd.DataFrame({
        "game_id": ["G1"] * 4,
        "game_date": pd.Timestamp("2024-06-01"),
        "batter_id": [100, 101, 100, 101],
        "pitcher_id": [200, 200, 201, 201],
        "pa_outcome": ["strikeout", "walk", "single", "out_in_play"],
    })


@pytest.fixture
def minimal_gold():
    """Minimal pitcher/batter Gold DataFrames for joins."""
    pitcher = pd.DataFrame({
        "player_id": [200, 201],
        "game_date": pd.Timestamp("2024-06-01"),
        "roll_k_pct": [0.25, 0.30],
        "roll_bb_pct": [0.08, 0.10],
    })
    batter = pd.DataFrame({
        "player_id": [100, 101],
        "game_date": pd.Timestamp("2024-06-01"),
        "roll_k_pct_batter": [0.22, 0.18],
        "roll_bb_pct_batter": [0.09, 0.12],
    })
    return pitcher, batter


class FakeModel:
    """Minimal mock model with configurable feature names."""

    def __init__(self, feature_names=None, n_classes=8):
        self.feature_names_in_ = np.array(feature_names) if feature_names else None
        self.n_classes = n_classes
        self.le = None
        self.calibrated_clf = None

    def predict_proba(self, X):
        n = len(X)
        probs = np.ones((n, self.n_classes)) / self.n_classes
        return probs

    def get_base_xgb_estimator(self):
        return self


class FakeModelWithCount(FakeModel):
    """Model that includes cnt_* features (simulates a retrained model)."""

    def __init__(self, n_classes=8):
        base_features = ["pitcher_roll_k_pct", "batter_roll_k_pct_batter", "elo_diff"]
        count_features = [f"cnt_{cs}" for cs in COUNT_STATES]
        super().__init__(feature_names=base_features + count_features, n_classes=n_classes)


# ── Test: FeaturePipeline Count Engineering ──────────────────────────────────


class TestFeaturePipelineCount:
    """Verify build_pa_matrix correctly one-hot encodes count_state."""

    def test_count_columns_present_when_balls_strikes_available(
        self, minimal_pas_df, minimal_gold
    ):
        pipeline = FeaturePipeline()
        pitcher, batter = minimal_gold
        X, y = pipeline.build_pa_matrix(minimal_pas_df, pitcher, batter)

        # All 12 cnt_* columns should be present
        cnt_cols = [c for c in X.columns if c.startswith("cnt_")]
        assert len(cnt_cols) == 12, f"Expected 12 cnt_ columns, got {len(cnt_cols)}: {cnt_cols}"

        # Verify one-hot encoding: each row should have exactly one cnt_* = 1.0
        cnt_matrix = X[cnt_cols].values
        for i, row in enumerate(cnt_matrix):
            assert row.sum() == pytest.approx(1.0, abs=1e-5), (
                f"Row {i}: cnt_* columns should sum to 1.0, got {row.sum()}"
            )

    def test_no_count_columns_without_balls_strikes(
        self, minimal_pas_df_no_count, minimal_gold
    ):
        pipeline = FeaturePipeline()
        pitcher, batter = minimal_gold
        X, y = pipeline.build_pa_matrix(minimal_pas_df_no_count, pitcher, batter)

        cnt_cols = [c for c in X.columns if c.startswith("cnt_")]
        assert len(cnt_cols) == 0, f"Should have no cnt_ columns, got {cnt_cols}"

    def test_specific_count_encoding(self, minimal_pas_df, minimal_gold):
        """Row 0 has balls=0, strikes=2 → cnt_0-2 should be 1.0."""
        pipeline = FeaturePipeline()
        pitcher, batter = minimal_gold
        X, y = pipeline.build_pa_matrix(minimal_pas_df, pitcher, batter)

        if "cnt_0-2" in X.columns:
            # First row: balls=0, strikes=2 → count_state "0-2"
            assert X["cnt_0-2"].iloc[0] == pytest.approx(1.0)
            assert X["cnt_0-0"].iloc[0] == pytest.approx(0.0)


# ── Test: GameState Count Tracking ──────────────────────────────────────────


class TestGameStateCount:

    def test_initial_count_is_zero(self):
        gs = GameState()
        assert gs.balls == 0
        assert gs.strikes == 0
        assert gs.count_state == "0-0"

    def test_count_state_property(self):
        gs = GameState()
        gs.balls = 3
        gs.strikes = 2
        assert gs.count_state == "3-2"

    def test_reset_count(self):
        gs = GameState()
        gs.balls = 2
        gs.strikes = 1
        gs.reset_count()
        assert gs.balls == 0
        assert gs.strikes == 0
        assert gs.count_state == "0-0"

    def test_count_independent_of_bases(self):
        gs = GameState()
        gs.balls = 1
        gs.strikes = 2
        gs.clear_bases()
        # clear_bases should NOT affect count
        assert gs.count_state == "1-2"


# ── Test: Simulate Count Markov Chain ────────────────────────────────────────


class TestSimulateCount:

    def test_simulate_count_terminates(self):
        """Count simulation must always terminate."""
        model = FakeModelWithCount()
        engine = SimulationEngine(pa_model=model, seed=42)
        engine._count_aware = True
        gs = GameState()

        for _ in range(1000):
            engine._simulate_count(gs)
            assert gs.balls <= 4
            assert gs.strikes <= 2
            assert gs.count_state in COUNT_STATES or gs.count_state in ["4-0", "4-1", "4-2", "0-3", "1-3", "2-3", "3-3"]

    def test_simulate_count_produces_valid_states(self):
        """Distribution check: should see most common count states appear."""
        model = FakeModelWithCount()
        engine = SimulationEngine(pa_model=model, seed=123)
        engine._count_aware = True
        gs = GameState()

        counts = {}
        for _ in range(10000):
            engine._simulate_count(gs)
            cs = gs.count_state
            counts[cs] = counts.get(cs, 0) + 1

        # 0-0 should appear rarely (only if terminal on first pitch)
        # Many states should be represented
        assert len(counts) >= 5, f"Expected diverse counts, only got: {list(counts.keys())}"

    def test_simulate_count_skipped_when_not_count_aware(self):
        """When _count_aware is False, count should always stay 0-0."""
        model = FakeModel()
        engine = SimulationEngine(pa_model=model, seed=42)
        engine._count_aware = False
        gs = GameState()

        engine._simulate_count(gs)
        assert gs.count_state == "0-0"


# ── Test: Sample PA Fallback Logic ───────────────────────────────────────────


class TestSamplePAFallback:

    def test_flat_cache_lookup(self):
        """Old-style flat cache should work when _count_aware is False."""
        model = FakeModel()
        engine = SimulationEngine(pa_model=model, seed=42)
        engine._count_aware = False

        probs = np.ones(8) / 8
        engine.matchup_cache[(1, 2)] = probs

        outcome = engine._sample_pa(1, 2, None, "home", "0-0")
        assert outcome in engine.outcome_map

    def test_3d_cache_lookup(self):
        """Count-conditional 3D cache should work when _count_aware is True."""
        model = FakeModel()
        engine = SimulationEngine(pa_model=model, seed=42)
        engine._count_aware = True

        probs = np.ones(8) / 8
        engine.matchup_cache[(1, 2, "0-2")] = probs

        outcome = engine._sample_pa(1, 2, None, "home", "0-2")
        assert outcome in engine.outcome_map

    def test_3d_fallback_to_neutral_count(self):
        """Missing count should fall back to 0-0 within 3D cache."""
        model = FakeModel()
        engine = SimulationEngine(pa_model=model, seed=42)
        engine._count_aware = True

        probs = np.ones(8) / 8
        engine.matchup_cache[(1, 2, "0-0")] = probs

        # Request "3-2" which isn't cached, should fallback to "0-0"
        outcome = engine._sample_pa(1, 2, None, "home", "3-2")
        assert outcome in engine.outcome_map

    def test_3d_fallback_to_flat_cache(self):
        """If 3D cache has nothing, should fall back to flat 2D cache."""
        model = FakeModel()
        engine = SimulationEngine(pa_model=model, seed=42)
        engine._count_aware = True

        probs = np.ones(8) / 8
        engine.matchup_cache[(1, 2)] = probs

        outcome = engine._sample_pa(1, 2, None, "home", "1-1")
        assert outcome in engine.outcome_map

    def test_cache_miss_raises(self):
        """Complete cache miss should raise RuntimeError."""
        model = FakeModel()
        engine = SimulationEngine(pa_model=model, seed=42)
        engine._count_aware = False

        with pytest.raises(RuntimeError, match="Failsafe triggered"):
            engine._sample_pa(999, 888, None, "home", "0-0")


# ── Test: Model Count Feature Detection ──────────────────────────────────────


class TestModelHasCountFeatures:

    def test_detects_count_features(self):
        model = FakeModelWithCount()
        engine = SimulationEngine(pa_model=model, seed=42)
        assert engine._model_has_count_features() is True

    def test_detects_no_count_features(self):
        model = FakeModel(feature_names=["pitcher_roll_k_pct", "batter_roll_k_pct_batter"])
        engine = SimulationEngine(pa_model=model, seed=42)
        assert engine._model_has_count_features() is False

    def test_handles_no_feature_names(self):
        model = FakeModel(feature_names=None)
        engine = SimulationEngine(pa_model=model, seed=42)
        assert engine._model_has_count_features() is False
