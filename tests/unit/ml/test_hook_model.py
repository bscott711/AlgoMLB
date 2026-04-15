"""
tests/unit/ml/test_hook_model.py

Contract tests for the HookModel integration. These tests lock:
  1. FEATURE_NAMES schema consistency between HookModel and BullpenManager
  2. Model fit/predict shape and probability validity
  3. Save/load round-trip fidelity
  4. BullpenManager ML-mode routing vs heuristic fallback
  5. Feature alignment tolerance (reindex handles missing/extra columns)
  6. Leverage index computation correctness
  7. SimulationEngine hook model injection (no crash)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from algomlb.ml.hook_model import HookModel, compute_leverage_index
from algomlb.ml.monte_carlo.bullpen import BullpenManager
from algomlb.ml.monte_carlo.state import (
    BatterSimState,
    GameState,
    ManagerHookProfile,
    PitcherSimState,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_X() -> pd.DataFrame:
    """Synthetic feature DataFrame with all FEATURE_NAMES present."""
    return pd.DataFrame(
        [
            {
                "inning": 5,
                "outs_at_hook": 1,
                "pitches_thrown": 85,
                "tto_at_hook": 2,
                "score_diff_at_hook": 1,
                "base_state_at_hook": 0,
                "leverage_index_at_hook": 0.82,
                "runs_allowed": 2,
                "hits_allowed": 5,
                "walks_allowed": 1,
                "strikeouts": 4,
                "is_starter": 1,
            },
            {
                "inning": 7,
                "outs_at_hook": 2,
                "pitches_thrown": 100,
                "tto_at_hook": 3,
                "score_diff_at_hook": -1,
                "base_state_at_hook": 3,
                "leverage_index_at_hook": 2.04,
                "runs_allowed": 4,
                "hits_allowed": 8,
                "walks_allowed": 3,
                "strikeouts": 6,
                "is_starter": 1,
            },
            {
                "inning": 8,
                "outs_at_hook": 0,
                "pitches_thrown": 30,
                "tto_at_hook": 1,
                "score_diff_at_hook": 2,
                "base_state_at_hook": 7,
                "leverage_index_at_hook": 1.96,
                "runs_allowed": 0,
                "hits_allowed": 1,
                "walks_allowed": 0,
                "strikeouts": 2,
                "is_starter": 0,
            },
        ]
    )


@pytest.fixture
def minimal_y() -> pd.Series:
    """Labels: first two pitchers were hooked, third completed the game."""
    return pd.Series([1, 1, 0], name="was_hooked")


@pytest.fixture
def trained_model(minimal_X, minimal_y) -> HookModel:
    """A minimally-trained HookModel for quick inference tests."""
    model = HookModel()
    # Use cv=2 for the tiny fixture dataset; calibrate=False avoids
    # CalibratedClassifierCV requiring multi-class splits on 3 samples.
    model.fit(minimal_X, minimal_y, calibrate=False)
    return model


@pytest.fixture
def dummy_profile() -> ManagerHookProfile:
    return ManagerHookProfile(
        manager_id=1,
        manager_name="Test Manager",
        avg_sp_pitch_count=90.0,
        pull_before_3rd_tto_pct=0.3,
    )


@pytest.fixture
def dummy_game() -> GameState:
    game = GameState()
    game.inning = 6
    game.outs = 1
    game.top_half = True
    game.home_score = 2
    game.away_score = 1
    return game


@pytest.fixture
def dummy_pitcher() -> PitcherSimState:
    return PitcherSimState(
        pitcher_id=999,
        pitches_thrown=88,
        current_tto=2,
        runs_allowed=2,
        hits_allowed=5,
        walks_allowed=1,
    )


@pytest.fixture
def dummy_bullpen_df() -> pd.DataFrame:
    return pd.DataFrame(
        [{"team_id": 0, "pitcher_id": 42, "availability_score": 1.0, "role": "mid_rel"}]
    )


# ── Test 1: Feature schema contract ──────────────────────────────────────────


def test_hook_model_feature_schema_locked():
    """FEATURE_NAMES must be a non-empty list of strings with no duplicates."""
    names = HookModel.FEATURE_NAMES
    assert isinstance(names, list)
    assert len(names) > 0
    assert all(isinstance(n, str) for n in names)
    assert len(names) == len(set(names)), "FEATURE_NAMES must not have duplicates"


def test_bullpen_feature_vector_matches_feature_names(
    dummy_bullpen_df, dummy_profile, dummy_game, dummy_pitcher
):
    """
    The columns produced by _build_hook_feature_vector must be a superset
    of HookModel.FEATURE_NAMES so that reindex() can align without ambiguity.
    """
    bm = BullpenManager(
        dummy_bullpen_df,
        {1: dummy_profile},
        starter_ids={999},
    )
    fv = bm._build_hook_feature_vector(dummy_pitcher, dummy_game)
    assert isinstance(fv, pd.DataFrame)
    assert len(fv) == 1

    missing = set(HookModel.FEATURE_NAMES) - set(fv.columns)
    assert missing == set(), (
        f"BullpenManager._build_hook_feature_vector is missing columns "
        f"required by HookModel.FEATURE_NAMES: {missing}"
    )


# ── Test 2: Fit and predict shape ─────────────────────────────────────────────


def test_hook_model_fit_predict_shape(trained_model, minimal_X):
    """predict_proba must return (n_samples, 2) with probabilities summing to 1."""
    proba = trained_model.predict_proba(minimal_X)
    assert proba.shape == (len(minimal_X), 2), f"Unexpected shape: {proba.shape}"
    np.testing.assert_allclose(
        proba.sum(axis=1),
        np.ones(len(minimal_X)),
        atol=1e-5,
        err_msg="Probabilities must sum to 1 per row.",
    )
    assert np.all(proba >= 0.0), "Probabilities must be non-negative."
    assert np.all(proba <= 1.0), "Probabilities must not exceed 1.0."


# ── Test 3: Save / load round-trip ───────────────────────────────────────────


def test_hook_model_save_load_roundtrip(trained_model, minimal_X):
    """Saved and re-loaded model must produce identical predict_proba output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "hook_model_test.joblib"
        trained_model.save(path)
        assert path.exists()

        loaded = HookModel.load(path)
        original_proba = trained_model.predict_proba(minimal_X)
        loaded_proba = loaded.predict_proba(minimal_X)
        np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-6)


def test_hook_model_load_schema_mismatch_warns(trained_model):
    """Loading a bundle whose feature_names differ from FEATURE_NAMES emits UserWarning."""
    import joblib

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "hook_stale.joblib"
        bundle = {
            "clf": trained_model.clf,
            "calibrated_clf": trained_model.calibrated_clf,
            "feature_names": ["old_feature_1", "old_feature_2"],  # deliberately wrong
        }
        joblib.dump(bundle, path)

        with pytest.warns(UserWarning, match="feature schema differs"):
            HookModel.load(path)


# ── Test 4: BullpenManager ML-mode routing ───────────────────────────────────


def test_bullpen_uses_hook_model_when_provided(
    dummy_bullpen_df, dummy_profile, dummy_game, dummy_pitcher
):
    """When hook_model is set, should_hook must call predict_proba, not heuristics."""
    mock_model = MagicMock(spec=HookModel)
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])  # 90% hook prob

    bm = BullpenManager(
        dummy_bullpen_df,
        {1: dummy_profile},
        hook_model=mock_model,
        rng=np.random.default_rng(0),  # seed 0 → first random() ≈ 0.637 < 0.9 → True
        starter_ids={999},
    )
    # Seed 0 for default_rng gives deterministic result; 0.637 < 0.9 → should hook
    result = bm.should_hook(dummy_pitcher, dummy_game, manager_id=1)
    mock_model.predict_proba.assert_called_once()
    assert isinstance(result, bool)


def test_bullpen_hook_prob_zero_never_hooks(
    dummy_bullpen_df, dummy_profile, dummy_game, dummy_pitcher
):
    """When model returns P(hook)=0, should_hook must always return False."""
    mock_model = MagicMock(spec=HookModel)
    mock_model.predict_proba.return_value = np.array([[1.0, 0.0]])  # 0% hook

    bm = BullpenManager(
        dummy_bullpen_df,
        {1: dummy_profile},
        hook_model=mock_model,
        rng=np.random.default_rng(99),
        starter_ids={999},
    )
    for _ in range(20):
        assert bm.should_hook(dummy_pitcher, dummy_game, manager_id=1) is False


# ── Test 5: Heuristic fallback when model is absent ──────────────────────────


def test_bullpen_fallback_to_heuristics_when_no_model(
    dummy_bullpen_df, dummy_profile, dummy_game, dummy_pitcher
):
    """With hook_model=None, should_hook uses the original rule-based heuristics."""
    bm = BullpenManager(dummy_bullpen_df, {1: dummy_profile}, hook_model=None)
    # dummy_pitcher has 88 pitches < 90 avg_sp_pitch_count; 2 runs < 4 threshold;
    # tto=2 < 3 → heuristic should NOT hook
    assert bm.should_hook(dummy_pitcher, dummy_game, manager_id=1) is False


def test_heuristic_hooks_on_pitch_count_exceeded(dummy_bullpen_df, dummy_profile):
    """Heuristic must hook when pitches_thrown >= avg_sp_pitch_count."""
    game = GameState()
    game.inning = 5
    pitcher = PitcherSimState(pitcher_id=1, pitches_thrown=95, runs_allowed=0)
    bm = BullpenManager(dummy_bullpen_df, {1: dummy_profile}, hook_model=None)
    # avg_sp_pitch_count=90; 95 >= 90 → True
    assert bm.should_hook(pitcher, game, manager_id=1) is True


# ── Test 6: Feature alignment tolerance ──────────────────────────────────────


def test_hook_model_reindex_fills_missing_columns(trained_model):
    """predict_proba must succeed even when input has missing/extra columns."""
    # Only provide 3 of the required 8 features; rest default to 0.0 via reindex
    partial_X = pd.DataFrame([{"inning": 6, "pitches_thrown": 90, "is_starter": 1}])
    proba = trained_model.predict_proba(partial_X)
    assert proba.shape == (1, 2)
    np.testing.assert_allclose(proba.sum(axis=1), [1.0], atol=1e-5)


def test_hook_model_reindex_drops_extra_columns(trained_model):
    """predict_proba must handle extra columns without error."""
    extra_X = pd.DataFrame(
        [
            {
                "inning": 5,
                "outs_at_hook": 0,
                "pitches_thrown": 70,
                "tto_at_hook": 2,
                "score_diff_at_hook": 0,
                "base_state_at_hook": 1,
                "leverage_index_at_hook": 1.2,
                "is_starter": 1,
                "extra_column_that_does_not_exist": 999.0,
            }
        ]
    )
    proba = trained_model.predict_proba(extra_X)
    assert proba.shape == (1, 2)


# ── Test 7: Leverage Index computation ───────────────────────────────────────


@pytest.mark.parametrize(
    "inning, outs, base_state, score_diff, expected_range",
    [
        (1, 0, 0, 0, (0.30, 0.80)),   # Early, bases empty, tied → low-mid LI
        (9, 2, 7, 0, (1.50, 5.00)),   # 9th, 2 outs, bases loaded, tied → very high LI
        (5, 1, 0, 5, (0.05, 0.30)),   # Mid-game, blowout → very low LI
        (8, 0, 3, 1, (0.80, 2.50)),   # Late, 1st+2nd, 1-run lead → high LI
    ],
)
def test_leverage_index_range(inning, outs, base_state, score_diff, expected_range):
    """compute_leverage_index must return values in domain-plausible ranges."""
    li = compute_leverage_index(inning, outs, base_state, score_diff)
    lo, hi = expected_range
    assert lo <= li <= hi, (
        f"LI={li:.3f} out of expected range [{lo}, {hi}] "
        f"for (inning={inning}, outs={outs}, base={base_state}, diff={score_diff})"
    )


def test_leverage_index_increases_with_closeness():
    """Tied game should give strictly higher LI than blowout, same base-out state."""
    li_tied = compute_leverage_index(inning=8, outs=1, base_state=1, score_diff=0)
    li_blowout = compute_leverage_index(inning=8, outs=1, base_state=1, score_diff=6)
    assert li_tied > li_blowout


def test_leverage_index_increases_with_inning():
    """LI in inning 9 should exceed LI in inning 1 (same base-out-score state)."""
    li_late = compute_leverage_index(inning=9, outs=1, base_state=1, score_diff=0)
    li_early = compute_leverage_index(inning=1, outs=1, base_state=1, score_diff=0)
    assert li_late > li_early


def test_leverage_index_extra_innings_clamped():
    """Extra innings (>9) must map to the inning-9 scale without KeyError."""
    li_10 = compute_leverage_index(inning=10, outs=0, base_state=0, score_diff=0)
    li_9 = compute_leverage_index(inning=9, outs=0, base_state=0, score_diff=0)
    assert li_10 == li_9


# ── Test 8: SimulationEngine hook model injection ────────────────────────────


def test_simulation_engine_accepts_hook_model():
    """SimulationEngine must accept hook_model kwarg without error."""
    from algomlb.ml.monte_carlo.engine import SimulationEngine

    mock_hook = MagicMock(spec=HookModel)
    engine = SimulationEngine(pa_model=None, hook_model=mock_hook, seed=42)
    assert engine.hook_model is mock_hook


def test_simulation_engine_hook_model_flows_to_bullpen(trained_model):
    """
    _setup_bullpen_manager must pass the hook_model to BullpenManager.
    Validates the dependency injection chain is complete.
    """
    from algomlb.ml.monte_carlo.engine import SimulationEngine
    from algomlb.ml.monte_carlo.loader import MatchupContext
    import datetime

    engine = SimulationEngine(pa_model=None, hook_model=trained_model, seed=0)

    # Build a minimal context just for _setup_bullpen_manager
    p_home = PitcherSimState(pitcher_id=1)
    p_away = PitcherSimState(pitcher_id=2)
    ctx = MatchupContext(
        game_pk=1,
        game_date=datetime.date(2025, 4, 1),
        home_lineup=[BatterSimState(player_id=i) for i in range(9)],
        away_lineup=[BatterSimState(player_id=i + 100) for i in range(9)],
        home_starter=p_home,
        away_starter=p_away,
        batter_features={},
        pitcher_features={},
        home_relievers=[],
        away_relievers=[],
        manager_profiles={},
        game_context={"temp": 70.0, "wind_speed": 5.0, "is_night": 1.0},
        matchup_features={},
    )

    bm = engine._setup_bullpen_manager(ctx)
    assert bm.hook_model is trained_model
    assert {1, 2} == bm.starter_ids
    assert bm.rng is not None
