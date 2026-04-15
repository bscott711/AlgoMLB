from __future__ import annotations

from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

from algomlb.ml.monte_carlo.state import PitcherSimState, GameState, ManagerHookProfile
from algomlb.ml.hook_model import HookModel, compute_leverage_index


class BullpenManager:
    """Manages in-game pitching changes based on leverage and manager profiles.

    Supports two decision modes:

    1. **ML Mode** (preferred): When a trained ``HookModel`` is provided, hook
       decisions are made by sampling from the predicted hook probability.
       This produces stochastic, calibrated substitution patterns consistent
       with each manager's historical tendencies captured in the training data.

    2. **Heuristic Mode** (fallback): When no model is available, the original
       rule-based thresholds (pitch count, runs allowed, TTO) are used. This
       preserves simulation stability even without a trained model artifact.
    """

    def __init__(
        self,
        bullpen_df: pd.DataFrame,
        hook_profiles: Dict[int, ManagerHookProfile],
        hook_model: Optional[HookModel] = None,
        rng: Optional[np.random.Generator] = None,
        starter_ids: Optional[Set[int]] = None,
    ) -> None:
        self.bullpen_df = bullpen_df
        self.hook_profiles = hook_profiles
        self.hook_model = hook_model
        self.rng = rng if rng is not None else np.random.default_rng(42)
        # Track which pitcher IDs are starters (for the is_starter feature)
        self.starter_ids: Set[int] = starter_ids or set()

    # ── Public API ────────────────────────────────────────────────────────────

    def should_hook(
        self, pitcher: PitcherSimState, game: GameState, manager_id: int
    ) -> bool:
        """
        Decide whether the current pitcher should be removed.

        Delegates to the ML model when available; falls back to heuristics
        gracefully so simulation never crashes due to a missing model artifact.
        """
        if self.hook_model is not None:
            features = self._build_hook_feature_vector(pitcher, game)
            hook_prob = float(self.hook_model.predict_proba(features)[0][1])
            return bool(self.rng.random() < hook_prob)
        return self._heuristic_hook(pitcher, game, manager_id)

    def select_arm(self, team_id: int, game: GameState) -> int:
        """Select the best available pitcher from the bullpen."""
        leverage = self._calculate_leverage(game)

        team_pen = self.bullpen_df[self.bullpen_df["team_id"] == team_id]
        if team_pen.empty:
            raise ValueError(f"No bullpen arms found for team_id {team_id}")

        if leverage == "high_lev":
            candidates = team_pen[team_pen["role"].isin(["closer", "setup"])]
        else:
            candidates = team_pen[team_pen["role"].isin(["mid_rel", "long_rel"])]

        if not candidates.empty:
            best_arm = candidates.sort_values(
                by="availability_score", ascending=False
            ).iloc[0]
            return int(best_arm["pitcher_id"])

        # Fallback if preferred roles are exhausted
        return int(team_pen.iloc[0]["pitcher_id"])

    # ── Feature Construction ──────────────────────────────────────────────────

    def _build_hook_feature_vector(
        self,
        pitcher: PitcherSimState,
        game: GameState,
    ) -> pd.DataFrame:
        """
        Build the feature row that the HookModel expects.

        Feature names are locked to ``HookModel.FEATURE_NAMES``. The
        ``leverage_index_at_hook`` is computed in real-time from the current
        base-out-inning-score state instead of using the DB placeholder (1.0).

        The ``score_diff_at_hook`` is sign-flipped so it always represents
        "pitching team's lead" (positive = pitching team is ahead).
        """
        # Score differential from the pitching team's perspective
        if game.top_half:
            # Top half → away team is batting, home team is pitching
            score_diff = game.home_score - game.away_score
        else:
            # Bottom half → home team is batting, away team is pitching
            score_diff = game.away_score - game.home_score

        # Occupied bases bitmask from current GameState
        base_state = sum(
            (1 << i) for i, b in enumerate(game.bases) if b is not None
        )

        li = compute_leverage_index(
            inning=game.inning,
            outs=game.outs,
            base_state=base_state,
            score_diff=score_diff,
        )

        row = {
            "inning": game.inning,
            "outs_at_hook": game.outs,
            "pitches_thrown": pitcher.pitches_thrown,
            "tto_at_hook": pitcher.current_tto,
            "score_diff_at_hook": score_diff,
            "base_state_at_hook": base_state,
            "leverage_index_at_hook": li,
            "runs_allowed": pitcher.runs_allowed,
            "hits_allowed": pitcher.hits_allowed,
            "walks_allowed": pitcher.walks_allowed,
            "strikeouts": pitcher.strikeouts,
            "is_starter": int(pitcher.pitcher_id in self.starter_ids),
        }
        return pd.DataFrame([row])

    # ── Heuristic Fallback ────────────────────────────────────────────────────

    def _heuristic_hook(
        self, pitcher: PitcherSimState, game: GameState, manager_id: int
    ) -> bool:
        """
        Original rule-based hook logic, preserved as the model-free fallback.

        Applied when no trained HookModel artifact is available. Thresholds
        mirror the historical averages captured in ManagerHookProfile.
        """
        profile = self.hook_profiles.get(manager_id)

        # 1. Fatigue Threshold
        limit = profile.avg_sp_pitch_count if profile else 95.0
        if pitcher.pitches_thrown >= limit:
            return True

        # 2. Performance Threshold
        if pitcher.runs_allowed >= 4:
            return True

        # 3. Time Through Order (TTO) / Strategy
        leverage = self._calculate_leverage(game)
        if pitcher.current_tto >= 3:
            # High leverage or quick-hook manager profiles pull early
            if leverage == "high_lev":
                return True
            if profile and profile.pull_before_3rd_tto_pct > 0.3:
                return True

        return False

    def _calculate_leverage(self, game: GameState) -> str:
        """Determine crude game leverage for bullpen tier selection."""
        score_diff = abs(game.home_score - game.away_score)
        if game.inning >= 8 and score_diff <= 2:
            return "high_lev"
        elif game.inning >= 6 and score_diff <= 4:
            return "mid_lev"
        return "low_lev"
