import pandas as pd
from algomlb.ml.monte_carlo.state import PitcherSimState, GameState, ManagerHookProfile
from typing import Dict


class BullpenManager:
    """Manages in-game pitching changes based on leverage and manager profiles."""

    def __init__(self, bullpen_df: pd.DataFrame, hook_profiles: Dict[int, ManagerHookProfile]):
        self.bullpen_df = bullpen_df
        self.hook_profiles = hook_profiles

    def _calculate_leverage(self, game: GameState) -> str:
        """Determines crude game leverage for bullpen tier selection."""
        score_diff = abs(game.home_score - game.away_score)
        if game.inning >= 8 and score_diff <= 2:
            return "high_lev"
        elif game.inning >= 6 and score_diff <= 4:
            return "mid_lev"
        return "low_lev"

    def should_hook(
        self, pitcher: PitcherSimState, game: GameState, manager_id: int
    ) -> bool:
        """Evaluates if the current pitcher should be removed."""
        profile = self.hook_profiles.get(manager_id)
        
        # 1. Fatigue Threshold
        limit = profile.avg_sp_pitch_count if profile else 95.0
        if pitcher.pitches_thrown >= limit:
            return True
            
        # 2. Performance Threshold
        if pitcher.runs_allowed >= 4: # slightly lower more realistic threshold
            return True
            
        # 3. Time Through Order (TTO) / Strategy
        leverage = self._calculate_leverage(game)
        if pitcher.current_tto >= 3:
            # High leverage or "Quick Hook" profiles pull early
            if leverage == "high_lev":
                return True
            if profile and profile.pull_before_3rd_tto_pct > 0.3:
                return True
                
        return False

    def select_arm(self, team_id: int, game: GameState) -> int:
        """Selects the best available pitcher from the bullpen."""
        leverage = self._calculate_leverage(game)

        # Filter for the team
        team_pen = self.bullpen_df[self.bullpen_df["team_id"] == team_id]

        if team_pen.empty:
            raise ValueError(f"No bullpen arms found for team_id {team_id}")

        if leverage == "high_lev":
            # Attempt to find closer or setup
            candidates = team_pen[team_pen["role"].isin(["closer", "setup"])]
        else:
            # Default to mid or long relief
            candidates = team_pen[team_pen["role"].isin(["mid_rel", "long_rel"])]

        if not candidates.empty:
            # Return the freshest arm (highest availability score)
            best_arm = candidates.sort_values(
                by="availability_score", ascending=False
            ).iloc[0]
            return int(best_arm["pitcher_id"])

        # Fallback if preferred roles are exhausted
        return int(team_pen.iloc[0]["pitcher_id"])
