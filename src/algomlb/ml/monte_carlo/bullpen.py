import pandas as pd
from algomlb.ml.monte_carlo.state import PitcherSimState, GameState


class BullpenManager:
    """Manages in-game pitching changes based on leverage and manager profiles."""

    def __init__(self, bullpen_df: pd.DataFrame, hook_profiles: pd.DataFrame):
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
        # Standard safety rails (in production, this checks the hook_profiles matrix)
        if pitcher.pitches_thrown >= 100:
            return True
        if pitcher.runs_allowed >= 5:
            return True
        if pitcher.current_tto >= 3 and self._calculate_leverage(game) == "high_lev":
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
