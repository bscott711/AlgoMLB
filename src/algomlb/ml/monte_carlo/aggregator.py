import pandas as pd
from typing import List, TYPE_CHECKING
from algomlb.ml.monte_carlo.state import SimulationResult

if TYPE_CHECKING:
    from algomlb.ml.monte_carlo.loader import MatchupContext


class SimulationAggregator:
    """Reduces raw Monte Carlo trial dicts into a structured results DataFrame."""

    def aggregate_results(
        self,
        game_pk: int,
        season: int,
        trial_results: List[SimulationResult],
        context: "MatchupContext",
    ) -> pd.DataFrame:
        """
        Collapses N trials into a single summary DataFrame ready for DB insertion.
        Includes Batter props, Pitcher props, and Team Win/Loss probabilities.
        """
        # 1. Delegate Prop Calculation to specialized PropAggregator
        import importlib
        from algomlb.ml.monte_carlo import prop_aggregator
        importlib.reload(prop_aggregator)
        from algomlb.ml.monte_carlo.prop_aggregator import PropAggregator
        
        prop_engine = PropAggregator(trial_results)
        player_stats = prop_engine.aggregate_player_stats()
        records = prop_engine.calculate_prop_probabilities(player_stats)

        # 2. Add Team Win Probability
        n_trials = len(trial_results)
        team_wins = {"home": 0, "away": 0}
        
        for trial in trial_results:
            if trial.home_score > trial.away_score:
                team_wins["home"] += 1
            elif trial.away_score > trial.home_score:
                team_wins["away"] += 1

        if n_trials > 0:
            for side, p_idx in [("home", 1), ("away", 0)]:
                prob = team_wins[side] / n_trials
                records.append({
                    "game_pk": game_pk,
                    "season": season,
                    "player_id": p_idx,
                    "stat_type": "WIN",
                    "mean": prob,
                    "median": 1.0 if prob > 0.5 else 0.0,
                    "p10": 0.0,
                    "p90": 1.0,
                    "prob_over_0_5": prob,
                    "prob_over_1_5": 0.0,
                    "prob_over_2_5": 0.0,
                    "prob_over_3_5": 0.0,
                    "prob_over_4_5": 0.0,
                    "trials": n_trials,
                })

        # Add game_pk/season back to player records
        for r in records:
            if "game_pk" not in r:
                r["game_pk"] = game_pk
                r["season"] = season

        return pd.DataFrame(records)
