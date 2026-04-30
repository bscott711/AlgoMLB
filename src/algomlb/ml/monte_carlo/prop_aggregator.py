import numpy as np
import pandas as pd
from typing import List, Dict, Any
from algomlb.ml.monte_carlo.state import SimulationResult, BatterSimState, PitcherSimState

class PropAggregator:
    """Calculates granular prop probabilities and distributions from Monte Carlo trials."""

    def __init__(self, trials: List[SimulationResult]):
        self.trials = trials
        self.n_trials = len(trials)

    def aggregate_player_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Collapses raw trial data into a per-player summary of distributions.
        Returns: {player_id: {stat_type: [values_across_trials]}}
        """
        player_data: Dict[int, Dict[str, List[float]]] = {}

        for trial in self.trials:
            for p_id_raw, state in trial.player_states.items():
                p_id = int(p_id_raw)
                if p_id not in player_data:
                    player_data[p_id] = self._init_stat_dict()
                
                pd_dict = player_data[p_id]
                if isinstance(state, BatterSimState):
                    pd_dict["H"].append(float(state.hits))
                    pd_dict["HR"].append(float(state.hr))
                    pd_dict["TB"].append(float(state.total_bases))
                    pd_dict["RBI"].append(float(state.rbi))
                    pd_dict["R"].append(float(state.runs))
                    pd_dict["K"].append(float(state.strikeouts))
                elif isinstance(state, PitcherSimState):
                    pd_dict["K_p"].append(float(state.strikeouts))
                    pd_dict["BB_p"].append(float(state.walks_allowed))
                    pd_dict["H_p"].append(float(state.hits_allowed))
                    pd_dict["Outs"].append(float(state.outs_recorded))
                elif isinstance(state, dict):
                    if "hits" in state: # Batter
                        pd_dict["H"].append(float(state.get("hits", 0)))
                        pd_dict["HR"].append(float(state.get("hr", 0)))
                        pd_dict["TB"].append(float(state.get("total_bases", 0)))
                        pd_dict["RBI"].append(float(state.get("rbi", 0)))
                        pd_dict["R"].append(float(state.get("runs", 0)))
                        pd_dict["K"].append(float(state.get("strikeouts", 0)))
                    elif "walks_allowed" in state: # Pitcher
                        pd_dict["K_p"].append(float(state.get("strikeouts", 0)))
                        pd_dict["BB_p"].append(float(state.get("walks_allowed", 0)))
                        pd_dict["H_p"].append(float(state.get("hits_allowed", 0)))
                        pd_dict["Outs"].append(float(state.get("outs_recorded", 0)))

        return player_data

    def calculate_prop_probabilities(self, player_stats: Dict[int, Dict[str, List[float]]]) -> List[Dict[str, Any]]:
        """
        Calculates mean, median, and O/U probabilities for each player/stat pair.
        """
        records = []
        for p_id_raw, stats in player_stats.items():
            p_id = int(p_id_raw)
            for stat_type, values in stats.items():
                if not values:
                    continue
                
                arr = np.array(values)
                record = {
                    "player_id": p_id,
                    "stat_type": stat_type,
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "p10": float(np.percentile(arr, 10)),
                    "p90": float(np.percentile(arr, 90)),
                    "prob_over_0_5": float(np.mean(arr > 0.5)),
                    "prob_over_1_5": float(np.mean(arr > 1.5)),
                    "prob_over_2_5": float(np.mean(arr > 2.5)),
                    "prob_over_3_5": float(np.mean(arr > 3.5)),
                    "prob_over_4_5": float(np.mean(arr > 4.5)),
                    "trials": self.n_trials
                }
                records.append(record)
        
        return records

    def _init_stat_dict(self) -> Dict[str, List[float]]:
        return {
            "H": [], "HR": [], "TB": [], "RBI": [], "R": [], "K": [],
            "K_p": [], "BB_p": [], "H_p": [], "Outs": []
        }
