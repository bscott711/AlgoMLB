import numpy as np
from typing import Dict


class PropAggregator:
    """Reduces raw Monte Carlo trial matrices into market-facing prop lines."""

    @staticmethod
    def calculate_over_under(trials_array: np.ndarray, line: float) -> Dict[str, float]:
        """Calculates exact probabilities for Over/Under markets."""
        over_prob = np.mean(trials_array > line)
        under_prob = np.mean(trials_array < line)
        push_prob = np.mean(trials_array == line)

        return {
            "p_over": round(float(over_prob), 4),
            "p_under": round(float(under_prob), 4),
            "p_push": round(float(push_prob), 4),
        }

    @staticmethod
    def summarize_stat(trials_array: np.ndarray) -> Dict[str, float]:
        """Calculates standard distribution metrics."""
        return {
            "mean": round(float(np.mean(trials_array)), 2),
            "median": float(np.median(trials_array)),
            "p25": float(np.percentile(trials_array, 25)),
            "p75": float(np.percentile(trials_array, 75)),
        }
