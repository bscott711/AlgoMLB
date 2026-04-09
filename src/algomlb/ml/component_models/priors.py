import pandas as pd


class PlayerPrior:
    """
    Defines the universal league-average baselines used as priors
    for Bayesian shrinkage to stabilize small sample sizes.
    """

    BASELINES = {
        "k_pct": {"value": 0.225, "denominator_col": "pa"},
        "bb_pct": {"value": 0.085, "denominator_col": "pa"},
        "iso": {"value": 0.155, "denominator_col": "ab"},
        "babip": {"value": 0.295, "denominator_col": "bip"},
    }


class BayesianShrinkage:
    """
    Applies Bayesian shrinkage to regress player-level metrics toward
    the league average based on their sample size (plate appearances, etc.).
    """

    def __init__(self, prior_weight: float = 250.0):
        # A weight of 250 means we add 250 "league average" PA's to the player's line.
        self.prior_weight = prior_weight

    def apply_shrinkage(
        self,
        df: pd.DataFrame,
        target_count_col: str,
        opp_count_col: str,
        metric_name: str,
    ) -> pd.Series:
        """
        Calculates the shrunk rate: (player_count + prior_weight * league_avg) / (total_count + prior_weight)
        """
        if metric_name not in PlayerPrior.BASELINES:
            raise ValueError(
                f"Unknown metric '{metric_name}'. Must be defined in PlayerPrior.BASELINES."
            )

        baseline = PlayerPrior.BASELINES[metric_name]["value"]

        player_count = df[target_count_col]
        total_count = df[opp_count_col]

        shrunk_rate = (player_count + self.prior_weight * baseline) / (
            total_count + self.prior_weight
        )
        return shrunk_rate
