from typing import Dict


class MarketAlignment:
    """Bridges internal AlgoMLB models with external sportsbook data structures."""

    # Simple hardcoded alias map (In production, this queries the DB alias table)
    PLAYER_ALIASES = {
        "shohei ohtani": 660271,
        "ronald acuna jr.": 660670,
        "ronald acuna jr": 660670,
        "matt olson": 621566,
    }

    MARKET_MAP = {
        "pitcher_strikeouts": "pitcher_strikeouts_over_under",
        "batter_hits": "batter_hits_over_under",
        "batter_home_runs": "batter_home_runs_over_under",
    }

    @classmethod
    def resolve_player_id(cls, sportsbook_name: str) -> int:
        """Maps a sportsbook string name to the canonical MLB player ID."""
        clean_name = sportsbook_name.strip().lower()
        if clean_name not in cls.PLAYER_ALIASES:
            raise ValueError(f"Player alias '{sportsbook_name}' not found in registry.")
        return cls.PLAYER_ALIASES[clean_name]

    @staticmethod
    def american_to_implied(american_odds: int) -> float:
        """Converts American odds (e.g., -110, +150) to implied probability."""
        if american_odds < 0:
            return -american_odds / (-american_odds + 100)
        else:
            return 100 / (american_odds + 100)

    @staticmethod
    def calculate_kelly_stake(
        calibrated_prob: float, american_odds: int, kelly_multiplier: float = 0.25
    ) -> float:
        """
        Calculates the recommended bankroll percentage to risk using the Kelly Criterion.
        Defaults to Quarter-Kelly (0.25) for bankroll safety.
        """
        if american_odds < 0:
            decimal_odds = (100 / -american_odds) + 1
        else:
            decimal_odds = (american_odds / 100) + 1

        b = decimal_odds - 1
        p = calibrated_prob
        q = 1.0 - p

        kelly_fraction = (b * p - q) / b

        # Never recommend betting if edge is negative
        return max(0.0, round(kelly_fraction * kelly_multiplier, 4))

    @classmethod
    def evaluate_edge(
        cls, calibrated_p_over: float, american_odds: int
    ) -> Dict[str, float]:
        """Calculates absolute edge and recommended unit sizing."""
        implied_prob = cls.american_to_implied(american_odds)
        edge = calibrated_p_over - implied_prob
        units = cls.calculate_kelly_stake(calibrated_p_over, american_odds)

        return {
            "implied_prob": round(implied_prob, 4),
            "edge": round(edge, 4),
            "recommended_units": units,
        }
