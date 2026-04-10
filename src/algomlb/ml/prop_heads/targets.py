import pandas as pd
from abc import ABC, abstractmethod


class BasePropHead(ABC):
    """
    Abstract base class for all prop-specific target definitions.
    Defines the contract for creating training labels from historical data.
    """

    @property
    @abstractmethod
    def internal_market_key(self) -> str:
        """The internal identifier for the prop market (e.g., 'pitcher_strikeouts')."""
        pass

    def generate_labels(self, actuals: pd.Series, lines: pd.Series) -> pd.Series:
        """
        Generates binary calibration targets (y_cal).
        1 if the actual outcome went strictly OVER the line.
        0 if the outcome went UNDER or PUSHED.
        """
        return (actuals > lines).astype(int)


class PitcherPropHead(BasePropHead):
    """Calibrates Pitcher Props like Strikeouts, Outs Recorded, Earned Runs."""

    def __init__(self, market_key: str):
        self._market_key = market_key

    @property
    def internal_market_key(self) -> str:
        return self._market_key


class BatterPropHead(BasePropHead):
    """Calibrates Batter Props like Hits, Total Bases, Home Runs."""

    def __init__(self, market_key: str):
        self._market_key = market_key

    @property
    def internal_market_key(self) -> str:
        return self._market_key


class GamePropHead(BasePropHead):
    """Calibrates Game/Team Props like Full Game ML, First 5 Innings, Totals."""

    def __init__(self, market_key: str):
        self._market_key = market_key

    @property
    def internal_market_key(self) -> str:
        return self._market_key
