from pydantic import BaseModel, Field
from typing import List, Dict


class PitcherState(BaseModel):
    """Tracks the mutable, in-game state of the active pitcher."""

    pitcher_id: int
    pitches_thrown: int = 0
    current_tto: int = 1
    runs_allowed: int = 0
    earned_runs_allowed: int = 0
    hits_allowed: int = 0
    walks_allowed: int = 0
    strikeouts: int = 0
    outs_recorded: int = 0
    fatigue_multipliers: Dict[str, float] = Field(default_factory=dict)
    manager_hook_prob: float = 0.0


class GameState(BaseModel):
    """Tracks the atomic base-out-score state of the game."""

    inning: int = 1
    top_half: bool = True
    outs: int = 0
    home_score: int = 0
    away_score: int = 0
    # Represents [1st Base, 2nd Base, 3rd Base]. True if occupied.
    bases: List[bool] = Field(default_factory=lambda: [False, False, False])

    def clear_bases(self):
        self.bases = [False, False, False]

    def process_event(self, event: str) -> int:
        """
        Advances runners based on the canonical PA outcome.
        Returns the number of runs scored on the play.
        """
        runs = 0
        if event in ["strikeout", "out_in_play"]:
            self.outs += 1
        elif event in ["walk", "hbp"]:
            runs = self._handle_walk_hbp()
        else:
            runs = self._handle_hit(event)

        if self.top_half:
            self.away_score += runs
        else:
            self.home_score += runs

        return runs

    def _handle_walk_hbp(self) -> int:
        """Logic for forced advancement (Walk/HBP)."""
        runs = 0
        if self.bases[0]:
            if self.bases[1]:
                if self.bases[2]:
                    runs = 1
                self.bases[2] = True
            self.bases[1] = True
        self.bases[0] = True
        return runs

    def _handle_hit(self, event: str) -> int:
        """Logic for home runs, triples, doubles, and singles."""
        runs = 0
        if event == "home_run":
            runs = sum(self.bases) + 1
            self.clear_bases()
        elif event == "triple":
            runs = sum(self.bases)
            self.bases = [False, False, True]
        elif event == "double":
            runs = (1 if self.bases[1] else 0) + (1 if self.bases[2] else 0)
            self.bases[2] = self.bases[0]  # Runner on 1st goes to 3rd
            self.bases[1] = True
            self.bases[0] = False
        elif event == "single":
            runs = 1 if self.bases[2] else 0
            self.bases[2] = self.bases[1]
            self.bases[1] = self.bases[0]
            self.bases[0] = True
        return runs
