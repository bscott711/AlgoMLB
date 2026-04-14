from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class BatterSimState(BaseModel):
    """Tracks the mutable, in-game stat accumulation of a batter."""

    player_id: int
    player_name: Optional[str] = None
    pa_count: int = 0
    hits: int = 0
    singles: int = 0
    doubles: int = 0
    triples: int = 0
    hr: int = 0
    rbi: int = 0
    runs: int = 0
    walks: int = 0
    hbp: int = 0
    strikeouts: int = 0
    total_bases: int = 0

    @property
    def hrr(self) -> int:
        """Hits + Runs + RBIs composite stat."""
        return self.hits + self.runs + self.rbi


class PitcherSimState(BaseModel):
    """Tracks the mutable, in-game stat accumulation and fatigue of a pitcher."""

    pitcher_id: int
    player_name: Optional[str] = None
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


class ManagerHookProfile(BaseModel):
    """Aggregated manager hook tendencies for a season."""

    manager_id: int
    manager_name: str
    avg_sp_pitch_count: float = 90.0
    pull_before_3rd_tto_pct: float = 0.1
    pull_with_lead_pct: float = 0.5
    bullpen_protective_pct: float = 0.2


class GameState(BaseModel):
    """Tracks the atomic base-out-score state of the game with identity-based runner tracking."""

    inning: int = 1
    top_half: bool = True
    outs: int = 0
    home_score: int = 0
    away_score: int = 0
    # Represents [1st Base, 2nd Base, 3rd Base]. Stores player_id if occupied, else None.
    bases: List[Optional[int]] = Field(default_factory=lambda: [None, None, None])

    def clear_bases(self):
        self.bases = [None, None, None]

    def process_event(self, event: str, batter_id: int) -> List[int]:
        """
        Advances runners based on the canonical PA outcome.
        Returns a list of player_ids who scored on the play.
        """
        scored_ids = []

        if event in ["strikeout", "out_in_play"]:
            self.outs += 1
        elif event in ["walk", "hbp"]:
            scored_ids = self._handle_walk_hbp(batter_id)
        else:
            scored_ids = self._handle_hit(event, batter_id)

        runs = len(scored_ids)
        if self.top_half:
            self.away_score += runs
        else:
            self.home_score += runs

        return scored_ids

    def _handle_walk_hbp(self, batter_id: int) -> List[int]:
        """Logic for forced advancement (Walk/HBP)."""
        scored_ids = []
        if self.bases[0] is not None:
            if self.bases[1] is not None:
                if self.bases[2] is not None:
                    scored_ids.append(self.bases[2])
                self.bases[2] = self.bases[1]
            self.bases[1] = self.bases[0]
        self.bases[0] = batter_id
        return scored_ids

    def _handle_hit(self, event: str, batter_id: int) -> List[int]:
        """Logic for hits with identity-based advancement."""
        scored_ids = []

        if event == "home_run":
            # Everyone scores including the batter
            scored_ids = [p for p in self.bases if p is not None]
            scored_ids.append(batter_id)
            self.clear_bases()

        elif event == "triple":
            # Everyone scores
            scored_ids = [p for p in self.bases if p is not None]
            self.bases = [None, None, batter_id]

        elif event == "double":
            # Runner on 2nd and 3rd score
            if self.bases[2] is not None:
                scored_ids.append(self.bases[2])
            if self.bases[1] is not None:
                scored_ids.append(self.bases[1])
            # Runner on 1st goes to 3rd
            self.bases[2] = self.bases[0]
            self.bases[1] = batter_id
            self.bases[0] = None

        elif event == "single":
            # Runner on 3rd scores
            if self.bases[2] is not None:
                scored_ids.append(self.bases[2])
            # Runner on 2nd scores (conservative sim: single scores from 2nd)
            if self.bases[1] is not None:
                scored_ids.append(self.bases[1])
            # Runner on 1st goes to 2nd
            self.bases[1] = self.bases[0]
            self.bases[0] = batter_id
            self.bases[2] = None

        return scored_ids


BatterSimState.model_rebuild()
PitcherSimState.model_rebuild()
ManagerHookProfile.model_rebuild()
GameState.model_rebuild()
