from __future__ import annotations

import logging
from datetime import date
from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RelieverProfile(BaseModel):
    player_id: int
    hand: str = Field(pattern="^(L|R)$")
    role: str = Field(pattern="^(closer|setup|middle|long|loogy)$")
    last_outing_date: Optional[date] = None
    pitches_yesterday: int = 0
    rest_days: int = Field(ge=0, default=0)


class BullpenOrchestrator:
    """Manages reliever availability, role/leverage assignment, and fatigue tracking."""

    ROLE_PRIORITY = {"closer": 5, "setup": 4, "middle": 3, "long": 2, "loogy": 1}

    def __init__(
        self,
        profiles: Sequence[RelieverProfile],
        rng: np.random.Generator,
        platoon_advantage: float = 1.12,
        fatigue_decay: float = 0.003,
        min_fatigue_floor: float = 0.82
    ):
        self.profiles = {p.player_id: p for p in profiles}
        self.rng = rng
        self.usage_today: dict[int, int] = {pid: 0 for pid in self.profiles}
        
        # Configuration for tuning
        self.platoon_advantage = platoon_advantage
        self.platoon_disadvantage = 1.0 / platoon_advantage if platoon_advantage != 0 else 0.88
        self.fatigue_decay = fatigue_decay
        self.min_fatigue_floor = min_fatigue_floor

    def filter_available(
        self, min_rest_days: int = 0, max_pitches_yesterday: int = 30
    ) -> list[int]:
        return [
            pid
            for pid, p in self.profiles.items()
            if p.rest_days >= min_rest_days and p.pitches_yesterday <= max_pitches_yesterday
        ]

    def select_next(
        self,
        available_ids: list[int],
        upcoming_batter_hands: Sequence[str],
        leverage_index: float,
        game_inning: int,
    ) -> Optional[int]:
        if not available_ids:
            logger.warning("No available relievers. Falling back to empty pool.")
            return None

        candidates = available_ids.copy()

        # 1. Leverage/Role Filtering
        if leverage_index >= 1.8:
            high_lev = [pid for pid in candidates if self.profiles[pid].role in ("closer", "setup")]
            if high_lev:
                candidates = high_lev
        elif leverage_index < 0.8 and game_inning < 7:
            long_rel = [pid for pid in candidates if self.profiles[pid].role == "long"]
            if long_rel:
                candidates = long_rel

        # 2. Platoon Matching (Prefer same hand as the upcoming batter)
        if upcoming_batter_hands:
            next_batter_hand = upcoming_batter_hands[0]
            platoon_matches = [
                pid for pid in candidates if self.profiles[pid].hand == next_batter_hand
            ]
            if platoon_matches:
                # If we have a loogy against a lefty, use them
                if next_batter_hand == "L":
                    loogies = [pid for pid in platoon_matches if self.profiles[pid].role == "loogy"]
                    if loogies:
                        return int(self.rng.choice(loogies))
                
                # Otherwise prefer platoon matches if we have multiple candidates
                if len(platoon_matches) >= 1:
                    candidates = platoon_matches

        # 3. Weighted selection based on role priority
        if len(candidates) > 1:
            weights = [self.ROLE_PRIORITY.get(self.profiles[pid].role, 2) for pid in candidates]
            total = sum(weights)
            probs = [w / total for w in weights]
            return int(self.rng.choice(candidates, p=probs))

        return candidates[0] if candidates else None

    def compute_platoon_fatigue_adjustment(
        self,
        pitcher_id: int,
        batter_hand: str,
        pitches_thrown: int,
    ) -> tuple[float, float, float]:
        """Returns (k_adj, bb_adj, hr_adj) multipliers for probability adjustment."""
        prof = self.profiles.get(pitcher_id)
        if not prof:
            return 1.0, 1.0, 1.0

        # Same-hand is advantage for pitcher
        is_advantage = prof.hand == batter_hand
        platoon = self.platoon_advantage if is_advantage else self.platoon_disadvantage
        fatigue = max(self.min_fatigue_floor, 1.0 - (pitches_thrown * self.fatigue_decay))

        k_adj = fatigue * platoon
        # Walks and HRs go up when fatigue is high (fatigue < 1.0) and down with platoon advantage
        bb_adj = (1.0 + (1.0 - fatigue) * 0.6) * (1.1 if not is_advantage else 1.0)
        hr_adj = (1.0 + (1.0 - fatigue) * 0.4) * (1.15 if not is_advantage else 1.0)

        return k_adj, bb_adj, hr_adj

    def record_usage(self, pitcher_id: int, pitches: int) -> None:
        if pitcher_id in self.usage_today:
            self.usage_today[pitcher_id] += pitches
