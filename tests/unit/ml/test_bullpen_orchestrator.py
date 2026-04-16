import numpy as np
import pytest
from datetime import date
from algomlb.ml.monte_carlo.bullpen_orchestrator import BullpenOrchestrator, RelieverProfile

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def profiles():
    return [
        RelieverProfile(player_id=1, hand="R", role="closer", rest_days=2, pitches_yesterday=0),
        RelieverProfile(player_id=2, hand="L", role="setup", rest_days=1, pitches_yesterday=15),
        RelieverProfile(player_id=3, hand="R", role="middle", rest_days=0, pitches_yesterday=45),
        RelieverProfile(player_id=4, hand="L", role="loogy", rest_days=1, pitches_yesterday=10),
        RelieverProfile(player_id=5, hand="R", role="long", rest_days=3, pitches_yesterday=0),
    ]

def test_filter_available(profiles, rng):
    orch = BullpenOrchestrator(profiles, rng)
    
    # Standard constraints
    available = orch.filter_available(min_rest_days=1, max_pitches_yesterday=20)
    assert 1 in available  # Closer (2 days rest, 0 yesterday)
    assert 2 in available  # Setup (1 day rest, 15 yesterday)
    assert 4 in available  # Loogy (1 day rest, 10 yesterday)
    assert 5 in available  # Long (3 days rest, 0 yesterday)
    assert 3 not in available # Middle (0 rest, 45 yesterday)

def test_select_next_high_leverage(profiles, rng):
    orch = BullpenOrchestrator(profiles, rng)
    available = [1, 2, 3, 4, 5]
    
    # High leverage (LI=2.0) should prefer closer/setup
    selected = orch.select_next(available, upcoming_batter_hands=["R"], leverage_index=2.0, game_inning=9)
    assert selected in [1, 2] # Role closer or setup

def test_select_next_platoon_match(profiles, rng):
    orch = BullpenOrchestrator(profiles, rng)
    available = [2, 4, 5] # L, L, R
    
    # Upcoming batter is L, should prefer L pitchers (2, 4)
    # Particularly loogy (4) for L batter
    selected = orch.select_next(available, upcoming_batter_hands=["L"], leverage_index=1.0, game_inning=7)
    assert selected == 4 # Loogy selected for L match if available

def test_compute_adjustments(profiles, rng):
    orch = BullpenOrchestrator(profiles, rng)
    
    # Test Pitcher 1 (RHP) vs RHB (Advantage)
    k, bb, hr = orch.compute_platoon_fatigue_adjustment(1, "R", 0)
    assert k > 1.0  # Platoon advantage
    assert bb == 1.0 # No fatigue inflation yet
    assert hr == 1.0
    
    # Test Pitcher 1 (RHP) vs LHB (Disadvantage)
    k, bb, hr = orch.compute_platoon_fatigue_adjustment(1, "L", 0)
    assert k < 1.0 # Platoon disadvantage
    assert bb > 1.0
    assert hr > 1.0
    
    # Test Fatigue (After 50 pitches)
    k_f, bb_f, hr_f = orch.compute_platoon_fatigue_adjustment(1, "R", 50)
    k_0, bb_0, hr_0 = orch.compute_platoon_fatigue_adjustment(1, "R", 0)
    assert k_f < k_0 # K rate should drop
    assert bb_f > bb_0 # BB rate should increase
    assert hr_f > hr_0 # HR rate should increase

def test_record_usage(profiles, rng):
    orch = BullpenOrchestrator(profiles, rng)
    orch.record_usage(1, 25)
    assert orch.usage_today[1] == 25
    orch.record_usage(1, 10)
    assert orch.usage_today[1] == 35
