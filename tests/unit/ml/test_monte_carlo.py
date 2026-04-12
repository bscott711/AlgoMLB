from algomlb.ml.monte_carlo.state import GameState, BatterSimState, PitcherSimState
from algomlb.ml.monte_carlo.engine import SimulationEngine


def test_hrr_attribution_logic():
    """Verify that Runs, RBIs, and Hits are accurately attributed in the Monte Carlo loop."""
    # Setup
    batter_id = 101
    runner_id = 202

    state = GameState()
    # Put runner on 3rd
    state.bases[2] = runner_id

    # Simulate a single (Runner on 3rd scores, batter moves to 1st)
    scored_ids = state.process_event("single", batter_id)

    assert scored_ids == [runner_id]
    assert state.bases[0] == batter_id
    assert state.bases[2] is None


def test_sim_state_aggregation():
    """Verify that composite stats like HRR are computed correctly inside the BatterSimState."""
    b = BatterSimState(player_id=1, hits=2, runs=1, rbi=2)
    assert b.hrr == 5


def test_engine_stat_accumulation():
    """Verify that multiple events properly accumulate in the registries."""
    engine = SimulationEngine(pa_model=None)

    batter_reg = {1: BatterSimState(player_id=1)}
    pitcher_reg = {9: PitcherSimState(pitcher_id=9)}

    # 1. First event: Single with nobody on
    engine._attribute_stats("single", 1, 9, [], batter_reg, pitcher_reg)

    assert batter_reg[1].hits == 1
    assert batter_reg[1].pa_count == 1
    assert pitcher_reg[9].hits_allowed == 1
    assert pitcher_reg[9].pitches_thrown == 4

    # 2. Second event: Strikeout
    engine._attribute_stats("strikeout", 1, 9, [], batter_reg, pitcher_reg)
    assert batter_reg[1].strikeouts == 1
    assert pitcher_reg[9].strikeouts == 1
    assert pitcher_reg[9].outs_recorded == 1


def test_home_run_attribution():
    """Verify HR sends everyone home and credits all RBIs correctly."""
    engine = SimulationEngine(pa_model=None)

    batter_reg = {
        1: BatterSimState(player_id=1),
        2: BatterSimState(player_id=2),
        3: BatterSimState(player_id=3),
    }
    pitcher_reg = {9: PitcherSimState(pitcher_id=9)}

    # Batter 1 hits a HR with Batter 2 on base
    engine._attribute_stats("home_run", 1, 9, [1, 2], batter_reg, pitcher_reg)

    assert batter_reg[1].hr == 1
    assert batter_reg[1].hits == 1
    assert batter_reg[1].runs == 1  # Batter scores
    assert batter_reg[2].runs == 1  # Runner scores
    assert batter_reg[1].rbi == 2  # 2 runs driven in
    assert pitcher_reg[9].runs_allowed == 2
