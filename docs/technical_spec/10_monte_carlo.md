# Module Specification: `algomlb.ml.monte_carlo`

The `monte_carlo` sub-module is the central inference layer of the `ml` tier. It bridges the deterministic feature pipeline (Gold and Uranium layers) with probabilistic prop distributions. By consuming Tier 3 (Gold) player-day features for atomic matchups and Tier 4 (Uranium) game-level matrices for global context, the system simulates game state transitions for every plate appearance over thousands of trials.

## 1. Architectural Integration

The Monte Carlo engine adheres to the strict unidirectional Import Ladder.

* **Upstream Dependencies**: Consumes `domain` models for state tracking, `db.repository` functions for persistence, and outputs from both `rolling_service.py` (Gold Layer) and `features.py` (Uranium Layer).
* **Downstream Consumers**: Outputs are consumed by the `ui` (for Streamlit probability visualizations), the `cli` (for batch inference tasks), and the final prop-specific calibration heads.
* **Performance Constraint**: To achieve the default 10,000 simulation trials per game within a reasonable time constraint, the core loop must utilize vectorized NumPy operations or Numba-compiled functions, avoiding standard Python `for` loops wherever possible.

---

## 2. Domain Models (`state_models.py`)

All simulation state variables must be defined as Pydantic models in the `domain` layer to ensure type safety. Because the state mutates rapidly during a simulation, these specific models will bypass the standard `frozen=True` constraint used elsewhere in the domain.

### `PitcherState`
Tracks the active pitcher's dynamic in-game deterioration.
* `pitcher_id`: integer
* `pitches_thrown`: integer, initialized to 0.
* `current_tto`: integer, represents Times Through Order (1, 2, or 3).
* `runs_allowed`: integer.
* `earned_runs_allowed`: integer.
* `hits_allowed`: integer.
* `walks_allowed`: integer.
* `strikeouts`: integer.
* `outs_recorded`: integer.
* `fatigue_multipliers`: dictionary or array modifying baseline velo, spin, whiff, and hard-hit risk (derived from Gold Layer fatigue indices).
* `manager_hook_prob`: float, dynamic probability of removal.

### `GameState`
Tracks the atomic base-out-score state.
* `inning`: integer, initialized to 1.
* `outs`: integer, bounded 0-2.
* `base_state`: integer bitmask (e.g., 000 for empty, 111 for loaded).
* `home_score`: integer.
* `away_score`: integer.

---

## 3. Core Engine Component (`engine.py`)

The `SimulationEngine` class coordinates the trial execution. It must be initialized with a fixed random seed for complete reproducibility.

### Input Contracts
Before starting trials, the engine must load the following materialized pregame features into memory:
* **Uranium Layer (Game Context)**: The game-level training matrix including Team Elo, Pythag, RE24, and the Matchup Spine for global priors.
* **Gold Layer (Player Context)**: The player-day grain features, including EMA trends, volatility, and fatigue indices for all 9 starting batters and the active bullpen.
* Confirmed or projected lineups and starting pitchers.
* `sp_fatigue_curves` and `sp_tto_splits`.
* `pitch_count_per_pa_dist`.
* `bullpen_availability` and `manager_hook_profiles`.
* Environmental park and weather data (Air Density Ratio and Tailwind Components).

### The Simulation Loop
The `run_trials(trials: int = 10000)` method executes the following state machine for each game simulation:
1. **Initialize**: Setup lineups, starters, bullpen queues, `GameState`, and `PitcherState` using the Uranium Matchup Spine as the base context.
2. **Inning Transition**: Identify the active pitcher for the top/bottom of the inning.
3. **Pre-PA Update**: Update `PitcherState` utilizing current pitch count bucket, TTO state, days rest baseline, Gold Layer fatigue indices, weather effects, and manager tolerance.
4. **Probability Generation**: Calculate specific PA outcome probabilities based on the multinomial PA Outcome Model, feeding it the Gold Layer batter/pitcher features.
5. **PA Sampling**: Sample exactly one canonical outcome (strikeout, walk, hbp, single, double, triple, home_run, out_in_play).
6. **Pitch Sampling**: Sample the pitches consumed during the PA from the `pitch_count_per_pa_dist` table.
7. **State Update**: Advance `GameState` (outs, runners, score) and `PitcherState` (counts, individual stat accumulation).
8. **Hook Evaluation**: Determine if the manager removes the pitcher before the next batter based on `manager_hook_profiles` and game leverage.
9. **Bullpen Transition**: If a hook is triggered, select a relief arm from `bullpen_availability`.
10. **Termination**: Loop through 9 innings, continuing to extra innings if required by the target market.

---

## 4. Sub-System Logic

### `BullpenManager`
The simulator must explicitly model bullpen transitions rather than using league-average noise.
* **Input**: Game score, inning, calculated leverage index, and `bullpen_availability`.
* **Logic**: Assigns available pitchers to role buckets (`closer`, `setup`, `high_lev`, `mid_lev`, `long_rel`).
* **Execution**: Pops the highest-suitability pitcher from the respective bucket based on the game's current leverage and lead/deficit size.

### `ScoringAttributor`
Official scoring attribution is critical for accurate pitcher prop evaluation.
* **Logic**: Applies Retrosheet's responsible-pitcher conventions to accurately assign inherited runners to the correct pitcher's ledger.
* **Outputs**: Calculates earned runs, quality starts, wins, losses, and no-decisions identically to official MLB scoring rules.

---

## 5. Aggregation Layer (`prop_aggregator.py`)

This component reduces the raw 10,000-trial matrix into actionable prop targets, functioning as the final bridge before market calibration.

* **Batter Props**: Calculates event probabilities for hits, hits 2+, total bases, HR, runs, RBI, walks, strikeouts, and stolen bases.
* **Pitcher Props**: Calculates medians and over/under probabilities for strikeouts, walks, hits allowed, earned runs, outs recorded, innings pitched, and pitch counts.
* **Team & Game Props**: Calculates moneyline support, derived win probability, team totals, first 5 innings support, and NRFI/YRFI probabilities.
* **Uranium Synthesis**: Appends the derived Monte Carlo probabilities back onto the game-level Uranium matrix (`X`) for the final calibration XGBoost models to consume.
