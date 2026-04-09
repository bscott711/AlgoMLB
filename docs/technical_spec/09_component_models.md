# Module Specification: `algomlb.ml.component_models`

The `component_models` sub-module forms the foundational predictive layer of the `ml` tier. It is responsible for training the isolated, atomic models and statistical priors that feed the downstream Monte Carlo simulation engine.

By strict architectural mandate, these models predict *player-level events* and *state transitions* rather than end-game prop outcomes.

## 1. The PA Outcome Model (`pa_model.py`)

This is the primary classification engine of the platform. It predicts the exact outcome of a single Plate Appearance (PA) given a specific batter, pitcher, and game context.

### A. Data Preparation & Target Engineering


* **Input Grain (`X`)**: `batter-pitcher-game-state`. This is constructed by joining the Tier 3 "Gold Layer" player-day features (batter skills, pitcher readiness, fatigue) with the Tier 4 "Uranium Layer" game matrix (park factors, weather density ratios, Elo context).
* **Target Grain (`y`)**: The model must map Retrosheet/Statcast raw `events` into exactly **eight canonical classes**. No other classes are permitted.
    1. `strikeout`
    2. `walk` (Must merge intentional and unintentional walks)
    3. `hbp` (Hit by pitch)
    4. `single`
    5. `double`
    6. `triple`
    7. `home_run`
    8. `out_in_play` (Must merge groundouts, flyouts, popouts, lineouts, double plays, etc.)

### B. Model Architecture
* **Algorithm**: Multinomial Gradient Boosted Trees (XGBoost or LightGBM).
* **Objective**: `multi:softprob` (XGBoost) or equivalent, ensuring the output is an array of 8 probabilities that sum to strictly 1.0 for every PA.
* **Hyperparameter Tuning**: Must prioritize multi-class Log Loss (`mlogloss`) to ensure the probability distributions are well-calibrated, rather than optimizing purely for absolute accuracy.

### C. Strict Time-Aware Validation (`validation.py`)
Random cross-validation (e.g., standard `train_test_split` or `KFold`) is **strictly prohibited** to prevent future data from leaking into the past.
* **Walk-Forward Methodology**: Training and validation sets must be split chronologically.
    * *Example Split 1*: Train on Seasons 2021-2022 -> Validate on April 2023.
    * *Example Split 2*: Train on Seasons 2021-2022 + April 2023 -> Validate on May 2023.
* **Leakage Guard**: A hard assertion must exist verifying that `max(train_date) < min(validation_date)`.

---

## 2. Statistical Priors & Generators (`priors.py`)

The Monte Carlo engine requires highly specific, dynamic lookup tables to manage in-game state transitions (pitch counts, fatigue, manager decisions). This module defines the generation of these materialized views.

### A. Pitch Count Distributions (`pitch_count_per_pa_dist`)
* **Purpose**: Simulates how many pitches are consumed during a specific PA outcome.
* **Grain**: `pitcher_archetype` + `PA_outcome` + `leverage_tier`.
* **Execution**: Aggregates historical Statcast data to compute the `mean`, `std`, `p25`, `p50`, `p75`, and `p95` of pitch counts for every combination of the above grain.
* *Note*: Leverage tiers must be derived from the base-out-score run expectancy (RE24) matrix.

### B. Pitcher Fatigue Curves (`sp_fatigue_curves`)
* **Purpose**: Maps how a pitcher's stuff degrades as pitch counts rise.
* **Grain**: `pitcher_id` + `pitch_type` + `pitch_count_bucket` (e.g., `0-20`, `21-40`, `41-60`, `61-80`, `81-100`, `100+`) + `season`.
* **Execution**: Computes the mean velocity, spin rate, vertical/horizontal movement, and whiff rate for each bucket. Must dynamically track the `release_speed_delta` compared to the pitcher's baseline (Bucket 0-20).

### C. Manager Hook Profiles (`manager_hook_profiles`)
* **Purpose**: Creates a probabilistic prior for when a manager will pull the active pitcher.
* **Grain**: `manager_id` + `season` + `cutoff_date` (Rolling prior).
* **Execution**: Calculates manager tendencies, specifically outputting:
    * `pull_before_3rd_tto_pct`: % of starts pulled before facing the order a 3rd time.
    * `pull_when_over_90_pitches_pct`: Tolerance for high workloads.
    * `bullpen_protective_pct`: Tendency to use quick hooks in high-leverage situations.

---

## 3. Serialization & Integration

* **Artifact Storage**: Once the PA Outcome model finishes its walk-forward tuning, the final production model is serialized via `joblib` and saved to the `models/` directory configured in `algomlb.config` (e.g., `models/pa_outcome_multi_v1.joblib`).
* **Database Sync**: The generated statistical priors (Pitch Counts, Fatigue, Hooks) are materialized and pushed to PostgreSQL via `db.repository` using the standard chunked UPSERT operations to ensure the Monte Carlo engine can query them instantly into memory before a simulation run.
