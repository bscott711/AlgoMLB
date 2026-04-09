# AlgoMLB ML System Specification

This document defines the full machine learning and simulation architecture for the AlgoMLB props platform. It is written as an implementation-grade specification for engineers building the data, modeling, simulation, and inference layers of the system. The platform is organized as a hierarchical pipeline in which pregame player and team state features feed batter and pitcher component models, those component models feed a Monte Carlo game simulation engine, and the simulator feeds calibrated prop-specific prediction models.

## System Objectives

The system must generate pregame probability distributions and market-facing probabilities for MLB batter props, pitcher props, team props, and game props. The architecture must support daily inference, rolling historical backtests, probabilistic calibration, and reproducible replay of any slate given a model version, feature snapshot, and market timestamp.

The design must prioritize calibration, leakage prevention, modularity, and temporal correctness over raw single-metric performance. The core premise is that game outcomes and player props are best modeled through player-level event generation and state transitions rather than through a single monolithic classifier.

## Guiding Principles

1. **Hierarchical modeling.** Predict low-level baseball events and player state first, then derive game and prop outcomes from simulation rather than directly modeling every market from raw features.
2. **Pregame information only.** Every feature used by a pregame model must be available prior to first pitch. No post-start information may leak into training or inference.
3. **Time-aware validation.** All model evaluation must use rolling or walk-forward splits. Random cross-validation across seasons is prohibited.
4. **Simulation-centered architecture.** The Monte Carlo engine is a first-class system component, not a post hoc wrapper.
5. **Prop-specific heads.** Each sportsbook prop family gets its own target definition, feature contract, and calibration layer.
6. **Versioned feature contracts.** Each engineered table must have explicit grain, freshness cadence, null policy, and leakage rules.
7. **Reproducibility.** Any prediction must be regenerable from stored source data, model artifacts, and configuration.
8. **Calibration over vanity metrics.** Log loss, Brier score, and calibration reliability matter more than accuracy alone for betting use.

## Scope

The platform covers four prediction domains:

| Domain | Outputs |
|---|---|
| Batter props | Hits, hits 2+, total bases, HR, runs, RBI, walks, strikeouts, stolen bases |
| Pitcher props | Strikeouts, walks, hits allowed, earned runs, outs recorded, innings pitched, quality start, win, loss, no-decisions, pitch count |
| Team props | Team runs, first 5 team runs, score first, NRFI/YRFI support |
| Game props | Moneyline support, totals support, first 5 outcomes, derived win probability |

## Canonical Data Sources

The system uses the following canonical source systems:

| Source | Purpose |
|---|---|
| `statcast_raw` | Pitch-level physics, pitch mix, release metrics, batted-ball quality, lineup context, TTO via `n_thruorder_pitcher` |
| `retrosheet_events` | PA-level game state, runner responsibility, earned/unearned attribution, substitutions, lineup position, score progression |
| `team_managers` | Manager tenure, active manager by game date, mid-season manager changes |
| `player_transactions` | IL status, recalls, options, roster availability |
| `openmeteo_weather_progression` | Weather progression T0-T4, headwind, crosswind, temperature deltas |
| `ballparks` | Park factors and orientation data |
| `game_results` | Master game identity, game type filtering, canonical MLB game PK |
| sportsbook odds snapshots | Market lines, prices, timestamps, CLV measurement |

`game_results` is the canonical game identity table. All model training rows, simulation rows, and prediction outputs must key to `game_pk` from `game_results`. Retrosheet-native IDs must be resolved into `game_pk` via a persisted registry before downstream modeling.

## Canonical Game Registry

A persisted `game_manager_registry` table must exist and serve as the canonical bridge between Retrosheet game identifiers and `game_results.game_pk`. The registry must be materialized and indexed; it must not exist only as a transient CTE.

### Table: `game_manager_registry`

| Column | Type | Description |
|---|---|---|
| `game_pk` | BIGINT | Canonical MLB game key |
| `retrosheet_game_id` | TEXT | Retrosheet game identifier |
| `game_date` | DATE | Official date |
| `season` | INT | Season |
| `game_type` | TEXT | Regular, postseason |
| `team_id` | INT | Team perspective row |
| `opponent_team_id` | INT | Opponent |
| `home_away` | TEXT | H/A |
| `doubleheader_num` | INT | Game number within doubleheader |
| `manager_id` | INT | Active manager for that team on game date |
| `manager_stint_start_date` | DATE | Start of current managerial stint |
| `manager_tenure_day` | INT | Team games managed in current stint, 1-indexed |
| `days_since_manager_change` | INT | Calendar days since stint start |

### Required rules

- Registry rows exist at team-game grain, not game grain.
- Inclusion rule: only games present in `game_results` and marked as regular season or postseason.
- Manager assignment rule: join the active manager whose start date is the latest start date on or before `game_date` for that team.
- `manager_tenure_day` is computed as `ROW_NUMBER()` partitioned by `(team_id, manager_id, manager_stint_start_date)` ordered by `(game_date, game_pk)`.
- `days_since_manager_change` is `game_date - manager_stint_start_date`.

## Engineered Feature Layers

All engineered data products must be materialized and versioned. Feature generation must be deterministic, idempotent, and partitionable by date.

### Layer A: Pregame Pitcher Readiness

Grain: pitcher-game

#### Table: `sp_pre_game_readiness`

| Feature group | Required fields |
|---|---|
| Rest | `days_rest`, `days_rest_since_start`, `last_appearance_days_ago` |
| Prior start stress | `last_start_pitch_count`, `last_start_ip`, `last_start_max_velo_delta`, `last_start_spin_drop_80_plus` |
| Rolling workload | `pitches_last_7d`, `pitches_last_15d`, `pitches_last_30d`, `starts_last_30d` |
| Season workload | `season_ip_to_date`, `season_pitches_to_date`, `season_batters_faced_to_date` |
| Health/availability | `days_since_il_return`, `il_stint_count_season`, `reported_pitch_limit`, `is_opener_flag` |
| Matchup | opponent handedness mix, opponent chase rate, opponent K rate, opponent BB rate, opponent hard-hit rate |
| Environment | `temp_f`, `headwind_t0_mph`, `crosswind_t0_mph`, humidity, pressure, park factors |
| Management | `manager_id`, `manager_tenure_day`, `days_since_manager_change` |

Leakage rule: all rolling values must be shifted to exclude the current game.

### Layer B: Pitcher Fatigue Curves

Grain: pitcher-pitch_type-pitch_count_bucket-season

#### Table: `sp_fatigue_curves`

| Field | Description |
|---|---|
| `pitcher_id` | Pitcher |
| `pitch_type` | FF, SI, SL, CH, CU, KC, etc. |
| `pitch_count_bucket` | `0-20`, `21-40`, `41-60`, `61-80`, `81-100`, `100+` |
| `sample_pitches` | Bucket support |
| `avg_release_speed` | Mean velo in bucket |
| `avg_spin_rate` | Mean spin in bucket |
| `avg_spin_axis` | Mean spin axis |
| `avg_pfx_x`, `avg_pfx_z` | Movement by bucket |
| `avg_extension` | Extension by bucket |
| `avg_rel_pos_x_delta`, `avg_rel_pos_z_delta` | Release drift vs pitcher baseline |
| `whiff_rate` | Swing-and-miss rate in bucket |
| `called_strike_rate` | Called strike rate |
| `hard_hit_rate` | EV >= 95 mph |
| `barrel_rate` | Barrel rate |
| `k_rate_bucket` | K rate when pitches in this bucket are thrown |
| `bb_rate_bucket` | BB rate when pitches in this bucket are thrown |

### Layer C: Times Through Order Splits

Grain: pitcher-TTO-season

#### Table: `sp_tto_splits`

| Field | Description |
|---|---|
| `pitcher_id` | Pitcher |
| `tto` | 1, 2, 3 where 3 means third-or-later |
| `sample_pa` | Support |
| `k_pct`, `bb_pct`, `h_pct`, `hr_pct`, `xwoba`, `hard_hit_rate`, `whiff_rate` | Raw split metrics |
| `k_pct_delta_vs_tto1`, `xwoba_delta_vs_tto1`, `hard_hit_delta_vs_tto1` | Penalties vs first TTO |

### Layer D: Pitch Count per Plate Appearance Distributions

Grain: pitcher_archetype-PA_outcome-leverage_tier

#### Table: `pitch_count_per_pa_dist`

Required outcomes:
- `strikeout`
- `walk`
- `hbp`
- `single`
- `double`
- `triple`
- `home_run`
- `out_in_play`

Required summary fields:
- `pitch_count_mean`
- `pitch_count_std`
- `pitch_count_p25`
- `pitch_count_p50`
- `pitch_count_p75`
- `pitch_count_p95`

### Layer E: Bullpen Availability

Grain: team-pitcher-game_date

#### Table: `bullpen_availability`

| Feature group | Required fields |
|---|---|
| Usage | `pitches_last_1d`, `pitches_last_2d`, `pitches_last_3d`, `appearances_3d` |
| Recovery | `consecutive_days_used`, `days_since_last_used` |
| Role | `role`, `available_for_high_lev`, `estimated_pitch_limit` |
| Health | `is_il_eligible` |

### Layer F: Manager Hook Events

Grain: pitcher removal event

#### Table: `manager_hook_events`

| Field | Description |
|---|---|
| `game_pk`, `season`, `game_date` | Identity |
| `team_id`, `manager_id`, `pitcher_id` | Decision actors |
| `inning`, `outs_at_hook` | Timing |
| `pitches_thrown` | Workload at removal |
| `tto_at_hook` | TTO state |
| `score_diff_at_hook` | Team score minus opponent score |
| `base_state_at_hook` | Encoded runners on base |
| `leverage_index_at_hook` | Derived leverage |
| `manager_tenure_day`, `days_since_manager_change` | Context |
| `bullpen_availability_snapshot_id` | Link to bullpen state |

### Layer G: Manager Hook Profiles

Grain: manager-season-cutoff_date

#### Table: `manager_hook_profiles`

Required fields:
- `avg_sp_pitch_count`
- `avg_ip_per_start`
- `pull_before_3rd_tto_pct`
- `pull_with_lead_pct`
- `pull_when_over_90_pitches_pct`
- `quick_hook_high_leverage_pct`
- `bullpen_protective_pct`
- `starter_leniency_index`

### Layer H: Batter Pregame State

Grain: batter-game

#### Table: `batter_pre_game_state`

Required groups:

| Group | Required fields |
|---|---|
| Contact quality | rolling EV, launch angle, hard-hit rate, barrel rate, sweet-spot rate |
| Plate discipline | chase rate, zone swing, contact rate, whiff rate, BB rate, K rate |
| Expected outcomes | xwOBA, xBA, xSLG, xISO |
| Split context | platoon split metrics vs SP handedness and pitch mix |
| Role context | projected lineup slot, expected PA count, speed score, steal attempt rate |
| Form windows | 3g, 7g, 15g, 30g, season-to-date windows |
| Availability | injury/rest flags, travel/day game after night game indicators |

### Layer I: Batter vs Pitcher Matchup State

Grain: batter-pitcher-game

#### Table: `bvp_matchup_state`

Required fields:
- batter skill features from `batter_pre_game_state`
- pitcher skill features from `sp_pre_game_readiness`
- SP pitch mix shares
- batter performance vs pitch type clusters
- platoon indicator
- park and weather context
- manager and bullpen context for later PA expectation adjustment

## Simulation Engine Specification

The Monte Carlo simulation engine is the central inference layer. It must produce game-level and player-level distributions by simulating plate appearances, pitch counts, lineup progression, fatigue, hooks, bullpen transitions, and scoring.

### Simulation goals

The simulator must output:
- team run distributions
- player stat count distributions
- pitcher stat count distributions
- event probabilities for bettable thresholds
- intermediate state distributions for calibration and debugging

### Simulation loop

For each game simulation trial:

1. Initialize lineups, starters, bullpen queues, game score, base state, inning, outs, and per-player counters.
2. For each half inning, identify the active pitcher.
3. Before each PA, update pitcher state using:
   - current pitch count bucket
   - TTO state
   - days rest baseline
   - weather effects
   - manager tolerance profile
4. Generate PA outcome probabilities from the batter-pitcher interaction model.
5. Sample a PA outcome.
6. Sample pitches consumed during the PA from `pitch_count_per_pa_dist`.
7. Update pitch count, fatigue state, base-out state, score, and individual stats.
8. Evaluate whether the manager pulls the pitcher before the next batter or inning.
9. If a hook occurs, select a bullpen arm based on role, availability, and leverage.
10. Continue through 9 innings and extras if required by the modeled market.
11. Record all tracked stat outputs for the trial.

## Model Families

The modeling stack contains component models and prop heads.

### Component Model 1: PA Outcome Model

Purpose: produce per-PA outcome probabilities for batter vs pitcher matchups.

Input grain: batter-pitcher-game-state

Output classes:
- strikeout
- walk
- hbp
- single
- double
- triple
- home_run
- out_in_play

Recommended model family: multinomial gradient boosted trees or equivalent multiclass model.
