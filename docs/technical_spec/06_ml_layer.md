# Module Specification: algomlb.ml

The `ml` module is the intelligence core of AlgoMLB. It transforms raw ingestion data into actionable features and machine learning models through a multi-tiered pipeline (Quant → Silver → Rolling → Uranium).

## 1. The Multi-Tiered Feature Pipeline

### Tier 1: The Quant Layer (`quant_service.py`)
Operating at the **pitch level**, this layer produces standardized performance metrics.
- **Pitch Movement Z-Scores**: Standardizes `pfx_x`, `pfx_z`, and `release_speed` into z-scores grouped by `(pitcher, pitch_type)` using a trailing baseline window (typically 365 days).
- **Coordinate Transforms**: Normalizes Statcast `hc_x/hc_y` to a Cartesian system where home plate is at `(0,0)`, and computes the `spray_angle_deg`.
- **xBA/xwOBA Calibration**: Rescales raw Statcast probabilities to the season mean/std to account for league-wide offensive environments.

### Tier 2: The Silver Layer (`silver_processor.py`)
Summarizes pitch-level data into **player-game logs**.
- **Performance Flags**: Calculates binary flags for `is_whiff`, `is_barrel`, `is_hard_hit` (EV ≥ 95), and `is_sweet_spot`.
- **Platoon Splits**: Aggregates metrics (xwOBA, PA) separately for RHB/LHB matchups.
- **Bayesian Shrinkage**: Regresses game-level metrics toward a player's prior-season mean to stabilize variance (especially for rookies or small samples).

### Tier 3: The Rolling Layer (`rolling_service.py`)
Computes the final **feature set** (Gold Layer) used for inference.
- **EMA & Trends**: Calculates Exponential Moving Averages (EMA 3G, 7G) for xwOBA, bat speed, and attack angles to capture momentum.
- **Volatility**: Tracks the standard deviation of performance (e.g., `std_launch_angle_15g`) to measure consistency.
- **Fatigue Indices**: Computes burden metrics based on pitch counts and days since last appearance.
- **Gold Layer Grain**: Outputs are at the **player-day** level.

### Tier 4: The Uranium Layer (`features.py`)
The ultimate stage where player-level features are transformed into a **game-level training matrix**.
- **Team Aggregation**: Collates the 9 starting batters from the boxscore/lineup and averages their Gold Layer features to produce a team-level offensive profile.
- **Matchup Spine**: Joins Home/Away starting pitcher Gold features and those for the relief corps, along with team-level metrics (Elo, Pythag, RE24), onto a single game row.
- **Matrix Finalization**: Selects model-ready features, performs final imputation (median-fill), and drops constant columns to prepare the `X` matrix for Uranium XGBoost models.
- **Uranium Layer Grain**: Outputs are at the **game** level, ready for training or inference.

---

## 2. Advanced Analytics & Physics

### Environmental Fortification (`weather_features.py`)
- **Air Density Ratio**: Computes the psychrometric density of air (Relative to standard 1013.25hPa, 15°C) to estimate flight carry.
- **Tailwind Component**: Decomposes wind vectors relative to the stadium's center field bearing and the ball's spray angle.

### Elo Rating System (`elo.py`)
- **Pure Outcome Rating**: A logistic Elo implementation that tracks team strength based strictly on wins/losses and home-field advantage (HFA).
- **No-Leakage Prior**: Specifically designed as a "dumb" prior that does not see market odds, ensuring the ML models can find alpha relative to the outcome history.

---

## 3. Model Management (`registry.py` & `model_io.py`)

### Manager Registry
- **Attribution**: Maps Retrosheet manager data to MLB game IDs.
- **Tenure Metrics**: Tracks `manager_tenure_day` and `days_since_manager_change` to model the "manager bump" or strategy shifts.

### Machine Learning I/O
- **Storage**: Models are stored in `/home/opc/AlgoMLB/models` using `joblib`.
- **Versioning**: Supports versioned assets (e.g., `batted_ball_baseline_v1.joblib` and `env_coefficients_v1.json`) to allow for A/B testing and rollbacks.

## Engineering Notes
- **Lookahead Guard**: All baseline and rolling calculations strictly enforce `game_date < target_date` to prevent data leakage during backtesting.
- **Idempotency**: Processors use a `StatcastProcessRegistry` table to track the `last_processed_ingested_at` checkpoint, allowing for safe resumes after failure.
