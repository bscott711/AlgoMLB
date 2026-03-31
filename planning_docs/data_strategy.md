This is a deep, multi-layered problem — here is a prioritized, production-grade blueprint organized by signal tier. The architecture assumes PostgreSQL as your persistent store and Python as your ETL/modeling layer.

***

## Tier 1: Advanced Pitching Metrics

### ERA Estimators — Ranked by Predictive Power

The sabermetric consensus on ERA estimator predictiveness (year-over-year) is well-established. Based on Baseball Prospectus's reliability vs. predictiveness matrix, the hierarchy is:[1]

| Metric | Reliability | Predictiveness | Notes |
|---|---|---|---|
| **DRA (Deserved Run Average)** | .53 | .26 | Highest predictive score; Statcast-adjusted |
| **cFIP** | .51 | .25 | Context-independent FIP |
| **SIERA** | .46 | .18 | Batted ball shape + strikeout interaction |
| **xFIP** | .44 | .19 | HR/FB normalized; portable across parks |
| **K-BB%** | — | Best single-stat | Strongest predictor of future ERA by correlation [2] |
| **ERA** | .13 | .10 | Highest noise; avoid as a model feature |

**Practical implication**: Use K-BB% as a primary feature, DRA/xFIP as the run-estimator backbone, and treat ERA itself as a target variable label, not a predictor.

### Statcast Pitch-Level Features (Must-Have)

These are your highest-signal raw features for pitcher modeling. Pull via `pybaseball.statcast_pitcher()`:[3]

- **release_speed** — Velocity; use delta from pitcher's own trailing mean (velocity drop = fatigue signal)
- **release_spin_rate** — Raw RPM; most predictive on fastballs and breaking balls
- **pfx_x / pfx_z** — Horizontal and vertical movement (inches from gravity-only baseline)
- **release_extension** — Perceived velocity proxy; 6.5+ feet is elite
- **effective_speed** — Accounts for extension; better than raw velo for batter reaction time
- **induced_vertical_break (IVB)** — Spin-axis-adjusted; filters seam-shifted wake effects
- **plate_x / plate_z** — Location at the plate; used to compute zone rates
- **launch_speed_against / launch_angle_against** — Pitch-level batted ball quality allowed

### Stuff+, Location+, Pitching+

Stuff+ evaluates physical pitch characteristics (velocity, movement, spin) independent of outcomes. It is **highly reliable (0.41)** for same-team pitchers but its predictive power collapses for team-switchers (0.14). This is a critical caveat for your model — **do not treat Stuff+ as team-portable for off-season acquisitions**. Location+ and Pitching+ are more stable across team changes (.24 and .23 respectively).[4][1]

**Engineering approach**: Don't use Stuff+ as a black-box feature. Instead, reconstruct its components (velocity percentile, IVB, horizontal break, extension) as individual features and let the model learn the interaction weights.

### Bullpen Fatigue & Availability Model

This is one of the most underpriced edges in ML/totals markets. Build a pitcher availability tensor updated daily:

```python
# Fatigue features per reliever
- pitches_thrown_last_1d / last_2d / last_3d   # recency-weighted pitch counts
- days_rest                                      # calendar days since last appearance
- inherited_runners_stranded_rate (rolling 30d)  # leverage quality
- high_leverage_appearances_L7d                  # usage context
- cum_pitches_rolling_14d                        # workload accumulation
- inning_pitched_today (game-state feature)      # live update during ETL
```

The bullpen exhaustion score should be a composite fatigue index fed as a game-level feature. Starters throwing 90+ pitches in prior starts dramatically change bullpen exposure, which directly prices totals differently than the naive starter-ERA approach most public models use.

***

## Tier 2: Advanced Hitting Metrics

### Stabilization Windows (Signal vs. Noise Thresholds)

The FanGraphs sample-size framework  gives you empirically-derived stabilization points. Use these as **minimum PA thresholds for feature inclusion**, not hard cutoffs — signal accumulates incrementally:[5][6]

| Metric | Stabilization PA | Use for Rolling Window |
|---|---|---|
| **K%** | ~60 PA | 150 PA rolling (fast-stabilizing) |
| **BB%** | ~120 PA | 200 PA rolling |
| **ISO (Power)** | ~160 PA | 250 PA rolling |
| **BABIP** | ~820 PA | 3-year regressed; not a short-term feature |
| **wOBA** | ~375 PA | 400–500 PA rolling [7] |
| **wRC+** | ~375 PA | 400 PA rolling; park-adjusted |
| **Barrel %** | ~200 PA | 300 PA rolling |
| **xwOBA (Statcast)** | ~150 PA | 200 PA rolling; lower variance than wOBA |

**Engineering insight**: Use **exponential decay weighting** rather than flat rolling windows. A game from 3 days ago is far more predictive than one from 60 days ago for in-season models. Use a half-life of ~15 games for hot/cold streaks.

### High-Alpha Statcast Hitting Features

From `pybaseball.statcast_batter()` or the Baseball Savant leaderboards:

- **xwOBA** — Expected wOBA based on EV/LA; removes defense and park from the equation; your primary batter quality signal
- **Barrel %** — EV ≥ 98 mph + optimal launch angle range (26–30°); most predictive HR/XBH metric
- **Hard Hit % (EV ≥ 95 mph)** — Broader than barrel; stabilizes faster
- **Average Exit Velocity** — Good for ranking batters; combine with launch angle for the full picture
- **Sprint Speed** — Relevant for BABIP on grounders; predictive for hit-based markets
- **Chase Rate / O-Swing%** — Plate discipline; leads discipline → BB% → OBP
- **Whiff %** — Predicts K%; high-value for matchup-level features (batter Whiff% vs. pitcher pitch-mix)

### Platoon Splits (Critical for Moneyline Pricing)

The most valuable platoon signal is **wOBA vs. LHP / wOBA vs. RHP**, not simple BA splits. Use a minimum of 200 PA per split before trusting the directional signal. Build these as **separate batter feature vectors**:

```python
batter_features = {
    'xwOBA_vs_RHP_L300PA': ...,
    'xwOBA_vs_LHP_L300PA': ...,
    'K_rate_vs_RHP_L150PA': ...,
    'K_rate_vs_LHP_L150PA': ...,
    'barrel_pct_vs_RHP_L300PA': ...,
}
```

Aggregate these to lineup-level features using batting order weighting (positions 1–4 weighted 1.5x, 5–9 weighted 0.8x) to approximate run-expectancy contribution.

***

## Tier 3: Environmental & Contextual Features

### Park Factors — Dynamic Adjustment

Static 3-year park factors (FanGraphs, Baseball Prospectus) are your baseline. But static PF misses game-day conditions. Build a **dynamic park factor adjustment**:[8]

```
Dynamic_PF = Static_PF_3yr × Temperature_Multiplier × Wind_Multiplier × Humidity_Modifier
```

Temperature effect on HR/run scoring is well-documented: approximately **+0.5% HR probability per 1°F above 60°F** (altitude-adjusted). At Coors Field specifically, strikeout rates are suppressed beyond altitude effects alone due to pitcher psychological adaptation. Pull per-game temperature from your weather API and apply as a continuous multiplier.[9]

### Weather Feature Engineering

Use the **Open-Meteo API** (free, hourly, historical) or **Tomorrow.io** for real-time stadium weather. Map weather to stadium geometry manually — wind direction matters directionally per park:

| Feature | How to Engineer |
|---|---|
| **Wind speed (mph)** | Continuous; ≥15 mph out = HR-positive; use `out_to_in` binary flag |
| **Wind direction vs. RF/LF** | Encode as dot product of wind vector and park's fly ball direction |
| **Temperature (°F)** | Continuous; center on 72°F for the multiplier |
| **Humidity** | Negatively correlated with ball carry; include as continuous |
| **Precipitation probability** | Binary + confidence level; affects total-side lean |
| **Roof status** | Binary (retractable roof open/closed) — overrides all outdoor factors |

### Travel Fatigue & Rest Advantage

Build a **rest differential feature** for each game:

```python
home_rest_days = days_since_last_game
away_rest_days = days_since_last_game
# NOW POSSIBLE: Latitude and longitude are stored in BallparkORM
travel_distance_km = haversine(prev_city_coords, current_city_coords)
timezone_change = abs(prev_tz_offset - current_tz_offset)
```

Research shows teams playing on 0 days rest after >1,500-mile travel lose ~0.15 runs of scoring equity on average.

### Umpire Tendencies — Critical 2026 Update

**Umpire tendency modeling is being deprecated as a signal for 2026.** MLB is implementing the Automated Ball-Strike (ABS) challenge system league-wide in 2026, which effectively kills historical umpire zone-tilt features. However, while ABS is transitioning, you can still model:[10][11]

- **Call accuracy rate** (historical, from Baseball Savant umpire scorecards)
- **Zone size tendency** (larger vs. smaller zone — affects K rates, BB rates)
- **Framing interaction** (certain catchers benefited disproportionately from certain umpires; now degraded signal)

Engineer these as a **trailing umpire profile** (`calls_above_avg_L50_games`, `extra_strike_rate_L50`) and include as a depreciating feature with a flag for ABS-affected games.

***

## Tier 4: Market Data & Odds Ingestion

To train a model that generates Closing Line Value (CLV), the target variable cannot just be the binary game outcome (Win/Loss). The model must be trained to recognize when the true probability diverges from the market's implied probability.

### Historical Odds (Backtesting Foundation)
You cannot backtest a betting strategy without accurate historical odds.
- **The Goal**: Acquire historical Pinnacle (PINN) closing lines for the past 3–5 MLB seasons. Pinnacle is the sharpest book in the world; beating their closing line is the mathematical definition of a winning model.
- **Sources**: The-Odds-API (Historical endpoint), OddsPortal (via targeted scrapers), or Kaggle MLB odds datasets.
- **Target Schema**: Store `open_odds`, `close_odds`, and `movement_delta`.

### Live Odds (Execution Engine)
For daily operations, we need continuous polling of the current market.
- **Primary Source**: The-Odds-API (`/v4/sports/baseball_mlb/odds/`).
- **Sharp vs. Soft Books**: The pipeline must separate books by type.
    - **Sharp Books**: Pinnacle, Circa. Use these to calculate the "true" efficient market price (vig-free).
    - **Soft Books**: DraftKings, FanDuel, BetMGM. These are the targets. If the model + Sharp Books agree on a probability, execute the bet against the lagging Soft Books.

### Market-Implied Features
Feed the market's opinion back into the ML model as contextual features:
- **Steam Moves**: Flag games where the sharp line moves > 10 cents in under an hour.
- **Implied Run Totals**: Derive the team's implied runs from the Over/Under and the Moneyline. This serves as a strong baseline feature for the model to "correct" rather than building a run prediction entirely from scratch.

***

## Tier 5: Gold Standard Data Sources

### Source Ranking by Reliability, Depth & ETL-Friendliness

| Rank | Source | Best For | Access Method | Cost |
|---|---|---|---|---|
| **1** | **Baseball Savant (Statcast)** | Pitch-level Statcast, EV/LA/spin, xStats | `pybaseball.statcast()` / direct CSV bulk download | Free |
| **2** | **The Odds API** | Historical closing lines, real-time sharp/soft odds polling | REST API (`/v4/sports/baseball_mlb/odds/`) | Paid (Tiered) |
| **3** | **MLB Stats API (undocumented)** | Live game state, lineups, roster, boxscores | `statsapi` Python wrapper / raw `requests` | Free |
| **4** | **FanGraphs** | DRA, SIERA, wRC+, park factors | `pybaseball.pitching_stats()` | Free (scrape) |
| **5** | **Open-Meteo API** | Hourly historical & live stadium weather | REST API | Free |
| **6** | **Baseball Reference** | Historical game logs, seasonal splits | `pybaseball.schedule_and_record()` | Free (scrape) |

### pybaseball Module Priority for ETL Pipelines

```python
# Tier 1 — Run daily
from pybaseball import statcast              # pitch-level data; cache aggressively
from pybaseball import pitching_stats        # FanGraphs season/date-range aggregates
from pybaseball import batting_stats         # FanGraphs aggregate hitting
from pybaseball import statcast_pitcher      # per-pitcher Statcast
from pybaseball import statcast_batter       # per-batter Statcast

# Tier 2 — Run daily (game-day)
import statsapi                              # lineups, probable starters, roster state
# statsapi.get('schedule', {...})
# statsapi.get('game', {'gamePk': ..., 'fields': 'liveData,boxscore'})

# Tier 3 — Run weekly or on-demand
from pybaseball import park_factors          # baseline park factor table
from pybaseball import playerid_lookup       # cross-source player ID mapping
```

**Critical ETL note**: Statcast data has a ~3-day lag on Baseball Savant's API for clean processed data. For same-day pitch characteristics, hit the MLB Stats API's `game` endpoint (GUMBO feed) directly using `gamePk` identifiers, which are real-time.

### PostgreSQL Schema Design Hints

Structure your database around four primary fact tables:
- `pitch_events` — one row per pitch; foreign key to `game_id`, `pitcher_id`, `batter_id`
- `odds_history` — time-series table tracking line movements. Must include sportsbook, timestamp, market (ML/RL/Total), and price.
- `game_results` — one row per game; stores all environmental features + final outcomes + closing lines.
- `player_rolling_features` — materialized view updated nightly; pre-compute all rolling windows to avoid in-query aggregation at inference time.

Use `PARTITION BY game_date` on `pitch_events` and `odds_history` for query performance — these tables will grow massively.

***

## Feature Priority Hierarchy (Model Input Ranking)

For XGBoost/LightGBM, feature importance in MLB moneyline/totals models historically ranks roughly as follows:

1. **Starter quality differential** (xFIP gap, K-BB% delta, DRA) — highest importance
2. **Lineup-weighted xwOBA vs. pitcher handedness** — 2nd tier
3. **Bullpen fatigue/availability score** — highest alpha in totals markets
4. **Market Implied Context** — deviation from sharp book implied run totals
5. **Park factor × weather multiplier** — totals-specific edge
6. **Team rolling run differential (L15)** — momentum/contextual signal
7. **Travel distance / rest differential** — small but consistent effect
8. **Starting pitcher days rest** — interacts strongly with velocity features

The professional edge comes not from having better features than the market, but from **more precise feature construction** — particularly the bullpen state tensor, dynamic park factors, and matchup-level platoon wOBA features that public models collapse to team averages.

***

## Modeling Imbalanced Betting Data

In MLB betting models, class imbalance manifests in two distinct ways: **outcome imbalance** (e.g., home teams win ~54% → mild imbalance) and **market-beating imbalance** (e.g., +EV opportunities occur in only 5–15% of games → severe imbalance). The treatment is fundamentally different depending on which problem you're solving, and the most critical constraint is **calibration** — without it, your predicted probabilities are meaningless for EV calculation.

***

### Why Calibration Trumps Everything

SMOTE and random oversampling are **destructive for betting models** specifically because they destroy probability calibration. A model predicting 62% win probability that actually resolves at 62% historically is worth everything; a model with better AUC but distorted probability outputs gives you wrong EV calculations and will bleed money even if directional accuracy is high. Research confirms SMOTE produces "strongly overestimated" minority-class probabilities with calibration slopes well below 1.0.[1][2][3]

**The rule for sports betting models:** Never use SMOTE. Fix imbalance through the model's loss function and post-hoc calibration, not through synthetic data.

***

### Strategy 1: `scale_pos_weight` (Native XGBoost — Primary Fix)

The canonical XGBoost approach for binary classification imbalance. It scales the gradient contribution from the positive (minority) class, forcing the model to over-correct errors on that class:[4]

```python
import numpy as np
from xgboost import XGBClassifier

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos  # e.g., 54/46 ≈ 1.17 for moneyline; higher for +EV signal

model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=spw,        # inverse class frequency
    max_delta_step=1,            # stabilizes gradient updates on imbalanced data
    eval_metric='logloss',       # NOT 'error' — you need probability quality
    use_label_encoder=False,
    n_estimators=500,
    learning_rate=0.05,
    early_stopping_rounds=30,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

`max_delta_step=1` is critical when `scale_pos_weight` is large — it bounds the maximum step update in the Newton boosting step, preventing the model from overcorrecting so aggressively it diverges. The XGBoost docs explicitly recommend it for highly imbalanced scenarios.[5][6]

***

Sources
[1] An Updated Evaluation of Hitting and Pitching (Including Stuff ... https://www.baseballprospectus.com/news/article/82426/prospectus-feature-an-updated-evaluation-of-hitting-and-pitching-including-stuff-metrics/
[2] Fun fact: The best predictor of future ERA is not FIP, xFIP, SIERA or ... https://www.reddit.com/r/baseball/comments/zngku4/fun_fact_the_best_predictor_of_future_era_is_not/
[3] pybaseball/README.md at master - GitHub https://github.com/jldbc/pybaseball/blob/master/README.md
[4] Stuff+, Location+, and Pitching+ Primer | Sabermetrics Library https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/
[5] Sample Size | Sabermetrics Library https://library.fangraphs.com/principles/sample-size/
[6] The Meaning of Small Sample Data | FanGraphs Baseball https://blogs.fangraphs.com/the-meaning-of-small-sample-data/
[7] Stabilization standard for wOBA, wRC+? : r/Sabermetrics - Reddit https://www.reddit.com/r/Sabermetrics/comments/1ndmsfb/stabilization_standard_for_woba_wrc/
[8] An Updated System of Park Factors (and Volatility) https://www.baseballprospectus.com/news/article/64534/an-updated-system-of-park-factors-and-volatility/
[9] Sharing my Monte Carlo MLB prop model architecture + 2024 ... https://www.reddit.com/r/algobetting/comments/1s2550e/sharing_my_monte_carlo_mlb_prop_model/
[10] MLB ABS System Explained: How the New Strike Zone Impacts ... https://vsin.com/the-vsin-daily/mlb-abs-system-explained-how-the-new-strike-zone-impacts-betting-in-2026/
[11] MLB ABS System Explained: How the New Strike Zone Impacts ... https://kdus1060.com/mlb-abs-system-explained-how-the-new-strike-zone-impacts-betting-in-2026/
[12] MLB-StatsAPI/statsapi/endpoints.py at master - GitHub https://github.com/toddrob99/MLB-StatsAPI/blob/master/statsapi/endpoints.py
[13] MLB Statistics Summary - Sportradar API https://developer.sportradar.com/baseball/reference/mlb-statistics-summary
[14] Get MLB Hitting Stats with the stats API in Python! - YouTube https://www.youtube.com/watch?v=SVhZBIxH1iw
[15] Major League Baseball Statcast, Visuals & Advanced Metrics https://baseballsavant.mlb.com/league?season=2024
[16] The Advanced Analytical Stack That Actually Predicts Pitcher ... https://www.instagram.com/p/DVzVRbRDWrZ/
[17] MLB Props ML Model https://www.reddit.com/r/algobetting/comments/1dxv6e3/mlb_props_ml_model/
[18] The Creation of Predictive Stuff Metrics: Unveiling the pSTFERA Suite https://www.prospectslive.com/the-creation-of-predictive-stuff-metrics-introducing-the-pstfera-suite/
[19] Pybaseball: a Python library for working with baseball data ... - Reddit https://www.reddit.com/r/Python/comments/79l3o7/pybaseball_a_python_library_for_working_with/
[20] How Much Does Weather & the Ballpark Itself Impact MLB Betting? https://help.outlier.bet/en/articles/12313109-how-much-does-weather-the-ballpark-itself-impact-mlb-betting
[21] Forrest31/Baseball-Betting-Model: Predictive machine ... https://github.com/Forrest31/Baseball-Betting-Model
[22] ERA Estimators, Pt. II: Present - RotoGraphs Fantasy Baseball https://fantasy.fangraphs.com/era-estimators-pt-ii-present/
[23] An Introduction to PyBaseball: Using Python to Analyze Baseball Data https://www.tdabaseball.com/post/an-introduction-to-pybaseball-using-python-to-analyze-baseball-data
[24] MLB API documentation : r/mlbdata - Reddit https://www.reddit.com/r/mlbdata/comments/1s4ibq2/mlb_api_documentation/
[25] MLB Stats API - Two Circles Help Center https://help.koresoftware.com/hc/en-us/articles/6026343263511-MLB-Stats-API
[26] MLB Data API Developer Portal - SportsDataIO https://sportsdata.io/developers/api-documentation/mlb
[27] MLB Stats API - Acquire List of Free Agents, Acquire Minor League ... https://www.youtube.com/watch?v=w1M0uMVg6sM
[28] Endpoints · toddrob99/MLB-StatsAPI Wiki - GitHub https://github.com/toddrob99/MLB-StatsAPI/wiki/Endpoints/a084ce1309dc525cbf90390f1fe10a744c351a02
[29] wOBA, wRC, and wRC+ - Baseball Basics - YouTube https://www.youtube.com/watch?v=99uphIh2WfA
[30] Umpire Analytics – Society for American Baseball Research https://sabr.org/journal/article/umpire-analytics/
