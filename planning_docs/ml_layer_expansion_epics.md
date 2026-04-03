# AlgoMLB ML Layer Expansion: Epic Implementation Plan

This document breaks down the ML feature expansion into **atomic, executable Epics**, ordered by feasibility and alpha potential.

Each epic strictly adheres to the **`just verify` framework**, meaning the completion of every epic must satisfy:

- `ruff` formatting and linting
- `pyright` strict type checking
- `deptry` & `vulture` dead-code analysis
- `lint-imports` architectural boundary checks
- `complexipy` cyclomatic complexity guards
- **`pytest` 100% unit test coverage (`--cov-fail-under=100`) on all new source layer components**

**Database Strategy:** Models will be mapped Epic-by-Epic to maintain agility. Each table will serve a strict, single purpose to avoid disrupting existing tables. StatsAPI 1st pitch weather will remain bundled with the main Game ingestion where it organically resides, while supplementary sources like Open-Meteo will receive their own tightly-scoped tables linking back to the Game ID.

---

## 🟢 TIER 1 — Build First (High Alpha, High Feasibility)

### Epic 1: Venue Wind Dynamics (Home Plate Bearing) [COMPLETED]

**Goal:** Implement the physical bearing of each stadium to decompose crude wind data into predictive `headwind` and `crosswind` vectors.

1. **Schema Update:** Add `hp_bearing_deg NUMERIC(5,2)` to the `venues` table. Generate an Alembic migration (`just revision message="add_hp_bearing_to_venues"`).
2. **Static Mapping:** Create a domain mapper in `src/algomlb/domain/stadium_bearings.py` containing the 30 stadium bearings.
3. **Core Logic:** Write pure Python physics functions `wind_components(wind_speed, wind_dir, hp_bearing)` in the domain/ML layer.
4. **Verification:** Write unit tests for trigonometry logic verifying headwind/crosswind conversions. Run `just verify`.

### Epic 2: Historical Player Transactions & IL Tracking [COMPLETED]

**Goal:** Build a completely isolated pipeline for tracking historical player IL stints and roster health.

1. **Schema Update:** Create a standalone `player_transactions` table with a computed `days_on_il` metric.
2. **Ingestion Layer:** Implement a monthly chunking pipeline for `StatsAPI /api/v1/transactions`, including regex-based extraction of `injury_body_part` and `injury_descriptor`.
3. **Verification:** Extensive unit tests to confirm robust pattern extraction of body parts from text descriptions. Run `just verify`.

### Epic 3: Open-Meteo Weather Progression & Forecast Surprise [COMPLETED]

**Goal:** Map the environmental arc from 1st to 9th inning (T0-T4) and extract market-surprise signals from opening odds, using Open-Meteo as the single source of truth for the trajectory.

1. **Schema Update:** Create a strictly isolated `openmeteo_weather_progression` table containing the T0-T4 and T-24hr forecast columns, with a foreign key back to the `game_id`. This prevents disrupting the existing StatsAPI game weather fields.
2. **Ingestion Expansion:** Utilize the Open-Meteo client to pull hourly windows (+0h through +4h) alongside the ERA5 / Forecast models.
3. **Aggregation Logic:** Connect Epic 1's `wind_components` logic over the T0-T4 arrays to derive instability flags and variance indices (`delta_headwind_mph`, `temp_drop_gt_10f`).
4. **Verification:** Mock multipoint array responses and verify aggregations. Run `just verify`.

### Epic 4: Contextual Situational Splits & Streaks

**Goal:** Offload computationally expensive situational processing to the StatsAPI `sitCodes` engine.

1. **Schema Updates:** Create caching tables for `player_sit_splits`, `player_streaks`, and `team_standings_snapshots` mapped to `date` logic (enforcing point-in-time constraints).
2. **Ingestion Pipelines:**

   - Pull `vl/vr` and `l7/l14/l30` splits.
   - Daily standing snapshots extraction.
   - Active streak pipelines.
3. **Integration Constraint:** Build logical guards ensuring that queries never pull splits/streaks updated *after* game time.
4. **Verification:** Test snapshot parsing logic to confirm no data leakage. Run `just verify`.

---

## 🟡 TIER 2 — Build Second (Medium Lift, Strong Supporting Signal)

### Epic 5: Leverage Modeling & Game Pace

**Goal:** Profile reliever competence in high-leverage scenarios and map systemic bullpen exhaustion via game durations/doubleheaders.

1. **Schema Updates:** Add 30-day trailing metric columns for bullpen performance and `gamePace` into a newly tailored or existing team statistics table.
2. **Ingestion Layer:** Process `game_winProbability` for play-by-play leverage indexes and `gamePace` endpoints for duration/pitch counts.
3. **Verification:** Mock complex trailing aggregations with boundary condition inputs. Run `just verify`.

### Epic 6: Deep Organization Context (Play-by-Play & Coaching)

**Goal:** Scrape baserunner state decisions (stolen bases, subs) and point-in-time coaching staff.

1. **Schema Updates:** Create tables/columns for coaching transitions (`pitching_coach_id`) and derived play-by-play team habits (SB aggression, pull rates).
2. **Ingestion Pipelines:** Parse `game_playByPlay` for baserunner states and pull `team_coaches` history per date. Incorporate true `attendance` counts.
3. **Verification:** Construct synthetic data tests for coaching changes and crowd density math. Run `just verify`.
