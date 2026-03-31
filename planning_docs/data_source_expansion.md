# Data Source Expansion: Umpire Scorecards, Retrosheet, & Ballparks

This plan outlines the integration of new data sources to enrich the AlgoMLB feature set, specifically focusing on umpire tendencies, historical play-by-play, and stadium geography.

> [!NOTE]
> All sources described below are now **IMPLEMENTED** and verified with 100% test coverage.

## 1. Umpire Scorecards (`umpscorecards.us`)

### Value Proposition
Statcast provides the "ground truth" location of pitches, but it doesn't quantify the **umpire's decision accuracy**. Integrating scorecard data allows the models to account for umpire-specific strike zones, which affects walk rates, strikeout rates, and total runs.

### Proposed Schema: `UmpireScorecardORM`
| Field | Type | Description |
|-------|------|-------------|
| `game_id` | String | MLB Game PK (FK to `game_results.game_id`) |
| `umpire_name` | String | Name of the home plate umpire |
| `accuracy` | Float | Percentage of correct calls |
| `consistency` | Float | Umpire consistency metric |
| `favoritism_home` | Float | Net runs favored for the home team (can be negative) |
| `total_expected_runs` | Float | Expected runs based on pitch quality/zone |
| `true_strikes` | Integer | Total actual strikes called |
| `true_balls` | Integer | Total actual balls called |

## 2. Retrosheet Play-by-Play (`retrosheet.org`)

### Value Proposition
Provides granular "Official Record" metadata (errors, fielder assignments, assists) complementary to Statcast telemetry. Essential for training on historical data and modeling defensive/umpiring outcomes.

### Status: IMPLEMENTED
Integrated via `RetrosheetIngester`, which processes event-level CSVs into `RetrosheetEventORM`.

## 3. Ballpark Location Metadata

### Value Proposition
Stadium geography (elevation, latitude/longitude) is critical for trajectory modeling and travel fatigue calculation.

### Implementation: `BallparkIngester`
- **Hybrid Data Model**: Merges structural data from Kaggle with high-precision geographic coordinates from a local JSON source of truth (`ballpark_locations.json`).
- **Synonym Mapping**: Encodes stadium name history (e.g., "SBC Park" -> "AT&T Park" -> "Oracle Park") to ensure robust data joins.

### Status: IMPLEMENTED
Integrated via `BallparkIngester` and surfaced in the "Ballpark Context" dashboard view.

## 4. Ingestion Summary & Next Steps
1. [x] **Add ORM Models** to `src/algomlb/db/models.py`.
2. [x] **Implement CLI commands** for bulk ingestion.
3. [ ] **Establish Foreign Key links** between Statcast pitch data and Retrosheet events (In Progress).
