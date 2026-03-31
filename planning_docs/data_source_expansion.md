# Data Source Expansion: Umpire Scorecards & Retrosheet

This plan outlines the integration of two new data sources to enrich the AlgoMLB feature set, specifically focusing on umpire tendencies and historical play-by-play.

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

### Value Proposition of Retrosheet
Provides granular "Official Record" metadata (errors, fielder assignments, assists) that is complementary to Statcast telemetry. Essential for training on historical data (pre-2015) and modeling outlier defensive/umpiring outcomes.

### Proposed Schema Details: `RetrosheetEventORM`
This table will store comprehensive play outcomes, including:

- **Game ID / ID**: Unique Retrosheet Game identifier.
- **Bat/Pit Identification**: Batter/Pitcher Retrosheet IDs.
- **Game State**: Inning, Outs (Pre/Post), Score (V/H), Baserunners (1st, 2nd, 3rd).
- **Play Result**: At-Bat (1/0), Hit Type (Single-HR), Outcome (ROE, FC, K, etc.).
- **Defensive Detail**: Specific fielders responsible for putouts, assists, and errors (E1-E9).
- **Umpire Map**: IDs for Home Plate, 1B, 2B, 3B, LF, RF umpires.

## 3. Ingestion Strategy
1. **Umpire Scorecards**:
    - Build `UmpireScorecardIngester` to fetch from `umpscorecards.us/data/games`.
    - Key challenge: Mapping Umpscorecard team names/dates to MLB Game PKs.
2. **Retrosheet**:
    - Utilize the [Chadwick tools](https://github.com/chadwickbureau/chadwick) or use pre-parsed CSV formats to load historical seasons.
    - Focus on adding the `RetrosheetEventORM` to store the detailed play list provided by the user.

## 4. Next Steps
1. **Add ORM Models** to `src/algomlb/db/models.py`.
2. **Implement CLI commands** for bulk ingestion.
3. **Establish Foreign Key links** between Statcast pitch data and Retrosheet events.
