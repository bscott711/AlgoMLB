# Module Specification: algomlb.domain

The `domain` module defines the fundamental entities and physical constants of the AlgoMLB universe. It is deliberately decoupled from any specific database or API implementation.

## 1. Core Models (`models.py`)

All domain models are immutable (frozen) Pydantic base models to ensure data integrity during processing.

### Major Entities

- **`Game`**: The central record for a baseball game.
    - Includes IDs for home/away teams and pitchers.
    - Tracks status (`SCHEDULED`, `IN_PROGRESS`, `COMPLETED`).
    - Handles doubleheaders via `doubleheader_num`.
- **`Odds`**: Represents a market snapshot from a sportsbook.
    - Fields: `sportsbook`, `market_type`, `price` (decimal), `point` (spread/total).
    - Properties: `implied_probability` and `american_odds` conversions.
- **`BankrollTransaction`**: Records for the internal financial ledger.
    - Tracks `stake`, `odds`, and `pnl` (Profit and Loss).
- **`Ballpark`**: Physical attributes of a stadium.
    - Fields: Dimensions (`left_field`, `center_field`, `right_field`), wall heights, elevation, and roof type.

---

## 2. Constants & Enums

### Strategic Enums
- `GameType`: `R` (Regular), `P` (Postseason), `S` (Spring), etc.
- `PlayerRole`: `PITCHER` or `BATTER`.
- `SurfaceType`: `GRASS` or `TURF`.
- `RoofType`: `OPEN`, `CLOSED`, or `RETRACTABLE`.

### Geographic & Organizational Data
- **`teams.py`**: Exports `TEAM_NAME_TO_ABB`, mapping full MLB team names (e.g., "New York Yankees") to internal 3-letter codes ("NYY").
- **`stadium_bearings.py`**: Exports `STADIUM_HP_BEARINGS`, providing the compass bearing from home plate toward center field for all 30 MLB stadiums. This is critical for wind calculations.

---

## 3. Wind Physics (`wind_physics.py`)

Calculates the impact of weather on gameplay based on stadium orientation.

### `WindComponents`
A NamedTuple consisting of:
- `headwind_mph`: Magnitude of wind along the home-plate-to-CF axis (>0 blowing OUT, <0 blowing IN).
- `crosswind_mph`: Magnitude of wind perpendicular to the axis (>0 left-to-right).

### `wind_components(...)`
Decomposes raw meteorological wind (Speed and Direction) into relative components based on the stadium's bearing.

### `circular_std(...)`
Computes the circular standard deviation of angles to measure wind stability throughout a game window.

## Engineering Notes
- **Unidirectional Logic**: Domain models must never import from `db`, `ingestion`, or `ml`.
- **Null Safety**: Optional fields have explicit defaults (usually `None`) to handle incomplete data from APIs gracefully.
