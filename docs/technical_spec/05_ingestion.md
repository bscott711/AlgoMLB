# Module Specification: algomlb.ingestion

The `ingestion` module is a sophisticated data acquisition engine. It orchestrates multiple external APIs and data sources to maintain a high-fidelity mirror of the MLB universe.

## 1. Architectural Foundations

### Infrastructure (`http_client.py`)

All API-based ingesters inherit from `BaseAPIClient`, which provides:

- **Resilient Requests**: Standardized `_request()` method using `httpx`.
- **Retry Logic**: Exponential backoff via `tenacity` (min 2s, max 10s, 3 attempts).
- **Circuit Breaker**: A token-bucket implementation that "opens" after 5 failures to protect external resources and the system's reputation.

### The Orchestrator (`orchestrator.py`)
The `IngestionOrchestrator` is the top-level service that coordinates all ingestion tasks. It ensures that data is fetched in the correct order (e.g., schedule before lineups).

---

## 2. Specialized Ingesters

### A. Statcast Layer (`statcast_ingester.py`)
- **Source**: `pybaseball` (wrapping Statcast).
- **Strategy**: 7-day chunking to avoid server timeouts.
- **Filtering**: Strictly limited to Regular Season (`R`) and Postseason (`F`, `D`, `L`, `W`) games.
- **Normalization**: Maps diverse Statcast CSV columns to the `statcast_raw` table schema.

### B. Live Game Context (`gumbo_ingester.py` & `lineup_ingester.py`)
- **Gumbo**: Fetches the MLB "Gumbo" live feed to extract canonical wall-clock timestamps (`start_time`, `end_time`) for every pitch.
- **Lineups**: Extracts the starting 9 batters per team from the boxscore endpoint.
- **Resolution**: Both use `game_pk` as the primary anchor.

### C. Market Layer (`odds_api.py` & `historical_odds.py`)
- **Live Odds**: Polls The-Odds-API for real-time Moneyline, Runline, and Total markets.
- **Historical Backfill**: Fetches two snapshots per game: **Opening** (~24h before) and **Closing** (5m before first pitch).

### D. Environmental Layer (`openmeteo_ingester.py`)
- **Source**: Open-Meteo ERA5 (Archive) and Historical Forecast APIs.
- **Core Logic**: Fetches a 5-hour progression (T0 to T4) for each game.
- **Calculations**: Integrates with `domain.wind_physics` to compute headwind/crosswind components based on stadium bearing.
- **Market Surprise**: Calculates the delta between the T-24h forecast and the actual T0 (First Pitch) weather.

### E. Health & Umpire Layer
- **Transactions (`transactions_ingester.py`)**: Parses transaction text using regex to extract injury body parts and IL duration.
- **Umpires (`umpire_ingester.py`)**: Scrapes `umpscorecards.us` for accuracy, consistency, and run-favoritism metrics.

---

## 3. Data Integrity & Syncing

### Idempotency
All ingesters use **UPSERT** logic. Re-running an ingestion task for a date range that has already been processed will update existing records (if changed) rather than creating duplicates.

### Bulk Operations
Ingesters leverage the `DatabaseRepository` for high-performance bulk operations. Records are processed in chunks (typically 20-500) to respect database limits.

## Engineering Notes
- **API Respect**: Many ingesters include `time.sleep()` throttling to avoid hitting rate limits on free or low-tier API plans.
- **Caching**: `pybaseball` and `requests-cache` are used for large historical backfills to minimize redundant network traffic.
- **Coordinate Precision**: `openmeteo_ingester` uses a hardcoded manifest of MLB stadium coordinates (proven centers) rather than relying on API geocoding.
