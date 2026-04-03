# Implementation Plan - Epic 2: Historical Player Transactions & IL Tracking

Build an isolated pipeline for tracking historical player IL stints to enable game-day roster health features.

## 1. Database Schema & Migration

### 1.1 Create `player_transactions` table

* **Table Name**: `player_transactions`
* **Columns**:
  * `transaction_id` (TEXT, PRIMARY KEY) - Corresponds to StatsAPI `id`.
  * `player_id` (INTEGER, NOT NULL) - Corresponds to `person.id`.
  * `team_id` (INTEGER, NOT NULL) - Corresponds to `toTeam.id`.
  * `transaction_date` (DATE, NOT NULL) - Corresponds to `date`.
  * `effective_date` (DATE) - Corresponds to `effectiveDate`.
  * `resolution_date` (DATE) - Corresponds to `resolutionDate`.
  * `type_desc` (TEXT, NOT NULL) - Corresponds to `typeDesc`.
  * `il_type` (TEXT) - '10day' | '60day' (if applicable).
  * `injury_body_part` (TEXT) - Extracted via `parse_injury`.
  * `injury_descriptor` (TEXT) - Extracted via `parse_injury`.
  * `raw_description` (TEXT) - Corresponds to `description`.
  * `days_on_il` (INTEGER, GENERATED ALWAYS AS (resolution_date - effective_date) STORED) - Using PostgreSQL computed columns.

> [!NOTE]
> Formal `REFERENCES` to `players` and `teams` will be omitted for now as these tables do not yet exist in the current project, but IDs will match other project tables.

### 1.2 Create `v_game_il_features` view

* **Base Table**: `game_results` (mapped as `g`).
* **Columns**:
  * `game_pk`, `game_date`, `home_team_id`, `away_team_id`.
  * `home_sp_il_stints_ytd`: Count of IL stints for the home starting pitcher as of `game_date`.
  * `home_sp_days_since_il_return`: Days since the last IL return for the home starting pitcher.
  * `home_team_active_il_count`: Current roster health.

## 2. ORM Expansion

### 2.1 Update `src/algomlb/db/models.py`

Add `PlayerTransactionORM` with appropriate fields and SQLAlchemy `Computed` for `days_on_il`.

### 2.2 Update `src/algomlb/db/repository.py`

Add `save_player_transaction(s)` method to persist transaction data.

## 3. Ingestion Pipeline

### 3.1 Create `src/algomlb/ingestion/transactions_ingester.py`

Implement the `PlayerTransactionsIngester` class with provided functions:

* `parse_injury()`: Regex-based extraction of body parts and descriptors.
* `monthly_date_chunks()`: Batching API calls.
* `fetch_transactions()`: Fetching data from StatsAPI (2019-present).
* `fetch_legacy_transactions()`: Fetching data from legacy BAM endpoint (2005-2018).

### 3.2 Update `src/algomlb/ingestion/orchestrator.py`

Add `run_transaction_ingestion` to facilitate backfilling and periodic ingestion.

## 4. Verification & Testing

### 4.1 Unit Tests (`tests/unit/test_transactions_ingester.py`)

* `parse_injury()`: Verify correct extraction for various description formats.
* `monthly_date_chunks()`: Verify chunk counts and boundary conditions.
* `test_ingest()`: Mock API integration to verify ORM mapping.

### 4.2 Script Verification

* Confirm Postgres generated column behavior.
* Execute `just verify` to ensure no regressions.
