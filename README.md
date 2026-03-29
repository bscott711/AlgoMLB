# ⚾ AlgoMLB: Project Roadmap

## 🎯 Project Overview

A highly structured, configuration-driven baseball quantitative betting system. The system ingests sportsbook odds and MLB statistics, utilizes state-of-the-art machine learning to calculate true game probabilities, identifies **+EV (Positive Expected Value)** opportunities, manages a paper bankroll, and publishes plays via a Bluesky bot.

## 🏗️ Architecture & Development Standards

### Core Technologies

- **Language:** Python 3.12+
- **Package Management:** `uv`
- **Task Runner:** `just` (via `Justfile` for orchestrating formatting, linting, testing, and execution)
- **CLI Framework:** `typer` for building a robust, subcommand-driven command-line interface.
- **Configuration & Security:** `pydantic-settings` and `python-dotenv` for strict environment variable validation. Credentials and API keys must never be hardcoded or checked into version control.
- **Infrastructure:** Oracle Cloud Compute (Workers orchestrated via APScheduler) + SQL Database.

### Architecture & Design

- **Unidirectional Import Ladder:** Enforced by `import-linter`.
  - *Flow:* `algomlb.config -> algomlb.domain -> algomlb.db -> algomlb.ingestion -> algomlb.ml -> algomlb.strategy -> algomlb.execution -> algomlb.social -> algomlb.cli`
- **Facade Pattern:** Each module must expose its public API strictly through its `__init__.py`. Internal implementations must be hidden to maintain clean module boundaries.
- **Class-Oriented:** Use classes wherever possible to encapsulate state, logic, and dependencies.

### Coding Standards

- **Absolute Imports Only:** Relative imports (e.g., `from . import models`) are strictly forbidden. All imports must be absolute starting with the package root.
- **Code Quality:**
  - 100% type-hinted (`pyright` strict mode).
  - Formatted and linted strictly via the `ruff` universe.
  - Dead code eliminated via `vulture`.
  - Complexity monitored via `complexipy`.
- **Testing:** `pytest`, `pytest-cov`, `pytest-asyncio`. 100% coverage required for Domain, Strategy, and ML metric calculations.

---

## 🗺️ Phase Tracker

### 🛠️ Phase 0: Project Scaffolding & CI/CD

*Objective: Set up the ironclad development environment and enforce structural rules.*

- [ ] Initialize `uv` project and define `pyproject.toml`.
- [ ] Configure `[dependency-groups]` (`jitsu`, `import-linter`, `deptry`, `pyright`, `vulture`, `pytest`, `typer`, `python-dotenv`, `apscheduler`, etc.).
- [ ] Set up pre-commit hooks for `ruff`, `mdformat`, and `pymarkdownlnt`.
- [ ] Create `Justfile` as the primary developer interface and CI/CD task runner.
- [ ] Configure `import-linter` contracts to strictly forbid circular imports and enforce the unidirectional ladder.
- [ ] Configure `ruff` rules to explicitly flag and prevent relative imports (`TID252`).
- [ ] Configure `pyright` for strict type checking.
- [ ] Establish directory structure (`src/algomlb/config`, `src/algomlb/domain`, `src/algomlb/db`, `src/algomlb/ingestion`, `src/algomlb/ml`, `src/algomlb/strategy`, `src/algomlb/execution`, `src/algomlb/social`, `src/algomlb/cli`).

### 🗄️ Phase 1: Domain, Configuration, & Persistence

*Objective: Build the central data structures that all other layers will rely on.*

- [ ] **Config:** Create pydantic `BaseSettings` utilizing `python-dotenv` to securely load DB credentials, API keys, and ML thresholds. Expose via `algomlb.config`.
- [ ] **Domain:** Define pure pydantic (v2) models (`Game`, `Odds`, `PitcherStats`, `PendingBet`, `SettledBet`). Expose via `algomlb.domain`.
- [ ] **CLI Entrypoint:** Set up the base `typer` application in `src/algomlb/cli/__init__.py` to serve as the orchestrator.
- [ ] **Database:** Provision Oracle Cloud SQL database.
- [ ] **ORM:** Create SQLAlchemy (2.0) declarative models mapping exactly to Domain models. Must include specific schema separation:
  - **HistoricalData**: Long-term storage for past seasons and baseline model training.
  - **LiveOdds**: Volatile table for daily odds pulls and active games.
  - **GradedGames**: Storage for completed/settled games migrated from the live table.
  - **BankrollLedger**: Persistent state for the paper bankroll, pending bets, and transaction history (survives instance restarts).
- [ ] **Migrations:** Set up `alembic` for database schema migrations.
- [ ] **Repository Pattern:** Implement database access interfaces to decouple SQLAlchemy from higher layers. Expose via `algomlb.db`.

### 🕷️ Phase 2: Ingestion Engine (API-First, Scraper-Fallback)

*Objective: Reliably acquire raw data via robust APIs (primary) and resilient scrapers (fallback).*

- [ ] **API Clients:** Build resilient `httpx` base clients with rate-limit handling, retry logic, and timeout management.
- [ ] **Stats Ingestion (Primary):** Integrate the official MLB Stats API for historical and daily statistics (pitching, hitting, weather, lineups).
- [ ] **Odds Ingestion (Primary):** Integrate The-Odds-API for live sportsbook odds. Execute daily to populate the `LiveOdds` table.
- [ ] **Scraper Fallback:** Build Playwright/httpx web scrapers only to be used if primary APIs fail or lack specific granular data.
- [ ] **ETL Pipeline:** Build the logic to validate ingested JSON/HTML against Domain models and load it into the Database via the Repository layer.
- [ ] **Testing:** Unit tests for API/scraper parsing logic and integration tests for database insertion.

### 🧠 Phase 3: Machine Learning & Backtesting (The Quant Layer)

*Objective: State-of-the-art probability modeling with zero data leakage.*

- [ ] **Feature Engineering:** Build pipelines for rolling averages, park factors, and pitcher/batter splits. Handle start-of-season cold starts.
- [ ] **Baseline Model:** Implement a simple, interpretable baseline (e.g., Logistic Regression on run differentials or Elo ratings) to establish a performance floor.
- [ ] **Backtest Engine:** Implement a strict, point-in-time temporal backtesting framework.
  - ⚠️ *Critical:* Computationally guarantee day $T$ predictions only use $\leq T-1$ data.
- [ ] **Hyperparameter Tuning:** Integrate `optuna` using time-series cross-validation (e.g., expanding window) to prevent look-ahead bias.
- [ ] **Model Training:** Train the primary probability model (XGBoost/LightGBM) and ensure it decisively beats the Baseline Model.
- [ ] **Evaluation:** Implement Closing Line Value (CLV) calculation to measure predictive edge against the market. Track CLV for every single game on the board.

### 📈 Phase 4: Strategy & Live Execution

*Objective: Sizing bets, managing risk, and maintaining the persistent bankroll.*

- [ ] **Strategy Logic:** Implement +EV detection (comparing ML probability vs. Ingestion odds).
- [ ] **Bet Sizing:** Implement the Kelly Criterion (full and fractional) for optimal risk management.
- [ ] **Bankroll Management:** Build the state manager to interface with the `BankrollLedger` database table, ensuring safe deductions and payouts.
- [ ] **Live Tracker Orchestration:** Implement APScheduler embedded within the `typer` CLI to manage continuous polling loops, comparing live odds to ML projections and triggering execution seamlessly.
- [ ] **Testing:** E2E testing simulating a full day of games, odds movement, and bet settlement.

### 🚀 Phase 5: Social Integration & Deployment

*Objective: Push the bot live and publish results.*

- [ ] **Bluesky Client:** Implement the `atproto` integration for the `BlueskyClient` class.
- [ ] **Alert Formatting:** Design clean, readable markdown/text templates for +EV play alerts (e.g., Team, Odds, Edge, Risk).
- [ ] **Settlement Loop & Data Migration:** Create an APScheduler background job that:
  - Checks game results and final scores via MLB Stats API.
  - Settles pending bets in the `BankrollLedger`.
  - Moves the completed games from the `LiveOdds` table into the `GradedGames` table.
  - Calculates daily ROI and market-wide CLV.
- [ ] **Deployment:** Containerize the application (Docker) and deploy to Oracle Cloud Compute instances.
- [ ] **Monitoring:** Set up basic alerting for API rate limits, scraper fallbacks, or database connection drops.
