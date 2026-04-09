# Module Specification: algomlb.db

The `db` module handles all persistence logic using SQLAlchemy 2.0. It follows a clean separation of concerns: `session` handles connectivity, `models` defines the schema, and `repository` encapsulates all I/O operations.

## 1. Database Session Management (`session.py`)

AlgoMLB supports both PostgreSQL (primary) and SQLite (for local testing/CI).

- **`Base` Class**: The modern SQLAlchemy `DeclarativeBase` from which all ORM models inherit.
- **`get_engine()`**: Singleton factory for the SQLAlchemy `Engine`.
    - Automatically configures connection pooling for PostgreSQL (default `pool_size=5`).
    - Disables pooling for SQLite to prevent locking issues.
- **`get_session_factory()`**: Singleton `sessionmaker` configured with:
    - `autoflush=False`
    - `autocommit=False`
    - `expire_on_commit=False` (to keep objects usable after a commit).

---

## 2. ORM Models (`models.py`)

This file contains the declarative mappings for all tables. It uses Type-Annotated `Mapped` and `mapped_column` syntax.

### Key Tables
- **`GameResultORM`**: Primary storage for MLB games, scores, and venue metadata.
- **`StatcastRawORM`**: The high-volume "Source of Truth" for pitch-level data.
- **`PitchEventORM`**: Enriched pitch data used for modeling.
- **`PlayerRollingFeaturesORM`**: The "Gold Layer" containing pre-computed rolling averages and momentum trends.
- **`BallparkORM`**: Permanent storage for stadium coordinates, bearings, and dimensions.
- **`UmpireScorecardORM`**: Granular umpire reliability metrics.
- **`OpenMeteoWeatherProgressionORM`**: Hourly weather progression (T0-T4) for every game.

---

## 3. The Repository Pattern (`repository.py`)

The `DatabaseRepository` is the **only** class that should interact with SQLAlchemy sessions at the application layer. It acts as a mediator:
- **Input**: Receives Domain objects or lists of ORM objects.
- **Logic**: Handles persistence specifics (e.g., bulk UPSERTs, resolving relationships).
- **Output**: Returns Domain objects (Pydantic models) where applicable.

### Critical Methods
- `save_game(game: Game)`: Upserts a game and intelligently resolves its `ballpark_id`.
- `save_live_odds(odds: Odds)`: Saves market snapshots.
- `save_statcast_raw(rows: List[dict])`: Optimized bulk PostgreSQL UPSERT for pitch data.
- `get_bankroll_balance()`: Computes total PnL from the ledger.

---

## 4. Database Health (`introspection.py`)

Provides a `SchemaInspector` service to audit the live database.

- **`list_tables()`**: Returns row counts for all user tables.
- **`column_report(table)`**: Generates a detailed audit of a table's columns, including the percentage of NULL values. Used to detect ingestion gaps.
- **`foreign_keys()`**: Maps out the relationships between tables for visualization.

---

## 5. Migrations (Alembic)

Schema changes are managed via Alembic.
- **Location**: `migrations/versions/`.
- **Naming**: Every migration has a unique hash and a descriptive slug.
- **State**: `alembic_version` table in the database tracks the current head.

### Strategy
1. Generate migration: `just db.migrate "description"`
2. Apply migration: `just db.upgrade`
3. Downgrade (if needed): `just db.downgrade -1`

## Engineering Notes
- **Upsert-First**: All saving methods in the repository should use UPSERT logic (Postgres `ON CONFLICT DO UPDATE`) to ensure idempotency.
- **Chunking**: Bulk saves (Statcast, Lineups) must use chunked execution to avoid SQL variable limit errors.
