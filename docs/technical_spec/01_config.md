# Module Specification: algomlb.config

The `config` module is the single source of truth for application settings. it uses `pydantic-settings` to aggregate and validate configuration from multiple sources.

## Core Components

### `Settings` Class
The root container for all sub-configuration groups. It orchestrates the loading and merging of settings.

- **Environment**: Controls the execution mode (e.g., `development`, `production`).
- **Sub-Configs**:
    - `database`: `DatabaseConfig`
    - `api`: `APIConfig`
    - `ml`: `MLConfig`
    - `db_health`: `DBHealthConfig`

### Configuration Sources
Settings are loaded in the following order of priority (highest priority first):
1. **Initialization Settings**: Passed directly to the `Settings()` constructor.
2. **Environment Variables**: Prefixed with `ALGOMLB__` (e.g., `ALGOMLB__DATABASE__URL`). Nested fields use a double underscore `__`.
3. **`.env` File**: Standard environment variable file.
4. **`config.yaml`**: Hierarchical YAML configuration file.
5. **Default Values**: Defined in the Pydantic models.

## Sub-Configuration Details

### 1. `DatabaseConfig`
Manages connections to PostgreSQL and SQLAlchemy settings.
- `url`: Full SQLAlchemy connection string.
- `host`, `port`, `user`, `password`, `name`: Individual components (synced from `url` if provided).
- `echo`: Boolean to enable SQL statement logging.
- `pool_size`: Number of connections in the pool.

### 2. `APIConfig`
Stores credentials for external services.
- `odds_api_key`: Required key for The-Odds-API.
- `mlb_stats_url`: Base URL for the official MLB Stats API.

### 3. `MLConfig`
Defines hyper-parameters and thresholds for the Quant/Silver/Rolling layers.
- `min_edge_percent`: Minimum Expected Value needed to flag a bet.
- `max_risk_units`: Safety limit for bankroll exposure.
- `quant_baseline_window`: Lookback period for baseline averages (default: 14 days).
- `quant_pitcher_shrinkage_k`: Bayesian shrinkage for pitchers (default: 75).
- `quant_batter_shrinkage_k`: Bayesian shrinkage for batters (default: 250).

### 4. `DBHealthConfig`
Used by the introspection tools to validate database integrity.
- `allow_null_columns`: Map of tables to columns where NULL is acceptable.
- `known_placeholders`: Tables expected to be empty.
- `table_naming_pattern`: Regex for enforcing naming conventions (default: `^[a-z][a-z0-9_]*$`).

## Usage Patterns

### Retrieving Settings
Always use the `get_settings()` singleton factory to ensure configuration is loaded and validated exactly once.

```python
from algomlb.config import get_settings

settings = get_settings()
print(settings.database.url)
```

### Validation
All configurations are validated at startup. If a required field (like `odds_api_key`) is missing or a value is out of range (like `min_edge_percent < 0`), a `PydanticValidationError` is raised immediately, preventing the system from running in an unstable state.
