# Module Specification: CLI & UI Layers

The interface layer provides the human and machine control panels for the AlgoMLB system. It translates high-level intentions (e.g., "sync today's data") into complex orchestrator calls.

## 1. CLI Control Panel (`algomlb.cli`)

The CLI is built using **Typer**, providing a modular, sub-command-based interface. It also supports a global `--agent-mode` flag for automated interaction.

### Main Entry Point (`main.py`)
Root CLI definition that aggregates all sub-modules.


- Usage: `algomlb [subcommand]`

### Sub-Modules
- **`ingest.py`**: Granular control over all data pipelines.
  - `algomlb ingest statcast --start [DATE] --end [DATE]`
  - `algomlb ingest odds --live`
  - `algomlb ingest weather --days 7`
- **`process.py`**: Triggers the ML feature generation layers.
  - `algomlb process silver`: Summarizes raw pitches to player-game logs.
  - `algomlb process rolling`: Computes EMA/Trending features (Gold layer).
- **`ml.py`**: Model lifecycle management.
  - `algomlb ml train --model-type xgb --version v2`
  - `algomlb ml calibrate`: Refreshes xBA/xwOBA probability baselines.
- **`sync.py`**: High-level daily automation.
  - `algomlb sync daily`: The "one-button" command to fetch schedule, lineups, odds, and weather for the current day.
- **`db.py`**: Database maintenance.
  - `algomlb db check`: Runs introspection audits (NULL counts, table health).
  - `algomlb db migrate`: Applies Alembic migrations.

---

## 2. UI Dashboard (`algomlb.ui`)

The UI is a **Streamlit** application designed for interactive analysis and system monitoring.

### Architecture
- **`app.py`**: The central router for the multi-page Streamlit application.
- **`styles.py`**: Custom CSS injections (Glassmorphism, dark-mode themes) to ensure a premium visual experience.
- **Components**: Reusable UI elements (e.g., Team Selectors, Stat Cards).

### Key Views
1. **Player Performance Lab**:
    - **Spray Charts**: High-fidelity field geometry using polar coordinates.
    - **Sticky selection**: Persists player context across team/season changes.
2. **Database Health View**:
    - Visualization of the `SchemaInspector` reports.
    - Identification of missing Statcast or Gumbo data gaps.
3. **Market Pulse**:
    - Real-time comparison of model predictions against live moneyline odds.

## Engineering Notes
- **Agent Mode**: When `--agent-mode` is activated, the CLI emits the `AgentResult` Pydantic model as JSON to `stdout`. All logs are redirected to `stderr` or a file, allowing another agent (or CI/CD system) to parse the output cleanly.
- **Throttling**: The CLI commands frequently use `just` task aliases (defined in the root `Justfile`) to manage environment variables and pre-flight checks.
