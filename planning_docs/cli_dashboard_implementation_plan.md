# Implementation Plan: CLI & Dashboard (Tracer Bullet)

This plan outlines the steps to implement the foundational CLI and UI layers for AlgoMLB, following the "Import Ladder" architecture.

## 1. Foundation Layer (`algomlb.core`)

Lower-level utilities with zero internal dependencies.

- [ ] **Step 1.1: Standardize Dependencies**
    - Update `pyproject.toml` to include: `loguru`, `streamlit`, `optuna`, `optuna-dashboard`.
    - Run `uv sync` to update the lockfile.
- [ ] **Step 1.2: Implementation of `algomlb.core.logger`**
    - Create `src/algomlb/core/logger.py` using `loguru`.
    - Implement dual sinks: formatted terminal output (Human) and JSONL file (Machine).
- [ ] **Step 1.3: Implementation of `algomlb.core.agent_io`**
    - Create `src/algomlb/core/agent_io.py`.
    - Define `AgentResult` Pydantic model for structured CLI responses.
    - Implement `emit_agent_result()` for automated agent consumption.

## 2. CLI Architecture (`algomlb.cli`)

A unified control panel for human and automated agents.

- [ ] **Step 2.1: Entry Point Definition**
    - Update `pyproject.toml` `[project.scripts]` to map `algomlb` to `algomlb.cli.main:app`.
- [ ] **Step 2.2: CLI Root App (`main.py`)**
    - Create `src/algomlb/cli/main.py`.
    - Implement the root logic with the `--agent-mode` callback.
- [ ] **Step 2.3: Command Scaffolding (Stubs)**
    - Create stub command files: `db.py`, `ingest.py`, `ml.py`, `run.py`, `ui.py`.
    - Each should have a simple "TODO" echo but be correctly mounted in `main.py`.

## 3. UI Layer (`algomlb.ui`)

Production-grade dashboard for data visualization and model monitoring.

- [ ] **Step 3.1: Streamlit Entry Point (`app.py`)**
    - Create `src/algomlb/ui/app.py`.
    - Implement `st.navigation()` to wire up the dashboard structure.
- [ ] **Step 3.2: Page Scaffolding**
    - Create `src/algomlb/ui/pages/` and the four target files: `optuna.py`, `picks.py`, `bankroll.py`, `data.py`.
    - Implement basic layouts for each.
- [ ] **Step 3.3: CLI-UI Link**
    - Implement the `algomlb ui launch` command in `src/algomlb/cli/ui.py` to trigger `streamlit run`.

## 4. Verification & Guardrails

- [ ] **Step 4.1: Import Linter Update**
    - Update `.importlinter` to enforce the position of `algomlb.core` at the bottom of the ladder.
- [ ] **Step 4.2: Full Pipeline Run**
    - Execute `just verify` to ensure all new files meet linting, type-safety, and complexity standards.
    - Manual smoke test: `algomlb --help` and `algomlb --agent-mode db status`.
