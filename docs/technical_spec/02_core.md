# Module Specification: algomlb.core

The `core` module provides foundational utilities used by every other layer in the application. It focus on observability and structured communication.

## 1. Logging System (`logger.py`)

AlgoMLB uses `loguru` for high-performance, flexible logging. The system is configured with dual sinks to satisfy both human developers and automated monitoring tools.

### Sinks

#### A. Human-Friendly Console (stderr)

- **Level**: Configurable via `ALGOMLB_LOG_LEVEL` (default: `INFO`).
- **Format**: Colorized string highlighting time, level, module name, and message.
- **Filtering**: Automatically excludes internal "agent mode" output to keep the console clean.

#### B. Machine-Friendly File (JSONL)

- **Path**: `logs/algomlb.log` (or `ALGOMLB_LOG_DIR`).
- **Level**: Always `DEBUG`.
- **Serialization**: Every log entry is written as a single line of JSON.
- **Rotation**: Rotates every 20 MB.
- **Retention**: Kept for 14 days and compressed (.gz).
- **Concurrency**: Asynchronous (`enqueue=True`) to remain thread and process safe.

### Logging Usage

```python
from algomlb.core.logger import logger

logger.info("Starting data ingestion", year=2024)
logger.error("API request failed", error=str(e))
```

---

## 2. Agent I/O Protocol (`agent_io.py`)

This protocol standardizes the output for automated agents (like Antigravity or CI/CD pipelines) that consume AlgoMLB's terminal output.

### `AgentResult` Model
A Pydantic model that encapsulates the result of any significant operation.

- `status`: One of `"success"`, `"error"`, or `"warning"`.
- `command`: The name of the operation performed (e.g., `ingest.statcast`).
- `duration_ms`: Execution time for performance tracking.
- `data`: A dictionary for structured payload (e.g., number of rows inserted).
- `errors`/`warnings`: Lists of strings for granular feedback.

### `emit_agent_result(result: AgentResult)`
Formats the result as JSON and writes it to `stdout`, followed by a flush. This ensures the output is captured immediately by parent processes.

### Agent Result Usage

```python
from algomlb.core.agent_io import AgentResult, emit_agent_result

result = AgentResult(
    status="success",
    command="db.sync",
    data={"rows_updated": 42}
)
emit_agent_result(result)
```
