## The Import Ladder Position

`algomlb.core` must sit **below `algomlb.config`** in the import hierarchy. The logger cannot import from config (circular dependency). Instead, it reads its own environment directly via `os.environ` or a dedicated `LogConfig` that is *not* the main settings object. The ladder looks like this:

```
algomlb.core ← logger, agent result types, base exceptions (NO deps on other algomlb modules)
algomlb.config ← Settings (can import from core)
algomlb.db ← depends on config
algomlb.ingest ← depends on config, db
algomlb.ml ← depends on config, db, ingest
algomlb.cli ← depends on everything (top of ladder)
algomlb.ui ← isolated; depends on config, db (Streamlit entry point)
```

Add `algomlb.core -> algomlb.config` as a forbidden import in `.importlinter`.[1]

***

## The Logger (`algomlb.core.logger`)

Loguru's `serialize=True` on the file sink produces valid JSONL natively — no custom formatter needed. The key is initializing once at the package level and making the singleton importable everywhere.[2]

```python
# src/algomlb/core/logger.py
import os
import sys
from pathlib import Path
from loguru import logger as _logger

LOG_DIR = Path(os.getenv("ALGOMLB_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remove default sink
_logger.remove()

# Human sink — rich formatting, no JSONL
_logger.add(
sys.stderr,
level=os.getenv("ALGOMLB_LOG_LEVEL", "INFO"),
format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
colorize=True,
filter=lambda r: not r["extra"].get("agent_mode", False),
)

# File sink — JSONL, rotating 20MB, 14-day retention
_logger.add(
LOG_DIR / "algomlb.log",
level="DEBUG",
serialize=True, # native JSONL via loguru
rotation="20 MB",
retention="14 days",
compression="gz",
enqueue=True, # multiprocess-safe for APScheduler
)

# Re-export as `logger` — every module does: from algomlb.core.logger import logger
logger = _logger.bind()
```

Every module imports `from algomlb.core.logger import logger`. No configuration boilerplate at call sites.[3]

***

## The CLI Skeleton (`algomlb/cli/`)

The key design decision is the **`--agent-mode` callback** on the root app, stored in `typer.Context` and threaded to subcommands. This cleanly gates all console output globally.[1]

```
src/algomlb/cli/
├── __init__.py
├── main.py ← root app, --agent-mode callback, mounts sub-apps
├── db.py ← algomlb db init / migrate / status
├── ingest.py ← algomlb ingest historical / odds / schedule
├── ml.py ← algomlb ml tune / backtest / evaluate
├── run.py ← algomlb run (APScheduler live loop)
└── ui.py ← algomlb ui (Streamlit launcher)
```

```python
# src/algomlb/cli/main.py
import typer
from typing import Optional
from algomlb.cli import db, ingest, ml, run, ui

app = typer.Typer(
name="algomlb",
help="AlgoMLB — MLB analytics and prediction engine.",
no_args_is_help=True,
)

app.add_typer(db.app, name="db")
app.add_typer(ingest.app, name="ingest")
app.add_typer(ml.app, name="ml")
app.add_typer(run.app, name="run")
app.add_typer(ui.app, name="ui")

@app.callback()
def root_callback(
ctx: typer.Context,
agent_mode: bool = typer.Option(
False, "--agent-mode", help="Suppress human output; emit structured JSON to stdout."
),
):
"""AlgoMLB control panel."""
ctx.ensure_object(dict)
ctx.obj["agent_mode"] = agent_mode
```

***

## The Agent Feedback System

This is where I'd push Agent Green's proposal further. Rather than an ad-hoc dict, define a **typed `AgentResult` Pydantic model** in `algomlb.core`. Every CLI command returns one, and the `--agent-mode` serializer knows exactly what to emit.

```python
# src/algomlb/core/agent_io.py
import json, time, sys
from typing import Any
from pydantic import BaseModel, Field

class AgentResult(BaseModel):
status: str # "success" | "error" | "warning"
command: str # e.g. "ingest.odds"
duration_ms: int = 0
dict[str, Any] = Field(default_factory=dict)
errors: list[str] = Field(default_factory=list)
warnings: list[str] = Field(default_factory=list)

def emit_agent_result(result: AgentResult) -> None:
"""Write structured JSON to stdout and exit. Called only in --agent-mode."""
sys.stdout.write(result.model_dump_json() + "\n")
sys.stdout.flush()
```

A command implementation looks like:

```python
# Inside algomlb/cli/ingest.py
@app.command()
def odds(ctx: typer.Context):
"""Fetch latest odds from The-Odds-API."""
t0 = time.monotonic()
agent_mode = ctx.obj.get("agent_mode", False)
try:
records = OddsIngester().run()
result = AgentResult(
status="success", command="ingest.odds",
duration_ms=int((time.monotonic() - t0) * 1000),
data={"records_inserted": len(records)},
)
except Exception as e:
result = AgentResult(
status="error", command="ingest.odds",
errors=[str(e)],
)
if agent_mode:
emit_agent_result(result)
else:
logger.success(f"Ingested {result.data.get('records_inserted', 0)} odds records.")
```

This gives Jitsu agents a perfectly parseable JSON dict on every run.

***

## The Streamlit Dashboard

Use Streamlit's **`st.navigation()` API** (1.29+) for multi-page structure — cleaner than the old `pages/` folder convention and lets `algomlb ui` point to a single entrypoint.[4]

```
src/algomlb/ui/
├── app.py ← entry point (streamlit run target)
├── pages/
│ ├── optuna.py ← Hyperparameter studies + Optuna plots
│ ├── picks.py ← Today's live model picks
│ ├── bankroll.py ← PnL, CLV drift, open bets
│ └── data.py ← Ingest health, DB row counts, pipeline status
└── components/
└── filters.py ← shared season/market/date filter widgets
```

The four dashboard tabs and what each owns:

| Tab | Key Content | Data Source |
|---|---|---|
| **Optuna Studies** | Slice plots, param importance, parallel coords, best trial per market | `optuna.load_study()` from Postgres storage |
| **Live Picks** | Today's scheduled games, model predictions, EV %, Kelly fractions | `ml.predictions` table |
| **Bankroll / CLV** | PnL curve, open bets, settled CLV by market, ROI rolling 7/30d | `bankroll_ledger` table |
| **Data Health** | Ingest timestamps, row counts, missing data flags, API quota | `db.status` views |

For Optuna, the native `optuna.visualization` module renders Plotly figures directly, so Streamlit just needs `st.plotly_chart(fig)`. No screen printing, no CLI noise.[5]

```python
# src/algomlb/ui/pages/optuna.py
import streamlit as st
import optuna
from optuna.visualization import plot_param_importances, plot_slice

@st.cache_resource(ttl=300) # refresh every 5 min
def load_study(market: str):
return optuna.load_study(
study_name=f"algomlb_{market}",
storage="postgresql://...",
)

study = load_study(st.selectbox("Market", ["moneyline", "total", "runline"]))
st.plotly_chart(plot_param_importances(study), use_container_width=True)
st.plotly_chart(plot_slice(study), use_container_width=True)
```

The `algomlb ui` CLI command simply shells out:

```python
# src/algomlb/cli/ui.py
import subprocess, sys
from pathlib import Path

@app.command()
def launch():
"""Start the Streamlit dashboard (port 8501)."""
ui_path = Path(__file__).parent.parent / "ui" / "app.py"
subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])
```

VS Code's Remote-SSH tunnel handles the port-forward to 8501 automatically, so you just open `localhost:8501` on your Mac.[4]

***

## Concrete Next Steps (in order)

1. **`pyproject.toml`** — add `loguru`, `streamlit`, `optuna`, `optuna-dashboard` to `[project.dependencies]`
2. **`.importlinter`** — add `algomlb.core` layer at the bottom, forbid imports back from `config`/`db`/etc. into `core`
3. **`algomlb/core/__init__.py`**, **`logger.py`**, **`agent_io.py`** — the foundation layer
4. **`pyproject.toml` `[project.scripts]`** — wire `algomlb = "algomlb.cli.main:app"` as the entry point
5. **CLI scaffold** — `main.py` + all sub-apps as empty stubs with `typer.echo("TODO")` bodies so agents have real targets immediately
6. **Streamlit skeleton** — `app.py` with `st.navigation()` wired to four empty page files

This ordering means after step 5, Jitsu agents can immediately start filling in subcommand bodies — they have the steering wheel before any real domain logic exists, which is exactly the point of the Tracer Bullet approach.

Sources
[1] Nested SubCommands - Typer https://typer.tiangolo.com/tutorial/subcommands/nested-subcommands/
[2] Delgan/loguru: Python logging made (stupidly) simple - GitHub https://github.com/Delgan/loguru
[3] A Complete Guide to Logging in Python with Loguru - Better Stack https://betterstack.com/community/guides/logging/loguru/
[4] 25+ Streamlit Tricks from a Co-founder Dashboard App - YouTube https://www.youtube.com/watch?v=pAPEP0j73QE
[5] optuna_dashboard.streamlit.render_objective_form_widgets https://optuna-dashboard.readthedocs.io/en/latest/_generated/optuna_dashboard.streamlit.render_objective_form_widgets.html
[6] Production-Grade Python Logging Made Easier with Loguru - Dash0 https://www.dash0.com/guides/python-logging-with-loguru
[7] python-logging-best-practices | Skil... - LobeHub https://lobehub.com/pt-BR/skills/neversight-skills_feed-python-logging-best-practices
[8] How to Use Loguru for Simpler Python Logging - Real Python https://realpython.com/python-loguru/
[9] Advice on logging libraries: Logfire, Loguru, or just Python's built-in ... https://www.reddit.com/r/Python/comments/1o4uyrv/advice_on_logging_libraries_logfire_loguru_or/
[10] How to Create a Python CLI Installable with Pip: Run Commands ... https://www.pythontutorials.net/blog/how-to-create-a-cli-in-python-that-can-be-installed-with-pip/
[11] SubCommands in a Single File - Typer https://typer.tiangolo.com/tutorial/subcommands/single-file/
[12] GitHub - algorandfoundation/algokit-cli: The Algorand AlgoKit CLI is the one-stop shop tool for developers building on the Algorand network. https://github.com/algorandfoundation/algokit-cli
[13] Implementing Structured Logging in Python: A Comprehensive Guide https://www.graphapp.ai/blog/implementing-structured-logging-in-python-a-comprehensive-guide
[14] How can I put typer commands in separate modules without ... https://stackoverflow.com/questions/73140193/how-can-i-put-typer-commands-in-separate-modules-without-becoming-sub-commands
[15] Best Practices! - Using Streamlit https://discuss.streamlit.io/t/streamlit-best-practices/57921
