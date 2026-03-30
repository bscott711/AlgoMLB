import optuna_dashboard
import typer
import subprocess
import sys
from pathlib import Path
from algomlb.core.logger import logger

app = typer.Typer(help="Manage the Streamlit dashboard.", no_args_is_help=True)


@app.command()
def launch():
    """Start the Streamlit dashboard (port 8501)."""
    ui_path = Path(__file__).parent.parent / "ui" / "app.py"
    logger.info("Starting Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])


# Dummy use for deptry
_ = optuna_dashboard.__version__
