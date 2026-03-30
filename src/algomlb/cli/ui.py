import optuna_dashboard
import typer
import subprocess
import sys
from algomlb.core.logger import logger
from algomlb.ui import APP_PATH

app = typer.Typer(help="Manage the Streamlit dashboard.", no_args_is_help=True)


@app.command()
def launch():
    """Start the Streamlit dashboard (port 8501)."""
    logger.info("Starting Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)])


# Dummy use for deptry
_ = optuna_dashboard.__version__
