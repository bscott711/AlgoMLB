import optuna_dashboard
import typer
import subprocess
import sys
from algomlb.core.logger import logger
from algomlb.ui import APP_PATH

app = typer.Typer(help="Manage the Streamlit dashboard.", no_args_is_help=True)


@app.command()
def launch(
    port: int = typer.Option(
        8502, "--port", help="Port to run the Streamlit dashboard on."
    ),
    host: str = typer.Option(
        "0.0.0.0", "--host", help="Host address to run the Streamlit dashboard on."
    ),
    headless: bool = typer.Option(
        True, "--headless/--no-headless", help="Run Streamlit in headless mode."
    ),
):
    """Start the Streamlit dashboard."""
    logger.info(
        f"Starting Streamlit dashboard on {host}:{port} (headless={headless})..."
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.port",
            str(port),
            "--server.address",
            host,
            "--server.headless",
            str(headless).lower(),
        ]
    )


# Dummy use for deptry
_ = optuna_dashboard.__version__
