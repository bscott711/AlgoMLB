from datetime import datetime

import optuna
import typer
from algomlb.core.agent_io import AgentResult, emit_agent_result
from algomlb.core.logger import logger
from algomlb.ml import HistoricalDataLoader

app = typer.Typer(help="Train and optimize ML models.", no_args_is_help=True)


@app.command()
def fetch_history(
    ctx: typer.Context,
    start_year: int = typer.Option(
        2023, "--start-year", help="Year to start fetching data from."
    ),
    end_year: int = typer.Option(
        datetime.now().year, "--end-year", help="Year to end fetching data at."
    ),
):
    """Fetch historical pitching and batting stats for a given year range."""
    agent_mode = ctx.obj.get("agent_mode", False)

    loader = HistoricalDataLoader()

    logger.info(f"Fetching historical data from {start_year} to {end_year}...")

    pitching_df = loader.fetch_pitching_stats(start_year, end_year)
    batting_df = loader.fetch_team_batting(start_year, end_year)

    logger.success(
        f"Fetched pitching data: {pitching_df.shape} and team batting data: {batting_df.shape}"
    )

    if agent_mode:
        emit_agent_result(
            AgentResult(
                status="success",
                command="ml.fetch-history",
                data={
                    "pitching_shape": list(pitching_df.shape),
                    "batting_shape": list(batting_df.shape),
                },
            )
        )


@app.command()
def train(ctx: typer.Context):
    """Train the model on historical data."""
    typer.echo("TODO: ml train")


@app.command()
def optimize(ctx: typer.Context):
    """Run Optuna optimization studies."""
    typer.echo("TODO: ml optimize")


# Dummy use for deptry
_ = optuna.Study
