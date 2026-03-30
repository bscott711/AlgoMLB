import optuna
import typer

app = typer.Typer(help="Train and optimize ML models.", no_args_is_help=True)


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
