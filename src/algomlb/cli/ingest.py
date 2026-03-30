import typer

app = typer.Typer(help="Ingest data from external APIs.", no_args_is_help=True)


@app.command()
def odds(ctx: typer.Context):
    """Fetch latest odds data."""
    typer.echo("TODO: ingest odds")


@app.command()
def games(ctx: typer.Context):
    """Fetch latest game results."""
    typer.echo("TODO: ingest games")
