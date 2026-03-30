import typer

app = typer.Typer(help="Run the live prediction pipeline.", no_args_is_help=True)


@app.command()
def live(ctx: typer.Context):
    """Start the APScheduler live loop."""
    typer.echo("TODO: run live")
