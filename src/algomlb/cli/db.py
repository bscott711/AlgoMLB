import typer

app = typer.Typer(help="Manage database migrations and status.", no_args_is_help=True)


@app.command()
def init(ctx: typer.Context):
    """Initialize the database schema."""
    typer.echo("TODO: db init")


@app.command()
def status(ctx: typer.Context):
    """Show current migration status."""
    typer.echo("TODO: db status")
