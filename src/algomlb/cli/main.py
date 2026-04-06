import typer
from algomlb.cli import db, sync, ingest, ml, process, run, ui

app = typer.Typer(
    name="algomlb",
    help="AlgoMLB — MLB analytics and prediction engine.",
    no_args_is_help=True,
)

app.add_typer(db.app, name="db")
app.add_typer(sync.app, name="sync")
app.add_typer(ingest.app, name="ingest")

app.add_typer(ml.app, name="ml")
app.add_typer(process.app, name="process")
app.add_typer(run.app, name="run")
app.add_typer(ui.app, name="ui")


@app.callback()
def root_callback(
    ctx: typer.Context,
    agent_mode: bool = typer.Option(
        False,
        "--agent-mode",
        help="Suppress human output; emit structured JSON to stdout.",
    ),
):
    """AlgoMLB control panel."""
    ctx.ensure_object(dict)
    ctx.obj["agent_mode"] = agent_mode


if __name__ == "__main__":
    app()
