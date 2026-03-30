import pytest
from typer.testing import CliRunner
from algomlb.cli.main import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AlgoMLB" in result.stdout


def test_cli_db_stubs():
    assert runner.invoke(app, ["db", "init"]).exit_code == 0
    assert runner.invoke(app, ["db", "status"]).exit_code == 0


def test_cli_ingest_stubs():
    assert runner.invoke(app, ["ingest", "odds"]).exit_code == 0
    assert runner.invoke(app, ["ingest", "games"]).exit_code == 0


def test_cli_ml_stubs():
    assert runner.invoke(app, ["ml", "train"]).exit_code == 0
    assert runner.invoke(app, ["ml", "optimize"]).exit_code == 0


def test_cli_run_stubs():
    assert runner.invoke(app, ["run", "live"]).exit_code == 0


def test_cli_ui_stubs(monkeypatch: pytest.MonkeyPatch):
    import subprocess

    # Use main app to avoid sub-app invocation issues
    from algomlb.cli.main import app as main_app

    # Mock subprocess.run to avoid starting a real process
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: None)

    result = runner.invoke(main_app, ["ui", "launch"])
    assert result.exit_code == 0
