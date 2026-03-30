from pathlib import Path


def test_ui_files_exist():
    # Only verify they exist and are valid python files
    ui_dir = Path("src/algomlb/ui")
    assert (ui_dir / "app.py").exists()
    assert (ui_dir / "views/optuna.py").exists()
    assert (ui_dir / "views/picks.py").exists()
    assert (ui_dir / "views/bankroll.py").exists()
    assert (ui_dir / "views/data.py").exists()


def test_ui_app_init():
    import algomlb.ui as ui

    assert ui is not None
