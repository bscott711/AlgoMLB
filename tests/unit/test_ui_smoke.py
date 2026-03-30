from pathlib import Path


def test_ui_files_exist():
    # Only verify they exist and are valid python files
    ui_dir = Path("src/algomlb/ui")
    assert (ui_dir / "app.py").exists()
    assert (ui_dir / "pages/optuna.py").exists()
    assert (ui_dir / "pages/picks.py").exists()
    assert (ui_dir / "pages/bankroll.py").exists()
    assert (ui_dir / "pages/data.py").exists()


def test_ui_app_init():
    import algomlb.ui as ui

    assert ui is not None
