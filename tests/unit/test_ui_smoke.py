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
    """Verify that all UI modules can at least be imported (smoke test for top-level code)."""
    # Import app.py top-level to hit st.Page calls
    # Note: Streamlit calls at top-level outside of a run will generally not fail immediately
    import algomlb.ui.app as app
    import algomlb.ui.views.optuna as optuna
    import algomlb.ui.views.picks as picks
    import algomlb.ui.views.bankroll as bankroll
    import algomlb.ui.views.data as data
    import algomlb.ui.views.player_health as health

    assert app.pages is not None
    assert optuna is not None
    assert picks is not None
    assert bankroll is not None
    assert data is not None
    assert health is not None
