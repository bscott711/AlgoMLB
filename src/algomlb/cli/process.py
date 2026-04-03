import typer
from typing import Optional

app = typer.Typer(help="Run processing and feature-engineering jobs.")


@app.command("quant")
def quant(
    game_date: Optional[str] = typer.Option(
        None, "--date", help="Process a single game date (YYYY-MM-DD)."
    ),
    game_pk: Optional[int] = typer.Option(
        None, "--game-pk", help="Process a single game by game_pk."
    ),
    start_date: Optional[str] = typer.Option(
        None, "--start-date", help="Start of date range for batch processing."
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end-date", help="End of date range for batch processing (inclusive)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview output without writing to the DB."
    ),
    window: Optional[int] = typer.Option(
        None, "--window", help="Rolling baseline window in days (overrides config)."
    ),
):
    """Calibrate Statcast quant features from statcast_raw."""
    from algomlb.ml.quant_processor import (
        process_quant_for_date,
        process_quant_for_game,
    )
    from datetime import date, timedelta

    total = 0

    if game_pk is not None:
        total = process_quant_for_game(game_pk, dry_run=dry_run)

    elif game_date is not None:
        d = date.fromisoformat(game_date)
        total = process_quant_for_date(d, dry_run=dry_run, baseline_window_days=window)

    elif start_date is not None and end_date is not None:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        current = start
        while current <= end:
            total += process_quant_for_date(
                current, dry_run=dry_run, baseline_window_days=window
            )
            current += timedelta(days=1)

    else:
        typer.echo(
            "Error: Provide one of --date, --game-pk, or --start-date + --end-date."
        )
        raise typer.Exit(code=1)

    typer.echo(f"Quant processing complete. Rows written: {total}")


@app.command("silver")
def silver(
    game_date: Optional[str] = typer.Option(
        None, "--date", help="Process a single game date (YYYY-MM-DD)."
    ),
    year: Optional[int] = typer.Option(
        None, "--year", help="Process an entire year (full backfill)."
    ),
    incremental: bool = typer.Option(
        False, "--incremental", help="Process only new data since last run."
    ),
    batch_size: int = typer.Option(
        50000, "--batch-size", help="Records to fetch in a single batch."
    ),
):
    """Summarize Statcast pitch data into player-game silver logs."""
    from algomlb.ml.silver_processor import (
        process_silver_incremental,
        summarize_to_silver,
        fetch_prior_year_stats,
    )
    from algomlb.db.session import get_engine
    from algomlb.db.models import StatcastRawORM
    from sqlalchemy import select
    from datetime import date
    import pandas as pd

    if incremental:
        process_silver_incremental(batch_size=batch_size)
        return

    if year is not None:
        engine = get_engine()
        # Fetch only dates that actually have game data for the year
        with engine.connect() as conn:
            from sqlalchemy import text

            query = text("""
                SELECT DISTINCT game_date 
                FROM statcast_raw 
                WHERE EXTRACT(year FROM game_date) = :year
                ORDER BY game_date
            """)
            dates_with_data = conn.execute(query, {"year": year}).fetchall()

        if not dates_with_data:
            typer.echo(f"No game data found for {year}.")
            return

        typer.echo(f"Summarizing {len(dates_with_data)} days of games for {year}...")
        for row in dates_with_data:
            current = row[0]
            typer.echo(f"Summarizing {current}...")
            # Reuse logic for each date
            from algomlb.ml.silver_processor import _upsert_silver

            query = select(StatcastRawORM).where(StatcastRawORM.game_date == current)
            df = pd.read_sql(query, engine)
            if not df.empty:
                prior = fetch_prior_year_stats(year - 1)
                summarized = summarize_to_silver(df, prior)
                _upsert_silver(summarized)
        return

    if game_date is not None:
        d = date.fromisoformat(game_date)
        engine = get_engine()
        query = select(StatcastRawORM).where(StatcastRawORM.game_date == d)
        df = pd.read_sql(query, engine)
        if not df.empty:
            from algomlb.ml.silver_processor import _upsert_silver

            prior = fetch_prior_year_stats(d.year - 1)
            summarized = summarize_to_silver(df, prior)
            _upsert_silver(summarized)
            typer.echo(f"Summarized {d} to silver.")
        else:
            typer.echo(f"No data found for {d}")
        return

    typer.echo("Error: Provide --date, --year, or --incremental.")
    raise typer.Exit(code=1)
