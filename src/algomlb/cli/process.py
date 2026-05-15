from datetime import date, timedelta
from typing import Optional

import typer

from algomlb.config.settings import get_settings
from algomlb.db.models import StatcastRawORM
from algomlb.db.repository import DatabaseRepository
from algomlb.db.session import get_engine
from algomlb.ml.rolling_processor import RollingProcessor
from algomlb.ml.rolling_service import RollingService
from sqlalchemy import select
import pandas as pd

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


@app.command("rolling")
def process_rolling(
    start_date: Optional[str] = typer.Option(
        None, "--start", "--start-date", help="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end", "--end-date", help="End date (YYYY-MM-DD)"
    ),
    single_date: Optional[str] = typer.Option(
        None, "--date", help="Single date to process (YYYY-MM-DD)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compute but do not save"),
):
    """
    Generate rolling statistics (Gold Layer) from Silver logs.
    Example: algomlb process rolling --start 2024-03-20 --end 2024-10-01
    """
    from algomlb.db.session import get_session_factory

    settings = get_settings()
    session = get_session_factory()()
    db = DatabaseRepository(session)
    processor = RollingProcessor(settings.ml)
    service = RollingService(db, processor)

    target_start: Optional[date] = None
    target_end: Optional[date] = None

    if single_date:
        target_start = date.fromisoformat(single_date)
        target_end = target_start
    elif start_date:
        target_start = date.fromisoformat(start_date)
        target_end = date.fromisoformat(end_date) if end_date else target_start
    else:
        typer.echo("Error: Please provide --date or --start", err=True)
        raise typer.Exit(1)

    typer.echo(f"Processing rolling features from {target_start} to {target_end}...")
    count = service.process_date_range(target_start, target_end, dry_run=dry_run)
    typer.echo(f"Done. Processed {count} rolling records.")


@app.command("silver")
def silver(
    game_date: Optional[str] = typer.Option(
        None, "--date", help="Process a single game date (YYYY-MM-DD)."
    ),
    start_date: Optional[str] = typer.Option(
        None, "--start", help="Start date for range processing (YYYY-MM-DD)."
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end", help="End date for range processing (YYYY-MM-DD, inclusive)."
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
        _upsert_silver,
    )

    if incremental:
        typer.echo(
            f"Running incremental silver processing (Batch Size: {batch_size})..."
        )
        process_silver_incremental(batch_size=batch_size)
        typer.echo("Incremental processing check complete.")
        return

    # Date range processing (--start / --end)
    if start_date is not None:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date) if end_date else s
        engine = get_engine()

        # Only query dates that actually have Statcast data
        from sqlalchemy import text
        with engine.connect() as conn:
            query = text("""
                SELECT DISTINCT game_date
                FROM statcast_raw
                WHERE game_date >= :start AND game_date <= :end
                ORDER BY game_date
            """)
            dates_with_data = conn.execute(query, {"start": s, "end": e}).fetchall()

        if not dates_with_data:
            typer.echo(f"No Statcast data found between {s} and {e}.")
            return

        typer.echo(f"Processing {len(dates_with_data)} game-days from {s} to {e}...")
        for row in dates_with_data:
            current = row[0]
            q = select(StatcastRawORM).where(StatcastRawORM.game_date == current)
            df = pd.read_sql(q, engine)
            if not df.empty:
                prior = fetch_prior_year_stats(current.year - 1)
                summarized = summarize_to_silver(df, prior)
                _upsert_silver(summarized)
                typer.echo(f"Summarized {current} to silver.")
        typer.echo("Range processing complete.")
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
            prior = fetch_prior_year_stats(d.year - 1)
            summarized = summarize_to_silver(df, prior)
            _upsert_silver(summarized)
            typer.echo(f"Summarized {d} to silver.")
        else:
            typer.echo(f"No data found for {d}")
        return

    typer.echo("Error: Provide --date, --start [--end], --year, or --incremental.")
    raise typer.Exit(code=1)
