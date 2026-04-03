#!/bin/bash
set -e
# Backfill Statcast data from 2026 down to 2019
# We start with 2026 as requested.

# Configure logging for the python processes
export PYTHONUNBUFFERED=1

for year in 2026 2025 2024 2023 2022 2021 2020 2019
do
    echo "========================================"
    echo "Processing Year: $year"
    echo "========================================"
    # Start and end dates for the season
    START_DATE="${year}-03-20"
    END_DATE="${year}-11-05"
    
    # Run the ingester
    uv run python -m algomlb.ingestion.statcast_ingester --start "$START_DATE" --end "$END_DATE"
done

echo "Historical backfill complete!"
