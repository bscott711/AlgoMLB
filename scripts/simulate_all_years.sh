#!/bin/bash
# scripts/simulate_all_years.sh
# Sequentially runs Monte Carlo backfills for all specified years.

YEARS=(2026 2024 2023 2022 2021 2020)
TRIALS=1000
WORKERS=2

for YEAR in "${YEARS[@]}"
do
    echo "Starting simulation for year: $YEAR"
    LOG_FILE="logs/simulation_${YEAR}.log"
    
    # Run the parallel simulation script
    # We use 'uv run' to ensure the correct environment
    uv run python3 scripts/batch_simulate_parallel.py --year "$YEAR" --trials "$TRIALS" --workers "$WORKERS" > "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed simulation for $YEAR."
    else
        echo "Simulation failed for $YEAR. Check $LOG_FILE for details."
    fi
done

echo "All simulations completed."
