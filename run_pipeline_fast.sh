#!/bin/bash
set -e
source .venv/bin/activate
echo "Starting Tune (Fast)..."
algomlb ml tune --target pa_outcome --version v1.6 --trials 3 --fast
echo "Starting Backtest (Fast)..."
algomlb ml backtest --target pa_outcome --version v1.6 --fast
echo "Starting Explain (Fast)..."
algomlb ml explain --target pa_outcome --version v1.6 --fast
echo "Starting Train (Fast)..."
algomlb ml train --target pa_outcome --version v1.6 --fast
echo "Pipeline complete!"
