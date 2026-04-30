#!/bin/bash
set -e
source .venv/bin/activate
echo "Starting Tune..."
algomlb ml tune --target pa_outcome --version v1.5 --trials 10
echo "Starting Backtest..."
algomlb ml backtest --target pa_outcome --version v1.5
echo "Starting Explain..."
algomlb ml explain --target pa_outcome --version v1.5
echo "Starting Train..."
algomlb ml train --target pa_outcome --version v1.5
echo "Pipeline complete!"
