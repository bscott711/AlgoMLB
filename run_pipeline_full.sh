#!/bin/bash
set -e
source .venv/bin/activate
echo "Starting Tune (Full Production Run)..."
algomlb ml tune --target pa_outcome --version v1.6 --trials 100
echo "Starting Backtest..."
algomlb ml backtest --target pa_outcome --version v1.6
echo "Starting Explain..."
algomlb ml explain --target pa_outcome --version v1.6
echo "Starting Train..."
algomlb ml train --target pa_outcome --version v1.6
echo "Full Production Pipeline complete!"
