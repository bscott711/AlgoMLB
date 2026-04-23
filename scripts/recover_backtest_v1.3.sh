#!/bin/bash
set -e

LOG="/home/opc/AlgoMLB/.data/models/backtest_pa_v1.3_recovery_v2.log"
CLI="/home/opc/AlgoMLB/.venv/bin/algomlb"
mkdir -p /home/opc/AlgoMLB/.data/models

echo "========================================" >> "$LOG"
echo "PA Outcome v1.3 Backtest Recovery"       >> "$LOG"
echo "Started: $(date -u)"                     >> "$LOG"
echo "========================================" >> "$LOG"

cd /home/opc/AlgoMLB

# Run only the backtest
$CLI ml backtest --target pa_outcome --version v1.3 >> "$LOG" 2>&1

echo ""                                                             >> "$LOG"
echo "========================================" >> "$LOG"
echo "BACKTEST COMPLETE: $(date -u)"            >> "$LOG"
echo "========================================" >> "$LOG"
