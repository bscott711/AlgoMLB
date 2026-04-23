#!/bin/bash
set -e

LOG="/home/opc/AlgoMLB/.data/models/train_pa_v1.3.log"
CLI="/home/opc/AlgoMLB/.venv/bin/algomlb"
mkdir -p /home/opc/AlgoMLB/.data/models

echo "========================================" >> "$LOG"
echo "PA Outcome v1.3 Training Pipeline"       >> "$LOG"
echo "Started: $(date -u)"                     >> "$LOG"
echo "========================================" >> "$LOG"

cd /home/opc/AlgoMLB

echo ""                                                             >> "$LOG"
echo "[1/3] TUNE — Optuna hyperparameter search (100 trials)..."    >> "$LOG"
echo "Started: $(date -u)"                                          >> "$LOG"
$CLI ml tune --target pa_outcome --version v1.3 --trials 100        >> "$LOG" 2>&1
echo "[1/3] TUNE complete: $(date -u)"                              >> "$LOG"

echo ""                                                             >> "$LOG"
echo "[2/3] TRAIN — Fit production model with best params..."       >> "$LOG"
echo "Started: $(date -u)"                                          >> "$LOG"
$CLI ml train --target pa_outcome --version v1.3                    >> "$LOG" 2>&1
echo "[2/3] TRAIN complete: $(date -u)"                             >> "$LOG"

echo ""                                                             >> "$LOG"
echo "[3/3] BACKTEST — Walk-forward validation..."                  >> "$LOG"
echo "Started: $(date -u)"                                          >> "$LOG"
$CLI ml backtest --target pa_outcome --version v1.3                 >> "$LOG" 2>&1
echo "[3/3] BACKTEST complete: $(date -u)"                          >> "$LOG"

echo ""                                                             >> "$LOG"
echo "========================================" >> "$LOG"
echo "ALL DONE: $(date -u)"                    >> "$LOG"
echo "Model: .data/models/pa_outcome_v1.3.joblib" >> "$LOG"
echo "========================================" >> "$LOG"
