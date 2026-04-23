#!/bin/bash
set -e

VERSION="v1.5"
LOG="/home/opc/AlgoMLB/.data/models/train_pa_${VERSION}.log"
CLI="/home/opc/AlgoMLB/.venv/bin/algomlb"
mkdir -p /home/opc/AlgoMLB/.data/models

echo "========================================" >> "$LOG"
echo "PA Outcome ${VERSION} Training Pipeline"  >> "$LOG"
echo "Strategy: Full Data + AUC Optimization"    >> "$LOG"
echo "Started: $(date -u)"                     >> "$LOG"
echo "========================================" >> "$LOG"

cd /home/opc/AlgoMLB

echo ""                                                             >> "$LOG"
echo "[1/3] TUNE — Optuna hyperparameter search (100 trials)..."    >> "$LOG"
echo "Started: $(date -u)"                                          >> "$LOG"
$CLI ml tune --target pa_outcome --version $VERSION --trials 100    >> "$LOG" 2>&1
echo "[1/3] TUNE complete: $(date -u)"                              >> "$LOG"

echo ""                                                             >> "$LOG"
echo "[2/3] TRAIN — Fit production model with best params..."       >> "$LOG"
echo "Started: $(date -u)"                                          >> "$LOG"
$CLI ml train --target pa_outcome --version $VERSION                >> "$LOG" 2>&1
echo "[2/3] TRAIN complete: $(date -u)"                             >> "$LOG"

echo ""                                                             >> "$LOG"
echo "[3/3] BACKTEST — Walk-forward validation..."                  >> "$LOG"
echo "Started: $(date -u)"                                          >> "$LOG"
$CLI ml backtest --target pa_outcome --version $VERSION              >> "$LOG" 2>&1
echo "[3/3] BACKTEST complete: $(date -u)"                          >> "$LOG"

echo ""                                                             >> "$LOG"
echo "========================================" >> "$LOG"
echo "ALL DONE: $(date -u)"                    >> "$LOG"
echo "Model: .data/models/pa_outcome_${VERSION}.joblib" >> "$LOG"
echo "========================================" >> "$LOG"
