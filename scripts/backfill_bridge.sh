#!/bin/bash
set -e

LOG="/home/opc/AlgoMLB/.data/process_backfill_bridge.log"
CLI="/home/opc/AlgoMLB/.venv/bin/algomlb"
START_DATE="2024-08-16"
END_DATE="2025-06-16"

echo "========================================" >> "$LOG"
echo "Uranium Data Bridge Backfill"            >> "$LOG"
echo "Range: $START_DATE to $END_DATE"         >> "$LOG"
echo "Started: $(date -u)"                     >> "$LOG"
echo "========================================" >> "$LOG"

cd /home/opc/AlgoMLB

echo "[1/2] SILVER — Summarizing Statcast Pitches for 2024 & 2025..." >> "$LOG"
$CLI process silver --year 2024 >> "$LOG" 2>&1
$CLI process silver --year 2025 >> "$LOG" 2>&1
echo "SILVER complete: $(date -u)" >> "$LOG"

echo "[2/2] GOLD — Generating Rolling Features..." >> "$LOG"
$CLI process rolling --start "$START_DATE" --end "$END_DATE" >> "$LOG" 2>&1
echo "GOLD complete: $(date -u)" >> "$LOG"

echo "========================================" >> "$LOG"
echo "BACKFILL DONE: $(date -u)"               >> "$LOG"
echo "========================================" >> "$LOG"
