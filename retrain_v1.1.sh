#!/usr/bin/env bash
#
# AlgoMLB v1.1 Model Retraining Pipeline
# =======================================
# Runs: tune → train → backtest for home_win v1.1
#
# Usage:
#   nohup bash retrain_v1.1.sh > retrain_v1.1.log 2>&1 &
#   disown
#
# Monitor:
#   tail -f retrain_v1.1.log
#
set -euo pipefail

cd /home/opc/AlgoMLB
export PYTHONPATH=src

TARGET="home_win"
VERSION="v1.1"
TRIALS=100
LOG_PREFIX="[retrain_v1.1]"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} $*"; }

# ── Step 1: Hyperparameter Tuning ────────────────────────────────────────
log "Starting hyperparameter tuning (${TRIALS} trials)..."
log "This is the longest step — expect ~2-3 min per trial."

uv run python -m algomlb.cli.main ml tune \
    --target "${TARGET}" \
    --version "${VERSION}" \
    --trials "${TRIALS}"

TUNE_EXIT=$?
if [ $TUNE_EXIT -ne 0 ]; then
    log "ERROR: Tuning failed with exit code ${TUNE_EXIT}. Aborting."
    exit 1
fi

log "Tuning complete. Best params saved to .data/models/optuna_best_params_${TARGET}_${VERSION}.json"
log "Best params:"
cat ".data/models/optuna_best_params_${TARGET}_${VERSION}.json"
echo ""

# ── Step 2: Train Production Model ──────────────────────────────────────
log "Training production model..."

uv run python -m algomlb.cli.main ml train \
    --target "${TARGET}" \
    --version "${VERSION}"

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    log "ERROR: Training failed with exit code ${TRAIN_EXIT}. Aborting."
    exit 1
fi

log "Training complete. Model saved to .data/models/${TARGET}_${VERSION}.joblib"
ls -lh ".data/models/${TARGET}_${VERSION}.joblib"

# ── Step 3: Walk-Forward Backtest ────────────────────────────────────────
log "Running walk-forward backtest..."

uv run python -m algomlb.cli.main ml backtest \
    --target "${TARGET}" \
    --version "${VERSION}"

BACKTEST_EXIT=$?
if [ $BACKTEST_EXIT -ne 0 ]; then
    log "ERROR: Backtest failed with exit code ${BACKTEST_EXIT}. Aborting."
    exit 1
fi

log "Backtest complete."

# ── Summary ──────────────────────────────────────────────────────────────
log "============================================"
log "  PIPELINE COMPLETE: ${TARGET} ${VERSION}"
log "============================================"
log "Artifacts:"
log "  Params:  .data/models/optuna_best_params_${TARGET}_${VERSION}.json"
log "  Model:   .data/models/${TARGET}_${VERSION}.joblib"
log "  Metrics persisted to PostgreSQL (uranium_eval_history)"
log ""
log "Compare v1.0 vs v1.1 with:"
log "  PYTHONPATH=src uv run python -c \\"
log "    \"from algomlb.ml.eval import fetch_eval_history; \\"
log "    from algomlb.db.session import get_session_factory; \\"
log "    e=get_session_factory().kw['bind']; \\"
log "    print(fetch_eval_history('home_win','v1.0',e)[['accuracy','auc','log_loss_val','brier']].to_string()); \\"
log "    print(fetch_eval_history('home_win','v1.1',e)[['accuracy','auc','log_loss_val','brier']].to_string())\""
