#!/bin/bash
# AlgoMLB ML Training Orchestration Script
# Targets: pa_outcome, home_win, total_runs_actual

set -e

declare -a targets=("pa_outcome" "home_win" "total_runs_actual")
TRIALS=100
VERSION="v1.0"

echo "🚀 Starting Full Tuning & Training Cycle..."
echo "Trials per study: $TRIALS"
echo "Model Version: $VERSION"

for TARGET in "${targets[@]}"; do
    echo "===================================================================="
    echo "🎯 TARGET: $TARGET"
    echo "===================================================================="
    
    echo "📝 Step 1: Optuna Hyperparameter Optimization..."
    uv run algomlb ml tune --target "$TARGET" --trials "$TRIALS" --version "$VERSION"
    
    echo "📊 Step 2: Walk-forward Backtesting (Model Training & Evaluation)..."
    uv run algomlb ml backtest --target "$TARGET" --version "$VERSION"
    
    echo "🔬 Step 3: SHAP Global Feature Importance & Diagnostics..."
    uv run algomlb ml explain --target "$TARGET" --version "$VERSION"
    
    echo "✅ Completed $TARGET"
    echo ""
done

echo "🎉 ALL TARGETS PROCESSED SUCCESSFULLY!"
