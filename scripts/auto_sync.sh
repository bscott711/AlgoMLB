#!/bin/bash
# scripts/auto_sync.sh
# ------------------------------------------------------------------
# Automated Daily Sync for AlgoMLB
# This script should be run via cron (e.g., daily at 4:30 AM ET).
# ------------------------------------------------------------------

set -e

# Change to the project directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

# Ensure environment is active
[[ -f ".venv/bin/activate" ]] && source .venv/bin/activate

echo "$(date): 🔄 Starting AlgoMLB automated daily sync..."

# Run the unified sync command
uv run algomlb sync daily

echo "$(date): ✅ Sync complete!"
