"""Verify training matrix has the right features for v1.3."""

import sys

sys.path.insert(0, "/home/opc/AlgoMLB/src")
from algomlb.db.session import get_session_factory
from algomlb.ml.features import FeaturePipeline
from algomlb.cli.ml import _load_ml_data

session_factory = get_session_factory()
engine = session_factory.kw["bind"]

# Load a small sample (2025 only)
data = _load_ml_data(engine, "2025")
print(
    f"PAs: {len(data['pas'])}, pitcher_gold: {len(data['pitcher_gold'])}, batter_gold: {len(data['batter_gold'])}"
)

pipeline = FeaturePipeline()
X, y = pipeline.build_pa_matrix(
    data["pas"],
    data["pitcher_gold"],
    data["batter_gold"],
    lineups_df=data["lineups"] if not data["lineups"].empty else None,
    elo_df=data["elo"],
)

print(f"\nMatrix shape: {X.shape}")
print(f"Outcome distribution:\n{y.value_counts()}")
print(f"\nFeature columns ({len(X.columns)}):")
for c in sorted(X.columns):
    print(f"  {c}")

# Check key features are present
key_feats = [
    "pitcher_n_games_used",
    "pitcher_days_since_last_game",
    "batter_n_games_used",
    "batter_days_since_last_game",
    "pitcher_fatigue_index_7d",
    "pitcher_fatigue_index_14d",
    "batter_fatigue_index_7d",
    "batter_fatigue_index_14d",
    "pitcher_window_games",
    "batter_window_games",
]
print("\nKey feature check:")
for f in key_feats:
    present = f in X.columns
    nz = (X[f] != 0).sum() if present else 0
    print(
        f"  {'✅' if present else '❌'} {f:40s} {'present' if present else 'MISSING'} non-zero={nz}"
    )

# Check NO cnt_ features
cnt_feats = [c for c in X.columns if c.startswith("cnt_")]
print(f"\ncnt_ features (should be 0): {len(cnt_feats)}")
if cnt_feats:
    print(f"  PROBLEM: {cnt_feats}")
