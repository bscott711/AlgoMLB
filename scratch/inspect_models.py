import joblib
from pathlib import Path


def inspect_model(name):
    path = Path(f".data/models/{name}.joblib")
    if not path.exists():
        print(f"{name} not found")
        return

    bundle = joblib.load(path)
    # XGBoost classifier
    clf = bundle.get("clf") if isinstance(bundle, dict) else bundle

    if hasattr(clf, "feature_names_in_"):
        print(f"Features for {name}: {list(clf.feature_names_in_)}")
    elif hasattr(clf, "feature_names"):
        print(f"Features for {name}: {list(clf.feature_names)}")
    else:
        print(f"Could not find feature names for {name}")


inspect_model("home_win_v1.0")
inspect_model("uranium_win_model")
