import joblib
from pathlib import Path
import sys

def deep_inspect(name):
    path = Path(f".data/models/{name}.joblib")
    if not path.exists():
        print(f"{name} not found")
        return
    
    try:
        sys.path.append("src")
        bundle = joblib.load(path)
        print(f"--- Inspecting {name} ---")
        
        if isinstance(bundle, dict):
            cclf = bundle.get("calibrated_clf")
            if cclf:
                print(f"Calibrated Clf fitted: {hasattr(cclf, 'calibrated_classifiers_')}")
                if hasattr(cclf, "feature_names_in_"):
                    print(f"Features: {list(cclf.feature_names_in_)}")
            else:
                clf = bundle.get("clf")
                print(f"Clf fitted: {hasattr(clf, 'feature_names_in_')}")
                if hasattr(clf, "feature_names_in_"):
                    print(f"Features: {list(clf.feature_names_in_)}")
                
    except Exception:
        import traceback
        traceback.print_exc()

deep_inspect("home_win_v1.0")
