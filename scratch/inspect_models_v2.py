import joblib
from pathlib import Path
import sys

def inspect_model(name):
    path = Path(f".data/models/{name}.joblib")
    if not path.exists():
        print(f"{name} not found at {path}")
        return
    
    try:
        # We need to add src to path so MLBModel can be loaded if it's a bundle
        sys.path.append("src")
        bundle = joblib.load(path)
        
        # If it's an MLBModel bundle
        if isinstance(bundle, dict) and "clf" in bundle:
            clf = bundle["clf"]
        else:
            clf = bundle
            
        if hasattr(clf, "feature_names_in_"):
            print(f"Features for {name}: {list(clf.feature_names_in_)}")
        else:
            print(f"No feature_names_in_ for {name}")
            
    except Exception as e:
        print(f"Error inspecting {name}: {e}")

inspect_model("uranium_win_model")
inspect_model("home_win_v1.0")
