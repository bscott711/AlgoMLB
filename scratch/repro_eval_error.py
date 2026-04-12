import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_fold_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


# Case 1: Standard numpy
y_true = np.array([0, 1])
y_prob = np.array([0.1, 0.9])
print("Case 1:", compute_fold_metrics(y_true, y_prob))

# Case 2: Object array of floats
y_prob_obj = np.array([0.1, 0.9], dtype=object)
print("Case 2:", compute_fold_metrics(y_true, y_prob_obj))

# Case 3: 2D array (should fail in metrics, but line 30?)
y_prob_2d = np.array([[0.9, 0.1], [0.2, 0.8]])
try:
    print("Case 3:", compute_fold_metrics(y_true, y_prob_2d))
except Exception as e:
    print("Case 3 failed:", type(e), e)

# Case 4: Series
y_prob_ser = pd.Series([0.1, 0.9])
print("Case 4:", compute_fold_metrics(y_true, y_prob_ser))
