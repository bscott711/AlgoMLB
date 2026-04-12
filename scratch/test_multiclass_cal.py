import numpy as np
import pandas as pd
from algomlb.ml.eval import compute_calibration_bins

# Multiclass Test Case
# 3 classes, 3 samples
# Top-1 predictions:
# Sample 0: Pred=0 (Prob=0.8), True=0 -> Correct
# Sample 1: Pred=1 (Prob=0.7), True=1 -> Correct
# Sample 2: Pred=2 (Prob=0.8), True=2 -> Correct
# Resulting confidence vector: [0.8, 0.7, 0.8]
# Resulting binary labels: [1, 1, 1]

y_true = np.array([0, 1, 2])
y_prob = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.1, 0.8]
])

print("Testing Multiclass Confidence Calibration...")
cal_df = compute_calibration_bins(y_true, y_prob)
print(cal_df[cal_df["sample_count"] > 0])

# Expected:
# Bins for 0.7 and 0.8 should be populated with actual_prob_mean = 1.0
assert not cal_df[cal_df["sample_count"] > 0].empty
assert np.all(cal_df[cal_df["sample_count"] > 0]["actual_prob_mean"] == 1.0)
print("SUCCESS: Multiclass Confidence Calibration verified.")
