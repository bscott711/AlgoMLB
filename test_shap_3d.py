import pandas as pd
import numpy as np

features = ["f1", "f2"]
mean_abs = np.random.rand(2, 8)  # 2 features, 8 classes
try:
    df = pd.DataFrame({"feature_name": features, "mean_abs_shap": mean_abs})
    print(df)
except Exception as e:
    print("Error:", e)
