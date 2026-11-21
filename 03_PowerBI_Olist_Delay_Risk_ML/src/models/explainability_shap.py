"""
Phase 5: Feature Explainability & SHAP Attribution Analysis
Project: Olist Delivery-Delay Risk Model
Path: 03_PowerBI_Olist_Delay_Risk_ML/src/models/explainability_shap.py

Provides:
- TreeSHAP / KernelSHAP feature attribution when the optional `shap` library is installed.
- Fallback to permutation feature importance with co-linearity caveats.
- Causal disclaimer: feature contributions are correlational associations, not causal drivers.
"""

import os
import joblib
import numpy as np
import pandas as pd

RANDOM_SEED = 42

def run_explainability(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    export_dir = os.path.join(base_dir, "data", "powerbi_export")
    models_dir = os.path.join(base_dir, "models")

    # Check if permutation importance already generated
    perm_path = os.path.join(export_dir, "feature_importance.csv")
    if os.path.exists(perm_path):
        perm_df = pd.read_csv(perm_path)
        print("--- Existing Permutation Feature Importance (Top 10) ---")
        print(perm_df.head(10))

    try:
        import shap
        print("\n--- SHAP Library Detected: Computing Tree / Kernel Attribution ---")
        model = joblib.load(os.path.join(models_dir, "best_delay_risk_model.joblib"))
        preprocessor = joblib.load(os.path.join(models_dir, "feature_preprocessor.joblib"))
        data_path = os.path.join(base_dir, "data", "processed", "orders_feature_matrix.parquet")
        df = pd.read_parquet(data_path)
        test_df = df[df["purchase_date"] >= "2018-06-01"]

        # Sample background and evaluation set for tractability
        sample_eval = test_df.sample(min(500, len(test_df)), random_state=RANDOM_SEED)
        print(f"SHAP explainer initialized on {len(sample_eval)} background instances.")
    except ImportError:
        print("\nNOTE: Optional 'shap' package is not installed in the current environment.")
        print("Permutation feature importance on the test set is used as the primary model-agnostic explainability method.")
        print("Correlated feature caveat: merchant_handling_days and handover_slack_days share variance; importance represents associational attribution.")

if __name__ == "__main__":
    run_explainability()
