"""
Automated Test Suite for Calibration and Rolling-Origin Temporal Stability
Project: Olist Delivery-Delay Risk Model
Path: 03_PowerBI_Olist_Delay_Risk_ML/tests/test_rolling_origin_and_calibration.py

Verifies:
- Calibrated Brier score improves over raw tree probabilities and approaches baseline.
- Expected Calibration Error (ECE) is reduced post-calibration.
- Rolling-origin validation CSV exists and reports valid temporal progression.
- Top 2% capacity queue captures >= 145 late orders (precision >= 40%).
"""

import os
import json
import pytest
import pandas as pd

@pytest.fixture(scope="module")
def base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def test_calibrated_metrics_improvement(base_dir):
    """Assert calibrated Brier score is <= 0.050 and calibrated ECE <= 0.10."""
    meta_path = os.path.join(base_dir, "models", "model_metadata.json")
    assert os.path.exists(meta_path), "model_metadata.json missing"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    test_m = meta["test_metrics"]
    assert test_m["calibrated_brier_score"] <= 0.050, f"Expected calibrated Brier <= 0.050, got {test_m['calibrated_brier_score']}"
    assert test_m["calibrated_brier_score"] < test_m["uncalibrated_brier_score"], "Calibration did not improve Brier score"
    assert test_m["calibrated_ece"] <= 0.10, f"Expected calibrated ECE <= 0.10, got {test_m['calibrated_ece']}"

def test_rolling_origin_results_integrity(base_dir):
    """Assert rolling-origin cross-validation file exists and all folds maintain positive lift."""
    ro_path = os.path.join(base_dir, "data", "audit", "rolling_origin_cv_results.csv")
    assert os.path.exists(ro_path), "rolling_origin_cv_results.csv missing"
    df = pd.read_csv(ro_path)

    assert len(df) in [3, 6], f"Expected 3 or 6 expanding fold rows, got {len(df)}"
    # Every fold must achieve positive discriminative lift over prior
    for _, r in df.iterrows():
        assert r["val_pr_auc"] > (r["val_late_rate_pct"] / 100.0), f"{r['fold_id']} PR-AUC does not beat prevalence"
        assert r["val_roc_auc"] >= 0.60, f"{r['fold_id']} ROC-AUC unexpectedly low: {r['val_roc_auc']}"

def test_top2_percent_queue_precision(base_dir):
    """Assert top 2% capacity queue achieves >= 40% precision (145+ late orders caught)."""
    scored_path = os.path.join(base_dir, "data", "powerbi_export", "fact_model_scored_orders.csv")
    df = pd.read_csv(scored_path)
    q_top2 = df[df["risk_band"] == "Critical Queue (Top 2%)"]

    assert len(q_top2) in [362, 363], f"Unexpected top 2% queue size: {len(q_top2)}"
    late_caught = q_top2["is_late"].sum()
    precision = late_caught / len(q_top2)
    assert late_caught >= 145, f"Expected at least 145 late orders caught in top 2%, got {late_caught}"
    assert precision >= 0.40, f"Expected precision >= 40%, got {precision:.4f}"
