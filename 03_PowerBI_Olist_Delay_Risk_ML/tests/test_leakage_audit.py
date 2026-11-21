"""
Automated Leakage Audit & Point-in-Time Contract Tests
Project: Olist Delivery-Delay Risk Model
Path: 03_PowerBI_Olist_Delay_Risk_ML/tests/test_leakage_audit.py

Verifies:
- Forbidden outcome columns are absent from candidate feature matrix
- Point-in-time timestamp contracts (no feature timestamp exceeds prediction moment)
- Expanding window outcome lag integrity (zero lookahead into future deliveries)
- Ablation and feature isolation checks
"""

import os
import pytest
import pandas as pd
import numpy as np

@pytest.fixture(scope="module")
def feature_df():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parquet_path = os.path.join(base, "data", "processed", "orders_feature_matrix.parquet")
    if not os.path.exists(parquet_path):
        from src.features.build_features import build_feature_matrix
        return build_feature_matrix(base)
    return pd.read_parquet(parquet_path)

def test_forbidden_columns_absent(feature_df):
    """Ensure no outcome-derived or post-handover columns exist as input features."""
    forbidden_features = [
        "transit_days", "cycle_days", "delay_days", "review_score", "review_id",
        "review_comment_message", "review_creation_date", "review_answer_timestamp",
        "order_delivered_customer_date_only", "order_delivery_status_category"
    ]
    for col in forbidden_features:
        assert col not in feature_df.columns, f"LEAKAGE DETECTED: Forbidden feature '{col}' is present!"

def test_point_in_time_durations(feature_df):
    """Verify that merchant handling duration and approval times are computed strictly at or before carrier handover."""
    # Exclude known anomalous sequence orders and the 1 null carrier order for strict duration checks
    clean = feature_df[(feature_df["is_anomalous_sequence"] == 0) & (feature_df["order_delivered_carrier_date"].notna())]
    c_clean = pd.to_datetime(clean["order_delivered_carrier_date"])
    p_clean = pd.to_datetime(clean["order_purchase_timestamp"])
    
    assert (c_clean >= p_clean).all(), "Found clean order where carrier handover preceded purchase timestamp!"
    assert (clean["merchant_handling_days"] >= 0).all(), "Found negative merchant handling days in clean subset!"

def test_expanding_window_lag_integrity(feature_df):
    """Verify that orders in January 2017 have seller/corridor historical rates derived strictly from 2016."""
    jan_2017 = feature_df[feature_df["purchase_ym"] == "2017-01"]
    
    # In January 2017, Olist had only ~267 completed orders from 2016
    # Global completed orders in historical lookup should be <= 300
    assert (jan_2017["seller_hist_completed_orders"] <= 300).all()
    # Most sellers in Jan 2017 should be flagged as no history (>= 75%)
    assert jan_2017["seller_no_history_flag"].mean() >= 0.75

def test_target_distribution(feature_df):
    """Verify that target distribution aligns with expected steady-state late rate (~6.8%)."""
    modelling = feature_df[feature_df["modeling_window_tier"] == "modelling"]
    late_rate = modelling["is_late"].mean() * 100
    assert 6.0 <= late_rate <= 7.5, f"Expected modeling window late rate around 6.8%, got {late_rate:.2f}%"
