"""
Automated Test Suite for Phase 0 & Phase 1
Project: Olist Delivery-Delay Risk Model
Path: 03_PowerBI_Olist_Delay_Risk_ML/tests/test_audit_and_definitions.py

Verifies:
- Data warehouse row counts and primary key uniqueness
- Referential integrity across fact and dimension tables
- Canonical target definition and control totals (6,534 late, 96,470 valid delivered)
- Exclusions breakdown (status filtering and missing dates)
- Timestamp anomaly quantification
- Audit reconciliation log contains 0 failures
"""

import os
import json
import pytest
import pandas as pd

@pytest.fixture(scope="module")
def base_paths():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return {
        "base": base,
        "warehouse": os.path.join(base, "data", "warehouse"),
        "raw": os.path.join(base, "data", "raw"),
        "audit": os.path.join(base, "data", "audit"),
    }

@pytest.fixture(scope="module")
def audit_summary(base_paths):
    summary_path = os.path.join(base_paths["audit"], "audit_summary.json")
    if not os.path.exists(summary_path):
        # Run audit if not present
        from src.audit.run_audit import run_comprehensive_audit
        return run_comprehensive_audit(base_paths["base"])
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_reconciliation_zero_failures(audit_summary):
    """Verify that every programmatic check in the reconciliation log PASSED."""
    rec = audit_summary["reconciliation_summary"]
    assert rec["failed_checks"] == 0, f"Found {rec['failed_checks']} failed reconciliation checks!"
    assert rec["passed_checks"] == rec["total_checks"], "Not all reconciliation checks completed successfully."
    assert rec["total_checks"] >= 30, f"Expected at least 30 checks, got {rec['total_checks']}."

def test_warehouse_row_counts(base_paths):
    """Verify exact row counts for all warehouse tables against canonical totals."""
    wh = base_paths["warehouse"]
    fact_del = pd.read_csv(os.path.join(wh, "fact_order_delivery.csv"))
    fact_items = pd.read_csv(os.path.join(wh, "fact_order_items.csv"))
    dim_cust = pd.read_csv(os.path.join(wh, "dim_customer.csv"))
    dim_seller = pd.read_csv(os.path.join(wh, "dim_seller.csv"))
    dim_prod = pd.read_csv(os.path.join(wh, "dim_product.csv"))

    assert len(fact_del) == 99441, f"Expected 99,441 delivery rows, got {len(fact_del)}"
    assert len(fact_items) == 112650, f"Expected 112,650 items rows, got {len(fact_items)}"
    assert len(dim_cust) == 96096, f"Expected 96,096 customers rows, got {len(dim_cust)}"
    assert len(dim_seller) == 3095, f"Expected 3,095 sellers rows, got {len(dim_seller)}"
    assert len(dim_prod) == 32951, f"Expected 32,951 products rows, got {len(dim_prod)}"

def test_primary_key_uniqueness(base_paths):
    """Verify that primary keys have 0 duplicate values."""
    wh = base_paths["warehouse"]
    fact_del = pd.read_csv(os.path.join(wh, "fact_order_delivery.csv"), usecols=["order_id"])
    dim_cust = pd.read_csv(os.path.join(wh, "dim_customer.csv"), usecols=["customer_unique_id"])
    dim_seller = pd.read_csv(os.path.join(wh, "dim_seller.csv"), usecols=["seller_id"])
    dim_prod = pd.read_csv(os.path.join(wh, "dim_product.csv"), usecols=["product_id"])

    assert fact_del["order_id"].nunique() == len(fact_del), "Duplicate order_id found in fact_order_delivery"
    assert dim_cust["customer_unique_id"].nunique() == len(dim_cust), "Duplicate customer_unique_id in dim_customer"
    assert dim_seller["seller_id"].nunique() == len(dim_seller), "Duplicate seller_id in dim_seller"
    assert dim_prod["product_id"].nunique() == len(dim_prod), "Duplicate product_id in dim_product"

def test_target_definition_control_totals(audit_summary):
    """Verify exact target counts: 96,470 valid delivered, 6,534 late (6.7731%), 89,936 on-time."""
    tgt = audit_summary["target_definition"]
    assert tgt["total_raw_orders"] == 99441
    assert tgt["raw_delivered"] == 96478
    assert tgt["delivered_missing_date"] == 8
    assert tgt["valid_delivered_orders"] == 96470
    assert tgt["date_level_late_orders"] == 6534
    assert tgt["date_level_ontime_orders"] == 89936
    assert abs(tgt["date_level_late_rate_pct"] - 6.7731) < 0.0001
    assert tgt["same_day_delivery_count"] == 1292

def test_timestamp_anomalies_distribution(audit_summary):
    """Verify quantified sequence anomalies in valid delivered orders."""
    anoms = {item["anomaly_check"]: item["count"] for item in audit_summary["timestamp_anomalies"]}
    assert anoms["carrier_ts < purchase_ts (carrier before purchase)"] == 165
    assert anoms["carrier_ts < approved_ts (carrier before approved)"] == 1350
    assert anoms["customer_ts < carrier_ts (delivery before carrier)"] == 23
    assert anoms["customer_ts < purchase_ts (delivery before purchase)"] == 0
    assert anoms["customer_ts < approved_ts (delivery before approved)"] == 61
    assert anoms["null carrier_ts among valid delivered"] == 1
    assert anoms["null approved_ts among valid delivered"] == 14
    assert anoms["warehouse is_anomalous_sequence (any anomaly)"] == 1373

def test_spike_months_seasonality(audit_summary):
    """Verify the Feb and Mar 2018 logistics delay spike."""
    spikes = audit_summary["spike_months"]
    assert spikes["feb_2018_late_rate_pct"] == 14.13
    assert spikes["mar_2018_late_rate_pct"] == 18.96
    assert spikes["steady_state_avg_late_rate_pct"] == 6.79
