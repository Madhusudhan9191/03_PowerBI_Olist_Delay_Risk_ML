"""
Phase 0 & Phase 1 Comprehensive Data Audit & Verification Script
Project: Olist Delivery-Delay Risk Model
Path: 03_PowerBI_Olist_Delay_Risk_ML/src/audit/run_audit.py

Executes all verifications for Phase 0 and Phase 1:
- Table row counts, key uniqueness, referential integrity
- Target definition & exclusions breakdown
- Timestamp sequence anomalies & sensitivity
- Monthly prevalence & 2018 spike detection
- Right-censoring & transit time distribution
- Multi-item / multi-seller order profiling
- Programmatic comparison against canonical control totals (no hardcoded PASS/FAIL)
"""

import os
import json
import numpy as np
import pandas as pd

def run_comprehensive_audit(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    warehouse_dir = os.path.join(base_dir, "data", "warehouse")
    raw_dir = os.path.join(base_dir, "data", "raw")
    audit_out_dir = os.path.join(base_dir, "data", "audit")
    os.makedirs(audit_out_dir, exist_ok=True)

    # 1. Load control totals from profiling_results.json
    profiling_path = os.path.join(warehouse_dir, "profiling_results.json")
    with open(profiling_path, "r", encoding="utf-8") as f:
        canonical_profile = json.load(f)
    control_totals = canonical_profile.get("control_totals", {})

    # 2. Load warehouse tables
    print("Loading warehouse tables...")
    fact_del = pd.read_csv(os.path.join(warehouse_dir, "fact_order_delivery.csv"))
    fact_items = pd.read_csv(os.path.join(warehouse_dir, "fact_order_items.csv"))
    fact_payments = pd.read_csv(os.path.join(warehouse_dir, "fact_order_payments.csv"))
    dim_cust = pd.read_csv(os.path.join(warehouse_dir, "dim_customer.csv"))
    dim_seller = pd.read_csv(os.path.join(warehouse_dir, "dim_seller.csv"))
    dim_prod = pd.read_csv(os.path.join(warehouse_dir, "dim_product.csv"))
    dim_geo = pd.read_csv(os.path.join(warehouse_dir, "dim_geography.csv"))
    dim_date = pd.read_csv(os.path.join(warehouse_dir, "dim_date.csv"))

    # Load raw orders to verify raw exclusions independently
    raw_orders = pd.read_csv(os.path.join(raw_dir, "olist_orders_dataset.csv"))

    reconciliation_rows = []

    def record_check(category, check_name, computed_val, expected_val, tolerance=0.0):
        if isinstance(computed_val, (int, np.integer)) and isinstance(expected_val, (int, np.integer)):
            passed = (computed_val == expected_val)
            diff = computed_val - expected_val
        elif isinstance(computed_val, (float, np.floating)) and isinstance(expected_val, (float, np.floating)):
            diff = abs(computed_val - expected_val)
            passed = (diff <= tolerance)
        else:
            passed = (computed_val == expected_val)
            diff = 0 if passed else -1
        
        reconciliation_rows.append({
            "category": category,
            "check_name": check_name,
            "computed_value": computed_val,
            "expected_value": expected_val,
            "difference": diff,
            "status": "PASS" if passed else "FAIL"
        })

    # ==========================================
    # VERIFICATION A: Table Row Counts & Keys
    # ==========================================
    record_check("Row Counts", "fact_order_delivery rows", len(fact_del), control_totals["total_orders"])
    record_check("Row Counts", "fact_order_items rows", len(fact_items), control_totals["total_order_items"])
    record_check("Row Counts", "dim_customer rows", len(dim_cust), control_totals["total_customer_unique_id_count"])
    record_check("Row Counts", "raw_orders rows", len(raw_orders), control_totals["total_orders"])
    record_check("Row Counts", "dim_seller rows", len(dim_seller), 3095)
    record_check("Row Counts", "dim_product rows", len(dim_prod), 32951)

    # Key Uniqueness
    record_check("Key Uniqueness", "fact_order_delivery.order_id unique", fact_del["order_id"].nunique(), len(fact_del))
    record_check("Key Uniqueness", "dim_customer.customer_unique_id unique", dim_cust["customer_unique_id"].nunique(), len(dim_cust))
    record_check("Key Uniqueness", "dim_seller.seller_id unique", dim_seller["seller_id"].nunique(), len(dim_seller))
    record_check("Key Uniqueness", "dim_product.product_id unique", dim_prod["product_id"].nunique(), len(dim_prod))

    # Referential Integrity
    item_orders_in_del = fact_items["order_id"].isin(fact_del["order_id"]).all()
    record_check("Referential Integrity", "All fact_order_items.order_id in fact_order_delivery", item_orders_in_del, True)
    item_sellers_in_dim = fact_items["seller_id"].isin(dim_seller["seller_id"]).all()
    record_check("Referential Integrity", "All fact_order_items.seller_id in dim_seller", item_sellers_in_dim, True)
    item_prods_in_dim = fact_items["product_id"].isin(dim_prod["product_id"]).all()
    record_check("Referential Integrity", "All fact_order_items.product_id in dim_product", item_prods_in_dim, True)
    payments_orders_in_del = fact_payments["order_id"].isin(fact_del["order_id"]).all()
    record_check("Referential Integrity", "All fact_order_payments.order_id in fact_order_delivery", payments_orders_in_del, True)

    # ==========================================
    # VERIFICATION B: Target Definition & Exclusions
    # ==========================================
    # Raw orders status breakdown
    status_counts = raw_orders["order_status"].value_counts().to_dict()
    total_raw_orders = len(raw_orders)
    
    # Check delivered orders with and without customer date
    raw_delivered = raw_orders[raw_orders["order_status"] == "delivered"]
    raw_delivered_count = len(raw_delivered)
    delivered_null_cust_date = raw_delivered["order_delivered_customer_date"].isna().sum()
    valid_delivered_count = raw_delivered["order_delivered_customer_date"].notna().sum()

    record_check("Target Definition", "Total Raw Orders", total_raw_orders, 99441)
    record_check("Target Definition", "Raw Delivered Orders", raw_delivered_count, 96478)
    record_check("Target Definition", "Delivered with Null Customer Date", int(delivered_null_cust_date), 8)
    record_check("Target Definition", "Valid Delivered Orders (with date)", int(valid_delivered_count), control_totals["delivered_orders_with_date"])

    # Non-delivered orders
    non_delivered = raw_orders[raw_orders["order_status"] != "delivered"]
    total_non_delivered = len(non_delivered)
    record_check("Target Definition", "Total Non-Delivered Orders", total_non_delivered, 99441 - 96478)

    # Define valid delivered subset strictly: order_status == 'delivered' and order_delivered_customer_date.notna()
    valid_delivered = fact_del[(fact_del["order_status"] == "delivered") & (fact_del["order_delivered_customer_date"].notna())].copy()
    record_check("Target Definition", "fact_order_delivery valid delivered count", len(valid_delivered), 96470)

    # Date-level Late vs On-time calculation
    # Date level: order_delivered_customer_date_only > order_estimated_delivery_date_only
    cust_del_date = pd.to_datetime(valid_delivered["order_delivered_customer_date"]).dt.date
    est_del_date = pd.to_datetime(valid_delivered["order_estimated_delivery_date"]).dt.date
    computed_late_date_level = int((cust_del_date > est_del_date).sum())
    computed_ontime_date_level = int((cust_del_date <= est_del_date).sum())
    computed_late_rate_pct = round(float(computed_late_date_level / len(valid_delivered) * 100), 4)

    record_check("Target Definition", "Date-level Late Orders", computed_late_date_level, control_totals["date_level_late_delivered_orders"])
    record_check("Target Definition", "Date-level On-Time Orders", computed_ontime_date_level, control_totals["date_level_ontime_delivered_orders"])
    record_check("Target Definition", "Date-level Late Rate %", computed_late_rate_pct, control_totals["date_level_late_delivery_rate_pct"], tolerance=0.001)

    # Timestamp-level comparison
    cust_del_ts = pd.to_datetime(valid_delivered["order_delivered_customer_date"])
    est_del_ts = pd.to_datetime(valid_delivered["order_estimated_delivery_date"])
    computed_late_ts_level = int((cust_del_ts > est_del_ts).sum())
    ts_vs_date_diff = computed_late_ts_level - computed_late_date_level
    same_day_delivery_count = int((cust_del_date == est_del_date).sum())

    # Exclusion breakdown table
    exclusion_records = []
    # Status breakdown
    for st, cnt in sorted(status_counts.items(), key=lambda x: -x[1]):
        if st == "delivered":
            exclusion_records.append({
                "filter_category": "order_status",
                "status_or_reason": "delivered (retained)",
                "orders_count": cnt,
                "pct_of_all_orders": round(cnt / total_raw_orders * 100, 4),
                "action": "Retain (subject to valid delivery date)"
            })
        else:
            exclusion_records.append({
                "filter_category": "order_status",
                "status_or_reason": f"status == '{st}'",
                "orders_count": cnt,
                "pct_of_all_orders": round(cnt / total_raw_orders * 100, 4),
                "action": "Excluded (unfulfilled / in-transit / canceled)"
            })
    exclusion_records.append({
        "filter_category": "data_quality",
        "status_or_reason": "delivered status but null order_delivered_customer_date",
        "orders_count": int(delivered_null_cust_date),
        "pct_of_all_orders": round(delivered_null_cust_date / total_raw_orders * 100, 4),
        "action": "Excluded (missing ground truth outcome)"
    })
    df_exclusions = pd.DataFrame(exclusion_records)
    df_exclusions.to_csv(os.path.join(audit_out_dir, "exclusion_breakdown.csv"), index=False)

    # ==========================================
    # VERIFICATION C: Timestamp Anomalies & Sensitivity
    # ==========================================
    ts_df = valid_delivered.copy()
    ts_df["purchase_ts"] = pd.to_datetime(ts_df["order_purchase_timestamp"])
    ts_df["approved_ts"] = pd.to_datetime(ts_df["order_approved_at"])
    ts_df["carrier_ts"] = pd.to_datetime(ts_df["order_delivered_carrier_date"])
    ts_df["customer_ts"] = pd.to_datetime(ts_df["order_delivered_customer_date"])
    ts_df["estimated_ts"] = pd.to_datetime(ts_df["order_estimated_delivery_date"])

    # Carrier missing among valid delivered
    null_carrier_count = int(ts_df["carrier_ts"].isna().sum())
    null_approved_count = int(ts_df["approved_ts"].isna().sum())

    # Sequence anomalies on valid delivered
    carrier_before_purchase = int((ts_df["carrier_ts"] < ts_df["purchase_ts"]).sum())
    carrier_before_approved = int(((ts_df["carrier_ts"] < ts_df["approved_ts"]) & ts_df["carrier_ts"].notna() & ts_df["approved_ts"].notna()).sum())
    delivery_before_carrier = int(((ts_df["customer_ts"] < ts_df["carrier_ts"]) & ts_df["carrier_ts"].notna()).sum())
    delivery_before_purchase = int((ts_df["customer_ts"] < ts_df["purchase_ts"]).sum())
    delivery_before_approved = int(((ts_df["customer_ts"] < ts_df["approved_ts"]) & ts_df["approved_ts"].notna()).sum())

    warehouse_anomalous_sequence = int(ts_df["is_anomalous_sequence"].sum())
    warehouse_clean_fulfillment = int(ts_df["is_clean_fulfillment"].sum())

    record_check("Timestamp Anomalies", "carrier_before_purchase in valid delivered", carrier_before_purchase, 165)
    record_check("Timestamp Anomalies", "carrier_before_approved in valid delivered", carrier_before_approved, 1350)
    record_check("Timestamp Anomalies", "delivery_before_carrier in valid delivered", delivery_before_carrier, 23)
    record_check("Timestamp Anomalies", "delivery_before_purchase in valid delivered", delivery_before_purchase, 0)
    record_check("Timestamp Anomalies", "delivery_before_approved in valid delivered", delivery_before_approved, 61)
    record_check("Timestamp Anomalies", "null carrier_ts in valid delivered", null_carrier_count, 1)
    record_check("Timestamp Anomalies", "null approved_ts in valid delivered", null_approved_count, 14)
    record_check("Timestamp Anomalies", "is_anomalous_sequence in valid delivered", warehouse_anomalous_sequence, 1373)

    # Late rates on clean vs anomalous
    clean_subset = ts_df[ts_df["is_anomalous_sequence"] == False]
    anom_subset = ts_df[ts_df["is_anomalous_sequence"] == True]
    
    late_full_pct = float(ts_df["is_late"].mean() * 100)
    late_clean_pct = float(clean_subset["is_late"].mean() * 100)
    late_anom_pct = float(anom_subset["is_late"].mean() * 100)

    anomaly_summary = [
        {"anomaly_check": "carrier_ts < purchase_ts (carrier before purchase)", "count": carrier_before_purchase, "pct_of_valid_delivered": round(carrier_before_purchase / len(ts_df) * 100, 4)},
        {"anomaly_check": "carrier_ts < approved_ts (carrier before approved)", "count": carrier_before_approved, "pct_of_valid_delivered": round(carrier_before_approved / len(ts_df) * 100, 4)},
        {"anomaly_check": "customer_ts < carrier_ts (delivery before carrier)", "count": delivery_before_carrier, "pct_of_valid_delivered": round(delivery_before_carrier / len(ts_df) * 100, 4)},
        {"anomaly_check": "customer_ts < purchase_ts (delivery before purchase)", "count": delivery_before_purchase, "pct_of_valid_delivered": round(delivery_before_purchase / len(ts_df) * 100, 4)},
        {"anomaly_check": "customer_ts < approved_ts (delivery before approved)", "count": delivery_before_approved, "pct_of_valid_delivered": round(delivery_before_approved / len(ts_df) * 100, 4)},
        {"anomaly_check": "null carrier_ts among valid delivered", "count": null_carrier_count, "pct_of_valid_delivered": round(null_carrier_count / len(ts_df) * 100, 4)},
        {"anomaly_check": "null approved_ts among valid delivered", "count": null_approved_count, "pct_of_valid_delivered": round(null_approved_count / len(ts_df) * 100, 4)},
        {"anomaly_check": "warehouse is_anomalous_sequence (any anomaly)", "count": warehouse_anomalous_sequence, "pct_of_valid_delivered": round(warehouse_anomalous_sequence / len(ts_df) * 100, 4)},
    ]
    df_anomaly = pd.DataFrame(anomaly_summary)
    df_anomaly.to_csv(os.path.join(audit_out_dir, "timestamp_anomalies.csv"), index=False)

    sensitivity_summary = {
        "full_valid_delivered_orders": len(ts_df),
        "full_valid_delivered_late_orders": int(ts_df["is_late"].sum()),
        "full_valid_delivered_late_rate_pct": round(late_full_pct, 4),
        "clean_fulfillment_orders": len(clean_subset),
        "clean_fulfillment_late_orders": int(clean_subset["is_late"].sum()),
        "clean_fulfillment_late_rate_pct": round(late_clean_pct, 4),
        "anomalous_fulfillment_orders": len(anom_subset),
        "anomalous_fulfillment_late_orders": int(anom_subset["is_late"].sum()),
        "anomalous_fulfillment_late_rate_pct": round(late_anom_pct, 4),
        "late_rate_difference_clean_vs_full_pp": round(late_clean_pct - late_full_pct, 4)
    }

    # ==========================================
    # VERIFICATION D: Monthly Prevalence & Seasonality
    # ==========================================
    ts_df["purchase_ym"] = ts_df["purchase_ts"].dt.to_period("M").astype(str)
    monthly_stats = ts_df.groupby("purchase_ym").agg(
        total_orders=("order_id", "count"),
        late_orders=("is_late", "sum"),
        late_rate_pct=("is_late", lambda x: round(float(x.mean() * 100), 2)),
        avg_promised_lead_time_days=("promised_lead_time_days", lambda x: round(float(x.mean()), 2)),
        avg_transit_days=("transit_days", lambda x: round(float(x.mean()), 2))
    ).reset_index()
    monthly_stats.to_csv(os.path.join(audit_out_dir, "monthly_prevalence.csv"), index=False)

    # Spike months
    feb_row = monthly_stats[monthly_stats["purchase_ym"] == "2018-02"]
    mar_row = monthly_stats[monthly_stats["purchase_ym"] == "2018-03"]
    feb_late_rate = float(feb_row["late_rate_pct"].values[0])
    mar_late_rate = float(mar_row["late_rate_pct"].values[0])
    steady_state_late_rate = round(float(ts_df[ts_df["purchase_ym"].between("2017-01", "2018-08")]["is_late"].mean() * 100), 2)

    # ==========================================
    # VERIFICATION E: Right-Censoring & Transit Distribution
    # ==========================================
    transit_clean = ts_df["transit_days"].dropna()
    transit_quantiles = {
        "p50_days": round(float(transit_clean.quantile(0.50)), 2),
        "p75_days": round(float(transit_clean.quantile(0.75)), 2),
        "p90_days": round(float(transit_clean.quantile(0.90)), 2),
        "p95_days": round(float(transit_clean.quantile(0.95)), 2),
        "p99_days": round(float(transit_clean.quantile(0.99)), 2),
        "max_days": round(float(transit_clean.max()), 2),
        "mean_days": round(float(transit_clean.mean()), 2)
    }

    last_purchase_date = ts_df["purchase_ts"].max().strftime("%Y-%m-%d %H:%M:%S")
    last_carrier_date = ts_df["carrier_ts"].max().strftime("%Y-%m-%d %H:%M:%S")
    last_customer_delivery_date = ts_df["customer_ts"].max().strftime("%Y-%m-%d %H:%M:%S")

    raw_last_purchase = pd.to_datetime(raw_orders["order_purchase_timestamp"]).max().strftime("%Y-%m-%d %H:%M:%S")
    raw_orders_after_aug22 = raw_orders[pd.to_datetime(raw_orders["order_purchase_timestamp"]) > "2018-08-22 23:59:59"]
    delivered_after_aug22 = raw_orders_after_aug22[raw_orders_after_aug22["order_status"] == "delivered"]
    non_delivered_after_aug22 = raw_orders_after_aug22[raw_orders_after_aug22["order_status"] != "delivered"]

    # ==========================================
    # VERIFICATION F: Multi-Seller & Multi-Item Orders
    # ==========================================
    items_per_order = fact_items.groupby("order_id")["order_item_id"].count()
    sellers_per_order = fact_items.groupby("order_id")["seller_id"].nunique()

    single_item_orders = int((items_per_order == 1).sum())
    multi_item_orders = int((items_per_order > 1).sum())
    max_items_in_order = int(items_per_order.max())

    single_seller_orders = int((sellers_per_order == 1).sum())
    multi_seller_orders = int((sellers_per_order > 1).sum())
    max_sellers_in_order = int(sellers_per_order.max())

    # Items per order distribution
    item_dist = {str(k): int(v) for k, v in items_per_order.value_counts().sort_index().items()}
    seller_dist = {str(k): int(v) for k, v in sellers_per_order.value_counts().sort_index().items()}

    # ==========================================
    # Save Final Reconciliation & Summary
    # ==========================================
    df_reconciliation = pd.DataFrame(reconciliation_rows)
    df_reconciliation.to_csv(os.path.join(audit_out_dir, "reconciliation_checks.csv"), index=False)

    summary = {
        "reconciliation_summary": {
            "total_checks": len(df_reconciliation),
            "passed_checks": int((df_reconciliation["status"] == "PASS").sum()),
            "failed_checks": int((df_reconciliation["status"] == "FAIL").sum()),
        },
        "target_definition": {
            "total_raw_orders": total_raw_orders,
            "raw_delivered": raw_delivered_count,
            "delivered_missing_date": int(delivered_null_cust_date),
            "valid_delivered_orders": int(valid_delivered_count),
            "date_level_late_orders": computed_late_date_level,
            "date_level_ontime_orders": computed_ontime_date_level,
            "date_level_late_rate_pct": computed_late_rate_pct,
            "timestamp_level_late_orders": computed_late_ts_level,
            "ts_vs_date_diff": ts_vs_date_diff,
            "same_day_delivery_count": same_day_delivery_count
        },
        "status_breakdown": status_counts,
        "timestamp_anomalies": anomaly_summary,
        "sensitivity_analysis": sensitivity_summary,
        "spike_months": {
            "feb_2018_late_rate_pct": feb_late_rate,
            "mar_2018_late_rate_pct": mar_late_rate,
            "steady_state_avg_late_rate_pct": steady_state_late_rate
        },
        "transit_distribution": transit_quantiles,
        "dates_coverage": {
            "raw_last_purchase": raw_last_purchase,
            "delivered_last_purchase": last_purchase_date,
            "delivered_last_carrier": last_carrier_date,
            "delivered_last_customer": last_customer_delivery_date,
            "raw_orders_after_2018_08_22": len(raw_orders_after_aug22),
            "delivered_orders_after_2018_08_22": len(delivered_after_aug22),
            "non_delivered_orders_after_2018_08_22": len(non_delivered_after_aug22)
        },
        "order_composition": {
            "total_orders_in_items": len(items_per_order),
            "single_item_orders": single_item_orders,
            "multi_item_orders": multi_item_orders,
            "multi_item_pct": round(multi_item_orders / len(items_per_order) * 100, 2),
            "max_items": max_items_in_order,
            "single_seller_orders": single_seller_orders,
            "multi_seller_orders": multi_seller_orders,
            "multi_seller_pct": round(multi_seller_orders / len(sellers_per_order) * 100, 2),
            "max_sellers": max_sellers_in_order,
            "items_per_order_distribution": item_dist,
            "sellers_per_order_distribution": seller_dist
        }
    }

    with open(os.path.join(audit_out_dir, "audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Audit completed: {summary['reconciliation_summary']['passed_checks']}/{summary['reconciliation_summary']['total_checks']} PASSED.")
    return summary

if __name__ == "__main__":
    run_comprehensive_audit()
