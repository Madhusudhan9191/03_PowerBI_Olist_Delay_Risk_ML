"""
Phase 2: Point-in-Time Feature Engineering Pipeline
Project: Olist Delivery-Delay Risk Model
Path: 03_PowerBI_Olist_Delay_Risk_ML/src/features/build_features.py

Constructs the point-in-time feature matrix evaluated at order_delivered_carrier_date:
- Order-level aggregation of items, products, sellers, payments
- Transit & lead time features observable at handover
- Expanding-window historical rates with lag (zero lookahead)
- Empirical Bayes smoothing for cold-start sellers and corridors
- Exports processed feature matrix to CSV & Parquet
"""

import os
import json
import numpy as np
import pandas as pd

def load_data(base_dir):
    wh_dir = os.path.join(base_dir, "data", "warehouse")
    print("Loading warehouse tables for feature engineering...")
    fact_del = pd.read_csv(os.path.join(wh_dir, "fact_order_delivery.csv"))
    fact_items = pd.read_csv(os.path.join(wh_dir, "fact_order_items.csv"))
    fact_payments = pd.read_csv(os.path.join(wh_dir, "fact_order_payments.csv"))
    dim_prod = pd.read_csv(os.path.join(wh_dir, "dim_product.csv"))
    dim_seller = pd.read_csv(os.path.join(wh_dir, "dim_seller.csv"))
    dim_geo = pd.read_csv(os.path.join(wh_dir, "dim_geography.csv"))
    return fact_del, fact_items, fact_payments, dim_prod, dim_seller, dim_geo

def aggregate_items(fact_items, dim_prod, dim_seller):
    print("Aggregating items to order level...")
    # Merge items with product dimensions
    items = fact_items.merge(
        dim_prod[["product_id", "product_category_name_en", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]],
        on="product_id",
        how="left"
    )
    items = items.merge(
        dim_seller[["seller_id", "seller_state", "seller_city"]],
        on="seller_id",
        how="left"
    )

    items["product_volume_cm3"] = items["product_length_cm"] * items["product_height_cm"] * items["product_width_cm"]
    items["shipping_limit_ts"] = pd.to_datetime(items["shipping_limit_date"])

    # Sort items by price descending to identify primary product & seller
    items_sorted = items.sort_values(["order_id", "price"], ascending=[True, False])
    primary_items = items_sorted.drop_duplicates(subset=["order_id"], keep="first")[
        ["order_id", "product_category_name_en", "seller_id", "seller_state", "seller_city"]
    ].rename(columns={
        "product_category_name_en": "primary_category",
        "seller_id": "primary_seller_id",
        "seller_state": "primary_seller_state",
        "seller_city": "primary_seller_city"
    })

    # Order-level aggregations
    agg_df = items.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        seller_count=("seller_id", "nunique"),
        total_order_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        max_route_distance_km=("route_distance_km", "max"),
        is_interstate=("is_interstate", "max"),
        min_shipping_limit_ts=("shipping_limit_ts", "min"),
        is_any_late_dispatch=("is_late_dispatch", "max"),
        max_product_weight_g=("product_weight_g", "max"),
        total_product_weight_g=("product_weight_g", "sum"),
        max_product_volume_cm3=("product_volume_cm3", "max"),
        total_product_volume_cm3=("product_volume_cm3", "sum")
    ).reset_index()

    agg_df["is_multi_seller"] = (agg_df["seller_count"] > 1).astype(int)
    agg_df["is_interstate"] = agg_df["is_interstate"].fillna(0).astype(int)
    agg_df["is_any_late_dispatch"] = agg_df["is_any_late_dispatch"].fillna(0).astype(int)
    agg_df["freight_to_price_ratio"] = agg_df["total_freight_value"] / np.maximum(agg_df["total_order_price"], 1.0)

    # Merge primary attributes
    order_items_features = agg_df.merge(primary_items, on="order_id", how="left")
    return order_items_features

def aggregate_payments(fact_payments):
    print("Aggregating payments to order level...")
    # Group payments by order
    pay_agg = fact_payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        payment_sequence_count=("payment_sequential", "max")
    ).reset_index()

    # Find dominant payment type by highest value
    pay_sorted = fact_payments.sort_values(["order_id", "payment_value"], ascending=[True, False])
    dom_pay = pay_sorted.drop_duplicates(subset=["order_id"], keep="first")[["order_id", "payment_type"]].rename(
        columns={"payment_type": "primary_payment_type"}
    )
    return pay_agg.merge(dom_pay, on="order_id", how="left")

def compute_expanding_window_history(df, lag_days=1):
    """
    Computes expanding-window historical delay rates with strict lag.
    For each order i with carrier handover on day D, only historical orders j whose
    customer delivery occurred on or before (D - lag_days) are included.
    Empirical Bayes smoothing is applied toward global/state priors.
    """
    print(f"Computing historical expanding rates with {lag_days}-day outcome lag...")
    df = df.copy()
    df["handover_date"] = pd.to_datetime(df["order_delivered_carrier_date"]).dt.date
    df["outcome_date"] = pd.to_datetime(df["order_delivered_customer_date"]).dt.date
    df["corridor"] = df["primary_seller_state"].fillna("UNKNOWN") + "_" + df["order_customer_state"].fillna("UNKNOWN")

    # Distinct dates in chronological order
    all_dates = sorted([d for d in df["handover_date"].dropna().unique()])
    min_date = all_dates[0]
    max_date = all_dates[-1]

    # Pre-index outcomes by outcome_date
    outcome_records = df[["order_id", "primary_seller_id", "corridor", "order_customer_state", "outcome_date", "is_late"]].dropna(subset=["outcome_date"])
    
    # Group by outcome date for quick incremental accumulation
    daily_outcomes = outcome_records.groupby("outcome_date")

    # Dictionaries to maintain running counts
    global_completed = 0
    global_late = 0
    seller_completed = {}
    seller_late = {}
    corridor_completed = {}
    corridor_late = {}

    # Map: date -> snapshot of running totals available BEFORE this date
    # i.e., stats finalized on or before (date - lag_days)
    daily_snapshots = {}

    date_range = pd.date_range(min_date - pd.Timedelta(days=5), max_date + pd.Timedelta(days=2)).date
    
    # Build timeline of finished outcomes
    for cur_date in date_range:
        # Check if outcomes finalized on (cur_date - pd.Timedelta(days=lag_days))
        finalized_date = cur_date - pd.Timedelta(days=lag_days)
        if finalized_date in daily_outcomes.groups:
            day_batch = daily_outcomes.get_group(finalized_date)
            for _, r in day_batch.iterrows():
                s_id = r["primary_seller_id"]
                corr = r["corridor"]
                is_l = r["is_late"]

                global_completed += 1
                global_late += is_l

                seller_completed[s_id] = seller_completed.get(s_id, 0) + 1
                seller_late[s_id] = seller_late.get(s_id, 0) + is_l

                corridor_completed[corr] = corridor_completed.get(corr, 0) + 1
                corridor_late[corr] = corridor_late.get(corr, 0) + is_l

        # Store snapshot available on cur_date
        daily_snapshots[cur_date] = {
            "global_completed": global_completed,
            "global_late": global_late,
            "seller_completed": seller_completed.copy(),
            "seller_late": seller_late.copy(),
            "corridor_completed": corridor_completed.copy(),
            "corridor_late": corridor_late.copy(),
        }

    # Now assign historical rates to each order based on its handover_date
    seller_hist_orders = []
    seller_hist_late_rate = []
    seller_no_history_flag = []
    corridor_hist_orders = []
    corridor_hist_late_rate = []
    corridor_no_history_flag = []

    m_seller = 10.0  # Empirical Bayes smoothing weight for sellers
    m_corridor = 20.0  # Empirical Bayes smoothing weight for corridors

    for _, row in df.iterrows():
        h_date = row["handover_date"]
        s_id = row["primary_seller_id"]
        corr = row["corridor"]

        if pd.isna(h_date) or h_date not in daily_snapshots:
            # Fallback for null handover date
            snap = daily_snapshots[min_date]
        else:
            snap = daily_snapshots[h_date]

        g_comp = snap["global_completed"]
        g_late = snap["global_late"]
        global_prior = (g_late / g_comp) if g_comp > 0 else 0.0

        # Seller history
        s_n = snap["seller_completed"].get(s_id, 0)
        s_k = snap["seller_late"].get(s_id, 0)
        s_rate = (s_k + m_seller * global_prior) / (s_n + m_seller)
        s_no_hist = 1 if s_n < 5 else 0

        # Corridor history
        c_n = snap["corridor_completed"].get(corr, 0)
        c_k = snap["corridor_late"].get(corr, 0)
        c_rate = (c_k + m_corridor * global_prior) / (c_n + m_corridor)
        c_no_hist = 1 if c_n < 10 else 0

        seller_hist_orders.append(s_n)
        seller_hist_late_rate.append(round(s_rate, 4))
        seller_no_history_flag.append(s_no_hist)
        corridor_hist_orders.append(c_n)
        corridor_hist_late_rate.append(round(c_rate, 4))
        corridor_no_history_flag.append(c_no_hist)

    df["seller_hist_completed_orders"] = seller_hist_orders
    df["seller_hist_late_rate"] = seller_hist_late_rate
    df["seller_no_history_flag"] = seller_no_history_flag
    df["corridor_hist_completed_orders"] = corridor_hist_orders
    df["corridor_hist_late_rate"] = corridor_hist_late_rate
    df["corridor_no_history_flag"] = corridor_no_history_flag

    return df

def build_feature_matrix(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    out_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    fact_del, fact_items, fact_payments, dim_prod, dim_seller, dim_geo = load_data(base_dir)

    # 1. Filter to valid delivered orders only
    valid_mask = (fact_del["order_status"] == "delivered") & (fact_del["order_delivered_customer_date"].notna())
    df = fact_del[valid_mask].copy()
    print(f"Valid delivered orders: {len(df)}")

    # 2. Add item-level aggregated features
    item_feats = aggregate_items(fact_items, dim_prod, dim_seller)
    df = df.merge(item_feats, on="order_id", how="left")

    # 3. Add payment features
    pay_feats = aggregate_payments(fact_payments)
    df = df.merge(pay_feats, on="order_id", how="left")

    # 4. Compute timestamps and point-in-time durations
    print("Computing point-in-time features at order_delivered_carrier_date...")
    purchase_ts = pd.to_datetime(df["order_purchase_timestamp"])
    approved_ts = pd.to_datetime(df["order_approved_at"])
    carrier_ts = pd.to_datetime(df["order_delivered_carrier_date"])
    estimated_ts = pd.to_datetime(df["order_estimated_delivery_date"])
    shipping_limit_ts = pd.to_datetime(df["min_shipping_limit_ts"])

    # Duration features (in days)
    df["promised_lead_time_days"] = (estimated_ts.dt.date - purchase_ts.dt.date).apply(lambda x: x.days if pd.notna(x) else np.nan)
    df["merchant_handling_days"] = ((carrier_ts - purchase_ts).dt.total_seconds() / 86400.0).round(2)
    df["approval_to_carrier_days"] = ((carrier_ts - approved_ts).dt.total_seconds() / 86400.0).round(2)
    df["handover_slack_days"] = ((shipping_limit_ts - carrier_ts).dt.total_seconds() / 86400.0).round(2)
    df["is_late_dispatch"] = (carrier_ts > shipping_limit_ts).astype(int)

    # Calendar features at carrier handover
    df["handover_dow"] = carrier_ts.dt.dayofweek
    df["handover_month"] = carrier_ts.dt.month
    df["handover_hour"] = carrier_ts.dt.hour
    df["is_weekend_handover"] = df["handover_dow"].isin([5, 6]).astype(int)

    # Brazilian Macro-Regions
    state_to_region = {
        "SP": "Sudeste", "RJ": "Sudeste", "MG": "Sudeste", "ES": "Sudeste",
        "PR": "Sul", "SC": "Sul", "RS": "Sul",
        "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
        "BA": "Nordeste", "PE": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
        "RN": "Nordeste", "AL": "Nordeste", "SE": "Nordeste", "PI": "Nordeste",
        "AM": "Norte", "PA": "Norte", "RO": "Norte", "TO": "Norte", "AC": "Norte", "AP": "Norte", "RR": "Norte"
    }
    df["customer_region"] = df["order_customer_state"].map(state_to_region).fillna("Other")

    # 5. Expanding-window historical rates with lag
    df = compute_expanding_window_history(df, lag_days=1)

    # 6. Target and split metadata
    c_dt = pd.to_datetime(df["order_delivered_customer_date"]).dt.date
    e_dt = pd.to_datetime(df["order_estimated_delivery_date"]).dt.date
    df["is_late"] = (c_dt > e_dt).astype(int)
    df["purchase_date"] = purchase_ts.dt.date.astype(str)
    df["purchase_ym"] = purchase_ts.dt.to_period("M").astype(str)

    # Anomaly indicator
    df["is_anomalous_sequence"] = df["is_anomalous_sequence"].astype(int)

    # Missing value imputation for physical dimensions and prices
    df["max_product_weight_g"] = df["max_product_weight_g"].fillna(df["max_product_weight_g"].median())
    df["total_product_weight_g"] = df["total_product_weight_g"].fillna(df["total_product_weight_g"].median())
    df["max_product_volume_cm3"] = df["max_product_volume_cm3"].fillna(df["max_product_volume_cm3"].median())
    df["total_product_volume_cm3"] = df["total_product_volume_cm3"].fillna(df["total_product_volume_cm3"].median())
    df["max_route_distance_km"] = df["max_route_distance_km"].fillna(df["max_route_distance_km"].median())
    df["freight_to_price_ratio"] = df["freight_to_price_ratio"].fillna(df["freight_to_price_ratio"].median())
    df["primary_category"] = df["primary_category"].fillna("unknown")
    df["primary_payment_type"] = df["primary_payment_type"].fillna("unknown")

    # Filter out modeling tail (right-censored orders after 2018-08-22)
    # Tag orders by modeling split window
    df["modeling_window_tier"] = np.where(
        df["purchase_date"] <= "2016-12-31", "warmup",
        np.where(df["purchase_date"] <= "2018-08-22", "modelling", "right_censored")
    )

    # Select final columns
    final_cols = [
        # Keys & metadata
        "order_id", "customer_id", "customer_unique_id", "purchase_date", "purchase_ym", "modeling_window_tier",
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
        # Quality flags
        "is_anomalous_sequence",
        # Point-in-time features
        "promised_lead_time_days", "merchant_handling_days", "approval_to_carrier_days",
        "handover_slack_days", "is_late_dispatch",
        "item_count", "seller_count", "is_multi_seller",
        "total_order_price", "total_freight_value", "freight_to_price_ratio",
        "max_route_distance_km", "is_interstate", "order_distance_band",
        "order_customer_state", "customer_region", "primary_seller_state",
        "max_product_weight_g", "total_product_weight_g", "max_product_volume_cm3", "total_product_volume_cm3",
        "primary_category", "primary_payment_type", "max_installments",
        "handover_dow", "handover_month", "handover_hour", "is_weekend_handover",
        # Historical expanding window features
        "seller_hist_completed_orders", "seller_hist_late_rate", "seller_no_history_flag",
        "corridor_hist_completed_orders", "corridor_hist_late_rate", "corridor_no_history_flag",
        # Target
        "is_late"
    ]

    feature_df = df[final_cols].copy()
    
    out_csv = os.path.join(out_dir, "orders_feature_matrix.csv")
    out_parquet = os.path.join(out_dir, "orders_feature_matrix.parquet")

    print(f"Saving feature matrix to {out_csv} and {out_parquet}...")
    feature_df.to_csv(out_csv, index=False)
    feature_df.to_parquet(out_parquet, index=False)

    # Print summary statistics
    print(f"Feature matrix built successfully: {len(feature_df)} rows, {len(final_cols)} columns.")
    print("Row breakdown by modeling window tier:")
    print(feature_df["modeling_window_tier"].value_counts())
    print(f"Target late rate in modelling window (2017-01 to 2018-08-22): {feature_df[feature_df['modeling_window_tier'] == 'modelling']['is_late'].mean()*100:.4f}%")

    return feature_df

if __name__ == "__main__":
    build_feature_matrix()
