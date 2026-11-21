"""
Phase 3: Rolling-Origin Time-Series Cross-Validation
Project: Olist Delivery-Delay Risk Model
Path: 03_PowerBI_Olist_Delay_Risk_ML/src/models/run_rolling_origin_validation.py

Implements expanding-window temporal cross-validation:
- Evaluates temporal stability across expanding training horizons.
- Prevents future-to-past lookahead leakage.
- Reports row counts, positive counts, prevalence, PR-AUC, and ROC-AUC per fold.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

RANDOM_SEED = 42

def run_rolling_origin_cv(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_path = os.path.join(base_dir, "data", "processed", "orders_feature_matrix.parquet")
    audit_dir = os.path.join(base_dir, "data", "audit")
    os.makedirs(audit_dir, exist_ok=True)

    df = pd.read_parquet(data_path)
    m_df = df[df["modeling_window_tier"] == "modelling"].copy().reset_index(drop=True)

    # Define expanding-window folds
    folds = [
        {
            "fold_id": "Fold_1",
            "train_start": "2017-01-01",
            "train_end": "2017-10-31",
            "val_start": "2017-11-01",
            "val_end": "2018-01-31",
            "notes": "Early marketplace expansion into Q4 2017 peak"
        },
        {
            "fold_id": "Fold_2",
            "train_start": "2017-01-01",
            "train_end": "2018-01-31",
            "val_start": "2018-02-01",
            "val_end": "2018-04-30",
            "notes": "Includes 2017 peak; tests during Feb-Mar 2018 logistics spike"
        },
        {
            "fold_id": "Fold_3",
            "train_start": "2017-01-01",
            "train_end": "2018-04-30",
            "val_start": "2018-05-01",
            "val_end": "2018-05-31",
            "notes": "Pre-test consolidation window"
        }
    ]

    num_features = [
        "promised_lead_time_days", "merchant_handling_days", "approval_to_carrier_days",
        "handover_slack_days", "is_late_dispatch",
        "item_count", "seller_count", "is_multi_seller",
        "total_order_price", "total_freight_value", "freight_to_price_ratio",
        "max_route_distance_km", "is_interstate",
        "max_product_weight_g", "total_product_weight_g", "max_product_volume_cm3", "total_product_volume_cm3",
        "max_installments", "handover_dow", "handover_month", "handover_hour", "is_weekend_handover",
        "seller_hist_completed_orders", "seller_hist_late_rate", "seller_no_history_flag",
        "corridor_hist_completed_orders", "corridor_hist_late_rate", "corridor_no_history_flag",
        "is_anomalous_sequence"
    ]
    cat_features = [
        "order_customer_state", "customer_region", "order_distance_band",
        "primary_payment_type", "primary_category"
    ]
    all_features = num_features + cat_features

    # Top 15 category capping
    top_categories = m_df["primary_category"].value_counts().nlargest(15).index.tolist()
    m_df["primary_category"] = m_df["primary_category"].apply(lambda x: x if x in top_categories else "other")

    results = []
    print("\n--- Running Rolling-Origin Time-Series Cross-Validation ---")

    for f in folds:
        train_mask = (m_df["purchase_date"] >= f["train_start"]) & (m_df["purchase_date"] <= f["train_end"])
        val_mask = (m_df["purchase_date"] >= f["val_start"]) & (m_df["purchase_date"] <= f["val_end"])

        train_split = m_df[train_mask]
        val_split = m_df[val_mask]

        X_train = train_split[all_features]
        y_train = train_split["is_late"].values
        X_val = val_split[all_features]
        y_val = val_split["is_late"].values

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
            ]
        )

        X_tr_proc = preprocessor.fit_transform(X_train)
        X_v_proc = preprocessor.transform(X_val)

        eval_models = [
            ("Logistic Regression", LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED)),
            ("HistGradientBoosting", HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=30,
                l2_regularization=1.0, random_state=RANDOM_SEED
            ))
        ]

        val_prev = y_val.mean()
        print(f"\n{f['fold_id']}: Train={len(train_split)} rows, Val={len(val_split)} rows | Val Prev={val_prev*100:.2f}%")

        clean_note = f["notes"].replace(",", ";")

        for m_name, m_obj in eval_models:
            m_obj.fit(X_tr_proc, y_train)
            val_prob = m_obj.predict_proba(X_v_proc)[:, 1]
            pr_auc = average_precision_score(y_val, val_prob)
            roc_auc = roc_auc_score(y_val, val_prob)
            brier = brier_score_loss(y_val, val_prob)
            lift = pr_auc / val_prev if val_prev > 0 else 0.0

            print(f"  {m_name:22s} | PR-AUC: {pr_auc:.4f} (Lift: {lift:.2f}x) | ROC-AUC: {roc_auc:.4f} | Brier: {brier:.4f}")

            results.append({
                "fold_id": f["fold_id"],
                "model_name": m_name,
                "train_window": f"{f['train_start']} to {f['train_end']}",
                "val_window": f"{f['val_start']} to {f['val_end']}",
                "train_rows": len(train_split),
                "train_late_count": int(y_train.sum()),
                "train_late_rate_pct": round(float(y_train.mean() * 100), 2),
                "val_rows": len(val_split),
                "val_late_count": int(y_val.sum()),
                "val_late_rate_pct": round(float(val_prev * 100), 2),
                "val_pr_auc": round(float(pr_auc), 4),
                "val_roc_auc": round(float(roc_auc), 4),
                "val_brier_score": round(float(brier), 4),
                "val_lift_vs_prior": round(float(lift), 2),
                "notes": clean_note
            })

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(audit_dir, "rolling_origin_cv_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nRolling-origin evaluation saved to {out_csv}.")
    return res_df

if __name__ == "__main__":
    run_rolling_origin_cv()
