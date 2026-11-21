"""
Phase 3, 4, 5: Model Training, Tuning, Baseline Comparison & Comprehensive Evaluation
Project: Olist Delivery-Delay Risk Model
Path: 03_PowerBI_Olist_Delay_Risk_ML/src/models/train_and_evaluate.py

Executes:
- Chronological time-based splits (Train, Validation, Test)
- Three baseline models: Prior, Heuristic, Regional Route Rule
- Model training: Logistic Regression vs HistGradientBoosting (with/without class weights)
- Validation tuning & decision threshold selection
- Out-of-time test evaluation: PR-AUC, ROC-AUC, Brier score, ECE, Recall @ Top-k%
- 1,000-sample bootstrap confidence intervals
- Parameterized cost analysis across cost ratios
- Permutation feature importance & explainability
- Segment performance with sample size suppression policy
- Power BI analytical table exports
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
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, brier_score_loss,
    confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
)
from sklearn.inspection import permutation_importance

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculates Expected Calibration Error across n equal-width bins preserving canonical schema."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        mask = bin_indices == b
        bin_count = np.sum(mask)
        if bin_count > 0:
            avg_prob = np.mean(y_prob[mask])
            avg_true = np.mean(y_true[mask])
            ece += (bin_count / len(y_true)) * abs(avg_prob - avg_true)
            bin_stats.append({
                "bin_id": b,
                "bin_min": round(bins[b], 2),
                "bin_max": round(bins[b+1], 2),
                "bin_count": int(bin_count),
                "avg_predicted_risk": round(float(avg_prob), 4),
                "empirical_late_rate": round(float(avg_true), 4),
                "calibration_gap": round(float(abs(avg_prob - avg_true)), 4)
            })
    return round(float(ece), 4), bin_stats

def top_k_metrics(y_true, y_prob, k_pct, heuristic_pr=0.1344):
    """Calculates recall, precision, alert rate, and lift when alerting top k% highest-risk orders."""
    n_alert = max(1, int(np.floor(len(y_true) * (k_pct / 100.0))))
    top_indices = np.argsort(y_prob)[::-1][:n_alert]
    true_positives = np.sum(y_true[top_indices])
    total_positives = np.sum(y_true)
    total_orders = len(y_true)
    
    prevalence = total_positives / total_orders
    recall = true_positives / total_positives if total_positives > 0 else 0.0
    precision = true_positives / n_alert
    lift_prior = precision / prevalence if prevalence > 0 else 0.0
    lift_heuristic = (precision / heuristic_pr) if heuristic_pr > 0 else 0.0
    alert_rate = (n_alert / total_orders) * 100.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "k_pct": k_pct,
        "n_alerted": int(n_alert),
        "alert_rate_pct": round(float(alert_rate), 2),
        "true_positives": int(true_positives),
        "recall": round(float(recall), 4),
        "precision": round(float(precision), 4),
        "f1": round(float(f1), 4),
        "lift_vs_prior": round(float(lift_prior), 2),
        "lift_vs_heuristic": round(float(lift_heuristic), 2)
    }

def bootstrap_confidence_intervals(y_true, y_prob, n_bootstraps=1000):
    """Computes 95% bootstrap confidence intervals for PR-AUC, ROC-AUC, and Recall@5%."""
    print(f"Running {n_bootstraps} bootstrap iterations for confidence intervals...")
    pr_aucs = []
    roc_aucs = []
    recalls_top5 = []
    n = len(y_true)

    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if np.sum(y_b) == 0 or np.sum(y_b) == len(y_b):
            continue
        pr_aucs.append(average_precision_score(y_b, p_b))
        roc_aucs.append(roc_auc_score(y_b, p_b))
        recalls_top5.append(top_k_metrics(y_b, p_b, 5.0)["recall"])

    return {
        "pr_auc": {
            "mean": round(float(np.mean(pr_aucs)), 4),
            "ci_lower": round(float(np.percentile(pr_aucs, 2.5)), 4),
            "ci_upper": round(float(np.percentile(pr_aucs, 97.5)), 4)
        },
        "roc_auc": {
            "mean": round(float(np.mean(roc_aucs)), 4),
            "ci_lower": round(float(np.percentile(roc_aucs, 2.5)), 4),
            "ci_upper": round(float(np.percentile(roc_aucs, 97.5)), 4)
        },
        "recall_at_top5": {
            "mean": round(float(np.mean(recalls_top5)), 4),
            "ci_lower": round(float(np.percentile(recalls_top5, 2.5)), 4),
            "ci_upper": round(float(np.percentile(recalls_top5, 97.5)), 4)
        }
    }

def run_pipeline(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    export_dir = os.path.join(base_dir, "data", "powerbi_export")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    # 1. Load feature matrix
    parquet_path = os.path.join(data_dir, "orders_feature_matrix.parquet")
    print(f"Loading feature matrix from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Filter to steady-state modeling window
    m_df = df[df["modeling_window_tier"] == "modelling"].copy().reset_index(drop=True)
    print(f"Steady-state cohort size: {len(m_df)} orders.")

    # 2. Chronological Splits
    train_mask = m_df["purchase_date"] < "2018-03-01"
    val_mask = (m_df["purchase_date"] >= "2018-03-01") & (m_df["purchase_date"] < "2018-06-01")
    test_mask = m_df["purchase_date"] >= "2018-06-01"

    train_df = m_df[train_mask].copy().reset_index(drop=True)
    val_df = m_df[val_mask].copy().reset_index(drop=True)
    test_df = m_df[test_mask].copy().reset_index(drop=True)

    split_summary = [
        {"split": "Train", "rows": len(train_df), "positives": int(train_df["is_late"].sum()), "prevalence_pct": round(float(train_df["is_late"].mean() * 100), 4), "min_date": train_df["purchase_date"].min(), "max_date": train_df["purchase_date"].max()},
        {"split": "Validation", "rows": len(val_df), "positives": int(val_df["is_late"].sum()), "prevalence_pct": round(float(val_df["is_late"].mean() * 100), 4), "min_date": val_df["purchase_date"].min(), "max_date": val_df["purchase_date"].max()},
        {"split": "Test", "rows": len(test_df), "positives": int(test_df["is_late"].sum()), "prevalence_pct": round(float(test_df["is_late"].mean() * 100), 4), "min_date": test_df["purchase_date"].min(), "max_date": test_df["purchase_date"].max()},
    ]
    print("\n--- Chronological Time Splits ---")
    print(pd.DataFrame(split_summary))

    # Feature definitions
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

    # Handle category frequency capping (top 15 + 'other')
    top_categories = train_df["primary_category"].value_counts().nlargest(15).index.tolist()
    for d in [train_df, val_df, test_df]:
        d["primary_category"] = d["primary_category"].apply(lambda x: x if x in top_categories else "other")

    y_train = train_df["is_late"].values
    y_val = val_df["is_late"].values
    y_test = test_df["is_late"].values

    # ==========================================
    # 3. BASELINES EVALUATION
    # ==========================================
    print("\n--- Evaluating Baselines on Test Set ---")
    # Baseline A: Prior Probability
    p_train_prior = float(y_train.mean())
    pred_base_a = np.full(len(y_test), p_train_prior)
    pr_base_a = average_precision_score(y_test, pred_base_a)
    roc_base_a = 0.5000
    brier_base_a = brier_score_loss(y_test, pred_base_a)

    # Baseline B: Historical Corridor Rule
    rule_table = train_df.groupby(["order_distance_band", "order_customer_state"])["is_late"].agg(["count", "mean"]).reset_index()
    state_table = train_df.groupby("order_customer_state")["is_late"].mean().to_dict()
    rule_map = {(r["order_distance_band"], r["order_customer_state"]): r["mean"] for _, r in rule_table.iterrows() if r["count"] >= 10}
    
    pred_base_b = []
    for _, r in test_df.iterrows():
        key = (r["order_distance_band"], r["order_customer_state"])
        if key in rule_map:
            pred_base_b.append(rule_map[key])
        elif r["order_customer_state"] in state_table:
            pred_base_b.append(state_table[r["order_customer_state"]])
        else:
            pred_base_b.append(p_train_prior)
    pred_base_b = np.array(pred_base_b)
    pr_base_b = average_precision_score(y_test, pred_base_b)
    roc_base_b = roc_auc_score(y_test, pred_base_b)
    brier_base_b = brier_score_loss(y_test, pred_base_b)

    # Baseline C: Promised Lead Time Heuristic (Inverted lead time: shorter lead time -> higher risk)
    lead_time_clean = test_df["promised_lead_time_days"].fillna(test_df["promised_lead_time_days"].median()).values
    pred_base_c = 1.0 / np.maximum(lead_time_clean, 1.0)
    # Normalize heuristic score to 0..1
    pred_base_c = (pred_base_c - pred_base_c.min()) / (pred_base_c.max() - pred_base_c.min())
    pr_base_c = average_precision_score(y_test, pred_base_c)
    roc_base_c = roc_auc_score(y_test, pred_base_c)

    print(f"Baseline A (Prior): PR-AUC = {pr_base_a:.4f}, ROC-AUC = {roc_base_a:.4f}, Brier = {brier_base_a:.4f}")
    print(f"Baseline B (Regional Rule): PR-AUC = {pr_base_b:.4f}, ROC-AUC = {roc_base_b:.4f}, Brier = {brier_base_b:.4f}")
    print(f"Baseline C (Lead Time Heuristic): PR-AUC = {pr_base_c:.4f}, ROC-AUC = {roc_base_c:.4f}")

    # ==========================================
    # 4. MODEL TRAINING & VALIDATION SELECTION
    # ==========================================
    print("\n--- Training Machine Learning Models & Cross-Candidate Selection ---")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ]
    )

    X_train_proc = preprocessor.fit_transform(train_df[all_features])
    X_val_proc = preprocessor.transform(val_df[all_features])
    X_test_proc = preprocessor.transform(test_df[all_features])

    candidates = [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED)
        },
        {
            "name": "Logistic Regression (Balanced)",
            "model": LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
        },
        {
            "name": "HistGradientBoosting",
            "model": HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=30,
                l2_regularization=1.0, random_state=RANDOM_SEED
            )
        },
        {
            "name": "HistGradientBoosting (Balanced)",
            "model": HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=30,
                l2_regularization=1.0, class_weight="balanced", random_state=RANDOM_SEED
            )
        }
    ]

    candidate_results = []
    print("\nValidation Performance Comparison across all 4 Candidates:")
    for cand in candidates:
        m = cand["model"]
        name = cand["name"]
        m.fit(X_train_proc, y_train)
        val_p = m.predict_proba(X_val_proc)[:, 1]
        test_p = m.predict_proba(X_test_proc)[:, 1]

        v_pr = average_precision_score(y_val, val_p)
        v_roc = roc_auc_score(y_val, val_p)
        t_pr = average_precision_score(y_test, test_p)
        t_roc = roc_auc_score(y_test, test_p)
        t_rec5 = top_k_metrics(y_test, test_p, 5.0)["recall"]

        print(f"  {name:32s} | Val PR: {v_pr:.4f} | Val ROC: {v_roc:.4f} | Test PR: {t_pr:.4f} | Test ROC: {t_roc:.4f} | Test Rec@5%: {t_rec5*100:.1f}%")

        candidate_results.append({
            "model_name": name,
            "model_obj": m,
            "val_pr_auc": round(float(v_pr), 4),
            "val_roc_auc": round(float(v_roc), 4),
            "test_pr_auc": round(float(t_pr), 4),
            "test_roc_auc": round(float(t_roc), 4),
            "test_recall_at_top5": round(float(t_rec5), 4)
        })

    # Export candidate comparison audit table
    audit_dir = os.path.join(base_dir, "data", "audit")
    os.makedirs(audit_dir, exist_ok=True)
    df_candidates = pd.DataFrame(candidate_results).drop(columns=["model_obj"])
    df_candidates.to_csv(os.path.join(audit_dir, "model_candidate_comparison.csv"), index=False)

    # Programmatic Selection Rule: Select candidate with highest Validation PR-AUC
    best_candidate_info = max(candidate_results, key=lambda c: c["val_pr_auc"])
    best_model_name = best_candidate_info["model_name"]
    best_model = best_candidate_info["model_obj"]
    print(f"\n>>> Shipped Model Selected Programmatically: {best_model_name} (Val PR-AUC = {best_candidate_info['val_pr_auc']:.4f})")

    # Fit Probability Calibration (Platt / Sigmoid Scaling) on Validation Set
    print("\n--- Fitting Probability Calibration on Validation Set ---")
    calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(best_model), method="sigmoid")
    calibrator.fit(X_val_proc, y_val)
    val_prob_cal = calibrator.predict_proba(X_val_proc)[:, 1]
    raw_val_prob = best_model.predict_proba(X_val_proc)[:, 1]

    # Calibration-in-the-large (Mean predicted probability vs observed late rate)
    val_observed_rate = float(y_val.mean())
    val_mean_pred_raw = float(raw_val_prob.mean())
    val_mean_pred_cal = float(val_prob_cal.mean())

    # Select optimal decision threshold on VALIDATION set using calibrated probabilities
    prec_v, rec_v, thresh_v = precision_recall_curve(y_val, val_prob_cal)
    f1_v = 2 * (prec_v[:-1] * rec_v[:-1]) / np.maximum(prec_v[:-1] + rec_v[:-1], 1e-9)
    best_idx = np.argmax(f1_v)
    optimal_f1_threshold = round(float(thresh_v[best_idx]), 4)
    best_val_f1 = round(float(f1_v[best_idx]), 4)
    top5_val_threshold = round(float(np.percentile(val_prob_cal, 95.0)), 4)

    print(f"Validation Calibration-in-the-Large: Observed = {val_observed_rate*100:.2f}%, Raw Pred = {val_mean_pred_raw*100:.2f}%, Platt Pred = {val_mean_pred_cal*100:.2f}%")
    print(f"Validation Optimal F1 Threshold (Calibrated): {optimal_f1_threshold} (F1 = {best_val_f1})")

    # ==========================================
    # 5. TEST SET EVALUATION
    # ==========================================
    print("\n--- Out-of-Time Test Set Evaluation ---")
    raw_test_prob = best_model.predict_proba(X_test_proc)[:, 1]
    test_prob = calibrator.predict_proba(X_test_proc)[:, 1]

    # Calibration-in-the-large on Test
    test_observed_rate = float(y_test.mean())
    test_mean_pred_raw = float(raw_test_prob.mean())
    test_mean_pred_cal = float(test_prob.mean())

    # Test Recalibration on Recent Window (May 2018 post-shock consolidation period)
    may_mask = (val_df["purchase_date"] >= "2018-05-01")
    X_may_proc = X_val_proc[may_mask]
    y_may = y_val[may_mask]
    calibrator_recent = CalibratedClassifierCV(estimator=FrozenEstimator(best_model), method="sigmoid")
    calibrator_recent.fit(X_may_proc, y_may)
    test_prob_recent = calibrator_recent.predict_proba(X_test_proc)[:, 1]
    test_brier_recent = brier_score_loss(y_test, test_prob_recent)
    test_mean_pred_recent = float(test_prob_recent.mean())

    # Primary Metrics
    test_pr_auc = average_precision_score(y_test, test_prob)
    test_roc_auc = roc_auc_score(y_test, test_prob)
    test_brier = brier_score_loss(y_test, test_prob)
    raw_test_brier = brier_score_loss(y_test, raw_test_prob)

    # Brier Skill Score (BSS) relative to the training prior baseline
    p_train_prior = float(y_train.mean())
    brier_prior_base = brier_score_loss(y_test, np.full_like(y_test, p_train_prior, dtype=float))
    bss_calibrated = 1.0 - (test_brier / brier_prior_base)
    bss_recent = 1.0 - (test_brier_recent / brier_prior_base)

    test_ece, cal_bin_records = calculate_ece(y_test, test_prob, n_bins=10)
    raw_test_ece, _ = calculate_ece(y_test, raw_test_prob, n_bins=10)

    # Threshold evaluation on test set using the VALIDATION-chosen threshold
    test_pred_binary = (test_prob >= optimal_f1_threshold).astype(int)
    cm = confusion_matrix(y_test, test_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    test_precision = precision_score(y_test, test_pred_binary)
    test_recall = recall_score(y_test, test_pred_binary)
    test_f1 = f1_score(y_test, test_pred_binary)
    test_alert_orders = int(tp + fp)
    test_alert_rate = float(test_alert_orders / len(y_test) * 100)

    # Capacity-constrained evaluation (Top 2%, 5%, 10%)
    top2_eval = top_k_metrics(y_test, test_prob, 2.0, heuristic_pr=pr_base_c)
    top5_eval = top_k_metrics(y_test, test_prob, 5.0, heuristic_pr=pr_base_c)
    top10_eval = top_k_metrics(y_test, test_prob, 10.0, heuristic_pr=pr_base_c)

    print(f"Primary Metric (PR-AUC): {test_pr_auc:.4f} (Lead Heuristic Baseline C = {pr_base_c:.4f}, Lift = {test_pr_auc/pr_base_c:.2f}x; Prior Baseline A = {pr_base_a:.4f}, Lift = {test_pr_auc/pr_base_a:.2f}x)")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"Test Calibration-in-the-Large: Observed = {test_observed_rate*100:.2f}%, Platt Pred = {test_mean_pred_cal*100:.2f}%, May-Recalibrated Pred = {test_mean_pred_recent*100:.2f}%")
    print(f"Test Brier Score: {test_brier:.4f} (Prior Baseline = {brier_prior_base:.4f}, BSS = {bss_calibrated:.4f}; May-Recalibrated Brier = {test_brier_recent:.4f}, BSS = {bss_recent:.4f})")
    print(f"Test ECE: Calibrated = {test_ece:.4f}, Raw = {raw_test_ece:.4f}")
    print(f"At Validation-Tuned F1 Threshold ({optimal_f1_threshold}):")
    print(f"  Alerted Orders: {test_alert_orders} ({test_alert_rate:.2f}% of cohort) | TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Precision: {test_precision*100:.2f}%, Recall: {test_recall*100:.2f}%, F1: {test_f1:.4f}")
    print(f"Capacity Operating Points:")
    print(f"  Top 2% Capacity: Alert={top2_eval['n_alerted']} ({top2_eval['alert_rate_pct']}%), Recall={top2_eval['recall']*100:.2f}%, Precision={top2_eval['precision']*100:.2f}%, F1={top2_eval['f1']:.4f}, Lift vs Prior={top2_eval['lift_vs_prior']}x, Lift vs Heuristic={top2_eval['lift_vs_heuristic']}x")
    print(f"  Top 5% Capacity: Alert={top5_eval['n_alerted']} ({top5_eval['alert_rate_pct']}%), Recall={top5_eval['recall']*100:.2f}%, Precision={top5_eval['precision']*100:.2f}%, F1={top5_eval['f1']:.4f}, Lift vs Prior={top5_eval['lift_vs_prior']}x, Lift vs Heuristic={top5_eval['lift_vs_heuristic']}x")
    print(f"  Top 10% Capacity: Alert={top10_eval['n_alerted']} ({top10_eval['alert_rate_pct']}%), Recall={top10_eval['recall']*100:.2f}%, Precision={top10_eval['precision']*100:.2f}%, F1={top10_eval['f1']:.4f}, Lift vs Prior={top10_eval['lift_vs_prior']}x, Lift vs Heuristic={top10_eval['lift_vs_heuristic']}x")

    # Bootstrap Confidence Intervals (1,000 iterations)
    boot_ci = bootstrap_confidence_intervals(y_test, test_prob, n_bootstraps=1000)
    boot_base_b = bootstrap_confidence_intervals(y_test, pred_base_b, n_bootstraps=1000)
    boot_base_c = bootstrap_confidence_intervals(y_test, pred_base_c, n_bootstraps=1000)

    print("\n--- 95% Bootstrap Confidence Intervals ---")
    print(f"{best_model_name} PR-AUC: {boot_ci['pr_auc']['mean']} [95% CI: {boot_ci['pr_auc']['ci_lower']} - {boot_ci['pr_auc']['ci_upper']}]")
    print(f"Baseline C (Heuristic) PR-AUC: {boot_base_c['pr_auc']['mean']} [95% CI: {boot_base_c['pr_auc']['ci_lower']} - {boot_base_c['pr_auc']['ci_upper']}]")
    print(f"{best_model_name} Recall@5%: {boot_ci['recall_at_top5']['mean']*100:.2f}% [95% CI: {boot_ci['recall_at_top5']['ci_lower']*100:.2f}% - {boot_ci['recall_at_top5']['ci_upper']*100:.2f}%]")

    # ==========================================
    # 6. PARAMETERIZED COST ANALYSIS
    # ==========================================
    print("\n--- Parameterized Scenario Cost Analysis ---")
    # Cost grid across thresholds and cost ratios
    # Cost = C_FP * FP + C_FN * FN. Assume C_FP = 1.0 (unit intervention cost)
    cost_ratios = [2.0, 5.0, 10.0, 20.0]
    thresh_grid = np.linspace(0.01, 0.50, 50)
    cost_grid_records = []

    for c_ratio in cost_ratios:
        c_fp = 1.0
        c_fn = c_ratio
        baseline_cost = c_fn * np.sum(y_test) # If no intervention (all positives missed)

        for t in thresh_grid:
            p_bin = (test_prob >= t).astype(int)
            c_m = confusion_matrix(y_test, p_bin)
            t_tn, t_fp, t_fn, t_tp = c_m.ravel()
            tot_cost = c_fp * t_fp + c_fn * t_fn
            cost_saving_pct = ((baseline_cost - tot_cost) / baseline_cost) * 100.0

            prec = precision_score(y_test, p_bin, zero_division=0)
            rec = recall_score(y_test, p_bin, zero_division=0)
            f1 = f1_score(y_test, p_bin, zero_division=0)

            cost_grid_records.append({
                "cost_ratio_cfn_cfp": c_ratio,
                "threshold": round(float(t), 3),
                "false_positives": int(t_fp),
                "false_negatives": int(t_fn),
                "true_positives": int(t_tp),
                "true_negatives": int(t_tn),
                "total_expected_cost": round(float(tot_cost), 1),
                "baseline_cost": round(float(baseline_cost), 1),
                "cost_saving_pct": round(float(cost_saving_pct), 2),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "alert_rate_pct": round(float((t_tp + t_fp) / len(y_test) * 100), 2)
            })

    df_cost_grid = pd.DataFrame(cost_grid_records)
    df_cost_grid.to_csv(os.path.join(export_dir, "fact_threshold_cost_grid.csv"), index=False)

    # ==========================================
    # 7. FEATURE IMPORTANCE & EXPLAINABILITY
    # ==========================================
    print("\n--- Computing Feature Importance & Attributions ---")
    perm_res = permutation_importance(
        best_model, X_test_proc, y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_SEED
    )

    feature_names = num_features + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features))
    feat_imp = pd.DataFrame({
        "feature_name": feature_names,
        "importance_mean": [round(float(x), 6) for x in perm_res.importances_mean],
        "importance_std": [round(float(x), 6) for x in perm_res.importances_std]
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    if hasattr(best_model, "coef_"):
        coef_map = dict(zip(feature_names, best_model.coef_[0]))
        feat_imp["standardized_coefficient"] = feat_imp["feature_name"].map(coef_map).round(4)

    feat_imp.to_csv(os.path.join(export_dir, "feature_importance.csv"), index=False)
    print("Top 10 Features by Permutation Importance:")
    print(feat_imp.head(10).to_string(index=False))

    # ==========================================
    # 8. SUBGROUP / SEGMENT PERFORMANCE
    # ==========================================
    print("\n--- Computing Subgroup Segment Performance ---")
    test_eval_df = test_df.copy()
    test_eval_df["predicted_probability"] = test_prob
    test_eval_df["predicted_binary"] = test_pred_binary

    segment_records = []
    
    # State segments
    for state, grp in test_eval_df.groupby("order_customer_state"):
        n_rows = len(grp)
        pos = int(grp["is_late"].sum())
        status = "INTERPRETABLE" if n_rows >= 100 else ("EXPLORATORY" if n_rows >= 30 else "SUPPRESSED")
        
        pr_auc = round(float(average_precision_score(grp["is_late"], grp["predicted_probability"])), 4) if pos > 0 and status != "SUPPRESSED" else np.nan
        roc = round(float(roc_auc_score(grp["is_late"], grp["predicted_probability"])), 4) if pos > 0 and (n_rows - pos) > 0 and status != "SUPPRESSED" else np.nan
        rec5 = round(float(top_k_metrics(grp["is_late"].values, grp["predicted_probability"].values, 5.0)["recall"]), 4) if pos > 0 and status != "SUPPRESSED" else np.nan

        segment_records.append({
            "segment_type": "customer_state",
            "segment_value": state,
            "order_count": n_rows,
            "late_orders": pos,
            "empirical_late_rate_pct": round(float(pos / n_rows * 100), 2),
            "sample_size_tier": status,
            "segment_pr_auc": pr_auc,
            "segment_roc_auc": roc,
            "segment_recall_at_top5": rec5
        })

    # Order Value Tiers
    test_eval_df["order_value_tier"] = pd.qcut(test_eval_df["total_order_price"], q=4, labels=["Q1_Low", "Q2_Medium", "Q3_High", "Q4_VeryHigh"])
    for val_tier, grp in test_eval_df.groupby("order_value_tier"):
        n_rows = len(grp)
        pos = int(grp["is_late"].sum())
        pr_auc = round(float(average_precision_score(grp["is_late"], grp["predicted_probability"])), 4)
        roc = round(float(roc_auc_score(grp["is_late"], grp["predicted_probability"])), 4)
        rec5 = round(float(top_k_metrics(grp["is_late"].values, grp["predicted_probability"].values, 5.0)["recall"]), 4)

        segment_records.append({
            "segment_type": "order_value_tier",
            "segment_value": str(val_tier),
            "order_count": n_rows,
            "late_orders": pos,
            "empirical_late_rate_pct": round(float(pos / n_rows * 100), 2),
            "sample_size_tier": "INTERPRETABLE",
            "segment_pr_auc": pr_auc,
            "segment_roc_auc": roc,
            "segment_recall_at_top5": rec5
        })

    df_segments = pd.DataFrame(segment_records)
    df_segments.to_csv(os.path.join(export_dir, "fact_segment_performance.csv"), index=False)

    # ==========================================
    # 9. MONTHLY STABILITY WITH BOOTSTRAP CIS
    # ==========================================
    print("\n--- Computing Monthly Stability with Bootstrap Confidence Intervals ---")
    monthly_records = []
    monthly_ci_records = []
    for ym, grp in test_eval_df.groupby("purchase_ym"):
        n_rows = len(grp)
        pos = int(grp["is_late"].sum())
        pr_auc = round(float(average_precision_score(grp["is_late"], grp["predicted_probability"])), 4)
        roc = round(float(roc_auc_score(grp["is_late"], grp["predicted_probability"])), 4)
        rec5 = round(float(top_k_metrics(grp["is_late"].values, grp["predicted_probability"].values, 5.0)["recall"]), 4)
        brier = round(float(brier_score_loss(grp["is_late"], grp["predicted_probability"])), 4)

        # Monthly bootstrap CIs (500 samples)
        m_pr_aucs = []
        m_rec_top5s = []
        y_m = grp["is_late"].values
        p_m = grp["predicted_probability"].values
        m_len = len(y_m)
        for _ in range(500):
            b_idx = np.random.choice(m_len, size=m_len, replace=True)
            if np.sum(y_m[b_idx]) > 0 and np.sum(y_m[b_idx]) < m_len:
                m_pr_aucs.append(average_precision_score(y_m[b_idx], p_m[b_idx]))
                m_rec_top5s.append(top_k_metrics(y_m[b_idx], p_m[b_idx], 5.0)["recall"])

        pr_ci_lo = round(float(np.percentile(m_pr_aucs, 2.5)), 4) if m_pr_aucs else pr_auc
        pr_ci_hi = round(float(np.percentile(m_pr_aucs, 97.5)), 4) if m_pr_aucs else pr_auc
        rec_ci_lo = round(float(np.percentile(m_rec_top5s, 2.5)), 4) if m_rec_top5s else rec5
        rec_ci_hi = round(float(np.percentile(m_rec_top5s, 97.5)), 4) if m_rec_top5s else rec5

        # Canonical 9 columns expected by Power Query
        monthly_records.append({
            "year_month": ym,
            "order_count": n_rows,
            "late_orders": pos,
            "late_rate_pct": round(float(pos / n_rows * 100), 2),
            "mean_predicted_prob": round(float(grp["predicted_probability"].mean()), 4),
            "pr_auc": pr_auc,
            "roc_auc": roc,
            "brier_score": brier,
            "recall_at_top5": rec5
        })

        # Separate audit table for statistical confidence intervals
        monthly_ci_records.append({
            "year_month": ym,
            "order_count": n_rows,
            "late_orders": pos,
            "pr_auc": pr_auc,
            "pr_auc_ci_lower": pr_ci_lo,
            "pr_auc_ci_upper": pr_ci_hi,
            "roc_auc": roc,
            "brier_score": brier,
            "recall_at_top5": rec5,
            "recall_top5_ci_lower": rec_ci_lo,
            "recall_top5_ci_upper": rec_ci_hi
        })
        print(f"  {ym}: Orders={n_rows}, Late={pos} ({pos/n_rows*100:.2f}%) | PR-AUC={pr_auc:.4f} [{pr_ci_lo}-{pr_ci_hi}] | Recall@5%={rec5*100:.1f}% [{rec_ci_lo*100:.1f}%-{rec_ci_hi*100:.1f}%]")

    df_monthly = pd.DataFrame(monthly_records)
    df_monthly.to_csv(os.path.join(export_dir, "fact_monthly_monitoring.csv"), index=False)

    # Save detailed bootstrap intervals to audit
    audit_dir = os.path.join(base_dir, "data", "audit")
    os.makedirs(audit_dir, exist_ok=True)
    pd.DataFrame(monthly_ci_records).to_csv(os.path.join(audit_dir, "monthly_bootstrap_cis.csv"), index=False)

    # ==========================================
    # 10. POWER BI ARTIFACT EXPORTS
    # ==========================================
    # PR Curve Points Export (Smooth sorted numerical recall axis with 101 points)
    prec_raw, rec_raw, th_raw = precision_recall_curve(y_test, test_prob)
    eval_recalls = np.linspace(0.0, 1.0, 101)
    eval_precisions = []
    eval_thresholds = []

    sort_idx = np.argsort(rec_raw)
    rec_sorted = rec_raw[sort_idx]
    prec_sorted = prec_raw[sort_idx]
    th_full = np.append(th_raw, 1.0)[sort_idx]

    test_prev = float(y_test.mean())
    for r in eval_recalls:
        mask = rec_sorted >= r
        if np.any(mask):
            eval_precisions.append(round(float(np.max(prec_sorted[mask])), 4))
            eval_thresholds.append(round(float(th_full[mask][0]), 4))
        else:
            eval_precisions.append(round(float(test_prev), 4))
            eval_thresholds.append(0.0)

    pr_pts_df = pd.DataFrame({
        "recall": [round(float(r), 2) for r in eval_recalls],
        "precision": eval_precisions,
        "threshold": eval_thresholds
    })
    pr_pts_df.to_csv(os.path.join(export_dir, "fact_pr_curve_points.csv"), index=False)

    # Calibration Bins Export
    df_cal = pd.DataFrame(cal_bin_records)
    df_cal.to_csv(os.path.join(export_dir, "fact_calibration_bins.csv"), index=False)

    # Scored Test Orders Export with EXACT rank-aligned capacity queues
    test_export_df = test_eval_df[[
        "order_id", "purchase_date", "purchase_ym", "order_delivered_carrier_date",
        "order_customer_state", "customer_region", "primary_category", "primary_payment_type",
        "total_order_price", "total_freight_value", "promised_lead_time_days", "merchant_handling_days",
        "handover_slack_days", "is_late_dispatch", "max_route_distance_km", "is_interstate",
        "predicted_probability", "predicted_binary", "is_late"
    ]].copy()

    # Exact rank-based risk band assignment: guarantees exactly 362 orders in Top 2% queue
    sorted_order_indices = test_export_df.sort_values(
        by=["predicted_probability", "order_id"], ascending=[False, True]
    ).index
    risk_bands = pd.Series(index=test_export_df.index, dtype="object")

    n_tot = len(test_export_df)
    n_2 = int(np.floor(0.02 * n_tot))    # 362 orders
    n_5 = int(np.floor(0.05 * n_tot))    # 905 orders
    n_10 = int(np.floor(0.10 * n_tot))   # 1810 orders

    risk_bands.loc[sorted_order_indices[:n_2]] = "Critical Queue (Top 2%)"
    risk_bands.loc[sorted_order_indices[n_2:n_5]] = "High Risk (95-98%)"
    risk_bands.loc[sorted_order_indices[n_5:n_10]] = "Moderate Risk (80-95%)"
    risk_bands.loc[sorted_order_indices[n_10:]] = "Low Risk (0-80%)"

    test_export_df["risk_band"] = risk_bands
    test_export_df.to_csv(os.path.join(export_dir, "fact_model_scored_orders.csv"), index=False)

    # ==========================================
    # 11. SAVE MODEL ARTIFACTS & METADATA
    # ==========================================
    joblib.dump(best_model, os.path.join(models_dir, "best_delay_risk_model.joblib"))
    joblib.dump(calibrator, os.path.join(models_dir, "calibrated_delay_risk_model.joblib"))
    joblib.dump(preprocessor, os.path.join(models_dir, "feature_preprocessor.joblib"))

    model_metadata = {
        "model_name": best_model_name,
        "algorithm": type(best_model).__name__,
        "selection_rule": "Programmatic selection by highest Validation PR-AUC across 4 candidates",
        "candidate_comparison": candidate_results,
        "calibration": "Sigmoid (Platt Scaling) fitted on validation set",
        "hyperparameters": {k: str(v) for k, v in best_model.get_params().items() if not k.startswith("_")},
        "training_data_period": f"{train_df['purchase_date'].min()} to {train_df['purchase_date'].max()}",
        "validation_data_period": f"{val_df['purchase_date'].min()} to {val_df['purchase_date'].max()}",
        "test_data_period": f"{test_df['purchase_date'].min()} to {test_df['purchase_date'].max()}",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "optimal_threshold_f1": optimal_f1_threshold,
        "top5_threshold": top5_val_threshold,
        "calibration_in_the_large": {
            "validation_observed_late_rate": round(val_observed_rate, 4),
            "validation_mean_predicted_prob": round(val_mean_pred_cal, 4),
            "test_observed_late_rate": round(test_observed_rate, 4),
            "test_mean_predicted_prob": round(test_mean_pred_cal, 4),
            "may_recalibrated_test_mean_predicted_prob": round(test_mean_pred_recent, 4)
        },
        "test_metrics": {
            "pr_auc": round(float(test_pr_auc), 4),
            "roc_auc": round(float(test_roc_auc), 4),
            "calibrated_brier_score": round(float(test_brier), 4),
            "uncalibrated_brier_score": round(float(raw_test_brier), 4),
            "brier_skill_score_vs_prior": round(float(bss_calibrated), 4),
            "may_recalibrated_brier_score": round(float(test_brier_recent), 4),
            "may_recalibrated_bss_vs_prior": round(float(bss_recent), 4),
            "calibrated_ece": test_ece,
            "uncalibrated_ece": raw_test_ece,
            "f1_optimal_threshold": optimal_f1_threshold,
            "precision_at_opt_threshold": round(float(test_precision), 4),
            "recall_at_opt_threshold": round(float(test_recall), 4),
            "f1_at_opt_threshold": round(float(test_f1), 4),
            "alerted_orders_at_opt_threshold": test_alert_orders,
            "alert_rate_at_opt_threshold": round(float(test_alert_rate), 2),
            "top2_pct_queue": top2_eval,
            "top5_pct_queue": top5_eval,
            "top10_pct_queue": top10_eval
        },
        "bootstrap_95_ci": boot_ci,
        "baseline_comparison": {
            "baseline_a_prior_pr_auc": round(float(pr_base_a), 4),
            "baseline_b_route_rule_pr_auc": round(float(pr_base_b), 4),
            "baseline_c_heuristic_pr_auc": round(float(pr_base_c), 4),
            "model_lift_vs_heuristic_primary": round(float(test_pr_auc / pr_base_c), 2),
            "model_lift_vs_prior": round(float(test_pr_auc / pr_base_a), 2),
            "model_lift_vs_route_rule": round(float(test_pr_auc / pr_base_b), 2)
        }
    }

    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2, default=str)

    # Save flat metadata CSV for Power BI Source
    meta_flat = [
        {"property": "Model Algorithm", "value": best_model_name},
        {"property": "Selection Rule", "value": "Highest Validation PR-AUC across 4 candidates"},
        {"property": "Calibration Method", "value": "Platt Scaling (Sigmoid)"},
        {"property": "Train Window", "value": model_metadata["training_data_period"]},
        {"property": "Validation Window", "value": model_metadata["validation_data_period"]},
        {"property": "Test Window", "value": model_metadata["test_data_period"]},
        {"property": "Test Sample Size", "value": str(len(test_df))},
        {"property": "Test Positive Prevalence", "value": f"{y_test.mean()*100:.2f}%"},
        {"property": "Test PR-AUC", "value": f"{test_pr_auc:.4f}"},
        {"property": "Primary Lift vs Lead Heuristic (0.1344)", "value": f"{test_pr_auc/pr_base_c:.2f}x"},
        {"property": "Lift vs Prior Baseline (0.0369)", "value": f"{test_pr_auc/pr_base_a:.2f}x"},
        {"property": "Test ROC-AUC", "value": f"{test_roc_auc:.4f}"},
        {"property": "Calibrated Brier Score", "value": f"{test_brier:.4f}"},
        {"property": "Brier Skill Score vs Prior", "value": f"{bss_calibrated:.4f}"},
        {"property": "May-Recalibrated Brier Score", "value": f"{test_brier_recent:.4f}"},
        {"property": "Calibrated ECE", "value": f"{test_ece:.4f}"},
        {"property": "Uncalibrated ECE", "value": f"{raw_test_ece:.4f}"},
        {"property": "Top 2% Queue Precision", "value": f"{top2_eval['precision']*100:.2f}%"},
        {"property": "Top 2% Queue Recall", "value": f"{top2_eval['recall']*100:.2f}%"},
        {"property": "Top 2% Queue Orders", "value": f"{top2_eval['n_alerted']} ({top2_eval['alert_rate_pct']}%)"},
        {"property": "Top 5% Queue Precision", "value": f"{top5_eval['precision']*100:.2f}%"},
        {"property": "Top 5% Queue Recall", "value": f"{top5_eval['recall']*100:.2f}%"},
        {"property": "Top 5% Queue Orders", "value": f"{top5_eval['n_alerted']} ({top5_eval['alert_rate_pct']}%)"},
        {"property": "Top 10% Queue Precision", "value": f"{top10_eval['precision']*100:.2f}%"},
        {"property": "Top 10% Queue Recall", "value": f"{top10_eval['recall']*100:.2f}%"},
        {"property": "Top 10% Queue Orders", "value": f"{top10_eval['n_alerted']} ({top10_eval['alert_rate_pct']}%)"},
        {"property": "F1 Optimal Threshold", "value": f"{optimal_f1_threshold:.4f}"},
        {"property": "PR-AUC 95% CI", "value": f"[{boot_ci['pr_auc']['ci_lower']} - {boot_ci['pr_auc']['ci_upper']}]"}
    ]
    pd.DataFrame(meta_flat).to_csv(os.path.join(export_dir, "model_metadata.csv"), index=False)

    print("\nTraining, evaluation, and Power BI package export completed successfully!")
    return model_metadata

if __name__ == "__main__":
    run_pipeline()
