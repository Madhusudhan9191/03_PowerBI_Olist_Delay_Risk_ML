# Olist Delivery-Delay Risk Model
**Machine Learning & Power BI Operational Intelligence Portfolio Project**  
**Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))  
**Random Seed:** 42  
**Scope:** Offline Observational ML & Analytical Dashboard  

---

## 1. Project Overview & Business Problem

In an e-commerce marketplace, late deliveries erode customer satisfaction and drive repeat-purchase churn. However, operational customer support and logistics partner teams have finite bandwidth: they cannot manually track or escalate all outbound shipments.

### Primary Business Question
> *At the moment an order is handed to the logistics carrier (`order_delivered_carrier_date`), can we score orders by their probability of arriving after the promised date, early enough for operations to intervene under fixed capacity constraints?*

### Solution Architecture
1. **Point-in-Time Feature Engineering:** Computes operational signals strictly observable at or before carrier handover (merchant handling time, handover slack versus `shipping_limit_date`, promised lead time, distance, physical bulk, and historical expanding-window rates with an outcome lag).
2. **Chronological Machine Learning:** Evaluates Gradient Boosted Trees (`HistGradientBoostingClassifier`) versus regularized Logistic Regression and 3 non-trivial baselines across strict chronological splits (Train: 2017-01 to 2018-02; Validation: 2018-03 to 2018-05; Test: 2018-06 to 2018-08-22).
3. **Capacity-Constrained Evaluation:** Rather than relying on an arbitrary 0.5 threshold, performance is evaluated through **Top-k% Alert Queues (Top 2%, 5%, 10%)** and parametric scenario cost curves.
4. **Data Quality & Drift Monitoring:** Automated dbt-style assertions and PSI/KS drift tests tracking distribution shifts across operational regimes.
5. **Power BI Package:** Star-schema semantic model, DAX measure library, What-If simulation parameters, and 6-page interactive report layout.
6. **Multi-Layer Three-Way Reconciliation:** Programmatic verification asserting exact numerical equivalence across Python, SQL (DuckDB), and Power BI DAX exports.

---

## 2. Key Empirical Findings & Model Results

All numbers below were computed programmatically from out-of-time test evaluation and verified against canonical warehouse totals:

### Benchmark Comparison (Held-Out Test Set: 18,107 Orders, 3.69% Prevalence)

| Evaluation Metric | Boosted Model | Baseline B (Regional Rule) | Baseline A (Prior) | Benchmark Impact |
|---|---|---|---|---|
| **PR-AUC (Average Precision)** | **0.2811** | 0.0407 | 0.0369 | **7.6x lift** over prior baseline. |
| **ROC-AUC** | **0.7773** | 0.4969 | 0.5000 | Strong discriminative ability. |
| **Brier Score** | **0.1119** | 0.0378 | 0.0364 | Proper scoring rule (lower is better; higher than prior reflects raw tree overconfidence). |
| **Expected Calibration Error (ECE)**| **0.2228** | N/A | N/A | 10 probability bins (indicates uncalibrated raw probabilities in low-prevalence setting). |
| **Top 2% Capacity Queue** | **Recall: 22.27%** | Recall: 2.10% | Recall: 2.00% | **Precision: 41.16%, Lift: 11.14x** (362 alerts, 149 late caught). |
| **Top 5% Capacity Queue** | **Recall: 34.38%** | Recall: 5.30% | Recall: 5.00% | **Precision: 25.41%, Lift: 6.88x** (905 alerts, 230 late caught). |
| **Top 10% Capacity Queue** | **Recall: 51.57%** | Recall: 10.40% | Recall: 10.00% | **Precision: 19.06%, Lift: 5.16x** (1,810 alerts, 345 late caught). |

### Calibration Reality & Post-Hoc Scaling (Platt Scaling)
While raw boosted decision tree models produce overconfident probabilities in rare-event settings, applying **Sigmoid (Platt) Calibration** on the validation set resolves probability distortion while preserving complete ranking fidelity:
- **Brier Score:** Improved from uncalibrated `0.1119` down to **`0.0379`** (approaching the prior baseline of `0.0364`).
- **Expected Calibration Error (ECE):** Reduced from `0.2228` down to **`0.0706`** (68% reduction in calibration error).
- **Discrimination & Ranking:** PR-AUC (`0.2811`) and ROC-AUC (`0.7773`) are completely preserved because sigmoid scaling is strictly monotonic.
- **Rolling-Origin Temporal Stability:** Evaluated across 3 expanding chronological folds (`Fold 1: 0.3284`, `Fold 2: 0.2481`, `Fold 3: 0.2960`, `Test: 0.2811`), confirming non-trivial lift above prevalence across all operational regimes.
- **Explainability:** Verified using test-set permutation importance with fixed seed 42 (and `explainability_shap.py` explainer wrapper). Correlated features (`merchant_handling_days` vs `handover_slack_days`) are reported as associational attributions, not causal drivers.

### 95% Bootstrap Confidence Intervals (1,000 iterations)
- **Boosted Model PR-AUC:** `0.2815` [95% CI: `0.2479` – `0.3152`]
- **Baseline B PR-AUC:** `0.0415` [95% CI: `0.0367` – `0.0475`]
- **Recall @ Top 5% Queue:** `34.48%` [95% CI: `31.27%` – `37.75%`]
- *Conclusion:* Model PR-AUC and recall demonstrate higher empirical performance than all heuristic and regional rule baselines on the test cohort.

---

## 3. Repository Structure

```
03_PowerBI_Olist_Delay_Risk_ML/
├── .gitignore                          # Ignores __pycache__, .venv, scratch files
├── requirements.txt                    # Pinned library dependencies
├── config.yaml                         # Central project configuration & random seed 42
├── data/
│   ├── raw/                            # Untouched Olist Kaggle raw CSVs
│   ├── warehouse/                      # Validated relational warehouse CSVs & profiles
│   ├── processed/                      # Feature matrices (Parquet & CSV)
│   ├── monitoring/                     # Data quality log and drift summary CSVs
│   └── powerbi_export/                 # Exported tables for Power BI ingestion
├── documentation/
│   ├── phase0_requirements_and_definitions.md
│   ├── phase1_data_audit_for_ml.md
│   ├── metric_definitions.md
│   ├── feature_availability_spec.md
│   ├── model_card.md
│   ├── data_dictionary.md
│   ├── data_quality_and_monitoring_report.md
│   ├── powerbi_package_spec.md
│   ├── interview_defense_notes.md
│   └── change_log.md
├── models/
│   ├── best_delay_risk_model.joblib    # Serialized HistGradientBoosting model
│   ├── feature_preprocessor.joblib     # Serialized ColumnTransformer
│   └── model_metadata.json             # Pinned hyperparameters and metrics
├── src/
│   ├── audit/
│   │   ├── run_audit.py                # Comprehensive data audit & control total check
│   │   └── run_reconciliation.py       # Three-way cross-layer numerical reconciliation
│   ├── features/
│   │   ├── build_features.py           # Point-in-time feature engineering pipeline
│   │   └── export_powerbi_dimensions.py# Bounded dimension tables export
│   ├── models/
│   │   └── train_and_evaluate.py       # Training, baseline comparison & evaluation
│   └── monitoring/
│       └── run_monitoring_and_data_quality.py # PSI/KS drift & dbt-style assertions
└── tests/
    ├── test_audit_and_definitions.py   # Pytest suite for control totals & anomalies
    └── test_leakage_audit.py           # Pytest suite for point-in-time contract & leakage
```

---

## 4. How to Reproduce

### Step 1: Environment Setup
Ensure Python 3.11+ is installed. Clone the repository and install pinned dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Run Data Audit & Verification Tests
Verify that all warehouse tables match canonical control totals (96,470 valid delivered orders, 6,534 late, 6.77%):
```bash
python src/audit/run_audit.py
pytest tests/test_audit_and_definitions.py -v
```

### Step 3: Execute Point-in-Time Feature Engineering
Build the feature matrix with 1-day outcome lagged expanding window:
```bash
python src/features/build_features.py
pytest tests/test_leakage_audit.py -v
```

### Step 4: Train Models & Run Out-of-Time Evaluation
Execute baseline benchmarks, model training, validation threshold selection, bootstrap CIs, and Power BI table exports:
```bash
python src/models/train_and_evaluate.py
```

### Step 5: Execute Monitoring & Data Quality Assertions
Run PSI/KS drift calculations and automated data quality assertions:
```bash
python src/monitoring/run_monitoring_and_data_quality.py
```

### Step 6: Run Three-Way Multi-Layer Reconciliation
Execute automated reconciliation between Python, SQL (DuckDB), and DAX measure outputs:
```bash
python src/audit/run_reconciliation.py
```

---

## 5. Power BI Implementation Guide

1. Open Power BI Desktop and create a new report.
2. In Power Query, set the `SourceFolder` parameter to point to `data/powerbi_export`.
3. Ingest `fact_model_scored_orders.csv`, `dim_date.csv`, `dim_geography.csv`, `dim_risk_band.csv`, `fact_pr_curve_points.csv`, `fact_calibration_bins.csv`, `fact_threshold_cost_grid.csv`, `fact_feature_drift.csv`, and `fact_data_quality_log.csv`.
4. Configure 1-to-many single-direction relationships from dimension tables to fact tables as detailed in [powerbi_package_spec.md](03_PowerBI_Olist_Delay_Risk_ML/documentation/powerbi_package_spec.md).
5. Create the What-If parameters `Threshold_Param` (0.01 to 0.50) and `Cost_Ratio_Param` (1 to 20).
6. Copy the DAX Measure Library formulas into a dedicated `_Measures` table.
7. Build the 6 pages: (1) Problem & Baseline, (2) Model Performance & Threshold What-If, (3) At-Risk Order Queue, (4) Segment Performance, (5) Monitoring & Data Quality, (6) Model Card & Governance.
8. Export measure results via DAX Studio / Power BI (`EVALUATE ROW(...)`) to `dax_measure_results_desktop.csv` and re-run `python src/audit/run_reconciliation.py` to achieve complete three-way cross-layer PASS status.

---

## 6. Observational Scope & Ethical Disclosures

- **Non-Causal Language:** All metrics, coefficients, and feature importances describe observational associations. Phrases like "caused by", "proves", or "confirms" are strictly avoided.
- **Offline Observational Study:** This project represents an offline portfolio data product. No live carrier dispatch algorithms or customer notification triggers were executed in historical data.
- **Parametric Cost Scenarios:** Expected cost metrics are illustrative scenario evaluations based on user-selected cost ratios ($C_{FN} / C_{FP}$); they do not represent realized financial savings since intervention outcome data is absent.
