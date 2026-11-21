# Project Change Log
**Project:** Olist Delivery-Delay Risk Model  
**Directory:** `03_PowerBI_Olist_Delay_Risk_ML/documentation/change_log.md`  

---

## [1.0.0] - Phase 0 & Phase 1: Requirements, Definitions & Data Audit
- Created `03_PowerBI_Olist_Delay_Risk_ML/` adhering strictly to non-negotiable rules.
- Replicated validated warehouse tables and canonical control totals (96,470 valid delivered orders, 6,534 late, 6.7731%).
- Quantified sequence anomalies: 1,373 anomalous orders (1.4232%). Verified clean vs full late rate sensitivity (+0.0726 pp).
- Quantified Feb–Mar 2018 delay surge: Feb (14.13%) and March (18.96%), driven by mean carrier transit surging to 13.4 days.
- Established right-censoring cutoff at 2018-08-22 to guarantee a 55-day outcome observation window (99th transit percentile = 40.98 days).
- Defined multi-item and multi-seller aggregation bottleneck rules (`max_route_distance_km`, `min_handover_slack_days`, `is_any_late_dispatch`).

## [2.0.0] - Phase 2: Point-in-Time Feature Engineering
- Implemented `src/features/build_features.py` constructing 47 point-in-time features evaluated strictly at `order_delivered_carrier_date`.
- Constructed expanding-window historical delay rates with an explicit **1-day outcome lag** (zero future lookahead).
- Implemented empirical Bayes (M-estimate) smoothing toward global and corridor priors, with explicit cold-start flags (`seller_no_history_flag`).
- Implemented automated leakage test suite in `tests/test_leakage_audit.py` (forbidden column absence, duration integrity, expanding window lag).

## [3.0.0] - Phase 3, 4, 5: Splits, Baselines, Model Training & Comprehensive Evaluation *(SUPERSEDED by programmatic selection in v6.0.0)*
- Structured strict chronological splits on steady-state cohort (95,707 orders):
  - Train: 57,050 orders (2017-01-05 to 2018-02-28, 6.63% late)
  - Validation: 20,550 orders (2018-03-01 to 2018-05-31, 10.11% late)
  - Test: 18,107 orders (2018-06-01 to 2018-08-22, 3.69% late)
- Evaluated three baselines on held-out test cohort:
  - Baseline A (Prior): PR-AUC = 0.0369, ROC-AUC = 0.5000, Brier = 0.0364
  - Baseline B (Regional Route Rule): PR-AUC = 0.0407, ROC-AUC = 0.4969, Brier = 0.0378
  - Baseline C (Lead Time Heuristic): PR-AUC = 0.1344, ROC-AUC = 0.7510
- *(Superseded Initial Run)*: Initially trained HistGradientBoostingClassifier. *(Note: Superseded in v6.0.0 by programmatic validation PR-AUC selection yielding Logistic Regression Balanced).*
- Superseded Initial Test Metrics:
  - **PR-AUC:** `0.2811` (superseded by `0.2933`)
  - **ROC-AUC:** `0.7773` (superseded by `0.8364`)
  - **Brier Score:** `0.1119`, **ECE:** `0.2228` (superseded by Calibrated Brier `0.0439`, Calibrated ECE `0.0991`)
  - **Capacity Queues:**
    - Top 2% Queue: Alert=362, Recall=22.27%, Precision=41.16%, Lift=11.14x
    - Top 5% Queue: Alert=905, Recall=34.38%, Precision=25.41%, Lift=6.88x
    - Top 10% Queue: Alert=1,810, Recall=51.57%, Precision=19.06%, Lift=5.16x
- Executed 1,000 bootstrap iterations indicating non-overlapping 95% confidence intervals versus baselines.
- Computed permutation feature importance identifying `merchant_handling_days` (+0.1960), `promised_lead_time_days` (+0.1698), and `handover_slack_days` (+0.0868) as top observational predictors.
- Computed parameterized cost analysis across cost ratios ($C_{FN}/C_{FP} \in [2, 5, 10, 20]$).
- Computed subgroup performance by state and order value tier enforcing sample-size policy (<30 suppress, 30-99 exploratory, 100+ interpretable).

## [4.0.0] - Phase 6: Monitoring, Drift & Data Quality
- Implemented `src/monitoring/run_monitoring_and_data_quality.py`.
- Computed PSI and KS statistics: observed that physical order attributes (`total_order_price`, `freight_to_price_ratio`, `max_route_distance_km`, `total_product_weight_g`, `max_product_volume_cm3`) remained stable ($\text{PSI} < 0.035$), while operational buffer features exhibited higher drift ($\text{PSI} > 0.25$) coincident with 2018 distribution changes.
- Implemented 15 dbt-style assertions: 14 PASSED, 1 real chronology anomaly finding on 158 historical sequences (DQ-15).

## [5.0.0] - Phase 7, 8, 9: Power BI Package, Reconciliation & Documentation
- Exported complete star-schema tables to `data/powerbi_export/`: `fact_model_scored_orders.csv`, `dim_date.csv` (bounded to 2018-10-17), `dim_geography.csv`, `dim_risk_band.csv`, `fact_threshold_cost_grid.csv`, `fact_pr_curve_points.csv`, `fact_calibration_bins.csv`, `fact_segment_performance.csv`, `fact_monthly_monitoring.csv`, `fact_data_quality_log.csv`, `fact_feature_drift.csv`, `model_metadata.csv`.
- Authored DAX Measure Library and Power BI visual layout in `documentation/powerbi_package_spec.md`.
- Implemented three-way reconciliation script `src/audit/run_reconciliation.py`: 12 out of 12 Python vs SQL checks PASSED with 100% exact numerical match. Formally logged DAX status as `PENDING_DESKTOP` per Rule 5.
- Completed Model Card (`documentation/model_card.md`), Data Dictionary (`documentation/data_dictionary.md`), Data Quality & Monitoring Report (`documentation/data_quality_and_monitoring_report.md`), Interview Defense Notes (`documentation/interview_defense_notes.md`), and comprehensive project `README.md`.

## [6.0.0] - Production Model Audit, Programmatic Selection & Visual Alignment
- **Programmatic Model Selection**: Replaced hardcoded model selection with algorithmic selection on Validation PR-AUC across all four candidates.
  - Winner: **Logistic Regression (Balanced)** (Validation PR-AUC `0.2905`).
  - *Statistical Selection Note:* On validation, Logistic Regression Balanced (0.2905) and plain Logistic Regression (0.2848) were within statistical noise (+0.0057 pp). While plain Logistic Regression scored higher on test (0.3344 vs 0.2933), the protocol strictly forbade peeking at test results during selection. The rule picked the balanced model honestly based on pre-test validation evidence.
- **Out-of-Time Test Benchmarks (Shipped Model)**:
  - PR-AUC: `0.2933` (95% CI: `[0.2603 - 0.3251]`), delivering a **2.18x lift** over the Lead-Time Heuristic (0.1344) and **7.94x lift** over the Prior (0.0369).
  - ROC-AUC: `0.8364` (vs 0.7692 for boosted trees).
  - Calibrated Brier Score: `0.0439`, Calibrated ECE: `0.0991`.
  - Operational Top 2% Capacity Queue (362 orders): **152 late orders captured**, 41.99% precision, 22.72% recall, **3.12x lift** over heuristic.
- **Power BI Canonical Schema Lock**:
  - Restored strict 7-column schema for `fact_calibration_bins.csv` and 9-column schema for `fact_monthly_monitoring.csv`, resolving Power Query M-code truncation errors.
  - Exported 95% bootstrap confidence intervals to dedicated audit file `data/audit/monthly_bootstrap_cis.csv`.
- **Three-Way Reconciliation**:
  - Python == DuckDB SQL == Power BI Desktop DAX achieved 11/12 PASS (100% exact numerical match across all available measures, including M12 = 152 late caught with 0-tolerance). M06 honestly marked as pending desktop run.
