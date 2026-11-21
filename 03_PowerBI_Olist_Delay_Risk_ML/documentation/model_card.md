# Model Card: Olist Delivery-Delay Risk Model
**Model Version:** 2.0.0 (Linear-Supervised Selection)  
**Model Date:** November 2026  
**Type:** Regularized Linear Classifier with Inverse-Frequency Class Weighting (`LogisticRegression`)  
**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
**Primary Metric:** PR-AUC (Average Precision)  
**Primary Benchmark:** Lead Time Heuristic Baseline C (PR-AUC = 0.1344)

---

## 1. Model Details

- **Developer:** Independent Data Science Portfolio Project.
- **Model Architecture:** Regularized Logistic Regression (`sklearn.linear_model.LogisticRegression`).
- **Selection Policy:** Programmatic selection across 4 trained candidates strictly based on out-of-time Validation PR-AUC.
- **Hyperparameters:**
  - `penalty`: `l2`
  - `C`: 0.1 (strong L2 regularization)
  - `class_weight`: `balanced` (inverse frequency weighting)
  - `max_iter`: 1000
  - `random_state`: 42
- **Preprocessors:** Scikit-learn ColumnTransformer with median imputation, standard scaling for numeric inputs, and one-hot encoding for categoricals.
- **Artifact Storage:** Serialized with `joblib` in [models/best_delay_risk_model.joblib](../models/best_delay_risk_model.joblib).

---

## 2. Intended Use & Target Audience

### Intended Use Cases
- **Operational Triage at Handover:** Evaluates package risk at the exact moment custody transfers to the logistics carrier (`order_delivered_carrier_date`).
- **Capacity-Constrained Queue Prioritization:** Sorts outbound shipments so operational teams with fixed investigative bandwidth (e.g. top 2%, 5%, or 10% of shipments) focus proactive customer support and carrier escalations on the highest-risk orders.
- **Scenario Simulation:** Evaluates cost-tradeoffs of proactive intervention policies using parameterized cost ratios ($C_{FN} / C_{FP}$).

### Out-of-Scope & Misuse
- **Real-Time Automated Dispatch:** This model is an offline observational prototype; it is not integrated into a live production carrier API.
- **Causal Intervention Claims:** The dataset contains no post-intervention counterfactuals. High predicted risk does not prove that an intervention will avert a delivery delay.
- **Automated Penalties:** Should not be used to penalize merchants or carriers without human operational review.

---

## 3. Data & Chronological Splits

- **Source Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) on Kaggle.
- **Total Valid Delivered Orders:** 96,470 orders.
- **Target Variable:** Date-level delivery delay indicator (`is_late = 1` if customer delivery calendar date > promised delivery calendar date).
- **Time Splits (Zero Random Leakage):**
  - **Warm-Up Period (2016-09-04 to 2016-12-31):** 267 valid orders used strictly to initialize expanding historical rates.
  - **Train Split (2017-01-05 to 2018-02-28):** 57,050 orders, 3,782 late (6.63% prevalence). Mature baseline operations.
  - **Validation Split (2018-03-01 to 2018-05-31):** 20,550 orders, 2,077 late (10.11% prevalence). Captures national postal disruptions; used for candidate selection, calibration, and decision threshold selection.
  - **Test Split (2018-06-01 to 2018-08-22):** 18,107 orders, 669 late (3.69% prevalence). Strictly held-out out-of-time evaluation cohort.
  - **Right-Censored Tail (2018-08-23 to 2018-10-17):** 496 delivered orders withheld from test evaluation to ensure a minimum 55-day observation follow-up.

---

## 4. Quantitative Performance Metrics

### Candidate Model Comparison on Out-of-Time Validation Set

| Model Candidate | Validation PR-AUC | Validation ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Recall @ 5% | Selection Decision |
|---|---|---|---|---|---|---|
| **Logistic Regression (Balanced)** | **0.2905** | **0.7254** | **0.2933** | **0.8364** | **37.67%** | **Selected Shipped Model** (Highest Val PR-AUC) |
| Logistic Regression (L2) | 0.2848 | 0.7142 | 0.3344 | 0.8390 | 41.11% | Candidate 2 |
| HistGradientBoosting (Balanced) | 0.2735 | 0.6734 | 0.2292 | 0.7649 | 24.36% | Candidate 4 |
| HistGradientBoosting (Standard) | 0.2144 | 0.6191 | 0.2850 | 0.7692 | 34.83% | Candidate 3 |

### Primary Benchmark Comparison (Held-Out Test Set: 18,107 Orders)

| Evaluation Metric | Shipped Model (LR Balanced) | Baseline C (Lead Time Heuristic) | Baseline B (Regional Rule) | Baseline A (Prior) | Benchmark Impact |
|---|---|---|---|---|---|
| **PR-AUC (Average Precision)** | **0.2933** | **0.1344** | 0.0407 | 0.0369 | **2.18x lift** vs Lead Heuristic; **7.94x lift** vs Prior. |
| **ROC-AUC** | **0.8364** | **0.7510** | 0.4969 | 0.5000 | Strong global discrimination. |
| **Brier Score (Calibrated)** | **0.0439** | N/A | 0.0378 | 0.0364 | Proper scoring rule (uncalibrated: 0.0650). |
| **Brier Skill Score vs Prior** | **-0.2035** | N/A | -0.0385 | 0.0000 | Reflects prevalence shift from validation calibration. |
| **May-Recalibrated Brier** | **0.0344** | N/A | N/A | 0.0364 | **Positive BSS (+0.0564)** when calibrated post-shock. |
| **Expected Calibration Error (ECE)**| **0.0991** | N/A | N/A | N/A | 10 probability bins (reduced from raw 0.5707). |

### Operational Capacity Operating Points vs Single Thresholds

| Operating Policy | Decision Rule | Alert Volume (Orders) | Alert Rate (% Cohort) | Precision (%) | Recall (%) | F1 Score | Lift vs Prior | Lift vs Heuristic |
|---|---|---|---|---|---|---|---|---|
| **Top 2% Capacity Queue** | Top 362 orders by risk | **362** | **2.00%** | **41.99%** | **22.72%** | **0.2949** | **11.36x** | **3.12x** (152 late caught) |
| **Top 5% Capacity Queue** | Top 905 orders by risk | **905** | **5.00%** | **27.85%** | **37.67%** | **0.3202** | **7.54x** | **2.07x** (252 late caught) |
| **Top 10% Capacity Queue**| Top 1,810 orders by risk | **1,810** | **10.00%** | **19.34%** | **52.32%** | **0.2824** | **5.23x** | **1.44x** (350 late caught) |
| *Validation-Optimal F1* | Probability $\ge 0.1581$ | 5,677 | 31.35% | 9.44% | 80.12% | 0.1689 | 2.55x | 0.70x (536 late caught) |

### Calibration Reality & Operational Limitations
- **Validation Calibration-in-the-Large:** Observed = 10.11%, Raw Mean Pred = 54.02%, Platt Mean Pred = 10.11%.
- **Test Set Base-Rate Shift:** In the test period, observed prevalence drops to 3.69%. The Platt calibrator fitted on validation overpredicts test risk (mean predicted 13.57%), resulting in a test Brier score of 0.0439 and negative Brier Skill Score.
- **Remediation:** Recalibrating on the recent consolidation month (May 2018) reduces mean predicted risk to 8.61% and Brier score to 0.0344 (positive BSS +0.0564).
- **Recommendation:** Use model output primarily for **ordinal triage priority**; run rolling recalibration if absolute probabilities are exposed to end users.

---

## 5. Feature Associations & Explainability

Top features by test-set permutation importance and standardized logistic regression coefficients:

| Feature Name | Permutation Importance | Standardized Coefficient | Operational Interpretation (Associational Only) |
|---|---|---|---|
| `merchant_handling_days` | +0.2306 | +0.8588 | Longer preparation time before carrier handover strongly associates with increased delay risk. |
| `promised_lead_time_days`| +0.1961 | -0.6897 | Shorter customer promised buffers leave less tolerance for logistics variance. |
| `corridor_hist_completed_orders` | +0.1490 | +0.2676 | High-volume delivery corridors with dense logistics volume. |
| `handover_slack_days` | +0.0940 | -0.4652 | Orders handed over past shipping limit date compress downstream transit SLAs. |
| `is_late_dispatch` | +0.0147 | +0.0410 | Binary indicator of dispatch past seller deadline. |
| `order_customer_state_MG` | +0.0061 | -0.6180 | Proximity to Southeast logistics hubs associates with lower delay rates. |

*Non-Causal Warning:* Features are observational associations. For example, high merchant handling days is associated with higher delivery delays, but the data cannot identify whether this was caused by merchant delay or courier pickup lag.

---

## 7. Limitations & Ethical Considerations

1. **Dataset Temporal Horizon:** The data spans 2016 to 2018 in Brazil. Logistics infrastructure, postal pricing, and carrier performance will differ in current e-commerce environments.
2. **Missing Granular Logistics Signals:** The dataset lacks intermediate hub tracking scans, carrier names (e.g., Correios vs private couriers), and package re-attempt logs.
3. **Distribution Shift:** Operational policies shifted significantly during mid-2018 (lead times narrowed from 25.5 days to 21.6 days, PSI = 0.4842), altering late prevalence.
4. **No Post-Intervention Feedback:** The data does not record what happens if customer service intervenes; all cost savings represent scenario assumptions.
