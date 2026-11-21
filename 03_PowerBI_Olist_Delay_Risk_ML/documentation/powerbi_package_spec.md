# Power BI Package Specification & DAX Measure Library
**Project:** Olist Delivery-Delay Risk Model  
**Directory:** `03_PowerBI_Olist_Delay_Risk_ML/documentation/powerbi_package_spec.md`  
**Data Export Path:** `03_PowerBI_Olist_Delay_Risk_ML/data/powerbi_export/`  
**Execution Note (Rule 5):** All DAX measure logic, threshold grids, and relationship schemas have been verified programmatically in Python. This document provides explicit step-by-step instructions for user assembly in Power BI Desktop.

---

## 1. Power Query Ingestion & Relative Parameter

To ensure zero absolute local paths (Rule 8), configure a single parameter in Power Query:
- **Parameter Name:** `SourceFolder`
- **Type:** `Text`
- **Current Value:** `data/powerbi_export` (relative to the project root, or absolute folder path selected upon local clone).
- **M-Code Formula:**
  ```powerquery
  let
      Source = Csv.Document(File.Contents(SourceFolder & "/fact_model_scored_orders.csv"), [Delimiter=",", Columns=19, Encoding=65001, QuoteStyle=QuoteStyle.None]),
      #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
  in
      #"Promoted Headers"
  ```

---

## 2. Semantic Model Architecture & Relationships

All relationships are strictly **one-to-many (1:*)**, with **Single** cross-filter direction pointing from Dimension tables to Fact tables (star-schema).

| Primary Table (Dimension, 1) | Foreign Key Table (Fact, *) | Dimension Key | Fact Foreign Key | Cardinality | Filter Direction | Operational Purpose |
|---|---|---|---|---|---|---|
| `dim_date` | `fact_model_scored_orders` | `full_date` | `purchase_date` | 1:* | Single | Temporal slicing across order purchase dates. |
| `dim_geography` | `fact_model_scored_orders` | `state_code` | `order_customer_state` | 1:* | Single | Geographic drill-down by customer destination UF & Macro-Region. |
| `dim_risk_band` | `fact_model_scored_orders` | `risk_band` | `risk_band` | 1:* | Single | Operational priority queue slicing and triage. |
| `dim_date` | `fact_monthly_monitoring` | `year_month` | `year_month` | 1:* | Single | Time-series drift tracking. |
| `dim_geography` | `fact_segment_performance` | `state_code` | `segment_value` | 1:* | Single (inactive) | Optional state-level segment benchmarking. |

---

## 3. What-If Simulation Parameters

### Parameter 1: Decision Threshold Slider (`Threshold_Param`)
- **Generation:** Modeling > New Parameter > Numeric Range
- **Name:** `Threshold_Param`
- **Data Type:** Decimal Number
- **Minimum:** `0.01`
- **Maximum:** `0.50`
- **Increment:** `0.01`
- **Default:** `0.11` (maps to validation-tuned F1 threshold `0.1124`)

### Parameter 2: Cost Ratio Slider (`Cost_Ratio_Param`)
- **Name:** `Cost_Ratio_Param`
- **Data Type:** Decimal Number
- **Minimum:** `1.0`
- **Maximum:** `20.0`
- **Increment:** `1.0`
- **Default:** `5.0` (Ratio $C_{FN} / C_{FP} = 5.0$, assuming a missed late order is 5x more costly than a needless proactive check)

---

## 4. DAX Measure Library (Organized by Display Folders)

### Display Folder: `01_Cohort_Volume`
```dax
[Total Test Orders] = 
COUNTROWS(fact_model_scored_orders)

[Actual Late Orders] = 
CALCULATE(
    COUNTROWS(fact_model_scored_orders),
    fact_model_scored_orders[is_late] = 1
)

[Actual On-Time Orders] = 
CALCULATE(
    COUNTROWS(fact_model_scored_orders),
    fact_model_scored_orders[is_late] = 0
)

[Empirical Late Rate] = 
DIVIDE([Actual Late Orders], [Total Test Orders], 0)
```

### Display Folder: `02_Model_Performance`
```dax
[Selected Threshold] = 
SELECTEDVALUE('Threshold_Param'[Threshold_Param Value], 0.1124)

[Flagged Orders] = 
CALCULATE(
    COUNTROWS(fact_model_scored_orders),
    fact_model_scored_orders[predicted_probability] >= [Selected Threshold]
)

[Alert Rate] = 
DIVIDE([Flagged Orders], [Total Test Orders], 0)

[True Positives (TP)] = 
CALCULATE(
    COUNTROWS(fact_model_scored_orders),
    fact_model_scored_orders[predicted_probability] >= [Selected Threshold],
    fact_model_scored_orders[is_late] = 1
)

[False Positives (FP)] = 
CALCULATE(
    COUNTROWS(fact_model_scored_orders),
    fact_model_scored_orders[predicted_probability] >= [Selected Threshold],
    fact_model_scored_orders[is_late] = 0
)

[False Negatives (FN)] = 
CALCULATE(
    COUNTROWS(fact_model_scored_orders),
    fact_model_scored_orders[predicted_probability] < [Selected Threshold],
    fact_model_scored_orders[is_late] = 1
)

[True Negatives (TN)] = 
CALCULATE(
    COUNTROWS(fact_model_scored_orders),
    fact_model_scored_orders[predicted_probability] < [Selected Threshold],
    fact_model_scored_orders[is_late] = 0
)

[Precision] = 
DIVIDE([True Positives (TP)], [Flagged Orders], 0)

[Recall] = 
DIVIDE([True Positives (TP)], [Actual Late Orders], 0)

[F1 Score] = 
VAR P = [Precision]
VAR R = [Recall]
RETURN
IF(P + R > 0, 2 * (P * R) / (P + R), 0)

[Model Lift] = 
DIVIDE([Precision], [Empirical Late Rate], 0)
```

### Display Folder: `03_Capacity_Queues`
```dax
[Top 2% Queue Precision] = 0.4199
[Top 2% Queue Recall] = 0.2272
[Top 2% Queue Lift vs Prior] = 11.36
[Top 2% Queue Lift vs Heuristic] = 3.12

[Top 5% Queue Precision] = 0.2785
[Top 5% Queue Recall] = 0.3767
[Top 5% Queue Lift vs Prior] = 7.54
[Top 5% Queue Lift vs Heuristic] = 2.07

[Top 10% Queue Precision] = 0.1934
[Top 10% Queue Recall] = 0.5232
[Top 10% Queue Lift vs Prior] = 5.23
[Top 10% Queue Lift vs Heuristic] = 1.44
```

### Display Folder: `04_Cost_Simulation`
```dax
[Selected Cost Ratio] = 
SELECTEDVALUE('Cost_Ratio_Param'[Cost_Ratio_Param Value], 5.0)

[Cost Per Needless Alert (C_FP)] = 1.0

[Cost Per Missed Late (C_FN)] = 
[Selected Cost Ratio] * [Cost Per Needless Alert (C_FP)]

[Expected Model Cost] = 
([Cost Per Needless Alert (C_FP)] * [False Positives (FP)]) + 
([Cost Per Missed Late (C_FN)] * [False Negatives (FN)])

[Do-Nothing Baseline Cost] = 
[Cost Per Missed Late (C_FN)] * [Actual Late Orders]

[Scenario Cost Reduction Pct] = 
DIVIDE([Do-Nothing Baseline Cost] - [Expected Model Cost], [Do-Nothing Baseline Cost], 0)
// Note: Evaluates hypothetical parameterized scenario cost vs unmitigated baseline; not empirical cash savings.
```

---

## 5. Dashboard Page Layout & Visual Specifications

### Page 1: Problem Statement & Baseline Comparison
- **Visual 1 (Cards):** Total Valid Orders (`96,470`), Canonical Late Rate (`6.77%`), Benchmark Baselines PR-AUC (`0.0369` Prior, `0.0407` Regional Rule, `0.1344` Lead-Time Heuristic).
- **Visual 2 (Clustered Column Chart):** PR-AUC comparison: Baselines vs. Shipped Model (`0.2933`, 2.18x lift vs Lead Heuristic).
- **Visual 3 (Text Box - Disclaimers):** Strictly observational dataset. Prediction moment = `order_delivered_carrier_date`. No causal claims.

### Page 2: Model Performance & Threshold What-If Simulator
- **Visual 1 (Slicers):** `Threshold_Param` slider (`0.01` to `0.50`, default `0.16`), `Cost_Ratio_Param` slider (`1` to `20`, default `5`).
- **Visual 2 (KPI Cards):** `[Precision]`, `[Recall]`, `[F1 Score]`, `[Alert Rate]`, `[Expected Model Cost]`, `[Scenario Cost Reduction Pct]`.
- **Visual 3 (Line Chart):** 
  - **Title:** `Precision-Recall Curve (PR-AUC = 0.2933, Python-verified)`
  - **X-Axis:** `fact_pr_curve_points[recall]` (Numeric, Continuous, sorted ascending `0.0` to `1.0`).
  - **Y-Axis:** `fact_pr_curve_points[precision]`.
- **Visual 4 (Line Chart):** 
  - **Title:** `Calibration Curve (ECE = 0.0991, Python-verified)`
  - **X-Axis:** `fact_calibration_bins[avg_predicted_risk]`.
  - **Y-Axis:** `fact_calibration_bins[empirical_late_rate]` (Set aggregation to **Average**, NOT Sum!).
  - **Analytics Line:** Diagonal reference line (Add X=Y or constant reference line via Analytics pane).
- **Visual 5 (Matrix):** Dynamic Confusion Matrix (`TP`, `FP`, `TN`, `FN`).

### Page 3: At-Risk Order Queue (Operational Triage)
- **Visual 1 (Slicer):** Risk Band (`Critical Queue (Top 2%)`, `High Risk (95-98%)`, `Moderate Risk (80-95%)`, `Low Risk (0-80%)`).
- **Visual 2 (Table Visual):** High-risk orders queue (exactly 362 orders in Top 2% queue, catching 152 late shipments): `order_id`, `purchase_date`, `customer_state`, `promised_lead_time_days`, `merchant_handling_days`, `handover_slack_days`, `predicted_probability`, `risk_band`.
- **Visual 3 (Donut Chart):** Orders by Risk Band.

### Page 4: Segment Performance & Geographic Disparity
- **Visual 1 (Shape Map / Bar Chart):** Customer Destination State performance (SP, RJ, MG, BA, RS, etc.).
- **Visual 2 (Table with Sample Size Policy):** Enforces data privacy and statistical power:
  - `< 30 rows`: Suppressed
  - `30 to 99 rows`: Exploratory
  - `100+ rows`: Interpretable
- **Visual 3 (Bar Chart):** Performance across Merchandise Value Quartiles (`Q1_Low` to `Q4_VeryHigh`).

### Page 5: Monitoring & Data Quality Audit
- **Visual 1 (Table):** 
  - **Title:** `Data Quality Assertions: 14 PASS / 1 Real Data Anomaly Finding (DQ-15 Chronology Audit)`
  - **Source:** `fact_data_quality_log.csv` showing 14 PASS / 1 FAIL (`DQ-15`: Real Data Anomaly Finding on 158 historical records).
- **Visual 2 (Clustered Bar Chart):** Feature Population Stability Index (PSI): Filtered strictly to `Train_vs_Test` in `fact_feature_drift.csv` to avoid multi-comparison summing (`promised_lead_time_days` = 0.484, `handover_slack_days` = 0.292).
- **Visual 3 (Line Chart):** Monthly Out-of-Time Stability (`fact_monthly_monitoring.csv`) tracking test months (2018-06, 2018-07, 2018-08) with dual axes (left Y-axis for PR-AUC and Recall [0 to 1], right Y-axis for late rate %).

### Page 6: Model Card & Governance
- **Visual 1 (Table/Cards):** Model Metadata (`Logistic Regression Balanced`, hyperparameters, training window: 2017-01 to 2018-02, validation window: 2018-03 to 2018-05, test window: 2018-06 to 2018-08-22, random seed 42).
- **Visual 2 (Bar Chart):** Permutation Feature Importance & Standardized Coefficients (`merchant_handling_days`, `promised_lead_time_days`, `corridor_hist_completed_orders`, `handover_slack_days`).
- **Visual 3 (Text Box - Model Card & Disclaimers):**
  ```text
  MODEL CARD — OLIST DELIVERY-DELAY RISK MODEL

  INTENDED USE: Portfolio demonstration only. Offline, observational, non-deployed.
  ALGORITHM: LogisticRegression (class_weight='balanced', C=0.1, scikit-learn 1.9.0, seed=42)
  SELECTION: Programmatic selection by validation PR-AUC across 4 candidates.
  TRAINING: 2017-01 to 2018-02 | VALIDATION: 2018-03 to 2018-05 | TEST: 2018-06 to 2018-08-22
  RIGHT-CENSORED: Orders handed over after 2018-08-22 excluded from evaluation.

  KEY RESULTS (Python-verified):
  PR-AUC = 0.2933 | 95% CI: [0.2603 – 0.3251] | Lift vs Heuristic: 2.18x | Lift vs Prior: 7.94x
  ROC-AUC = 0.8364 | Calibrated Brier Score = 0.0439 | Calibrated ECE = 0.0991
  Top 2% Capacity Queue: 362 orders | 152 late caught | Precision = 41.99% | Recall = 22.72%

  LIMITATIONS:
  • Data covers 2016–2018 Brazilian e-commerce only. Distribution shift likely in deployment.
  • No intervention outcomes — cost estimates are parameterised assumptions, not empirical.
  • Feature associations are correlational. No causal claims about congestion or seller intent.
  • Multi-seller orders attributed to primary seller (highest freight). Some ambiguity remains.
  • Calibrated on validation strike window; overpredicts absolute risk on test (13.6% vs 3.7%). Use for ranking.
  DATA: Olist Brazilian E-Commerce Dataset. Kaggle. License: CC BY-NC-SA 4.0.
  ```
