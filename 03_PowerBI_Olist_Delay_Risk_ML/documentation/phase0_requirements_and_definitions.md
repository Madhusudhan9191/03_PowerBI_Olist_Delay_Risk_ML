# Phase 0: Requirements and Definitions
**Project:** Olist Delivery-Delay Risk Model  
**Directory:** `03_PowerBI_Olist_Delay_Risk_ML/`  
**Status:** Completed & Reconciled (Verified via Python & Pytest)  
**Random Seed:** 42  

---

## 1. Source Dataset and License Citation

This project builds upon the open public dataset:
- **Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Publisher:** Olist and André Sionek on Kaggle
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))
- **Terms:** Non-commercial use, attribution required, share-alike under the same license terms.
- **Context:** Anonymized transactional data of 99,441 orders placed between September 2016 and October 2018 on the Olist marketplace across Brazil.

---

## 2. Business Problem & Core Question

### Primary Business Question
> *At the moment an order is handed to the logistics carrier (`order_delivered_carrier_date`), can we score orders by their probability of arriving after the promised delivery date, early enough for an operations team to intervene hypothetically?*

### Operational Decision Context
In an e-commerce marketplace, when an order is handed over to the carrier:
1. The merchant handling phase is complete, fixing the exact handover timestamp and handover slack against the merchant dispatch SLA (`shipping_limit_date`).
2. The package enters the postal / 3PL logistics network.
3. If an order has elevated delivery-delay risk, operational teams could hypothetically prioritize tracking escalations, alert customer support, or trigger proactive customer communications before the delivery SLA breaches.
4. **Offline Scope Disclosure:** This project is an observational offline study. No real-time interventions were enacted in historical data, and no causal post-intervention outcomes exist. All cost and capacity analyses represent hypothetical scenario simulations.

---

## 3. Simulated Stakeholder Roles

All stakeholder roles are simulated for portfolio realism and operational alignment:

| Simulated Role | Objective | Primary Concern / Constraint |
|---|---|---|
| **Simulated Head of Operations** | Reduce surprise delivery delays and proactive support costs. | Capacity is bounded; customer service teams can investigate at most 5% to 10% of weekly order volume (top-k capacity framing). High alert false alarms waste operational time. |
| **Simulated Logistics Partner Manager** | Monitor carrier SLAs across corridors and detect fulfillment bottlenecks. | Needs transparent regional breakdown (interstate vs intrastate) and merchant dispatch compliance without black-box bias. |
| **Simulated Data Science Lead** | Oversee model validity, governance, and auditability. | Strictly enforces point-in-time correctness, no future leakage, calibrated probabilities (Brier score / ECE), rolling chronological validation, and zero causal claims. |

---

## 4. Analytical Hypotheses

All hypotheses explore observational associations (no causal claims):

- **Hypothesis H1 (Handover Slack):** Orders where the merchant handover occurred late or with compressed buffer relative to the seller dispatch SLA (`shipping_limit_date`) are associated with a higher observed rate of customer delivery delays.
- **Hypothesis H2 (Corridor / Geographic Distance):** Orders involving long-haul, interstate corridors (specifically from Southeastern hubs to Northern and Northeastern states) are associated with elevated transit times and higher delay probability compared to intra-state shipments.
- **Hypothesis H3 (Promised Buffer Elasticity):** Orders assigned shorter promised lead times (`order_estimated_delivery_date` minus purchase date) exhibit a higher probability of late arrival due to lower absorption capacity for logistical transit variances.
- **Hypothesis H4 (Physical Cargo Bulk):** Bulky and heavy freight (elevated product weight and cubic volume) is associated with higher delivery delay frequency due to specialized carrier handling.

---

## 5. Scope, Out-of-Scope, and Assumptions

### In Scope
- Offline point-in-time feature engineering at `order_delivered_carrier_date`.
- Chronological train/validation/test splits (time-ordered, zero lookahead).
- Benchmark comparisons: Prior baseline, heuristic rule baseline, logistic regression, and gradient boosting (`HistGradientBoostingClassifier`).
- Capacity-constrained evaluation: PR-AUC, ROC-AUC, Brier score, calibration curve/ECE, recall at top-k% (top 2%, 5%, 10%), lift, and parametric cost curves.
- Subgroup analysis across state corridors, seller volume tiers, category groups, and order value bands.
- Feature and label drift monitoring (PSI / KS tests), automated data-quality verification scripts, and a Power BI export package.

### Out of Scope
- Real-time API deployment, streaming Kafka ingestion, or live inference servers.
- Dynamic route dispatch, carrier reassignment, or automated customer refund disbursements.
- Causal inference: The dataset contains no counterfactuals, randomized control trials, or logged interventions.

### Key Assumptions
1. **Prediction Moment Definition:** The moment of package handover (`order_delivered_carrier_date`) represents the earliest instant when all seller handling durations and dispatch slacks are observable.
2. **Alternative Handover Fallback:** If `order_delivered_carrier_date` is unrecorded or defective, `order_approved_at` represents the earliest defensible alternative (payment confirmed, inventory assigned, routing fixed, but pre-dispatch).
3. **Observation Completeness:** A transit observation window of at least 55 days from purchase to dataset cutoff is required to ensure right-censoring does not bias outcome labels.

---

## 6. Ground-Truth Target Definition & Control Totals

### Definition
An order is labeled **Late (`is_late = 1`)** if and only if the package was handed to the customer on a calendar date strictly after the promised estimated delivery date:
$$\text{is\_late} = \begin{cases} 1 & \text{if } \text{order\_delivered\_customer\_date\_only} > \text{order\_estimated_delivery\_date\_only} \\ 0 & \text{otherwise} \end{cases}$$

### Grain and Population Exclusions
- **Grain:** One row per unique order (`order_id`).
- **Population:** Valid delivered orders only (`order_status == 'delivered'` AND `order_delivered_customer_date IS NOT NULL`).

### Control Totals Verification Table
The figures below were computed programmatically by `03_PowerBI_Olist_Delay_Risk_ML/src/audit/run_audit.py` and verified against the canonical warehouse profiling totals:

| Metric Name | Computed Value | Canonical Control Total | Difference | Programmatic Status |
|---|---|---|---|---|
| **Total Raw Orders** | 99,441 | 99,441 | 0 | **PASS** |
| **Raw Delivered Status** | 96,478 | 96,478 | 0 | **PASS** |
| **Delivered with Null Customer Date** | 8 | 8 | 0 | **PASS** |
| **Valid Delivered Orders (with outcome date)** | 96,470 | 96,470 | 0 | **PASS** |
| **Total Non-Delivered Orders Excluded** | 2,963 | 2,963 | 0 | **PASS** |
| **On-Time Delivered Orders (`is_late = 0`)** | 89,936 | 89,936 | 0 | **PASS** |
| **Late Delivered Orders (`is_late = 1`)** | 6,534 | 6,534 | 0 | **PASS** |
| **Date-Level Late Delivery Rate (%)** | **6.7731%** | **6.7731%** | 0.0000 pp | **PASS** |

### Date-Level vs Timestamp-Level Comparison
- **Date-level Late Orders (`date > date`):** 6,534 orders (6.7731%).
- **Timestamp-level Late Orders (`ts > ts`):** 7,826 orders (8.1124%).
- **Difference:** Exactly **1,292 orders** were delivered on the promised calendar date, but with a delivery time after `00:00:00`. Because Olist estimated delivery dates are stored with truncated time (`00:00:00`), timestamp-level comparison erroneously penalizes same-day deliveries as "late". The business SLA treats delivery on the promised date as on-time. Therefore, date-level comparison is the only defensible definition.

---

## 7. Exclusions Breakdown

Every exclusion from raw orders to the modeling population is accounted for:

| Filter Stage | Category / Condition | Order Count | % of Raw Orders (99,441) | Operational Rationale |
|---|---|---|---|---|
| Retained | `order_status == 'delivered'` (with date) | **96,470** | **97.0123%** | Finalized fulfillment with validated ground-truth arrival date. |
| Excluded | `order_status == 'shipped'` | 1,107 | 1.1132% | Order in transit; outcome unknown at data collection cutoff. |
| Excluded | `order_status == 'canceled'` | 625 | 0.6285% | Canceled before or during fulfillment; no delivery outcome. |
| Excluded | `order_status == 'unavailable'` | 609 | 0.6124% | Merchant inventory unavailable; never dispatched. |
| Excluded | `order_status == 'invoiced'` | 314 | 0.3158% | Order billed but not handed over to logistics carrier. |
| Excluded | `order_status == 'processing'` | 301 | 0.3027% | Order being assembled; not dispatched. |
| Excluded | `order_status == 'created'` | 5 | 0.0050% | Order initiated but payment/workflow not completed. |
| Excluded | `order_status == 'approved'` | 2 | 0.0020% | Payment confirmed but fulfillment not started. |
| Excluded | `status == 'delivered'` but `null delivery date` | 8 | 0.0080% | Missing ground-truth customer delivery timestamp. |
| **Total** | **All Ingested Records** | **99,441** | **100.0000%** | Reconciles to 96,470 valid delivered + 2,971 excluded. |

---

## 8. Modelling Window & Warm-up Period

The Olist dataset spans from September 2016 through October 2018. The modeling window is structured as follows:

```
[2016-09-04 to 2016-12-31]      [2017-01-01 to 2018-08-22]               [2018-08-23 to 2018-10-17]
--------------------------      --------------------------               --------------------------
   Warm-Up / Cold-Start          Steady-State Modelling Window               Right-Censored Tail
     267 valid orders               95,707 valid delivered orders            496 delivered orders
  (Historical stats only)         (Chronological Train/Val/Test)          (Outcome observation cutoff)
```

### 1. Warm-Up Period (2016-09-04 to 2016-12-31): 267 valid delivered orders
- **Justification:** Olist launched marketplace operations during late 2016. Order volumes were exploratory (1 order in Sep, 265 in Oct, 0 in Nov, 1 in Dec).
- **Usage:** Not used for model training or evaluation. Retained strictly to warm up historical seller completion counts and initial corridor transit statistics so that orders starting 2017-01-01 have historical lookup data.

### 2. Steady-State Modeling Window (2017-01-01 to 2018-08-22): 95,707 valid delivered orders
- **Justification:** Consistent marketplace operations, national carrier network coverage, and stable month-over-month volume (from 750 orders/month in Jan 2017 to 6,500+ orders/month in 2018).
- **Target Rate:** 6,509 late orders out of 95,707 valid delivered orders (**6.8009%** late rate).

### 3. Right-Censoring Cutoff (2018-08-22):
- **Justification:** The dataset extraction occurred on 2018-10-17. Between 2018-08-23 and 2018-08-29, 496 delivered orders were recorded. After 2018-08-29, exactly 25 orders were placed in raw data, of which 0 reached delivered status (24 canceled, 1 shipped).
- Setting the cutoff at 2018-08-22 ensures that every test order has at least 55 days of follow-up observation before dataset termination. Since 99% of all deliveries complete within 40.98 days, a 55-day window guarantees complete outcome capture without censoring bias.

---

## 9. Acceptance Criteria

| Phase | Milestone / Deliverable | Success Criteria | Verification Method |
|---|---|---|---|
| **Phase 0** | Requirements, Definitions & Target | 100% reconciliation with warehouse control totals (96,470 valid, 6,534 late, 6.7731%). Zero manual metrics. | Pytest test suite & automated reconciliation script. |
| **Phase 1** | Data Audit for ML | Complete availability audit, quantified sequence anomalies, right-censoring distribution, multi-seller aggregation rules. | Python audit execution & JSON artifact logging. |
| **Phase 2** | Point-in-Time Feature Engineering | Automated unit tests asserting zero feature timestamp > prediction moment. Expanding window historical rates with lag. | Automated leakage audit script. |
| **Phase 3** | Splits & Baselines | Strictly chronological splits (train/val/test). Three non-trivial baselines (prior, heuristic, corridor rate). | Train/val/test distribution logs. |
| **Phase 4** | Models & Training | Regularized Logistic Regression vs. Gradient Boosting. Fixed seeds, tuned on validation only. | Pinned model artifacts & metadata. |
| **Phase 5** | Evaluation & Calibration | PR-AUC primary metric, Brier score, ECE, recall at top-k%, bootstrap CIs, cost sensitivity grid. | Evaluation scripts & confidence interval tables. |
| **Phase 6** | Monitoring & Data Quality | PSI/KS drift tests, label drift, calibration drift, automated data quality assertions. | Quality CSV log & drift report. |
| **Phase 7** | Power BI Package | Clean relational model, DAX measure library, What-If parameter, 6 analytical pages. | Measure export script & user validation guide. |
| **Phase 8** | Three-Way Reconciliation | Automated comparison between Python, SQL, and DAX measure outputs with explicit tolerances. | Reconciliation report. |
| **Phase 9** | Documentation & Model Card | README, Model Card, Data Dictionary, Interview Defence Notes. | Markdown documentation in repo. |
