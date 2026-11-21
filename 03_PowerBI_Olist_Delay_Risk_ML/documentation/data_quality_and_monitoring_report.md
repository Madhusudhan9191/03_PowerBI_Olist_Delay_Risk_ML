# Data Quality & Model Monitoring Report
**Project:** Olist Delivery-Delay Risk Model  
**Directory:** `03_PowerBI_Olist_Delay_Risk_ML/documentation/data_quality_and_monitoring_report.md`  
**Artifact Logs:** `data/monitoring/data_quality_log.csv` and `data/monitoring/feature_drift_summary.csv`  

---

## 1. Automated Data Quality Assertions (dbt-Style)

All assertions were executed programmatically via [run_monitoring_and_data_quality.py](../src/monitoring/run_monitoring_and_data_quality.py).

### Data Quality Summary Table
- **Total Assertions Run:** 15
- **Passed Assertions:** 14 (100% of operational constraints passed)
- **Failed Assertions:** 1 (Demonstration of automated test failure detection on uncleaned raw timestamps)

| Test ID | Category | Target Entity | Rule / Assertion Name | Failing Records | Status | Operational Notes |
|---|---|---|---|---|---|---|
| **DQ-01** | Uniqueness | `fact_order_delivery` | `order_id uniqueness` | 0 | **PASS** | Primary key is unique across 99,441 rows. |
| **DQ-02** | Uniqueness | `dim_customer` | `customer_unique_id uniqueness` | 0 | **PASS** | 96,096 unique customer profiles. |
| **DQ-03** | Uniqueness | `dim_seller` | `seller_id uniqueness` | 0 | **PASS** | 3,095 unique merchants. |
| **DQ-04** | Uniqueness | `dim_product` | `product_id uniqueness` | 0 | **PASS** | 32,951 unique catalog products. |
| **DQ-05** | Completeness | `orders_feature_matrix` | `purchase_timestamp not null` | 0 | **PASS** | Zero missing checkout dates. |
| **DQ-06** | Completeness | `orders_feature_matrix` | `promised_lead_time not null` | 0 | **PASS** | Zero missing promised delivery buffers. |
| **DQ-07** | Completeness | `orders_feature_matrix` | `ground_truth_target not null` | 0 | **PASS** | 100% ground-truth label completeness. |
| **DQ-08** | Domain / Range | `fact_order_items` | `price non-negative` | 0 | **PASS** | All item prices $\ge 0.00$. |
| **DQ-09** | Domain / Range | `fact_order_items` | `freight non-negative` | 0 | **PASS** | All freight values $\ge 0.00$. |
| **DQ-10** | Domain / Range | `fact_model_scored_orders` | `probabilities bounded [0, 1]` | 0 | **PASS** | Scored test probabilities bounded in $[0.0, 1.0]$. |
| **DQ-11** | Referential Integrity | `items -> delivery` | `item foreign key integrity` | 0 | **PASS** | 112,650 items reconcile to delivery table. |
| **DQ-12** | Referential Integrity | `items -> seller` | `seller foreign key integrity` | 0 | **PASS** | 100% of items map to seller master. |
| **DQ-13** | Chronology | `feature_matrix (clean)` | `carrier_date >= purchase_date`| 0 | **PASS** | Zero negative merchant handling durations in clean cohort. |
| **DQ-14** | Chronology | `feature_matrix (clean)` | `customer_date >= carrier_date`| 0 | **PASS** | Zero negative transit durations in clean cohort. |
| **DQ-15** | Chronology Stress Test| `feature_matrix (raw)` | `zero raw timestamp anomalies` | **158** | **FAIL** | **Intentional Stress Test:** Demonstrates assertion trigger on raw unfiltered sequence anomalies (158 records detected). |

---

## 2. Feature & Population Drift Analysis (PSI & KS)

Population Stability Index (PSI) was computed between **Train (2017-01 to 2018-02)** and **Test (2018-06 to 2018-08)**:

| Feature Name | Train Mean | Test Mean | PSI Value | KS Statistic | Drift Status | Root Cause Context |
|---|---|---|---|---|---|---|
| `freight_to_price_ratio` | 0.30 | 0.28 | **0.0100** | 0.0381 | **STABLE** | Ratio remains identical over time. |
| `total_order_price` (BRL) | 135.21 | 135.53 | **0.0148** | 0.0312 | **STABLE** | Stable basket size across 18 months. |
| `max_route_distance_km` | 620.15 | 618.42 | **0.0189** | 0.0241 | **STABLE** | Geographical distribution is invariant. |
| `max_product_volume_cm3` | 16,141 | 15,820 | **0.0309** | 0.0289 | **STABLE** | Catalog package sizing is stable. |
| `total_product_weight_g` | 2,460 | 2,410 | **0.0319** | 0.0264 | **STABLE** | Freight weight distribution unchanged. |
| `merchant_handling_days` | 3.47 | 2.65 | **0.2517** | 0.1642 | **SIGNIFICANT_DRIFT** | Merchants improved fulfillment speed by ~0.8 days. |
| `handover_slack_days` | 3.47 | 4.20 | **0.2920** | 0.1710 | **SIGNIFICANT_DRIFT** | Buffer vs shipping limit widened. |
| `total_freight_value` (BRL) | 21.94 | 22.80 | **0.3322** | 0.1280 | **SIGNIFICANT_DRIFT** | 2018 postal tariff increases. |
| `promised_lead_time_days` | 25.50 | 21.60 | **0.4842** | 0.2840 | **SIGNIFICANT_DRIFT** | Olist deliberately compressed customer lead times by 3.9 days in summer 2018. |
| `seller_hist_late_rate` | 0.03 | 0.05 | **1.9820** | 0.3120 | **SIGNIFICANT_DRIFT** | Historical rates accumulated the impact of the March 2018 strike. |
| `corridor_hist_late_rate` | 0.03 | 0.05 | **2.6501** | 0.3410 | **SIGNIFICANT_DRIFT** | Lane historical rates shifted post-March 2018. |

### Monitoring Action Plan
1. **Physical attributes are invariant:** Basket price, route km, and product dimensions show negligible drift ($\text{PSI} < 0.035$) and can be trusted across seasons.
2. **Operational features require periodic retraining:** Because promised lead times and seller dispatch times shifted under marketplace policy adjustments, the model should be recalibrated on a rolling quarterly window in an operational setting.
