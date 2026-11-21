# Data Dictionary: Olist Delivery-Delay Risk Model
**Project:** Olist Delivery-Delay Risk Model  
**Directory:** `03_PowerBI_Olist_Delay_Risk_ML/documentation/data_dictionary.md`  
**Status:** Validated & Aligned with Warehouse Schema  

---

## 1. Processed Feature Matrix (`orders_feature_matrix`)
- **Storage:** `data/processed/orders_feature_matrix.parquet` & `.csv`
- **Grain:** One row per valid delivered order (`order_id`). Total: 96,470 rows.

| Column Name | Data Type | Nullable | Source / Definition | Operational Purpose |
|---|---|---|---|---|
| `order_id` | String (UUID) | No | `fact_order_delivery.order_id` | Primary key of the order. |
| `customer_id` | String (UUID) | No | `fact_order_delivery.customer_id` | Transaction-level customer identifier. |
| `customer_unique_id` | String (UUID) | No | `dim_customer.customer_unique_id` | Master customer entity identifier. |
| `purchase_date` | Date (YYYY-MM-DD) | No | `order_purchase_timestamp::date` | Calendar date of purchase checkout. |
| `purchase_ym` | String (YYYY-MM) | No | Format `YYYY-MM` | Monthly partition bucket. |
| `modeling_window_tier` | String | No | Category: `warmup`, `modelling`, `right_censored` | Chronological split allocation. |
| `order_purchase_timestamp` | Timestamp | No | Raw checkout timestamp | Order initiation instant. |
| `order_approved_at` | Timestamp | Yes | Payment clearance timestamp | Fallback prediction moment. |
| `order_delivered_carrier_date` | Timestamp | Yes | Handover timestamp to logistics carrier | Primary prediction moment ($\mathbf{T_{\text{predict}}}$). |
| `order_delivered_customer_date` | Timestamp | No | Customer parcel receipt timestamp | Ground truth arrival timestamp. |
| `order_estimated_delivery_date` | Timestamp | No | Promised delivery SLA timestamp | Customer-facing promised date. |
| `is_anomalous_sequence` | Integer (0/1) | No | Flag for reversed sequence timestamps | Data quality indicator. |
| `promised_lead_time_days` | Float | No | Estimated date minus purchase date (days) | Promised delivery buffer. |
| `merchant_handling_days` | Float | Yes | Carrier date minus purchase timestamp (days) | Elapsed merchant fulfillment duration. |
| `approval_to_carrier_days` | Float | Yes | Carrier date minus approval timestamp (days) | Merchant dispatch duration after payment. |
| `handover_slack_days` | Float | Yes | `shipping_limit_date` minus carrier date (days) | Merchant SLA buffer remaining at handover. |
| `is_late_dispatch` | Integer (0/1) | No | $\mathbb{I}(\text{carrier} > \text{shipping\_limit})$ | Binary indicator of merchant dispatch breach. |
| `item_count` | Integer | No | `COUNT(order_item_id)` | Number of line items in order. |
| `seller_count` | Integer | No | `COUNT(DISTINCT seller_id)` | Number of unique merchants in order. |
| `is_multi_seller` | Integer (0/1) | No | $\mathbb{I}(\text{seller\_count} > 1)$ | Multi-package shipment indicator. |
| `total_order_price` | Float (BRL) | No | `SUM(price)` | Total merchandise value. |
| `total_freight_value` | Float (BRL) | No | `SUM(freight_value)` | Total shipping fees paid. |
| `freight_to_price_ratio` | Float | No | `freight / price` | Freight economic burden ratio. |
| `max_route_distance_km` | Float | No | Geodesic km between seller and customer zip | Bottleneck route distance. |
| `is_interstate` | Integer (0/1) | No | $\mathbb{I}(\text{seller\_state} \ne \text{customer\_state})$ | Interstate transit flag. |
| `order_distance_band` | String | No | Distance category: Local, Regional, Remote | Macro-distance segment. |
| `order_customer_state` | String (2-UF) | No | Destination state code (e.g. SP, RJ, BA) | Delivery geographic destination. |
| `customer_region` | String | No | Brazilian macro-region (Sudeste, Sul, etc.) | Regional macro-cluster. |
| `primary_seller_state` | String (2-UF) | Yes | State of primary merchant | Origin state code. |
| `max_product_weight_g` | Float | No | Max item weight in order (grams) | Heavy parcel classification. |
| `total_product_weight_g` | Float | No | Sum of item weights in order (grams) | Total shipment weight. |
| `max_product_volume_cm3` | Float | No | Max $L \times H \times W$ of items ($\text{cm}^3$) | Largest package volume. |
| `total_product_volume_cm3` | Float | No | Sum of $L \times H \times W$ ($\text{cm}^3$) | Total cargo volume. |
| `primary_category` | String | No | Category of highest-priced item | Dominant catalog department. |
| `primary_payment_type` | String | No | Dominant payment method | Payment instrument. |
| `max_installments` | Integer | No | Max credit card installment count | Payment term complexity. |
| `handover_dow` | Integer | No | Day of week of carrier handover (0..6) | Day-of-week seasonality. |
| `handover_month` | Integer | No | Calendar month of carrier handover (1..12) | Annual seasonality. |
| `handover_hour` | Integer | No | Hour of day of carrier handover (0..23) | Handover timing within day. |
| `is_weekend_handover` | Integer (0/1) | No | $\mathbb{I}(\text{handover\_dow} \in [5, 6])$ | Weekend carrier drop-off flag. |
| `seller_hist_completed_orders` | Integer | No | Historical finalized orders before this handover | Merchant maturity indicator. |
| `seller_hist_late_rate` | Float | No | Smoothed historical seller delay rate | Empirical Bayes merchant rate. |
| `seller_no_history_flag` | Integer (0/1) | No | $\mathbb{I}(\text{completed} < 5)$ | Cold-start merchant flag. |
| `corridor_hist_completed_orders`| Integer | No | Historical orders in seller_UF -> cust_UF | Corridor route maturity. |
| `corridor_hist_late_rate` | Float | No | Smoothed historical corridor delay rate | Regional lane delay rate. |
| `corridor_no_history_flag` | Integer (0/1) | No | $\mathbb{I}(\text{corridor\_orders} < 10)$ | Cold-start route flag. |
| `is_late` | Integer (0/1) | No | $\mathbb{I}(\text{actual\_date} > \text{estimated\_date})$ | **Ground-Truth Target Label**. |

---

## 2. Power BI Export Tables

### 2.1 Scored Test Orders (`fact_model_scored_orders`)
- **Storage:** `data/powerbi_export/fact_model_scored_orders.csv`
- **Grain:** One row per out-of-time test order (18,107 rows).
- **Key Columns:** `order_id`, `purchase_date`, `predicted_probability`, `risk_band`, `is_late`, `order_customer_state`, `promised_lead_time_days`, `merchant_handling_days`, `handover_slack_days`.

### 2.2 Threshold & Cost Simulation Grid (`fact_threshold_cost_grid`)
- **Storage:** `data/powerbi_export/fact_threshold_cost_grid.csv`
- **Grain:** One row per threshold (50 steps) per cost ratio ($C_{FN}/C_{FP} \in [2, 5, 10, 20]$). Total: 200 rows.
- **Key Columns:** `cost_ratio_cfn_cfp`, `threshold`, `false_positives`, `false_negatives`, `total_expected_cost`, `cost_saving_pct`, `precision`, `recall`, `f1_score`, `alert_rate_pct`.

### 2.3 Feature Drift Summary (`fact_feature_drift`)
- **Storage:** `data/powerbi_export/fact_feature_drift.csv`
- **Grain:** One row per feature per drift comparison window.
- **Key Columns:** `feature_name`, `comparison`, `train_mean`, `test_mean`, `psi_value`, `ks_statistic`, `drift_severity`.

### 2.4 Data Quality Assertions Log (`fact_data_quality_log`)
- **Storage:** `data/powerbi_export/fact_data_quality_log.csv`
- **Grain:** One row per automated assertion rule (15 rows).
- **Key Columns:** `test_id`, `category`, `entity`, `assertion_name`, `failing_record_count`, `status`, `operational_notes`.
