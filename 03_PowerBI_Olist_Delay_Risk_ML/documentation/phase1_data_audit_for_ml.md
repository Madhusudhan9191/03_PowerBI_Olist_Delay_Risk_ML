# Phase 1: Data Audit for Machine Learning
**Project:** Olist Delivery-Delay Risk Model  
**Directory:** `03_PowerBI_Olist_Delay_Risk_ML/documentation/phase1_data_audit_for_ml.md`  
**Status:** Completed & Reconciled (Verified via Python Code & Pytest)  
**Output Data Artifacts:** `03_PowerBI_Olist_Delay_Risk_ML/data/audit/`  

---

## 1. Feature Availability Audit & Prediction Moment

### 1.1 The Operational Prediction Moment
The chosen operational moment for model scoring is:
$$\mathbf{T_{\text{predict}}} = \text{order\_delivered\_carrier\_date}$$

At this timestamp:
1. The merchant has prepared the package and physically transferred custody to the carrier.
2. The merchant handling duration (`order_delivered_carrier_date` minus `order_purchase_timestamp`) is fully realized.
3. The merchant dispatch compliance (`order_delivered_carrier_date` versus `shipping_limit_date`) is known.
4. The remaining fulfillment cycle consists entirely of carrier transit from the seller origin hub to the customer destination address.

### 1.2 Earliest Defensible Alternative: `order_approved_at`
- **Assessment of `order_delivered_carrier_date` Reliability:**
  - In our audit of 96,470 valid delivered orders, **only 1 single order (0.001%)** has a null `order_delivered_carrier_date`.
  - However, **1,350 orders (1.399%)** have carrier timestamps logged prior to the payment approval timestamp, and **165 orders (0.171%)** have carrier timestamps logged before the purchase timestamp.
- **Why `order_approved_at` is the Earliest Defensible Alternative:**
  - Occurs before any physical movement. At payment approval (`order_approved_at`), customer shipping address, seller assignment, product dimensions, price, freight, and shipping limit date are all immutably established.
  - **Trade-off:** Evaluating at `order_approved_at` occurs earlier, but discards the strongest operational signals (actual merchant handling duration and handover slack against `shipping_limit_date`). Because carrier date is present for 99.999% of valid delivered orders, `order_delivered_carrier_date` remains the primary prediction moment, with anomalous records handled transparently.

---

## 2. Timestamp Anomalies & Sensitivity Analysis

### 2.1 Anomaly Quantification Table
All metrics below were computed directly by `run_audit.py` on the 96,470 valid delivered orders:

| Anomaly Condition | Code Expression | Order Count | % of Valid Delivered (96,470) | Operational Interpretation |
|---|---|---|---|---|
| **Carrier before Purchase** | `carrier_ts < purchase_ts` | 165 | 0.1710% | Clock skew, timezone mismatch, or manual backdating. |
| **Carrier before Approved** | `carrier_ts < approved_ts` | 1,350 | 1.3994% | Merchant dispatched before payment gateway webhook sync. |
| **Delivery before Carrier** | `customer_ts < carrier_ts` | 23 | 0.0238% | Carrier delivered parcel before recording origin handover scan. |
| **Delivery before Purchase** | `customer_ts < purchase_ts` | 0 | 0.0000% | Zero instances observed. |
| **Delivery before Approved** | `customer_ts < approved_ts` | 61 | 0.0632% | Delivery completed prior to formal payment approval log. |
| **Null Carrier Handover Date** | `carrier_ts IS NULL` | 1 | 0.0010% | Missing handover scan. |
| **Null Payment Approval Date**| `approved_ts IS NULL` | 14 | 0.0145% | Missing approval timestamp. |
| **Any Anomalous Sequence** | `is_anomalous_sequence == True` | **1,373** | **1.4232%** | **Total unique orders with at least one sequence flaw.** |

### 2.2 Sensitivity Analysis: Clean Fulfillment vs. All Valid Delivered
We compared the target late delivery rate across the full population versus the clean-sequence population:

| Population Segment | Total Orders | Late Orders | Late Rate (%) | Diff vs Full (pp) |
|---|---|---|---|---|
| **Full Valid Delivered Orders** | 96,470 | 6,534 | **6.7731%** | 0.0000 pp |
| **Clean Fulfillment Sequence (`is_anomalous = False`)** | 95,097 | 6,510 | **6.8456%** | **+0.0726 pp** |
| **Anomalous Sequence (`is_anomalous = True`)** | 1,373 | 24 | **1.7480%** | **−5.0251 pp** |

### Policy Decision
- The late rate discrepancy between the clean fulfillment subset (6.85%) and the full population (6.77%) is only **0.0726 percentage points** (a negligible 1.07% relative difference).
- **Decision:** Keep all 96,470 orders in the modeling warehouse to preserve complete sample coverage and control total integrity, but add an explicit binary flag `is_anomalous_sequence` and test model sensitivity with and without anomalous records during Phase 4/5.

---

## 3. Target Prevalence and Temporal Seasonality

### 3.1 Monthly Prevalence Breakdown (2016-09 to 2018-08)
Computed from `03_PowerBI_Olist_Delay_Risk_ML/data/audit/monthly_prevalence.csv`:

| Year-Month | Valid Delivered Orders | Late Delivered Orders | Late Delivery Rate (%) | Mean Promised Lead Time (Days) | Mean Carrier Transit (Days) |
|---|---|---|---|---|---|
| **2016-09** | 1 | 1 | 100.00% | 19.00 | 1.61 |
| **2016-10** | 265 | 2 | 0.75% | 56.31 | 5.73 |
| **2016-12** | 1 | 0 | 0.00% | 27.00 | 1.40 |
| **2017-01** | 750 | 22 | 2.93% | 40.16 | 9.20 |
| **2017-02** | 1,653 | 49 | 2.96% | 32.50 | 9.54 |
| **2017-03** | 2,546 | 116 | 4.56% | 25.37 | 9.83 |
| **2017-04** | 2,303 | 151 | 6.56% | 27.99 | 11.24 |
| **2017-05** | 3,545 | 106 | 2.99% | 24.92 | 8.34 |
| **2017-06** | 3,135 | 95 | 3.03% | 24.66 | 8.88 |
| **2017-07** | 3,872 | 108 | 2.79% | 23.96 | 8.48 |
| **2017-08** | 4,193 | 122 | 2.91% | 24.11 | 8.06 |
| **2017-09** | 4,150 | 182 | 4.39% | 23.07 | 8.65 |
| **2017-10** | 4,478 | 187 | 4.18% | 23.66 | 8.41 |
| **2017-11** | 7,288 | 904 | **12.40%** | 23.20 | 11.09 |
| **2017-12** | 5,513 | 411 | 7.46% | 28.31 | 11.67 |
| **2018-01** | 7,069 | 403 | 5.70% | 26.94 | 10.48 |
| **2018-02** | 6,555 | 926 | **14.13%** | 25.16 | 13.38 |
| **2018-03** | 7,003 | 1,328 | **18.96%** | 22.67 | 12.97 |
| **2018-04** | 6,798 | 306 | 4.50% | 24.31 | 8.67 |
| **2018-05** | 6,749 | 443 | 6.56% | 23.53 | 8.73 |
| **2018-06** | 6,096 | 71 | **1.16%** | 28.41 | 6.68 |
| **2018-07** | 6,156 | 208 | 3.38% | 20.33 | 6.14 |
| **2018-08** | 6,351 | 393 | 6.19% | 15.82 | 5.21 |

### 3.2 Analysis of the Feb–Mar 2018 Delay Surge
- **Observed Pattern:** Between February and March 2018, late deliveries spiked to **14.13% (926 late)** and **18.96% (1,328 late)**, compared to the steady-state baseline average of **6.79%**.
- **Underlying Logistics Dynamics:**
  - Mean carrier transit time surged to **13.38 days in Feb** and **12.97 days in March** (vs. ~8.5 days in summer/autumn 2017).
  - Simultaneously, mean promised lead times were compressed to **22.67 days in March 2018** (the second-lowest in 2017–2018), leaving minimal buffer to absorb carrier transit volatility.
  - **External Context:** Corresponds to the nationwide Brazilian postal service (*Correios*) logistical backlogs and regional strikes reported in early 2018.
- **Modeling Implication:** Time-based validation must account for this period. A model trained prior to 2018 will face strong temporal distribution shift when tested during Q1 2018, making rolling-origin evaluation essential.

---

## 4. Right-Censoring Analysis

### 4.1 Carrier Transit Time Distribution
To determine an unbiased cutoff date, we evaluated the empirical transit duration distribution (`transit_days`) across valid delivered orders:
- **Median (50th percentile):** 7.10 days
- **75th percentile:** 12.03 days
- **90th percentile:** 18.90 days
- **95th percentile:** 24.20 days
- **99th percentile:** 40.98 days
- **Maximum observed:** 205.19 days

### 4.2 Dataset Cutoff Date & Right-Censoring Window
- The raw dataset export occurred on **2018-10-17 17:30:18**.
- The last valid delivered order in the dataset was purchased on **2018-08-29 15:00:37** and delivered to the customer on **2018-08-30 18:05:27**.
- Between **2018-08-30** and **2018-10-17**, 25 orders were placed in raw data: 24 are `canceled` and 1 is `shipped`. Exactly **0 delivered orders** exist after August 2018.
- **Chosen Cutoff Date:** **2018-08-22 23:59:59**.
  - Orders purchased up to 2018-08-22 had **at least 55 calendar days** of potential observation before the dataset terminated on 2018-10-17.
  - Because 99% of deliveries complete within 40.98 days, an observation runway of 55 days provides $>99.5\%$ certainty of outcome finalization.
  - Orders placed after 2018-08-22 (496 delivered orders) are withheld from test evaluation to avoid survival/censoring bias.

---

## 5. Multi-Seller and Multi-Item Aggregation Rules

### 5.1 Observed Order Composition
From `fact_order_items` across 98,666 distinct orders with items:
- **Single-Item Orders:** 88,863 orders (**90.06%** of orders with items; 89.36% of all orders).
- **Multi-Item Orders:** 9,803 orders (**9.94%**; max items in a single order = 21).
- **Single-Seller Orders:** 97,388 orders (**98.70%**).
- **Multi-Seller Orders:** 1,278 orders (**1.30%**; max sellers in a single order = 5).

### 5.2 Deterministic Order-Level Aggregation Rules
Because the target `is_late` is defined at the **order level** (`order_id`), all item-level and seller-level features must be aggregated to order-level using point-in-time safe operators:

| Feature Name | Source Level | Aggregation Rule | Operational Justification |
|---|---|---|---|
| `item_count` | Items | `COUNT(order_item_id)` | Captures order complexity and picking/packing overhead. |
| `seller_count` | Items | `COUNT(DISTINCT seller_id)` | Quantifies multi-merchant coordination risk. |
| `is_multi_seller` | Items | $\mathbb{I}(\text{seller\_count} > 1)$ | Binary indicator for multi-package split shipments. |
| `total_order_price` | Items | `SUM(price)` | Total merchandise value. |
| `total_freight_value` | Items | `SUM(freight_value)` | Total freight cost paid. |
| `freight_to_price_ratio` | Items | `total_freight_value / total_order_price` | Economic freight burden ratio. |
| `max_route_distance_km` | Items / Geo | `MAX(route_distance_km)` | In multi-seller orders, the farthest package dictates arrival SLA. |
| `min_handover_slack_days`| Items | `MIN(shipping_limit_date - carrier_date)` | The slowest seller handover creates the critical fulfillment path. |
| `is_any_late_dispatch` | Items | `MAX(is_late_dispatch)` | Flagged True if any merchant in the order breached their dispatch SLA. |
| `max_product_weight_g` | Items / Dim | `MAX(product_weight_g)` | Heaviest parcel dictates transport classification. |
| `total_product_weight_g`| Items / Dim | `SUM(product_weight_g)` | Total freight weight. |
| `max_product_volume_cm3`| Items / Dim | `MAX(length * height * width)` | Largest volumetric parcel. |
| `primary_product_category`| Items / Dim | Category of item with highest `price` | Dominant merchandise category of the order. |
| `primary_seller_id` | Items | Seller associated with highest item `price` | Attribute order to principal merchant. |
