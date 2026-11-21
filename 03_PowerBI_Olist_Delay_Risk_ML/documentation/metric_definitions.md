# Metric Definitions Sheet
**Project:** Olist Delivery-Delay Risk Model  
**Directory:** `03_PowerBI_Olist_Delay_Risk_ML/documentation/metric_definitions.md`  
**Scope:** Shared Business & Analytical Metric Definitions  
**Status:** Validated  

This document serves as the single source of truth for business and machine learning metrics across Python, SQL, and Power BI DAX implementations.

---

## 1. Fulfillment & Logistics Performance Metrics

### 1.1 Valid Delivered Orders
- **Formula:** $\sum \mathbb{I}(\text{order\_status} = \text{'delivered'} \land \text{order\_delivered\_customer\_date} \text{ IS NOT NULL})$
- **Grain:** Order level (`order_id`).
- **Exclusions:** Canceled, unavailable, shipped, invoiced, processing, created, approved orders, and delivered orders with null delivery date.
- **Warehouse Benchmark:** Exactly 96,470 orders.

### 1.2 Late Delivery Flag (`is_late`)
- **Formula:** $\mathbb{I}(\text{CAST}(\text{order\_delivered\_customer\_date AS DATE}) > \text{CAST}(\text{order\_estimated\_delivery\_date AS DATE}))$
- **Grain:** Order level (`order_id`).
- **Exclusions:** Non-delivered orders.
- **Warehouse Benchmark:** Exactly 6,534 late orders.

### 1.3 On-Time Delivery Flag (`is_on_time`)
- **Formula:** $\mathbb{I}(\text{CAST}(\text{order\_delivered\_customer\_date AS DATE}) \le \text{CAST}(\text{order\_estimated\_delivery\_date AS DATE}))$
- **Grain:** Order level (`order_id`).
- **Exclusions:** Non-delivered orders.
- **Warehouse Benchmark:** Exactly 89,936 on-time orders.

### 1.4 Late Delivery Rate (%)
- **Formula:** $\frac{\sum \text{is\_late}}{\text{Valid Delivered Orders}} \times 100\%$
- **Grain:** Aggregated (national, monthly, regional, or segment).
- **Exclusions:** Non-delivered orders.
- **Warehouse Benchmark:** **6.7731%** across all 96,470 valid delivered orders.

### 1.5 Promised Lead Time (Days)
- **Formula:** $\text{DATE\_DIFF}(\text{day}, \text{order\_purchase\_timestamp}, \text{order\_estimated\_delivery\_date})$
- **Grain:** Order level.
- **Exclusions:** Orders with negative or corrupt timestamps.
- **Benchmark Mean:** 24.37 days.

### 1.6 Merchant Handling Duration (Days)
- **Formula:** $\text{DATE\_DIFF}(\text{day}, \text{order\_purchase\_timestamp}, \text{order\_delivered\_carrier\_date})$
- **Grain:** Order level.
- **Exclusions:** Non-delivered orders, null carrier handover dates.
- **Benchmark Mean:** 3.23 days across valid delivered orders.

### 1.7 Carrier Transit Duration (Days)
- **Formula:** $\text{DATE\_DIFF}(\text{day}, \text{order\_delivered\_carrier\_date}, \text{order\_delivered\_customer\_date})$
- **Grain:** Order level.
- **Exclusions:** Non-delivered orders, null carrier dates, negative transit anomalies.
- **Benchmark Median:** 7.10 days (Mean: 9.33 days, 95th percentile: 24.20 days).

### 1.8 Merchant Handover Slack (Days)
- **Formula:** $\text{DATE\_DIFF}(\text{day}, \text{order\_delivered\_carrier\_date}, \text{shipping\_limit\_date})$
- **Grain:** Order level (minimum across items for multi-item orders).
- **Interpretation:** Positive value indicates handover occurred before seller SLA deadline (buffer remaining); negative value indicates late handover to carrier.

### 1.9 Late Dispatch Flag (`is_late_dispatch`)
- **Formula:** $\mathbb{I}(\text{order\_delivered\_carrier\_date} > \text{shipping\_limit\_date})$
- **Grain:** Order level (True if ANY item in the order was dispatched late).
- **Interpretation:** Indicator of merchant fulfillment SLA breach prior to carrier custody.

---

## 2. Machine Learning Evaluation Metrics

### 2.1 PR-AUC (Average Precision)
- **Definition:** Area under the Precision-Recall curve: $\sum_n (R_n - R_{n-1}) P_n$.
- **Primary Metric Justification:** High class imbalance (positive prevalence = 6.77%). PR-AUC directly evaluates the trade-off between precision and recall on the rare positive class without being inflated by true negatives.
- **Benchmark Baseline:** Constant prior prediction achieves PR-AUC equal to the positive prevalence ($\approx 0.068$).

### 2.2 ROC-AUC
- **Definition:** Area under the Receiver Operating Characteristic curve (True Positive Rate vs. False Positive Rate).
- **Secondary Metric Justification:** Evaluates global ranking discrimination across all thresholds.
- **Benchmark Baseline:** Random guess achieves $0.500$.

### 2.3 Brier Score
- **Definition:** Mean squared error of predicted probabilities: $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$.
- **Interpretation:** Strictly proper scoring rule assessing both discrimination and calibration. Lower is better.
- **Benchmark Baseline:** Constant prior prediction ($p = 0.0677$) yields a Brier score of $p(1-p) = 0.0631$.

### 2.4 Expected Calibration Error (ECE)
- **Definition:** Weighted absolute difference between predicted confidence and empirical accuracy across $M$ equal-frequency bins: $\sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$.
- **Target:** ECE $< 0.02$ indicates well-calibrated probabilities for operational decision-making.

### 2.5 Capacity-Constrained Metrics: Recall @ Top-k%
- **Definition:** Recall achieved when flagging the top $k\%$ highest-risk orders: $\frac{\text{True Positives in top } k\% \text{ queue}}{\text{Total Positives in population}}$.
- **Operational Alignment:** Operations teams have fixed capacity (e.g. investigating top 2%, 5%, or 10% of shipments).
- **Target Capacity Tiers:**
  - **Top 2% Capacity:** Immediate high-urgency escalation.
  - **Top 5% Capacity:** Standard proactive carrier intervention.
  - **Top 10% Capacity:** Extended customer service notification queue.

### 2.6 Lift @ Top-k%
- **Definition:** Ratio of positive prevalence in the top $k\%$ queue to overall population prevalence: $\frac{\text{Precision @ Top-}k\%}{\text{Population Prevalence}}$.
- **Baseline:** Lift = $1.0$ represents random selection.

### 2.7 Expected Operational Cost
- **Formula:** $\text{Cost} = C_{FP} \cdot FP + C_{FN} \cdot FN$
  - $C_{FP}$: Assumed cost of needless operational intervention (e.g., proactive SMS/call investigation time).
  - $C_{FN}$: Assumed cost of an unflagged late delivery (e.g., customer churn risk, concession vouchers).
- **Disclosure:** Parametric scenario model based on explicitly stated cost ratios (e.g., $C_{FN}/C_{FP} \in [2, 5, 10, 20]$). Not a claim of realized business savings.

---

## 3. Monitoring & Drift Metrics

### 3.1 Population Stability Index (PSI)
- **Formula:** $\sum_{b=1}^B \left( \% \text{Actual}_b - \% \text{Expected}_b \right) \times \ln\left( \frac{\% \text{Actual}_b}{\% \text{Expected}_b} \right)$
- **Thresholds:**
  - $\text{PSI} < 0.10$: Negligible shift (stable distribution).
  - $0.10 \le \text{PSI} < 0.25$: Moderate shift (monitor closely).
  - $\text{PSI} \ge 0.25$: Significant distribution shift (action required / model recalibration).
