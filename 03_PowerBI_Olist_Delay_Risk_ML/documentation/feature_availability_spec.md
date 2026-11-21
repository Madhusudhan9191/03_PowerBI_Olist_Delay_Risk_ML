# Feature Availability & Leakage Risk Specification
**Project:** Olist Delivery-Delay Risk Model  
**Directory:** `03_PowerBI_Olist_Delay_Risk_ML/documentation/feature_availability_spec.md`  
**Prediction Moment ($\mathbf{T_{\text{predict}}}$):** `order_delivered_carrier_date`  
**Status:** Approved for Feature Pipeline Design  

---

## 1. Candidate Features Allowed at Prediction Moment

All candidate features below satisfy the point-in-time contract: **their information is strictly observable at or before `order_delivered_carrier_date`**.

| Feature Name | Source Table & Column | Timestamp Available | Definition / Formula | Leakage Risk & Safeguard |
|---|---|---|---|---|
| `promised_lead_time_days` | `fact_order_delivery.order_estimated_delivery_date` | `order_purchase_timestamp` | Estimated delivery date minus purchase date (days) | Safe. Assigned at purchase before carrier dispatch. |
| `merchant_handling_days` | `fact_order_delivery.order_delivered_carrier_date` | `order_delivered_carrier_date` | Carrier date minus purchase date (days) | Safe. Realized at the prediction moment. |
| `approval_to_carrier_days`| `fact_order_delivery.order_delivered_carrier_date` | `order_delivered_carrier_date` | Carrier date minus approval timestamp (days) | Safe. Realized at the prediction moment. |
| `handover_slack_days` | `fact_order_items.shipping_limit_date` | `order_delivered_carrier_date` | `shipping_limit_date` minus carrier date (days) | Safe. Compares dispatch SLA against handover time. |
| `is_late_dispatch` | `fact_order_items.shipping_limit_date` | `order_delivered_carrier_date` | $\mathbb{I}(\text{carrier\_date} > \text{shipping\_limit\_date})$ | Safe. Indicator of merchant SLA breach. |
| `max_route_distance_km` | `fact_order_items.route_distance_km` | `order_purchase_timestamp` | Maximum route km between seller and customer | Safe. Computed from zip code geolocations. |
| `order_distance_band` | `fact_order_delivery.order_distance_band` | `order_purchase_timestamp` | Categorical distance bucket (Local, Regional, Remote) | Safe. Static geography lookup. |
| `is_interstate` | `fact_order_items.is_interstate` | `order_purchase_timestamp` | $\mathbb{I}(\text{seller\_state} \ne \text{customer\_state})$ | Safe. Fixed at order checkout. |
| `customer_state` | `fact_order_delivery.order_customer_state` | `order_purchase_timestamp` | Destination state code (27 Brazilian UFs) | Safe. Known at purchase. |
| `customer_region` | Derived from `customer_state` | `order_purchase_timestamp` | Macro-region (Sudeste, Sul, Nordeste, Norte, Centro-Oeste) | Safe. Static mapping. |
| `total_order_price` | `fact_order_items.price` | `order_purchase_timestamp` | Sum of item prices in order (BRL) | Safe. Fixed at transaction time. |
| `total_freight_value` | `fact_order_items.freight_value` | `order_purchase_timestamp` | Sum of freight charges in order (BRL) | Safe. Fixed at transaction time. |
| `freight_to_price_ratio` | Derived from price & freight | `order_purchase_timestamp` | `total_freight_value / total_order_price` | Safe. Fixed at transaction time. |
| `item_count` | `fact_order_items.order_item_id` | `order_purchase_timestamp` | Count of items in the order | Safe. Fixed at transaction time. |
| `seller_count` | `fact_order_items.seller_id` | `order_purchase_timestamp` | Count of distinct sellers in the order | Safe. Fixed at transaction time. |
| `is_multi_seller` | Derived from `seller_count` | `order_purchase_timestamp` | $\mathbb{I}(\text{seller\_count} > 1)$ | Safe. Fixed at transaction time. |
| `max_product_weight_g` | `dim_product.product_weight_g` | `order_purchase_timestamp` | Max weight among items in order (grams) | Safe. Catalog attribute. |
| `total_product_weight_g`| `dim_product.product_weight_g` | `order_purchase_timestamp` | Sum of item weights in order (grams) | Safe. Catalog attribute. |
| `max_product_volume_cm3`| `dim_product` dimensions | `order_purchase_timestamp` | Max $(L \times H \times W)$ among items in order ($\text{cm}^3$) | Safe. Catalog attribute. |
| `primary_product_category`| `dim_product.product_category_name_en`| `order_purchase_timestamp` | Category of the highest-price item | Safe. Catalog attribute. |
| `payment_type` | `fact_order_payments.payment_type` | `order_approved_at` | Primary payment instrument (credit_card, boleto, voucher) | Safe. Known upon payment clearance. |
| `payment_installments` | `fact_order_payments.payment_installments`| `order_approved_at` | Number of payment installments | Safe. Known upon payment clearance. |
| `handover_dow` | Derived from carrier date | `order_delivered_carrier_date` | Day of week of carrier handover (0=Mon ... 6=Sun) | Safe. Calendar feature of the prediction instant. |
| `handover_month` | Derived from carrier date | `order_delivered_carrier_date` | Month of carrier handover (1 to 12) | Safe. Calendar feature of the prediction instant. |
| `is_weekend_handover` | Derived from carrier date | `order_delivered_carrier_date` | $\mathbb{I}(\text{handover\_dow} \in [5, 6])$ | Safe. Handover on Saturday/Sunday. |

---

## 2. Historical Aggregated Features (Expanding Window with Lag)

To prevent target leakage, all historical aggregations (seller late rate, corridor late rate) must adhere to a strict chronological lag rule:
$$\text{Historical Rate}(t) = \frac{\sum_{i \in \mathcal{H}(t)} \text{is\_late}_i}{|\mathcal{H}(t)|}, \quad \text{where } \mathcal{H}(t) = \{ i : \text{order\_delivered\_customer\_date}_i < t \}$$

Notice that an order's outcome is only included if its ground-truth delivery timestamp occurred **strictly before** the target order's prediction moment $t = \text{order\_delivered\_carrier\_date}$.

| Feature Name | Computation Window | Lag Requirement | Smoothing / Handling | Leakage Risk & Safeguard |
|---|---|---|---|---|
| `seller_hist_completed_orders` | Expanding window up to $t$ | Outcome known before $t$ | Direct count | Safe. Zero lookahead. |
| `seller_hist_late_rate` | Expanding window up to $t$ | Outcome known before $t$ | Empirical Bayes / M-estimate smoothing toward global prior | Safe. Prevent small-sample overfitting. |
| `seller_no_history_flag` | Expanding window up to $t$ | Outcome known before $t$ | Binary flag: $\mathbb{I}(\text{orders} < 5)$ | Safe. Explicit cold-start indicator. |
| `corridor_hist_completed_orders` | Expanding window up to $t$ | Outcome known before $t$ | Count by (seller_state, customer_state) | Safe. Zero lookahead. |
| `corridor_hist_late_rate` | Expanding window up to $t$ | Outcome known before $t$ | M-estimate smoothing toward state prior | Safe. Captures regional carrier congestion. |
| `corridor_no_history_flag`| Expanding window up to $t$ | Outcome known before $t$ | Binary flag: $\mathbb{I}(\text{orders} < 10)$ | Safe. Cold-start corridor flag. |

---

## 3. Strictly Forbidden Features (Target Leakage & Post-Event Signals)

The following columns and derived metrics are **STRICTLY PROHIBITED** from entering feature matrices:

| Column / Concept | Source Table | Reason for Prohibition | Leakage Mechanism |
|---|---|---|---|
| `order_delivered_customer_date` | `fact_order_delivery` | Post-event ground truth. | Direct target definition component. |
| `order_delivered_customer_date_only` | `fact_order_delivery` | Post-event ground truth. | Direct target definition component. |
| `transit_days` | `fact_order_delivery` | Depends on customer delivery timestamp. | Future knowledge of transit completion. |
| `cycle_days` | `fact_order_delivery` | Depends on customer delivery timestamp. | Future knowledge of total cycle duration. |
| `delay_days` | `fact_order_delivery` | Computed from customer delivery timestamp. | Direct target leakage. |
| `is_late` / `is_on_time` | `fact_order_delivery` | Ground-truth target labels. | Direct target leakage. |
| `order_delivery_status_category` | `fact_order_delivery` | Derived from delivery outcome. | Post-event classification. |
| `review_id` / `review_score` | `fact_order_delivery` | Review submitted post-purchase or post-delivery. | High target correlation and future leakage. |
| `review_creation_date` | `fact_order_delivery` | Review event timestamp. | Post-handover customer interaction. |
| `review_answer_timestamp` | `fact_order_delivery` | Customer response timestamp. | Post-handover customer interaction. |
| `review_created_before_delivery` | `fact_order_delivery` | Depends on customer delivery date. | Post-event interaction flag. |
| Future carrier tracking milestones | External / Carrier | Intermediate scans between carrier and delivery. | Occur after $T_{\text{predict}}$. |
