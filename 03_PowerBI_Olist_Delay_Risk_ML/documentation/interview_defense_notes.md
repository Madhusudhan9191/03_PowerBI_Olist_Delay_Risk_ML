# Interview Defense & Technical Architecture Notes
**Project:** Olist Delivery-Delay Risk Model  
**Target Roles:** Data Analyst / Junior Data Scientist / Analytics Engineer  
**Directory:** `03_PowerBI_Olist_Delay_Risk_ML/documentation/interview_defense_notes.md`  

---

## 1. Core Problem & System Design Decisions

### Q1: Why predict at `order_delivered_carrier_date` rather than at checkout (`order_purchase_timestamp`)?
> **Answer:** At checkout, the only available information is customer address, seller location, and catalog dimensions. Crucially, the actual merchant handling time—how many days the seller takes to pack and hand over the parcel—is completely unknown. In our empirical feature importance analysis, `merchant_handling_days` (+0.1960 PR-AUC impact) and `handover_slack_days` (+0.0868) are two of the three most powerful predictors of delivery delay. Predicting at the moment custody transfers to the carrier captures whether the merchant already consumed the safety buffer, while leaving the entire carrier transit window open for operational intervention.

### Q2: Why is date-level comparison (`date > date`) the only valid ground truth, rather than timestamp comparison (`ts > ts`)?
> **Answer:** In Olist, estimated delivery dates are stored with truncated time (`YYYY-MM-DD 00:00:00`). Actual deliveries occur during normal working hours (e.g. 14:00:00). A timestamp-level comparison falsely labels packages delivered on the promised date as "late" because `14:00:00 > 00:00:00`. In our audit, exactly **1,292 orders** were delivered on the promised calendar date. Treating them as late artificially inflates the delay rate from 6.77% to 8.11% and distorts SLA reporting. Date-level evaluation correctly mirrors commercial contractual SLAs.

### Q3: How did you ensure point-in-time correctness and eliminate target leakage in historical rates?
> **Answer:** We implemented an expanding-window historical lookup with an explicit **1-day outcome lag**. For an order handed over on calendar day $D$, its historical seller and corridor rates only include orders whose delivery outcome occurred on or before $D - 1$. An order can never see its own outcome or any future outcome. Furthermore, to handle cold-start merchants and rare lanes without overfitting, we implemented empirical Bayes (M-estimate) smoothing toward global and regional priors, coupled with explicit boolean flags (`seller_no_history_flag`).

### Q4: Why is PR-AUC (Average Precision) the primary evaluation metric rather than ROC-AUC or Accuracy, and what is the primary baseline?
> **Answer:** With an imbalanced positive class (3.69% in test, 6.77% overall), accuracy is meaningless (predicting "never late" achieves 96.3% accuracy but zero operational utility). ROC-AUC is overly optimistic on rare events because its false-positive rate denominator is dominated by the huge majority class of on-time orders (89,936 orders). PR-AUC directly evaluates the trade-off between precision (minimizing wasted proactive alerts) and recall (capturing true delays) on the positive class.
>
> Critically, rather than comparing solely against an uninformative prior baseline (PR-AUC 0.0369), our primary operational benchmark is **Baseline C (Promised Lead Time Heuristic, PR-AUC = 0.1344)**, which captures operational scheduling slack. Our model achieves a PR-AUC of **0.2933**, delivering a **2.18x lift over the lead-time heuristic** and a **7.94x lift over prior prevalence**.

### Q4b: Why did you ship regularized Logistic Regression over Gradient Boosted Trees?
> **Answer:** We enforced a strict, non-negotiable model selection rule: **train candidates on the training window and select the best model programmatically based on out-of-time Validation PR-AUC**. Across the 4 evaluated candidates:
> 1. Logistic Regression (Balanced): **Validation PR-AUC = 0.2905** | Test PR-AUC = 0.2933 | Test ROC-AUC = 0.8364
> 2. Logistic Regression (L2): Validation PR-AUC = 0.2848 | Test PR-AUC = 0.3344 | Test ROC-AUC = 0.8390
> 3. HistGradientBoosting (Balanced): Validation PR-AUC = 0.2735 | Test PR-AUC = 0.2292 | Test ROC-AUC = 0.7649
> 4. HistGradientBoosting (Standard): Validation PR-AUC = 0.2144 | Test PR-AUC = 0.2850 | Test ROC-AUC = 0.7692
>
> Boosting did not beat regularized linear modeling. Logistic Regression (Balanced) was programmatically selected. In addition to winning on validation, it proved significantly more robust during operational regime shifts (e.g., in Rolling-Origin Fold 2 during the logistics shock, LR scored PR-AUC 0.3291 vs 0.2481 for boosting) and provides direct interpretability via standardized coefficients. In real-world data science, choosing a simpler, more robust model that outperforms complex architectures is a hallmark of engineering maturity.

---

## 2. Temporal Drift & Validation Strategy

### Q5: Why did the late delivery rate fluctuate from 6.63% in Train to 10.11% in Validation and 3.69% in Test?
> **Answer:** The Olist marketplace experienced severe external operational shocks:
1. **Validation Window (2018-03 to 2018-05):** In March 2018, nationwide postal disruptions and strikes occurred across the Brazilian postal service (*Correios*), combined with regional flooding. In our monthly audit, late delivery rate spiked to **18.96% in March 2018**, with carrier transit times surging from 8.5 days to 13.4 days.
2. **Test Window (2018-06 to 2018-08):** In response, marketplace operations adjusted policies, expanding lead time buffers (mean buffer rose to 28.4 days in June), and transit normalized, causing late rate to plummet to **1.16% in June** and averaging 3.69% across the test cohort.
By using chronological time splits rather than random k-fold splits, we stress-tested the model against real-world regime shifts. Even with this substantial distribution shift, the model maintained an ROC-AUC of **0.7773** and PR-AUC of **0.2811** on the out-of-time test cohort.

### Q6: How did you determine the right-censoring cutoff date of 2018-08-22?
> **Answer:** The raw Olist dataset export ended on 2018-10-17. In our transit analysis, the 99th percentile of carrier transit duration is 40.98 days. If we evaluated orders placed in September or October 2018, many would still be in transit at the moment the database was dumped, creating severe survival bias (only quick deliveries would be observed as completed). By fixing the cutoff at 2018-08-22, every order in the test set has a **minimum 55-day observation follow-up window**, ensuring $>99.5\%$ outcome completion before data collection ceased.

---

## 3. Operational Translation & Business Framing

### Q7: If the validation-tuned F1 threshold yields an impractical alert rate, how should operations deploy this model?
> **Answer:** Operating at a single probability threshold tuned on validation (e.g. 0.1581) alerts 5,677 orders (31.35% of the test cohort) with under 10% precision (9.44%), which completely overwhelms human operations. Therefore, we framed production deployment around **Capacity-Constrained Queue Prioritization (Top-k%)**, which guarantees fixed operational workload:
> - **Top 2% Capacity Queue (362 orders):** Delivers **41.99% precision** (152 late orders caught) with **11.36x lift vs prior** and **3.12x lift vs lead-time heuristic**. Ideal for high-touch, urgent carrier dispatch escalations.
> - **Top 5% Capacity Queue (905 orders):** Delivers **27.85% precision** (252 late orders caught) with **37.67% recall** (7.54x lift vs prior, 2.07x vs heuristic). Suitable for proactive customer messaging and parcel tracking pings.
> - **Top 10% Capacity Queue (1,810 orders):** Captures **52.32% of all late deliveries** with 19.34% precision (5.23x lift vs prior).
> This decouples model scoring from arbitrary probability cutoffs and aligns it directly with operational labor constraints.

### Q8: Does this model prove that slow merchant handling causes carrier delays?
> **Answer:** **No.** This project strictly avoids causal claims ("causes", "proves", "confirms"). The data is purely observational and contains no randomized interventions or carrier route telemetry. An extended `merchant_handling_days` is *associated with* delivery delay, but the underlying reason could be delayed carrier pickup scans, seller inventory stockouts, or unobserved weekend batching. We report associations, not causal mechanisms.

### Q9: How did you enforce three-way reconciliation between Python, SQL, and Power BI DAX?
> **Answer:** We implemented an automated reconciliation script ([run_reconciliation.py](../src/audit/run_reconciliation.py)) that connects to the warehouse via DuckDB and asserts that row counts, currency sums, late order counts, and rate calculations match between Python and SQL to within 0.001 tolerance. Furthermore, per Rule 5, the script explicitly inspects whether the user has exported DAX measure outputs from Power BI Desktop. If absent, it logs `DESKTOP_EXPORT_PENDING` without fabricating or substituting Python numbers. Metrics M01–M08 originate from the validated relational warehouse export (`dax_all_63_export.csv`), and SQL executes DuckDB queries on exported warehouse CSVs.

### Q10: Why did calibration on the validation set overpredict test probabilities, and how do you defend that in an interview?
> **Answer:** This is a classic real-world demonstration of **temporal regime shift and calibration-in-the-large**:
> - Platt scaling (Sigmoid calibration) was fitted on the out-of-time validation window (March–May 2018), where observed late prevalence was **10.11%** due to the nationwide postal strike and severe weather disruptions. The calibrator learned that base risk is ~10%.
> - When applied to the test cohort (June–August 2018), observed prevalence normalized to **3.69%**. Consequently, the model's mean predicted probability was **13.57%**, overpredicting in the large.
> - Because of this base-rate shift, the test Brier score was **0.0439** compared to guessing the training prior (0.0364), resulting in a negative Brier Skill Score ($BSS = -0.2035$).
> - When recalibrated on a recent post-shock window (May 2018, 6.56% late rate), mean predicted probability drops to **8.61%** and Brier score improves to **0.0344**, achieving a positive Brier Skill Score (**BSS = +0.0564** vs prior).
> - **Operational Takeaway:** In an interview, explaining this temporal dynamic proves that you understand probability calibration in dynamic environments, rather than blindly assuming stationary test distributions. Operations should use the model as an **ordinal ranker** for queue triage, and run rolling 30-day recalibrations if presenting probabilities to end users.
