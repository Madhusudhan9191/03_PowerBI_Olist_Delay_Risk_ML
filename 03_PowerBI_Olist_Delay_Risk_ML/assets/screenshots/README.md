# Dashboard Screenshots & Visual Index
**Directory:** `03_PowerBI_Olist_Delay_Risk_ML/assets/screenshots/`

This directory stores unfiltered canvas captures of the 6 Power BI report pages:

1. `page1_problem_and_baseline.png` — Context cards, baseline comparisons (PR-AUC 0.2933 vs 0.1344 lead heuristic and 0.0369 prior), and study limitations.
2. `page2_model_performance.png` — Live Threshold and Cost Ratio sliders, PR Curve, Calibrated Calibration Curve, and Confusion Matrix.
3. `page3_at_risk_queue.png` — Operational triage queue with conditional risk scores and risk-band donut distribution (Top 2% Critical Queue: 362 orders, 152 late caught).
4. `page4_segment_performance.png` — Geographic disparity by customer state and order value tiers with sample-size suppression tags.
5. `page5_monitoring_and_data_quality.png` — 15 Data Quality test assertions (14 PASS / 1 real data anomaly finding on DQ-15) and feature drift (PSI).
6. `page6_model_card_and_governance.png` — Model metadata, standardized coefficients / permutation feature importance ranking, and ethical/operational disclaimers.

*To export clean canvas screenshots from Power BI Desktop:*
`File -> Export -> Export to PDF`, or use Windows Snipping Tool (`Win + Shift + S`) across each unfiltered canvas view.
