"""
Export Dimension Tables for Power BI Semantic Model
Project: Olist Delivery-Delay Risk Model
Path: 03_PowerBI_Olist_Delay_Risk_ML/src/features/export_powerbi_dimensions.py

Generates:
- dim_date: Bounded to 2016-01-01 to 2018-10-17 (never TODAY())
- dim_geography: 27 Brazilian states with macro-regions and capital cities
- dim_risk_band: Risk tier sorting and operational queue definitions
"""

import os
import pandas as pd
import numpy as np

def export_dimensions(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    export_dir = os.path.join(base_dir, "data", "powerbi_export")
    os.makedirs(export_dir, exist_ok=True)

    # 1. dim_date: Bounded strictly from 2016-09-01 to 2018-10-17
    dates = pd.date_range("2016-09-01", "2018-10-17", freq="D")
    df_date = pd.DataFrame({
        "date_key": dates.strftime("%Y%m%d").astype(int),
        "full_date": dates.strftime("%Y-%m-%d"),
        "year": dates.year,
        "quarter": "Q" + dates.quarter.astype(str),
        "year_quarter": dates.year.astype(str) + "-Q" + dates.quarter.astype(str),
        "month_num": dates.month,
        "month_name": dates.strftime("%B"),
        "year_month": dates.strftime("%Y-%m"),
        "day_of_month": dates.day,
        "day_of_week": dates.dayofweek,
        "day_name": dates.strftime("%A"),
        "is_weekend": np.where(dates.dayofweek >= 5, 1, 0)
    })
    df_date.to_csv(os.path.join(export_dir, "dim_date.csv"), index=False)

    # 2. dim_geography: 27 Brazilian Federation Units (UFs)
    states_data = [
        {"state_code": "SP", "state_name": "São Paulo", "region": "Sudeste", "is_hub": 1},
        {"state_code": "RJ", "state_name": "Rio de Janeiro", "region": "Sudeste", "is_hub": 0},
        {"state_code": "MG", "state_name": "Minas Gerais", "region": "Sudeste", "is_hub": 0},
        {"state_code": "ES", "state_name": "Espírito Santo", "region": "Sudeste", "is_hub": 0},
        {"state_code": "PR", "state_name": "Paraná", "region": "Sul", "is_hub": 0},
        {"state_code": "SC", "state_name": "Santa Catarina", "region": "Sul", "is_hub": 0},
        {"state_code": "RS", "state_name": "Rio Grande do Sul", "region": "Sul", "is_hub": 0},
        {"state_code": "DF", "state_name": "Distrito Federal", "region": "Centro-Oeste", "is_hub": 0},
        {"state_code": "GO", "state_name": "Goiás", "region": "Centro-Oeste", "is_hub": 0},
        {"state_code": "MT", "state_name": "Mato Grosso", "region": "Centro-Oeste", "is_hub": 0},
        {"state_code": "MS", "state_name": "Mato Grosso do Sul", "region": "Centro-Oeste", "is_hub": 0},
        {"state_code": "BA", "state_name": "Bahia", "region": "Nordeste", "is_hub": 0},
        {"state_code": "PE", "state_name": "Pernambuco", "region": "Nordeste", "is_hub": 0},
        {"state_code": "CE", "state_name": "Ceará", "region": "Nordeste", "is_hub": 0},
        {"state_code": "MA", "state_name": "Maranhão", "region": "Nordeste", "is_hub": 0},
        {"state_code": "PB", "state_name": "Paraíba", "region": "Nordeste", "is_hub": 0},
        {"state_code": "RN", "state_name": "Rio Grande do Norte", "region": "Nordeste", "is_hub": 0},
        {"state_code": "AL", "state_name": "Alagoas", "region": "Nordeste", "is_hub": 0},
        {"state_code": "SE", "state_name": "Sergipe", "region": "Nordeste", "is_hub": 0},
        {"state_code": "PI", "state_name": "Piauí", "region": "Nordeste", "is_hub": 0},
        {"state_code": "PA", "state_name": "Pará", "region": "Norte", "is_hub": 0},
        {"state_code": "AM", "state_name": "Amazonas", "region": "Norte", "is_hub": 0},
        {"state_code": "RO", "state_name": "Rondônia", "region": "Norte", "is_hub": 0},
        {"state_code": "TO", "state_name": "Tocantins", "region": "Norte", "is_hub": 0},
        {"state_code": "AC", "state_name": "Acre", "region": "Norte", "is_hub": 0},
        {"state_code": "AP", "state_name": "Amapá", "region": "Norte", "is_hub": 0},
        {"state_code": "RR", "state_name": "Roraima", "region": "Norte", "is_hub": 0}
    ]
    df_geo = pd.DataFrame(states_data)
    df_geo.to_csv(os.path.join(export_dir, "dim_geography.csv"), index=False)

    # 3. dim_risk_band: Risk classification mapping
    risk_data = [
        {"risk_band": "Low Risk (0-80%)", "sort_order": 1, "action_tier": "Standard Automated Notification", "sla_target": "Normal Monitoring"},
        {"risk_band": "Moderate Risk (80-95%)", "sort_order": 2, "action_tier": "Proactive Carrier Ping", "sla_target": "Check Tracking Milestones"},
        {"risk_band": "High Risk (95-98%)", "sort_order": 3, "action_tier": "Operational Support Escalation", "sla_target": "Priority Route Follow-Up"},
        {"risk_band": "Critical Queue (Top 2%)", "sort_order": 4, "action_tier": "Urgent Manual Investigation", "sla_target": "Proactive Customer SLA Notice"}
    ]
    df_risk = pd.DataFrame(risk_data)
    df_risk.to_csv(os.path.join(export_dir, "dim_risk_band.csv"), index=False)

    print("Dimension tables exported successfully: dim_date, dim_geography, dim_risk_band.")

if __name__ == "__main__":
    export_dimensions()
