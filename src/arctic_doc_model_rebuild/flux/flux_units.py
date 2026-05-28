from __future__ import annotations

import pandas as pd


KG_C_PER_DAY_PER_MG_L_M3_S = 86.4


def doc_q_to_flux_kg_day(doc_mg_c_l: pd.Series, q_m3s: pd.Series) -> pd.Series:
    """Convert DOC concentration and discharge to kg C/day."""
    return pd.to_numeric(doc_mg_c_l, errors="coerce") * pd.to_numeric(q_m3s, errors="coerce") * KG_C_PER_DAY_PER_MG_L_M3_S


def kg_day_to_mg_day(flux_kg_day: pd.Series) -> pd.Series:
    return pd.to_numeric(flux_kg_day, errors="coerce") / 1000.0


def kg_day_to_tg_day(flux_kg_day: pd.Series) -> pd.Series:
    return pd.to_numeric(flux_kg_day, errors="coerce") / 1e9
