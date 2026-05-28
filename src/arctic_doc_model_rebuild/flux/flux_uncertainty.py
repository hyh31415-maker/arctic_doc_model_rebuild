from __future__ import annotations

import pandas as pd

from .flux_units import doc_q_to_flux_kg_day, kg_day_to_tg_day


def attach_flux_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for level in ["80", "90", "95"]:
        lower_doc = f"DOC_prediction_interval_{level}_lower"
        upper_doc = f"DOC_prediction_interval_{level}_upper"
        lower_kg = f"daily_flux_{level}_lower_kgC_day"
        upper_kg = f"daily_flux_{level}_upper_kgC_day"
        out[lower_kg] = doc_q_to_flux_kg_day(out[lower_doc], out["Q_m3s"])
        out[upper_kg] = doc_q_to_flux_kg_day(out[upper_doc], out["Q_m3s"])
        out[f"daily_flux_{level}_lower_TgC_day"] = kg_day_to_tg_day(out[lower_kg])
        out[f"daily_flux_{level}_upper_TgC_day"] = kg_day_to_tg_day(out[upper_kg])
    return out
