from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def validation_scheme_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "validation_scheme": "leave_one_year_out",
                "validation_role": "primary",
                "group_column": "year",
                "description": "Leave one sampled year out at a time.",
                "stress_test": False,
                "primary_for_model_selection": True,
            },
            {
                "validation_scheme": "river_year_groupkfold",
                "validation_role": "secondary_structure_check",
                "group_column": "river_year",
                "description": "Five-fold GroupKFold using river-year groups.",
                "stress_test": False,
                "primary_for_model_selection": False,
            },
            {
                "validation_scheme": "leave_one_river_out",
                "validation_role": "stress_test",
                "group_column": "river",
                "description": "Held-out river stress test; not primary selection with only six rivers.",
                "stress_test": True,
                "primary_for_model_selection": False,
            },
        ]
    )


def validation_splits(frame: pd.DataFrame):
    years = sorted(pd.Series(frame["year"]).dropna().astype(int).unique())
    for fold_idx, year in enumerate(years, start=1):
        test_mask = frame["year"].astype(int).eq(year).to_numpy()
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        if len(train_idx) and len(test_idx):
            yield "leave_one_year_out", f"loyo_{year}", train_idx, test_idx, {"fold_year": int(year), "fold_group": int(year)}

    groups = frame["river"].astype(str) + "_" + frame["year"].astype(int).astype(str)
    n_groups = groups.nunique()
    if n_groups >= 5 and len(frame) >= 5:
        splitter = GroupKFold(n_splits=5)
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(frame, groups=groups), start=1):
            fold_groups = sorted(groups.iloc[test_idx].unique())
            yield "river_year_groupkfold", f"river_year_gkf_{fold_idx}", train_idx, test_idx, {"fold_group": ";".join(fold_groups)}

    rivers = sorted(frame["river"].dropna().astype(str).unique())
    for river in rivers:
        test_mask = frame["river"].astype(str).eq(river).to_numpy()
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        if len(train_idx) and len(test_idx):
            yield "leave_one_river_out", f"loro_{river}", train_idx, test_idx, {"fold_river": river, "fold_group": river}
