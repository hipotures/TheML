import numpy as np
import pandas as pd

NUMERIC_INPUTS = (
    "sleep_duration",
    "exercise_duration",
    "step_count",
)

TRAIN_ID_MAX = 690087
EPSILON = 1.0e-6


def add_movement_sleep_compositional_logratios(raw, deps, aux):
    index = raw.index
    n_rows = len(raw)

    train_mask = None
    if "id" in raw.columns:
        train_mask = pd.to_numeric(raw["id"], errors="coerce") <= TRAIN_ID_MAX

    filled = {}
    for col in NUMERIC_INPUTS:
        values = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else pd.Series(np.nan, index=index)

        if train_mask is not None and bool(train_mask.any()):
            median_value = values.loc[train_mask].median()
        else:
            median_value = values.median()

        if pd.isna(median_value):
            if col == "sleep_duration":
                median_value = 7.0
            elif col == "exercise_duration":
                median_value = 30.0
            else:
                median_value = 7000.0

        filled[col] = values.fillna(median_value).astype(float)

    sleep_hours = filled["sleep_duration"].clip(lower=3.0, upper=10.0)
    structured_activity_hours = filled["exercise_duration"].clip(lower=0.0, upper=120.0) / 60.0
    steps = filled["step_count"].clip(lower=1000.0, upper=15000.0)

    cadence_activity_hours = steps / 100.0 / 60.0
    ambulatory_hours = (cadence_activity_hours - structured_activity_hours).clip(lower=0.0)

    inactive_waking_hours = (
        24.0 - sleep_hours - structured_activity_hours - ambulatory_hours
    ).clip(lower=0.25)

    parts = pd.DataFrame(
        {
            "sleep": sleep_hours.clip(lower=EPSILON),
            "structured": structured_activity_hours.clip(lower=EPSILON),
            "ambulatory": ambulatory_hours.clip(lower=EPSILON),
            "inactive": inactive_waking_hours.clip(lower=EPSILON),
        },
        index=index,
    )

    part_sum = parts.sum(axis=1).replace(0.0, np.nan)
    composition = parts.div(part_sum, axis=0) * 24.0
    shares = composition / 24.0

    total_active_hours = composition["structured"] + composition["ambulatory"]
    entropy_terms = shares * np.log(shares.clip(lower=EPSILON))
    composition_entropy = -entropy_terms.sum(axis=1) / np.log(4.0)

    new_features = pd.DataFrame(index=index)
    new_features["log_sleep_to_inactive"] = np.log(
        composition["sleep"].clip(lower=EPSILON) / composition["inactive"].clip(lower=EPSILON)
    )
    new_features["log_structured_to_inactive"] = np.log(
        composition["structured"].clip(lower=EPSILON) / composition["inactive"].clip(lower=EPSILON)
    )
    new_features["log_ambulatory_to_inactive"] = np.log(
        composition["ambulatory"].clip(lower=EPSILON) / composition["inactive"].clip(lower=EPSILON)
    )
    new_features["log_total_active_to_sleep"] = np.log(
        total_active_hours.clip(lower=EPSILON) / composition["sleep"].clip(lower=EPSILON)
    )
    new_features["active_day_share"] = total_active_hours / 24.0
    new_features["inactive_day_share"] = composition["inactive"] / 24.0
    new_features["composition_entropy"] = composition_entropy

    if len(new_features) != n_rows:
        raise ValueError("Feature group changed row count.")

    return new_features


FEATURE_GROUPS = [
    {
        "name": "movement_sleep_compositional_logratios",
        "fn": add_movement_sleep_compositional_logratios,
        "depends_on": [],
        "description": "Daily movement-sleep compositional log-ratio features from sleep, exercise, step count, and inactive waking time.",
    }
]