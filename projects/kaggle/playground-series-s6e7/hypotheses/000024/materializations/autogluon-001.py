import numpy as np
import pandas as pd


_ARCHETYPE_NAMES = (
    "balanced_fit",
    "depleted_multirisk",
    "active_overstrained",
    "low_risk_inactive",
    "moderate_mixed",
)

_ARCHETYPE_PROTOTYPES = (
    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0, 0.0, 0.0, -1.0),
    (0.0, -1.0, 1.0, 1.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)


def _numeric(raw, column):
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce").astype("float64")


def _category(raw, column):
    if column not in raw.columns:
        return pd.Series(pd.NA, index=raw.index, dtype="object")
    return raw[column].astype("string").str.strip().str.lower()


def _map_category(raw, column, mapping):
    values = _category(raw, column).map(mapping)
    return pd.to_numeric(values, errors="coerce").astype("float64")


def _nanmean_frame(series_list, index):
    frame = pd.concat(series_list, axis=1)
    return frame.mean(axis=1, skipna=True).reindex(index).astype("float64")


def add_campus_lifestyle_archetype_margins(raw, deps, aux):
    index = raw.index

    sleep_duration = _numeric(raw, "sleep_duration")
    sleep_duration_score = pd.Series(np.nan, index=index, dtype="float64")
    sleep_duration_score = sleep_duration_score.mask((sleep_duration >= 7.0) & (sleep_duration < 9.0), 1.0)
    sleep_duration_score = sleep_duration_score.mask(
        ((sleep_duration >= 6.0) & (sleep_duration < 7.0))
        | ((sleep_duration >= 9.0) & (sleep_duration < 10.0)),
        0.4,
    )
    sleep_duration_score = sleep_duration_score.mask((sleep_duration >= 5.0) & (sleep_duration < 6.0), 0.0)
    sleep_duration_score = sleep_duration_score.mask((sleep_duration < 5.0) | (sleep_duration >= 10.0), -1.0)

    sleep_quality_score = _map_category(raw, "sleep_quality", {"poor": -1.0, "average": 0.0, "good": 1.0})
    rest_score = _nanmean_frame([sleep_duration_score, sleep_quality_score], index)

    step_count = _numeric(raw, "step_count")
    step_score = pd.Series(np.nan, index=index, dtype="float64")
    step_score = step_score.mask(step_count < 5000.0, -1.0)
    step_score = step_score.mask((step_count >= 5000.0) & (step_count < 10000.0), 0.0)
    step_score = step_score.mask(step_count >= 10000.0, 1.0)

    exercise_duration = _numeric(raw, "exercise_duration")
    exercise_score = pd.Series(np.nan, index=index, dtype="float64")
    exercise_score = exercise_score.mask(exercise_duration < 20.0, -1.0)
    exercise_score = exercise_score.mask((exercise_duration >= 20.0) & (exercise_duration < 60.0), 0.0)
    exercise_score = exercise_score.mask(exercise_duration >= 60.0, 1.0)

    calorie_expenditure = _numeric(raw, "calorie_expenditure")
    calorie_score = pd.Series(np.nan, index=index, dtype="float64")
    calorie_score = calorie_score.mask(calorie_expenditure < 1800.0, -1.0)
    calorie_score = calorie_score.mask((calorie_expenditure >= 1800.0) & (calorie_expenditure < 2600.0), 0.0)
    calorie_score = calorie_score.mask(calorie_expenditure >= 2600.0, 1.0)

    reported_activity_score = _map_category(
        raw,
        "physical_activity_level",
        {"sedentary": -1.0, "moderate": 0.0, "active": 1.0},
    )
    activity_score = _nanmean_frame([step_score, exercise_score, calorie_score, reported_activity_score], index)

    pressure_score = _map_category(raw, "stress_level", {"high": -1.0, "medium": 0.0, "low": 1.0})
    substance_score = _map_category(raw, "smoking_alcohol", {"yes": -1.0, "occasional": 0.0, "no": 1.0})

    diet_score = _map_category(raw, "diet_type", {"non-veg": 0.0, "veg": 0.5, "balanced": 1.0})
    water_intake = _numeric(raw, "water_intake")
    water_score = pd.Series(np.nan, index=index, dtype="float64")
    water_score = water_score.mask(water_intake < 1.5, -1.0)
    water_score = water_score.mask((water_intake >= 1.5) & (water_intake < 2.0), 0.0)
    water_score = water_score.mask((water_intake >= 2.0) & (water_intake <= 4.0), 1.0)
    water_score = water_score.mask(water_intake > 4.0, 0.0)
    nourishment_score = _nanmean_frame([diet_score, water_score], index)

    bmi = _numeric(raw, "bmi")
    bmi_score = pd.Series(np.nan, index=index, dtype="float64")
    bmi_score = bmi_score.mask((bmi >= 18.5) & (bmi < 25.0), 1.0)
    bmi_score = bmi_score.mask(((bmi >= 16.0) & (bmi < 18.5)) | ((bmi >= 25.0) & (bmi < 30.0)), 0.0)
    bmi_score = bmi_score.mask(bmi >= 30.0, -1.0)

    heart_rate = _numeric(raw, "heart_rate")
    heart_rate_score = pd.Series(np.nan, index=index, dtype="float64")
    heart_rate_score = heart_rate_score.mask(heart_rate <= 70.0, 1.0)
    heart_rate_score = heart_rate_score.mask((heart_rate > 70.0) & (heart_rate <= 85.0), 0.0)
    heart_rate_score = heart_rate_score.mask(heart_rate > 85.0, -1.0)
    physiology_score = _nanmean_frame([bmi_score, heart_rate_score], index)

    domain_frame = pd.DataFrame(
        {
            "rest_score": rest_score,
            "activity_score": activity_score,
            "pressure_score": pressure_score,
            "substance_score": substance_score,
            "nourishment_score": nourishment_score,
            "physiology_score": physiology_score,
        },
        index=index,
    )

    domain_values = domain_frame.to_numpy(dtype="float64")
    observed = np.isfinite(domain_values)
    observed_domain_count = observed.sum(axis=1)

    affinity_columns = {}
    affinity_arrays = []
    for archetype_name, prototype_values in zip(_ARCHETYPE_NAMES, _ARCHETYPE_PROTOTYPES):
        prototype = np.asarray(prototype_values, dtype="float64")
        component_affinity = np.clip(1.0 - np.abs(domain_values - prototype) / 2.0, 0.0, 1.0)
        component_affinity = np.where(observed, component_affinity, np.nan)
        affinity = np.nanmean(component_affinity, axis=1)
        affinity = np.where(observed_domain_count == 0, 0.5, affinity)
        affinity_columns["affinity_" + archetype_name] = affinity.astype("float64")
        affinity_arrays.append(affinity.astype("float64"))

    affinity_matrix = np.column_stack(affinity_arrays)
    best_position = np.argmax(affinity_matrix, axis=1)
    best_affinity = affinity_matrix[np.arange(len(index)), best_position]
    sorted_affinity = np.sort(affinity_matrix, axis=1)
    best_minus_second = sorted_affinity[:, -1] - sorted_affinity[:, -2]
    best_archetype = np.asarray(_ARCHETYPE_NAMES, dtype=object)[best_position]

    features = domain_frame.copy()
    features["observed_domain_count"] = observed_domain_count.astype("int16")
    for column_name, values in affinity_columns.items():
        features[column_name] = values
    features["best_archetype"] = pd.Series(best_archetype, index=index, dtype="object")
    features["best_affinity"] = best_affinity.astype("float64")
    features["best_minus_second_affinity_margin"] = best_minus_second.astype("float64")
    features["balanced_minus_multirisk_affinity_margin"] = (
        features["affinity_balanced_fit"] - features["affinity_depleted_multirisk"]
    ).astype("float64")
    features["active_overstrained_minus_balanced_affinity_margin"] = (
        features["affinity_active_overstrained"] - features["affinity_balanced_fit"]
    ).astype("float64")

    return features


FEATURE_GROUPS = [
    {
        "name": "campus_lifestyle_archetype_margins",
        "fn": add_campus_lifestyle_archetype_margins,
        "depends_on": [],
        "description": "Deterministic affinities and margins to interpretable campus lifestyle archetypes built from raw health behavior domains.",
    }
]