import numpy as np
import pandas as pd


TRAIN_ID_MAX = 690087
MIN_REFERENCE_VALUES = 100

DIET_TYPE_PROTECTIVE = {
    "balanced": 1.0,
    "veg": 0.8,
    "non-veg": 0.5,
}

STRESS_LEVEL_PROTECTIVE = {
    "low": 1.0,
    "medium": 0.5,
    "high": 0.0,
}

SLEEP_QUALITY_PROTECTIVE = {
    "good": 1.0,
    "average": 0.5,
    "poor": 0.0,
}

PHYSICAL_ACTIVITY_PROTECTIVE = {
    "active": 1.0,
    "moderate": 0.6,
    "sedentary": 0.0,
}

SMOKING_ALCOHOL_PROTECTIVE = {
    "no": 1.0,
    "occasional": 0.5,
    "yes": 0.0,
}

KNOWN_GENDERS = ("female", "male", "other")


def _numeric_col(raw, name):
    if name in raw.columns:
        return pd.to_numeric(raw[name], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=raw.index, dtype="float64")


def _clean_category_col(raw, name):
    if name not in raw.columns:
        return pd.Series(pd.NA, index=raw.index, dtype="string")
    cleaned = raw[name].astype("string").str.strip().str.lower()
    return cleaned.mask(cleaned == "")


def _mapped_category_score(raw, name, mapping):
    mapped = _clean_category_col(raw, name).map(mapping)
    return pd.to_numeric(mapped, errors="coerce").astype("float64")


def _known_category_flag(raw, name, allowed_values):
    cleaned = _clean_category_col(raw, name)
    return cleaned.isin(allowed_values).astype("int8")


def _plateau_score(values, low_zero, low_one, high_one, high_zero):
    score = pd.Series(np.nan, index=values.index, dtype="float64")
    observed = values.notna()

    score.loc[observed & (values <= low_zero)] = 0.0
    score.loc[observed & (values >= high_zero)] = 0.0
    score.loc[observed & (values >= low_one) & (values <= high_one)] = 1.0

    rising = observed & (values > low_zero) & (values < low_one)
    falling = observed & (values > high_one) & (values < high_zero)

    score.loc[rising] = (values.loc[rising] - low_zero) / (low_one - low_zero)
    score.loc[falling] = (high_zero - values.loc[falling]) / (high_zero - high_one)

    return score.clip(lower=0.0, upper=1.0)


def _lower_is_better_score(values, one_at_or_below, zero_at_or_above):
    score = pd.Series(np.nan, index=values.index, dtype="float64")
    observed = values.notna()

    score.loc[observed & (values <= one_at_or_below)] = 1.0
    score.loc[observed & (values >= zero_at_or_above)] = 0.0

    middle = observed & (values > one_at_or_below) & (values < zero_at_or_above)
    score.loc[middle] = (
        zero_at_or_above - values.loc[middle]
    ) / (zero_at_or_above - one_at_or_below)

    return score.clip(lower=0.0, upper=1.0)


def _higher_is_better_score(values, zero_at_or_below, one_at_or_above):
    score = pd.Series(np.nan, index=values.index, dtype="float64")
    observed = values.notna()

    score.loc[observed & (values <= zero_at_or_below)] = 0.0
    score.loc[observed & (values >= one_at_or_above)] = 1.0

    middle = observed & (values > zero_at_or_below) & (values < one_at_or_above)
    score.loc[middle] = (
        values.loc[middle] - zero_at_or_below
    ) / (one_at_or_above - zero_at_or_below)

    return score.clip(lower=0.0, upper=1.0)


def _calorie_reference_values(raw, calories):
    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        train_mask = ids.notna() & (ids <= TRAIN_ID_MAX)
        train_values = calories.loc[train_mask & calories.notna()]
        if train_values.shape[0] >= MIN_REFERENCE_VALUES:
            return train_values
    return calories.loc[calories.notna()]


def _calorie_score(raw, calories):
    reference = _calorie_reference_values(raw, calories)

    if reference.shape[0] < MIN_REFERENCE_VALUES:
        return pd.Series(np.nan, index=raw.index, dtype="float64")

    q01 = float(reference.quantile(0.01))
    q25 = float(reference.quantile(0.25))
    q75 = float(reference.quantile(0.75))
    q99 = float(reference.quantile(0.99))

    valid_bounds = (
        np.isfinite(q01)
        and np.isfinite(q25)
        and np.isfinite(q75)
        and np.isfinite(q99)
        and q01 < q25
        and q25 <= q75
        and q75 < q99
    )

    if not valid_bounds:
        return pd.Series(np.nan, index=raw.index, dtype="float64")

    return _plateau_score(calories, q01, q25, q75, q99)


def add_health_evidence_uncertainty_envelope(raw, deps, aux):
    sleep_duration = _numeric_col(raw, "sleep_duration")
    heart_rate = _numeric_col(raw, "heart_rate")
    bmi = _numeric_col(raw, "bmi")
    calorie_expenditure = _numeric_col(raw, "calorie_expenditure")
    step_count = _numeric_col(raw, "step_count")
    exercise_duration = _numeric_col(raw, "exercise_duration")
    water_intake = _numeric_col(raw, "water_intake")

    scores = pd.DataFrame(index=raw.index)

    scores["sleep_duration_protective_score"] = _plateau_score(
        sleep_duration, 5.0, 7.0, 9.0, 10.0
    )
    scores["heart_rate_protective_score"] = _lower_is_better_score(
        heart_rate, 70.0, 100.0
    )
    scores["bmi_protective_score"] = _plateau_score(
        bmi, 16.0, 18.5, 24.9, 34.82
    )
    scores["calorie_expenditure_protective_score"] = _calorie_score(
        raw, calorie_expenditure
    )
    scores["step_count_protective_score"] = _higher_is_better_score(
        step_count, 2000.0, 10000.0
    )
    scores["exercise_duration_protective_score"] = _higher_is_better_score(
        exercise_duration, 0.0, 45.0
    )
    scores["water_intake_protective_score"] = _plateau_score(
        water_intake, 0.75, 2.0, 3.5, 4.72
    )
    scores["diet_type_protective_score"] = _mapped_category_score(
        raw, "diet_type", DIET_TYPE_PROTECTIVE
    )
    scores["stress_level_protective_score"] = _mapped_category_score(
        raw, "stress_level", STRESS_LEVEL_PROTECTIVE
    )
    scores["sleep_quality_protective_score"] = _mapped_category_score(
        raw, "sleep_quality", SLEEP_QUALITY_PROTECTIVE
    )
    scores["physical_activity_level_protective_score"] = _mapped_category_score(
        raw, "physical_activity_level", PHYSICAL_ACTIVITY_PROTECTIVE
    )
    scores["smoking_alcohol_protective_score"] = _mapped_category_score(
        raw, "smoking_alcohol", SMOKING_ALCOHOL_PROTECTIVE
    )

    numeric_score_cols = [
        "sleep_duration_protective_score",
        "heart_rate_protective_score",
        "bmi_protective_score",
        "calorie_expenditure_protective_score",
        "step_count_protective_score",
        "exercise_duration_protective_score",
        "water_intake_protective_score",
    ]
    categorical_score_cols = [
        "diet_type_protective_score",
        "stress_level_protective_score",
        "sleep_quality_protective_score",
        "physical_activity_level_protective_score",
        "smoking_alcohol_protective_score",
    ]
    all_score_cols = numeric_score_cols + categorical_score_cols

    observed_mask = scores[all_score_cols].notna()
    observed_count = observed_mask.sum(axis=1).astype("float64")
    total_count = float(len(all_score_cols))
    missing_count = total_count - observed_count

    numeric_observed_count = scores[numeric_score_cols].notna().sum(axis=1).astype("float64")
    categorical_observed_count = (
        scores[categorical_score_cols].notna().sum(axis=1).astype("float64")
    )
    numeric_missing_count = float(len(numeric_score_cols)) - numeric_observed_count
    categorical_missing_count = (
        float(len(categorical_score_cols)) - categorical_observed_count
    )

    observed_sum = scores[all_score_cols].sum(axis=1, skipna=True)
    numeric_observed_sum = scores[numeric_score_cols].sum(axis=1, skipna=True)
    categorical_observed_sum = scores[categorical_score_cols].sum(axis=1, skipna=True)

    observed_mean = observed_sum / observed_count.replace(0.0, np.nan)
    numeric_observed_mean = numeric_observed_sum / numeric_observed_count.replace(
        0.0, np.nan
    )
    categorical_observed_mean = (
        categorical_observed_sum / categorical_observed_count.replace(0.0, np.nan)
    )

    new_features = scores.copy()
    new_features["observed_mean_score"] = observed_mean.fillna(0.5)
    new_features["numeric_observed_mean_score"] = numeric_observed_mean.fillna(0.5)
    new_features["categorical_observed_mean_score"] = categorical_observed_mean.fillna(
        0.5
    )
    new_features["worst_case_score"] = observed_sum / total_count
    new_features["best_case_score"] = (observed_sum + missing_count) / total_count
    new_features["midpoint_score"] = (
        new_features["worst_case_score"] + new_features["best_case_score"]
    ) / 2.0
    new_features["interval_width"] = (
        new_features["best_case_score"] - new_features["worst_case_score"]
    )
    new_features["observed_evidence_fraction"] = observed_count / total_count
    new_features["numeric_observed_evidence_fraction"] = numeric_observed_count / float(
        len(numeric_score_cols)
    )
    new_features["categorical_observed_evidence_fraction"] = (
        categorical_observed_count / float(len(categorical_score_cols))
    )
    new_features["missing_field_count"] = missing_count
    new_features["numeric_missing_field_count"] = numeric_missing_count
    new_features["categorical_missing_field_count"] = categorical_missing_count
    new_features["numeric_missing_uncertainty_share"] = numeric_missing_count / total_count
    new_features["categorical_missing_uncertainty_share"] = (
        categorical_missing_count / total_count
    )
    new_features["all_health_evidence_missing"] = (observed_count == 0.0).astype("int8")
    new_features["complete_health_evidence"] = (missing_count == 0.0).astype("int8")
    new_features["gender_known"] = _known_category_flag(raw, "gender", KNOWN_GENDERS)

    return new_features


FEATURE_GROUPS = [
    {
        "name": "health_evidence_uncertainty_envelope",
        "fn": add_health_evidence_uncertainty_envelope,
        "depends_on": [],
        "description": "Builds bounded protective health scores and summarizes missing-field best/worst uncertainty.",
    }
]