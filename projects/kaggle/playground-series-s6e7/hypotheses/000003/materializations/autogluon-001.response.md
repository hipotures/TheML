import pandas as pd


NUMERIC_RESPONSE_FEATURES = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
)

CATEGORICAL_RESPONSE_FEATURES = (
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
)

ALL_RESPONSE_FEATURES = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
)

WELLNESS_REPORT_FEATURES = (
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "diet_type",
    "gender",
)


def _missing_mask(raw, column_name, check_blank):
    values = raw[column_name]
    mask = values.isna()
    if check_blank:
        blank_mask = values.astype("string").str.strip().eq("").fillna(False)
        mask = mask | blank_mask
    return mask


def _safe_ratio(counts, denominator):
    if denominator:
        return counts.astype("float32") / float(denominator)
    return pd.Series(0.0, index=counts.index, dtype="float32")


def add_response_completeness_signature(raw, deps, aux):
    features = pd.DataFrame(index=raw.index)

    all_available = []
    numeric_available = []
    categorical_available = []
    wellness_available = []

    for column_name in ALL_RESPONSE_FEATURES:
        if column_name not in raw.columns:
            continue

        check_blank = column_name in CATEGORICAL_RESPONSE_FEATURES
        missing = _missing_mask(raw, column_name, check_blank)

        features[column_name + "_is_missing"] = missing.astype("int8")
        all_available.append(column_name)

        if column_name in NUMERIC_RESPONSE_FEATURES:
            numeric_available.append(column_name)
        if column_name in CATEGORICAL_RESPONSE_FEATURES:
            categorical_available.append(column_name)
        if column_name in WELLNESS_REPORT_FEATURES:
            wellness_available.append(column_name)

    if all_available:
        all_indicator_columns = [column_name + "_is_missing" for column_name in all_available]
        missing_count = features[all_indicator_columns].sum(axis=1).astype("int16")
    else:
        missing_count = pd.Series(0, index=raw.index, dtype="int16")

    if numeric_available:
        numeric_indicator_columns = [column_name + "_is_missing" for column_name in numeric_available]
        numeric_missing_count = features[numeric_indicator_columns].sum(axis=1).astype("int16")
    else:
        numeric_missing_count = pd.Series(0, index=raw.index, dtype="int16")

    if categorical_available:
        categorical_indicator_columns = [column_name + "_is_missing" for column_name in categorical_available]
        categorical_missing_count = features[categorical_indicator_columns].sum(axis=1).astype("int16")
    else:
        categorical_missing_count = pd.Series(0, index=raw.index, dtype="int16")

    if wellness_available:
        wellness_indicator_columns = [column_name + "_is_missing" for column_name in wellness_available]
        wellness_missing_count = features[wellness_indicator_columns].sum(axis=1).astype("int16")
    else:
        wellness_missing_count = pd.Series(0, index=raw.index, dtype="int16")

    features["missing_count"] = missing_count
    features["missing_ratio"] = _safe_ratio(missing_count, len(all_available))

    features["numeric_missing_count"] = numeric_missing_count
    features["numeric_missing_ratio"] = _safe_ratio(numeric_missing_count, len(numeric_available))

    features["categorical_missing_count"] = categorical_missing_count
    features["categorical_missing_ratio"] = _safe_ratio(categorical_missing_count, len(categorical_available))

    features["wellness_report_missing_count"] = wellness_missing_count
    features["wellness_report_missing_ratio"] = _safe_ratio(wellness_missing_count, len(wellness_available))

    completeness_bin = pd.Series("complete", index=raw.index, dtype="object")
    completeness_bin = completeness_bin.mask((missing_count >= 1) & (missing_count <= 2), "low_missing")
    completeness_bin = completeness_bin.mask((missing_count >= 3) & (missing_count <= 4), "moderate_missing")
    completeness_bin = completeness_bin.mask(missing_count >= 5, "high_missing")
    features["completeness_bin"] = completeness_bin

    return features


FEATURE_GROUPS = [
    {
        "name": "response_completeness_signature",
        "fn": add_response_completeness_signature,
        "depends_on": [],
        "description": "Encodes per-field and row-level missingness patterns across health and lifestyle responses.",
    }
]