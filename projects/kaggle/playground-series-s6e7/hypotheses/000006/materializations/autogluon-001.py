import numpy as np
import pandas as pd


NUMERIC_FEATURES = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
)

CATEGORICAL_FEATURES = (
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
)

CLASS_ORDER = ("unhealthy", "at-risk", "fit")

CLASS_SUFFIX = {
    "unhealthy": "unhealthy",
    "at-risk": "at_risk",
    "fit": "fit",
}


def add_external_health_archetype_affinity(raw, deps, aux):
    row_count = len(raw)
    out = pd.DataFrame(index=raw.index)

    for cls in CLASS_ORDER:
        out["affinity_" + CLASS_SUFFIX[cls]] = 0.0
    out["affinity_best_margin"] = 0.0
    out["affinity_fit_minus_unhealthy"] = 0.0

    if (
        not isinstance(aux, pd.DataFrame)
        or aux.empty
        or "health_condition" not in aux.columns
    ):
        return out

    target = aux["health_condition"].astype("string").str.strip().str.lower()
    valid_target = target.isin(CLASS_ORDER).fillna(False)
    if not bool(valid_target.any()):
        return out

    aux_ref = aux.loc[valid_target]
    target_ref = target.loc[valid_target]
    class_masks = {
        cls: (target_ref == cls).fillna(False).to_numpy(dtype=bool)
        for cls in CLASS_ORDER
    }

    numeric_sums = {
        cls: np.zeros(row_count, dtype=np.float64)
        for cls in CLASS_ORDER
    }
    categorical_sums = {
        cls: np.zeros(row_count, dtype=np.float64)
        for cls in CLASS_ORDER
    }
    numeric_count = np.zeros(row_count, dtype=np.float64)
    categorical_count = np.zeros(row_count, dtype=np.float64)

    for col in NUMERIC_FEATURES:
        if col not in raw.columns or col not in aux_ref.columns:
            continue

        aux_num = pd.to_numeric(aux_ref[col], errors="coerce")
        observed_aux = aux_num.notna()
        if not bool(observed_aux.any()):
            continue

        q01 = aux_num.quantile(0.01)
        q25 = aux_num.quantile(0.25)
        q75 = aux_num.quantile(0.75)
        q99 = aux_num.quantile(0.99)
        overall_median = aux_num.median()

        if pd.isna(q01) or pd.isna(q99) or pd.isna(overall_median):
            continue

        scale = q75 - q25
        if pd.isna(scale) or scale < 1e-6:
            scale = 1e-6

        raw_num = pd.to_numeric(raw[col], errors="coerce")
        raw_observed = raw_num.notna().to_numpy(dtype=bool)
        numeric_count += raw_observed.astype(np.float64)

        clipped = raw_num.clip(lower=float(q01), upper=float(q99))

        for cls in CLASS_ORDER:
            class_values = aux_num.loc[class_masks[cls]]
            class_median = class_values.median()
            if pd.isna(class_median):
                class_median = overall_median

            distance = ((clipped - float(class_median)).abs() / float(scale)).clip(
                upper=6.0
            )
            numeric_sums[cls] += distance.fillna(0.0).to_numpy(dtype=np.float64)

    for col in CATEGORICAL_FEATURES:
        if col not in raw.columns or col not in aux_ref.columns:
            continue

        aux_cat = aux_ref[col].astype("string").str.strip().str.lower()
        valid_aux_cat = aux_cat.notna()
        if not bool(valid_aux_cat.any()):
            continue

        levels = pd.Index(aux_cat.loc[valid_aux_cat].unique())
        level_count = len(levels)
        if level_count == 0:
            continue

        overall_total = int(valid_aux_cat.sum())
        overall_counts = aux_cat.loc[valid_aux_cat].value_counts(dropna=True)

        raw_cat = raw[col].astype("string").str.strip().str.lower()
        raw_observed = raw_cat.notna().to_numpy(dtype=bool)
        categorical_count += raw_observed.astype(np.float64)

        for cls in CLASS_ORDER:
            class_valid = pd.Series(class_masks[cls], index=aux_ref.index) & valid_aux_cat
            class_total = int(class_valid.sum())
            if class_total == 0:
                continue

            class_counts = aux_cat.loc[class_valid].value_counts(dropna=True)
            lift_map = {}

            for level in levels:
                class_rate = (class_counts.get(level, 0) + 1.0) / (
                    class_total + level_count
                )
                overall_rate = (overall_counts.get(level, 0) + 1.0) / (
                    overall_total + level_count
                )
                lift_map[level] = float(np.log(class_rate / overall_rate))

            mapped = raw_cat.map(lift_map).fillna(0.0)
            categorical_sums[cls] += mapped.to_numpy(dtype=np.float64)

    numeric_distances = {}
    categorical_lifts = {}
    has_numeric = numeric_count > 0.0
    has_categorical = categorical_count > 0.0

    for cls in CLASS_ORDER:
        numeric_component = np.zeros(row_count, dtype=np.float64)
        categorical_component = np.zeros(row_count, dtype=np.float64)

        numeric_component[has_numeric] = (
            numeric_sums[cls][has_numeric] / numeric_count[has_numeric]
        )
        categorical_component[has_categorical] = (
            categorical_sums[cls][has_categorical] / categorical_count[has_categorical]
        )

        numeric_distances[cls] = numeric_component
        categorical_lifts[cls] = categorical_component

    combined = {}
    for cls in CLASS_ORDER:
        combined[cls] = categorical_lifts[cls] - numeric_distances[cls]
        out["affinity_" + CLASS_SUFFIX[cls]] = combined[cls]

    score_matrix = np.column_stack([combined[cls] for cls in CLASS_ORDER])
    if row_count:
        sorted_scores = np.sort(score_matrix, axis=1)
        out["affinity_best_margin"] = sorted_scores[:, -1] - sorted_scores[:, -2]

    out["affinity_fit_minus_unhealthy"] = combined["fit"] - combined["unhealthy"]

    return out


FEATURE_GROUPS = [
    {
        "name": "external_health_archetype_affinity",
        "fn": add_external_health_archetype_affinity,
        "depends_on": [],
        "description": "Classwise external student-health archetype affinity scores from auxiliary labeled health-condition prototypes.",
    }
]