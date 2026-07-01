import numpy as np
import pandas as pd


TRAIN_ID_MAX = 690087
DIET_CONTEXT_COL = "diet_type"
DIET_CONTEXTS = ("balanced", "veg", "non-veg", "__missing__")
NUMERIC_FEATURES = (
    "bmi",
    "heart_rate",
    "calorie_expenditure",
    "water_intake",
    "step_count",
    "exercise_duration",
    "sleep_duration",
)
MIN_CONTEXT_OBS = 100
MIN_IQR = 1.0e-6
RESIDUAL_CLIP = 4.0
MISMATCH_THRESHOLDS = (0.5, 1.0, 2.0)


def _training_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids.le(TRAIN_ID_MAX)
    if not bool(mask.any()):
        return pd.Series(True, index=raw.index)
    return mask.fillna(False)


def _diet_context(raw):
    if DIET_CONTEXT_COL not in raw.columns:
        return pd.Series("__missing__", index=raw.index, dtype="object")

    context = raw[DIET_CONTEXT_COL].astype("object").where(raw[DIET_CONTEXT_COL].notna(), "__missing__")
    context = context.where(context.isin(DIET_CONTEXTS), "__missing__")
    return context


def _context_stats(values, context, train_mask):
    train_values = values.loc[train_mask]
    train_context = context.loc[train_mask]

    global_observed = train_values.dropna()
    if global_observed.empty:
        global_median = 0.0
        global_iqr = 1.0
    else:
        global_median = float(global_observed.median())
        global_q75 = float(global_observed.quantile(0.75))
        global_q25 = float(global_observed.quantile(0.25))
        global_iqr = max(global_q75 - global_q25, MIN_IQR)

    medians = {}
    iqrs = {}
    for diet_context in DIET_CONTEXTS:
        observed = train_values.loc[train_context.eq(diet_context)].dropna()
        if len(observed) < MIN_CONTEXT_OBS:
            medians[diet_context] = global_median
            iqrs[diet_context] = global_iqr
        else:
            median = float(observed.median())
            q75 = float(observed.quantile(0.75))
            q25 = float(observed.quantile(0.25))
            iqr = q75 - q25
            if iqr <= MIN_IQR:
                medians[diet_context] = global_median
                iqrs[diet_context] = global_iqr
            else:
                medians[diet_context] = median
                iqrs[diet_context] = iqr

    return medians, iqrs


def add_diet_metabolic_alignment(raw, deps, aux):
    index = raw.index
    train_mask = _training_mask(raw)
    diet_context = _diet_context(raw)

    features = pd.DataFrame(index=index)
    residuals = {}
    observed_masks = {}

    for column in NUMERIC_FEATURES:
        if column in raw.columns:
            values = pd.to_numeric(raw[column], errors="coerce")
        else:
            values = pd.Series(np.nan, index=index, dtype="float64")

        medians, iqrs = _context_stats(values, diet_context, train_mask)
        context_median = diet_context.map(medians).astype("float64")
        context_iqr = diet_context.map(iqrs).astype("float64").clip(lower=MIN_IQR)

        observed = values.notna()
        z = ((values - context_median) / context_iqr).clip(-RESIDUAL_CLIP, RESIDUAL_CLIP)
        z = z.where(observed, 0.0).astype("float32")

        feature_name = "diet_z_" + column
        features[feature_name] = z
        residuals[column] = z
        observed_masks[column] = observed

    residual_matrix = pd.DataFrame(residuals, index=index)
    observed_matrix = pd.DataFrame(observed_masks, index=index)
    observed_count = observed_matrix.sum(axis=1).replace(0, np.nan)

    squared_sum = residual_matrix.pow(2).where(observed_matrix, 0.0).sum(axis=1)
    features["diet_context_rms"] = np.sqrt(squared_sum / observed_count).fillna(0.0).astype("float32")

    activity_cols = ("step_count", "exercise_duration")
    activity_sum = residual_matrix.loc[:, activity_cols].where(observed_matrix.loc[:, activity_cols], 0.0).sum(axis=1)
    activity_count = observed_matrix.loc[:, activity_cols].sum(axis=1).replace(0, np.nan)
    activity_mean = (activity_sum / activity_count).fillna(0.0)

    calorie_z = residual_matrix["calorie_expenditure"].where(observed_matrix["calorie_expenditure"], 0.0)
    features["diet_energy_activity_gap"] = (calorie_z - activity_mean).astype("float32")

    tail_components = pd.DataFrame(index=index)
    tail_observed = pd.DataFrame(index=index)
    tail_components["high_bmi"] = np.maximum(0.0, residual_matrix["bmi"] - 0.5)
    tail_observed["high_bmi"] = observed_matrix["bmi"]
    tail_components["low_bmi"] = np.maximum(0.0, -residual_matrix["bmi"] - 1.0)
    tail_observed["low_bmi"] = observed_matrix["bmi"]
    tail_components["high_heart_rate"] = np.maximum(0.0, residual_matrix["heart_rate"] - 0.5)
    tail_observed["high_heart_rate"] = observed_matrix["heart_rate"]
    tail_components["low_water_intake"] = np.maximum(0.0, -residual_matrix["water_intake"] - 0.5)
    tail_observed["low_water_intake"] = observed_matrix["water_intake"]
    tail_components["low_step_count"] = np.maximum(0.0, -residual_matrix["step_count"] - 0.5)
    tail_observed["low_step_count"] = observed_matrix["step_count"]
    tail_components["low_exercise_duration"] = np.maximum(0.0, -residual_matrix["exercise_duration"] - 0.5)
    tail_observed["low_exercise_duration"] = observed_matrix["exercise_duration"]
    tail_components["low_sleep_duration"] = np.maximum(0.0, -residual_matrix["sleep_duration"] - 0.5)
    tail_observed["low_sleep_duration"] = observed_matrix["sleep_duration"]

    tail_count = tail_observed.sum(axis=1).replace(0, np.nan)
    tail_score = tail_components.where(tail_observed, 0.0).sum(axis=1) / tail_count
    features["diet_adverse_tail"] = tail_score.fillna(0.0).astype("float32")

    thresholds = list(MISMATCH_THRESHOLDS)
    features["diet_mismatch_bin"] = np.searchsorted(
        thresholds,
        features["diet_context_rms"].to_numpy(dtype="float64"),
        side="right",
    ).astype("int8")

    return features


FEATURE_GROUPS = [
    {
        "name": "diet_metabolic_alignment",
        "fn": add_diet_metabolic_alignment,
        "depends_on": [],
        "description": "Diet-conditioned robust residual features measuring metabolic, hydration, activity, and sleep alignment.",
    }
]