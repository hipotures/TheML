import numpy as np
import pandas as pd


_NUMERIC_CONTEXT_COLUMNS = [
    "sleep_duration",
    "bmi",
    "step_count",
    "exercise_duration",
    "calorie_expenditure",
]

_CATEGORICAL_CONTEXT_COLUMNS = [
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "diet_type",
    "smoking_alcohol",
    "gender",
]

_QUANTILE_PROBS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_MIN_CONTEXT_COUNT = 200
_TRAIN_ID_MAX = 690087


def _training_mask(raw):
    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        mask = ids.le(_TRAIN_ID_MAX)
        if mask.any():
            return mask.fillna(False)
    return pd.Series(True, index=raw.index)


def _context_frame(raw, fit_mask):
    ctx = pd.DataFrame(index=raw.index)

    for col in _CATEGORICAL_CONTEXT_COLUMNS:
        if col in raw.columns:
            ctx[col + "_level"] = raw[col].astype("string").fillna("unknown").astype(str)
        else:
            ctx[col + "_level"] = "unknown"

    for col in _NUMERIC_CONTEXT_COLUMNS:
        if col not in raw.columns:
            ctx[col + "_band"] = "missing"
            continue

        values = pd.to_numeric(raw[col], errors="coerce")
        fit_values = values.loc[fit_mask & values.notna()]
        if fit_values.nunique(dropna=True) < 2:
            ctx[col + "_band"] = np.where(values.notna(), "known", "missing")
            continue

        quantiles = fit_values.quantile(_QUANTILE_PROBS).to_numpy()
        edges = np.unique(quantiles)
        if len(edges) < 3:
            ctx[col + "_band"] = np.where(values.notna(), "known", "missing")
            continue

        binned = pd.cut(
            values,
            bins=edges,
            include_lowest=True,
            duplicates="drop",
            labels=False,
        )
        ctx[col + "_band"] = binned.astype("Int64").astype("string").fillna("missing")
        ctx[col + "_band"] = np.where(
            values.notna(),
            col + "_q" + ctx[col + "_band"].astype(str),
            "missing",
        )

    return ctx


def _reference_table(frame, fit_mask, keys):
    fit = frame.loc[fit_mask & frame["heart_rate"].notna(), list(keys) + ["heart_rate"]]
    if fit.empty:
        return pd.DataFrame(columns=list(keys) + ["expected_hr", "hr_iqr", "context_count"])

    grouped = fit.groupby(list(keys), dropna=False)["heart_rate"]
    table = grouped.agg(
        expected_hr="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        context_count="size",
    ).reset_index()
    table["hr_iqr"] = table["q75"] - table["q25"]
    return table.drop(columns=["q25", "q75"])


def _apply_reference(features, table, keys, unresolved):
    if table.empty or not unresolved.any():
        return unresolved

    eligible = table.loc[table["context_count"].ge(_MIN_CONTEXT_COUNT)]
    if eligible.empty:
        return unresolved

    matched = features.loc[unresolved, list(keys)].merge(
        eligible,
        how="left",
        on=list(keys),
        sort=False,
    )
    has_match = matched["expected_hr"].notna().to_numpy()
    target_index = features.index[unresolved][has_match]

    if len(target_index) > 0:
        features.loc[target_index, "expected_hr"] = matched.loc[has_match, "expected_hr"].to_numpy()
        features.loc[target_index, "hr_context_iqr"] = matched.loc[has_match, "hr_iqr"].to_numpy()
        features.loc[target_index, "context_count"] = matched.loc[has_match, "context_count"].to_numpy()
        unresolved.loc[target_index] = False

    return unresolved


def add_contextual_heart_rate_strain(raw, deps, aux):
    fit_mask = _training_mask(raw)
    ctx = _context_frame(raw, fit_mask)

    heart_rate = pd.to_numeric(raw["heart_rate"], errors="coerce") if "heart_rate" in raw.columns else pd.Series(np.nan, index=raw.index)
    work = ctx.copy()
    work["heart_rate"] = heart_rate

    features = pd.DataFrame(index=raw.index)
    features["expected_hr"] = np.nan
    features["hr_context_iqr"] = np.nan
    features["context_count"] = np.nan

    hierarchy = [
        [
            "sleep_quality_level",
            "stress_level_level",
            "sleep_duration_band",
            "physical_activity_level_level",
            "exercise_duration_band",
            "step_count_band",
            "bmi_band",
        ],
        [
            "sleep_quality_level",
            "stress_level_level",
            "sleep_duration_band",
            "physical_activity_level_level",
            "exercise_duration_band",
        ],
        [
            "physical_activity_level_level",
            "exercise_duration_band",
            "step_count_band",
            "calorie_expenditure_band",
        ],
        [
            "bmi_band",
            "gender_level",
            "diet_type_level",
            "smoking_alcohol_level",
        ],
        [
            "sleep_quality_level",
            "stress_level_level",
            "physical_activity_level_level",
        ],
    ]

    unresolved = pd.Series(True, index=raw.index)
    for keys in hierarchy:
        table = _reference_table(work, fit_mask, keys)
        unresolved = _apply_reference(pd.concat([ctx, features], axis=1), table, keys, unresolved)

    train_hr = heart_rate.loc[fit_mask & heart_rate.notna()]
    if train_hr.empty:
        global_median = np.nan
        global_iqr = np.nan
        global_count = 0
    else:
        global_median = train_hr.median()
        global_iqr = train_hr.quantile(0.75) - train_hr.quantile(0.25)
        global_count = int(train_hr.shape[0])

    features.loc[unresolved, "expected_hr"] = global_median
    features.loc[unresolved, "hr_context_iqr"] = global_iqr
    features.loc[unresolved, "context_count"] = global_count

    scale = (features["hr_context_iqr"] / 1.349).clip(lower=1.0)
    residual = heart_rate - features["expected_hr"]
    robust_z = residual / scale

    out = pd.DataFrame(index=raw.index)
    out["expected_hr"] = features["expected_hr"]
    out["hr_context_residual"] = residual
    out["hr_context_robust_z"] = robust_z
    out["abs_hr_context_robust_z"] = robust_z.abs()
    out["elevated_hr_context_flag"] = robust_z.ge(1.5).fillna(False)
    out["suppressed_hr_context_flag"] = robust_z.le(-1.5).fillna(False)
    out["log_context_count"] = np.log1p(features["context_count"].fillna(0.0))

    return out


FEATURE_GROUPS = [
    {
        "name": "contextual_heart_rate_strain",
        "fn": add_contextual_heart_rate_strain,
        "depends_on": [],
        "description": "Peer-context heart-rate residuals and robust strain flags using non-target recovery, activity, and body-context reference groups.",
    }
]