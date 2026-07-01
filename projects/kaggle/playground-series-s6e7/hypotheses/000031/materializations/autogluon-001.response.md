import numpy as np
import pandas as pd


GROUP_NAME = "gender_lifestyle_category_typicality"
GENDER_COL = "gender"
ID_COL = "id"
TRAIN_ID_UPPER_EXCLUSIVE = 690088
MISSING_TOKEN = "__missing__"
UNKNOWN_TOKEN = "__unknown__"
ALPHA = 1.0
LOW_PROB_THRESHOLD = 0.20

LIFESTYLE_COLS = (
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
)


def _as_token_series(df, col):
    if col not in df.columns:
        return pd.Series(MISSING_TOKEN, index=df.index, dtype="object")
    values = df[col].astype("object").where(df[col].notna(), MISSING_TOKEN)
    return values.astype(str)


def _fit_mask(raw):
    if ID_COL not in raw.columns:
        return pd.Series(True, index=raw.index)

    ids = pd.to_numeric(raw[ID_COL], errors="coerce")
    mask = ids.lt(TRAIN_ID_UPPER_EXCLUSIVE)
    if bool(mask.any()):
        return mask.fillna(False)
    return pd.Series(True, index=raw.index)


def add_gender_lifestyle_category_typicality(raw, deps, aux):
    fit_mask = _fit_mask(raw)

    gender_all = _as_token_series(raw, GENDER_COL)
    gender_fit = gender_all.loc[fit_mask]

    out = pd.DataFrame(index=raw.index)

    cond_surprises = []
    overall_surprises = []
    log_lifts = []
    cond_log_probs = []
    low_prob_flags = []

    n_fit = int(fit_mask.sum())

    for col in LIFESTYLE_COLS:
        value_all = _as_token_series(raw, col)
        value_fit = value_all.loc[fit_mask]

        levels = pd.Index(value_fit.unique()).union(pd.Index([MISSING_TOKEN]))
        genders = pd.Index(gender_fit.unique()).union(pd.Index([MISSING_TOKEN]))

        n_levels = int(len(levels))
        n_genders = int(len(genders))

        overall_counts = value_fit.value_counts(dropna=False)
        overall_denom = float(n_fit + ALPHA * n_levels)
        overall_probs = (overall_counts.reindex(levels, fill_value=0).astype(float) + ALPHA) / overall_denom
        default_overall_prob = float(ALPHA / overall_denom)

        pair_counts = pd.crosstab(gender_fit, value_fit, dropna=False)
        pair_counts = pair_counts.reindex(index=genders, columns=levels, fill_value=0).astype(float)
        gender_counts = gender_fit.value_counts(dropna=False).reindex(genders, fill_value=0).astype(float)
        cond_probs = pair_counts.add(ALPHA).div(gender_counts.add(ALPHA * n_levels), axis=0)

        row_index = pd.MultiIndex.from_arrays([gender_all, value_all])
        cond_lookup = cond_probs.stack(dropna=False)
        cond_prob = pd.Series(row_index.map(cond_lookup), index=raw.index, dtype="float64")

        overall_prob = value_all.map(overall_probs).astype("float64").fillna(default_overall_prob)
        cond_prob = cond_prob.fillna(overall_prob)

        cond_prob = cond_prob.clip(lower=np.finfo(float).tiny)
        overall_prob = overall_prob.clip(lower=np.finfo(float).tiny)

        cond_surprise = -np.log(cond_prob)
        overall_surprise = -np.log(overall_prob)
        log_lift = np.log(cond_prob) - np.log(overall_prob)

        cond_surprises.append(cond_surprise)
        overall_surprises.append(overall_surprise)
        log_lifts.append(log_lift)
        cond_log_probs.append(np.log(cond_prob))
        low_prob_flags.append(cond_prob.lt(LOW_PROB_THRESHOLD).astype("int8"))

    cond_surprise_df = pd.concat(cond_surprises, axis=1)
    overall_surprise_df = pd.concat(overall_surprises, axis=1)
    log_lift_df = pd.concat(log_lifts, axis=1)
    cond_log_prob_df = pd.concat(cond_log_probs, axis=1)
    low_prob_df = pd.concat(low_prob_flags, axis=1)

    out["mean_conditional_surprise"] = cond_surprise_df.mean(axis=1)
    out["max_conditional_surprise"] = cond_surprise_df.max(axis=1)
    out["sum_conditional_surprise"] = cond_surprise_df.sum(axis=1)
    out["sum_overall_surprise"] = overall_surprise_df.sum(axis=1)
    out["low_gender_probability_count"] = low_prob_df.sum(axis=1).astype("int8")
    out["mean_gender_log_lift"] = log_lift_df.mean(axis=1)
    out["sum_abs_gender_log_lift"] = log_lift_df.abs().sum(axis=1)
    out["summed_conditional_log_probability"] = cond_log_prob_df.sum(axis=1)

    return out


FEATURE_GROUPS = [
    {
        "name": "gender_lifestyle_category_typicality",
        "fn": add_gender_lifestyle_category_typicality,
        "depends_on": [],
        "description": "Smoothed same-gender typicality aggregates for lifestyle category values.",
    }
]