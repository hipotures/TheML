import numpy as np
import pandas as pd


TRAIN_ID_MAX = 690087

SLEEP_QUALITY_MAP = {
    "poor": 0,
    "average": 1,
    "good": 2,
}

PHYSICAL_ACTIVITY_MAP = {
    "sedentary": 0,
    "moderate": 1,
    "active": 2,
}

STRESS_LEVEL_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def _training_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids.le(TRAIN_ID_MAX)
    if bool(mask.any()):
        return mask.fillna(False)
    return pd.Series(True, index=raw.index)


def _ordered_category(series, mapping):
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(mapping).astype("float64")


def _tertile_bins(values, fit_mask, higher_is_higher=True):
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    fit_values = numeric.loc[fit_mask & numeric.notna()]

    out = pd.Series(np.nan, index=values.index, dtype="float64")
    if fit_values.empty:
        return out

    q1 = float(fit_values.quantile(1.0 / 3.0))
    q2 = float(fit_values.quantile(2.0 / 3.0))

    if not np.isfinite(q1) or not np.isfinite(q2):
        return out

    if q2 < q1:
        q1, q2 = q2, q1

    ranked = pd.Series(1.0, index=values.index, dtype="float64")
    ranked.loc[numeric.le(q1)] = 0.0
    ranked.loc[numeric.gt(q2)] = 2.0
    ranked.loc[numeric.isna()] = np.nan

    if not higher_is_higher:
        ranked = 2.0 - ranked
        ranked.loc[numeric.isna()] = np.nan

    return ranked


def _percentile_rank(series, fit_mask):
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    fit_values = numeric.loc[fit_mask & numeric.notna()]

    out = pd.Series(np.nan, index=series.index, dtype="float64")
    if fit_values.empty:
        return out

    ranks = fit_values.rank(method="average", pct=True)
    reference = pd.DataFrame({"value": fit_values.to_numpy(), "rank": ranks.to_numpy()})
    reference = reference.groupby("value", sort=True, as_index=False)["rank"].mean()

    x = numeric.to_numpy(dtype="float64")
    xp = reference["value"].to_numpy(dtype="float64")
    fp = reference["rank"].to_numpy(dtype="float64")

    valid = np.isfinite(x)
    if len(xp) == 1:
        out.loc[valid] = fp[0]
    else:
        out.loc[valid] = np.interp(x[valid], xp, fp, left=fp[0], right=fp[-1])

    return out


def _domain_features(name, reported, observed):
    valid = reported.notna() & observed.notna()

    gap = pd.Series(-1.0, index=reported.index, dtype="float64")
    abs_gap = pd.Series(-1.0, index=reported.index, dtype="float64")
    pair_code = pd.Series(-1, index=reported.index, dtype="int16")

    signed = reported - observed
    gap.loc[valid] = signed.loc[valid]
    abs_gap.loc[valid] = signed.loc[valid].abs()
    pair_code.loc[valid] = (reported.loc[valid] * 3.0 + observed.loc[valid]).astype("int16")

    return {
        f"{name}_reported_ord": reported.fillna(-1).astype("int8"),
        f"{name}_objective_bin": observed.fillna(-1).astype("int8"),
        f"{name}_gap": gap,
        f"{name}_abs_gap": abs_gap,
        f"{name}_pair_code": pair_code,
    }


def add_perceived_observed_alignment(raw, deps, aux):
    fit_mask = _training_mask(raw)

    sleep_reported = _ordered_category(raw["sleep_quality"], SLEEP_QUALITY_MAP)
    activity_reported = _ordered_category(raw["physical_activity_level"], PHYSICAL_ACTIVITY_MAP)
    stress_reported = _ordered_category(raw["stress_level"], STRESS_LEVEL_MAP)

    sleep_observed = _tertile_bins(raw["sleep_duration"], fit_mask, higher_is_higher=True)
    stress_observed = _tertile_bins(raw["heart_rate"], fit_mask, higher_is_higher=True)

    step_pct = _percentile_rank(raw["step_count"], fit_mask)
    exercise_pct = _percentile_rank(raw["exercise_duration"], fit_mask)
    calorie_pct = _percentile_rank(raw["calorie_expenditure"], fit_mask)
    activity_score = pd.concat([step_pct, exercise_pct, calorie_pct], axis=1).mean(axis=1, skipna=True)
    activity_observed = _tertile_bins(activity_score, fit_mask, higher_is_higher=True)

    feature_parts = {}
    feature_parts.update(_domain_features("sleep", sleep_reported, sleep_observed))
    feature_parts.update(_domain_features("activity", activity_reported, activity_observed))
    feature_parts.update(_domain_features("stress", stress_reported, stress_observed))

    gaps = pd.DataFrame(
        {
            "sleep": sleep_reported - sleep_observed,
            "activity": activity_reported - activity_observed,
            "stress": stress_reported - stress_observed,
        },
        index=raw.index,
    )
    available = gaps.notna()
    available_count = available.sum(axis=1).astype("int8")

    above_count = gaps.gt(0).sum(axis=1).astype("int8")
    equal_count = gaps.eq(0).sum(axis=1).astype("int8")
    below_count = gaps.lt(0).sum(axis=1).astype("int8")

    mean_abs_gap = gaps.abs().mean(axis=1, skipna=True).fillna(-1.0)
    mean_signed_gap = gaps.mean(axis=1, skipna=True).fillna(0.0)

    feature_parts["alignment_available_domain_count"] = available_count
    feature_parts["alignment_report_above_count"] = above_count
    feature_parts["alignment_report_equal_count"] = equal_count
    feature_parts["alignment_report_below_count"] = below_count
    feature_parts["alignment_mean_abs_gap"] = mean_abs_gap
    feature_parts["alignment_mean_signed_gap"] = mean_signed_gap

    return pd.DataFrame(feature_parts, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "perceived_observed_alignment",
        "fn": add_perceived_observed_alignment,
        "depends_on": [],
        "description": "Measures agreement and disagreement between ordered self-reports and adjacent objective health signals.",
    }
]