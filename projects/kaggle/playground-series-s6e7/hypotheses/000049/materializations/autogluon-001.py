import numpy as np
import pandas as pd


TRAIN_ID_MAX = 690087
EPSILON = 1e-6

ITEM_DEFINITIONS = (
    ("adequate_sleep_window", "sleep_duration", "between", 7.0, 9.0),
    ("favorable_sleep_quality", "sleep_quality", "isin", ("good",), None),
    ("low_stress", "stress_level", "isin", ("low",), None),
    ("non_sedentary_activity_report", "physical_activity_level", "isin", ("moderate", "active"), None),
    ("active_activity_report", "physical_activity_level", "isin", ("active",), None),
    ("moderate_steps", "step_count", "ge", 7000.0, None),
    ("high_steps", "step_count", "ge", 10000.0, None),
    ("moderate_exercise", "exercise_duration", "ge", 30.0, None),
    ("high_exercise", "exercise_duration", "ge", 60.0, None),
    ("no_smoking_alcohol", "smoking_alcohol", "isin", ("no",), None),
    ("balanced_or_vegetarian_diet", "diet_type", "isin", ("balanced", "veg"), None),
    ("adequate_hydration", "water_intake", "ge", 2.0, None),
    ("high_hydration", "water_intake", "ge", 3.0, None),
    ("normal_bmi_range", "bmi", "between", 18.5, 24.9),
    ("non_elevated_resting_pulse", "heart_rate", "le", 90.0, None),
    ("low_resting_pulse", "heart_rate", "le", 75.0, None),
    ("non_low_energy_expenditure", "calorie_expenditure", "ge", 1800.0, None),
)


def _training_mask(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    mask = ids.le(TRAIN_ID_MAX)
    if bool(mask.any()):
        return mask.fillna(False)

    return pd.Series(True, index=raw.index)


def _evaluate_item(raw, column, op, left, right):
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")

    values = raw[column]

    if op == "between":
        numeric = pd.to_numeric(values, errors="coerce")
        passed = numeric.ge(left) & numeric.le(right)
        observed = numeric.notna()
    elif op == "ge":
        numeric = pd.to_numeric(values, errors="coerce")
        passed = numeric.ge(left)
        observed = numeric.notna()
    elif op == "le":
        numeric = pd.to_numeric(values, errors="coerce")
        passed = numeric.le(left)
        observed = numeric.notna()
    elif op == "isin":
        normalized = values.astype("string").str.strip().str.lower()
        allowed = set(left)
        passed = normalized.isin(allowed)
        observed = normalized.notna()
    else:
        return pd.Series(np.nan, index=raw.index, dtype="float64")

    item = pd.Series(np.nan, index=raw.index, dtype="float64")
    item.loc[observed] = passed.loc[observed].astype("float64")
    return item


def add_cumulative_habit_ladder_fit(raw, deps, aux):
    item_frame = pd.DataFrame(index=raw.index)

    for item_name, column, op, left, right in ITEM_DEFINITIONS:
        item_frame[item_name] = _evaluate_item(raw, column, op, left, right)

    train_mask = _training_mask(raw)
    train_items = item_frame.loc[train_mask]

    prevalence = train_items.mean(axis=0, skipna=True).fillna(item_frame.mean(axis=0, skipna=True))
    prevalence = prevalence.fillna(0.5).clip(EPSILON, 1.0 - EPSILON)

    ordered_items = list(prevalence.sort_values(ascending=False, kind="mergesort").index)
    ordered = item_frame[ordered_items]
    ordered_prev = prevalence.loc[ordered_items].to_numpy(dtype="float64")
    difficulty_weights = -np.log(ordered_prev + EPSILON)
    ranks = np.arange(len(ordered_items), dtype="float64")
    rank_percentiles = ranks / max(len(ordered_items) - 1, 1)

    values = ordered.to_numpy(dtype="float64")
    observed = ~np.isnan(values)
    passed = values == 1.0
    failed = values == 0.0

    observed_count = observed.sum(axis=1).astype("float64")
    pass_count = passed.sum(axis=1).astype("float64")
    fail_count = failed.sum(axis=1).astype("float64")

    weighted_pass_sum = np.where(passed, difficulty_weights.reshape(1, -1), 0.0).sum(axis=1)
    weighted_possible_sum = np.where(observed, difficulty_weights.reshape(1, -1), 0.0).sum(axis=1)

    pass_fraction = np.divide(
        pass_count,
        observed_count,
        out=np.zeros(len(raw), dtype="float64"),
        where=observed_count > 0,
    )
    weighted_pass_score = np.divide(
        weighted_pass_sum,
        weighted_possible_sum,
        out=np.zeros(len(raw), dtype="float64"),
        where=weighted_possible_sum > 0,
    )

    passed_rank_values = np.where(passed, rank_percentiles.reshape(1, -1), np.nan)
    failed_rank_values = np.where(failed, rank_percentiles.reshape(1, -1), np.nan)

    hardest_passed_rank = np.nanmax(passed_rank_values, axis=1)
    easiest_failed_rank = np.nanmin(failed_rank_values, axis=1)

    hardest_passed_rank = np.where(np.isnan(hardest_passed_rank), 0.0, hardest_passed_rank)
    easiest_failed_rank = np.where(np.isnan(easiest_failed_rank), 1.0, easiest_failed_rank)

    easy_fail_counts = np.zeros(len(raw), dtype="float64")
    easy_observed_counts = np.zeros(len(raw), dtype="float64")
    hard_pass_counts = np.zeros(len(raw), dtype="float64")
    hard_observed_counts = np.zeros(len(raw), dtype="float64")
    inversion_counts = np.zeros(len(raw), dtype="float64")
    inversion_possible = np.zeros(len(raw), dtype="float64")

    for row_idx in range(len(raw)):
        row_observed = observed[row_idx]
        row_passed = passed[row_idx]
        row_failed = failed[row_idx]

        passed_positions = np.flatnonzero(row_passed)
        failed_positions = np.flatnonzero(row_failed)

        if passed_positions.size:
            hard_pos = passed_positions.max()
            easy_region = row_observed.copy()
            easy_region[(hard_pos + 1):] = False
            easy_observed_counts[row_idx] = easy_region.sum()
            easy_fail_counts[row_idx] = (easy_region & row_failed).sum()

        if failed_positions.size:
            easy_fail_pos = failed_positions.min()
            hard_region = row_observed.copy()
            hard_region[:easy_fail_pos] = False
            hard_observed_counts[row_idx] = hard_region.sum()
            hard_pass_counts[row_idx] = (hard_region & row_passed).sum()

        if passed_positions.size and failed_positions.size:
            for fail_pos in failed_positions:
                inversion_counts[row_idx] += (passed_positions > fail_pos).sum()
            inversion_possible[row_idx] = float(failed_positions.size * passed_positions.size)

    easy_foundation_fail_rate = np.divide(
        easy_fail_counts,
        easy_observed_counts,
        out=np.zeros(len(raw), dtype="float64"),
        where=easy_observed_counts > 0,
    )
    hard_tail_pass_rate = np.divide(
        hard_pass_counts,
        hard_observed_counts,
        out=np.zeros(len(raw), dtype="float64"),
        where=hard_observed_counts > 0,
    )
    inversion_rate = np.divide(
        inversion_counts,
        inversion_possible,
        out=np.zeros(len(raw), dtype="float64"),
        where=inversion_possible > 0,
    )

    features = pd.DataFrame(index=raw.index)
    features["observed_item_count"] = observed_count
    features["raw_pass_fraction"] = pass_fraction
    features["prevalence_weighted_pass_score"] = weighted_pass_score
    features["hardest_passed_rank_pct"] = hardest_passed_rank
    features["easiest_failed_rank_pct"] = easiest_failed_rank
    features["easy_foundation_fail_rate"] = easy_foundation_fail_rate
    features["hard_tail_pass_rate"] = hard_tail_pass_rate
    features["inversion_count"] = inversion_counts
    features["inversion_rate"] = inversion_rate
    features["pass_count"] = pass_count
    features["fail_count"] = fail_count

    return features


FEATURE_GROUPS = [
    {
        "name": "cumulative_habit_ladder_fit",
        "fn": add_cumulative_habit_ladder_fit,
        "depends_on": [],
        "description": "Cumulative healthy-habit ladder features based on prevalence-ordered lifestyle and physiology endorsements.",
    }
]