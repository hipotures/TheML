import numpy as np
import pandas as pd


UNKNOWN_TOKEN = "__missing__"
MIN_CONTEXT_ROWS = 50
SLEEP_GAP_CLIP = 4.0
TYPICAL_SLEEP_GAP_HOURS = 1.0
CDC_ADULT_MIN_SLEEP_HOURS = 7.0
ACTIVITY_BIN_LABELS = ("low", "middle", "high")


def _existing_columns(frame, columns):
    return [col for col in columns if col in frame.columns]


def _series_or_nan(frame, column, index):
    if column in frame.columns:
        return frame[column]
    return pd.Series(np.nan, index=index)


def _safe_numeric(frame, column, index):
    return pd.to_numeric(_series_or_nan(frame, column, index), errors="coerce")


def _context_column(frame, column, index):
    if column in frame.columns:
        return frame[column].astype("object").where(frame[column].notna(), UNKNOWN_TOKEN).astype(str)
    return pd.Series(UNKNOWN_TOKEN, index=index, dtype="object")


def _train_mask_from_id(raw):
    if "id" not in raw.columns:
        return pd.Series(True, index=raw.index)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    valid = ids.notna()
    if valid.sum() == 0:
        return pd.Series(True, index=raw.index)

    max_id = ids[valid].max()
    known_train_end = 690087
    if max_id > known_train_end and (ids <= known_train_end).sum() >= MIN_CONTEXT_ROWS:
        return ids <= known_train_end

    return pd.Series(True, index=raw.index)


def _percentile_rank_by_train(values, train_values):
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    train_clean = train_values.dropna().sort_values()
    n_train = len(train_clean)
    if n_train == 0:
        return result

    sorted_values = train_clean.to_numpy()
    raw_values = values.to_numpy(dtype="float64")
    valid = ~np.isnan(raw_values)
    if valid.any():
        ranks = np.searchsorted(sorted_values, raw_values[valid], side="right") / float(n_train)
        result.iloc[np.flatnonzero(valid)] = np.clip(ranks, 0.0, 1.0)
    return result


def _activity_demand_bin(activity_demand, train_mask):
    result = pd.Series("unknown", index=activity_demand.index, dtype="object")
    train_values = activity_demand[train_mask & activity_demand.notna()]
    if len(train_values) < MIN_CONTEXT_ROWS:
        result.loc[activity_demand.notna()] = "middle"
        return result

    q1 = train_values.quantile(1.0 / 3.0)
    q2 = train_values.quantile(2.0 / 3.0)
    known = activity_demand.notna()

    if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
        result.loc[known] = "middle"
        return result

    result.loc[known & (activity_demand <= q1)] = ACTIVITY_BIN_LABELS[0]
    result.loc[known & (activity_demand > q1) & (activity_demand <= q2)] = ACTIVITY_BIN_LABELS[1]
    result.loc[known & (activity_demand > q2)] = ACTIVITY_BIN_LABELS[2]
    return result


def _make_context_table(base, keys, value_col):
    observed = base.dropna(subset=[value_col])
    if not keys:
        values = observed[value_col]
        if len(values) == 0:
            return None
        q75 = values.quantile(0.75)
        q25 = values.quantile(0.25)
        return {
            "median": values.median(),
            "iqr": q75 - q25,
            "count": len(values),
        }

    grouped = observed.groupby(keys, dropna=False)[value_col]
    return grouped.agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        count="size",
    ).reset_index()


def _lookup_context(rows, table, keys, value_name, min_count):
    result = pd.Series(np.nan, index=rows.index, dtype="float64")
    if table is None or len(table) == 0:
        return result

    eligible = table.loc[table["count"] >= min_count, list(keys) + [value_name]]
    if len(eligible) == 0:
        return result

    merged = rows[list(keys)].merge(eligible, on=list(keys), how="left", sort=False)
    result[:] = pd.to_numeric(merged[value_name], errors="coerce").to_numpy()
    return result


def add_contextual_sleep_need_gap(raw, deps, aux):
    index = raw.index
    train_mask = _train_mask_from_id(raw)

    sleep = _safe_numeric(raw, "sleep_duration", index)
    step_count = _safe_numeric(raw, "step_count", index)
    exercise_duration = _safe_numeric(raw, "exercise_duration", index)

    train_rows = train_mask & sleep.notna()
    step_rank = _percentile_rank_by_train(step_count, step_count[train_mask])
    exercise_rank = _percentile_rank_by_train(exercise_duration, exercise_duration[train_mask])
    activity_demand = pd.concat([step_rank, exercise_rank], axis=1).mean(axis=1, skipna=True)
    activity_bin = _activity_demand_bin(activity_demand, train_mask)

    feature_base = pd.DataFrame(index=index)
    feature_base["sleep_duration"] = sleep
    feature_base["stress_level"] = _context_column(raw, "stress_level", index)
    feature_base["sleep_quality"] = _context_column(raw, "sleep_quality", index)
    feature_base["physical_activity_level"] = _context_column(raw, "physical_activity_level", index)
    feature_base["activity_demand_bin"] = activity_bin
    feature_base["smoking_alcohol"] = _context_column(raw, "smoking_alcohol", index)

    train_base = feature_base.loc[train_mask].copy()
    global_stats = _make_context_table(train_base.loc[train_rows], [], "sleep_duration")
    if global_stats is None:
        global_median = sleep.median()
        global_iqr = sleep.quantile(0.75) - sleep.quantile(0.25)
    else:
        global_median = global_stats["median"]
        global_iqr = global_stats["iqr"]

    if not np.isfinite(global_median):
        global_median = CDC_ADULT_MIN_SLEEP_HOURS
    if not np.isfinite(global_iqr) or global_iqr <= 0:
        global_iqr = 1.0

    full_keys = [
        "stress_level",
        "sleep_quality",
        "physical_activity_level",
        "activity_demand_bin",
        "smoking_alcohol",
    ]
    mid_keys = ["stress_level", "sleep_quality", "physical_activity_level"]
    short_keys = ["stress_level", "sleep_quality"]

    full_table = _make_context_table(train_base, full_keys, "sleep_duration")
    mid_table = _make_context_table(train_base, mid_keys, "sleep_duration")
    short_table = _make_context_table(train_base, short_keys, "sleep_duration")

    if full_table is not None:
        full_table["iqr"] = full_table["q75"] - full_table["q25"]
    if mid_table is not None:
        mid_table["iqr"] = mid_table["q75"] - mid_table["q25"]
    if short_table is not None:
        short_table["iqr"] = short_table["q75"] - short_table["q25"]

    expected = _lookup_context(feature_base, full_table, full_keys, "median", MIN_CONTEXT_ROWS)
    matched_iqr = _lookup_context(feature_base, full_table, full_keys, "iqr", MIN_CONTEXT_ROWS)

    expected = expected.fillna(_lookup_context(feature_base, mid_table, mid_keys, "median", MIN_CONTEXT_ROWS))
    matched_iqr = matched_iqr.fillna(_lookup_context(feature_base, mid_table, mid_keys, "iqr", MIN_CONTEXT_ROWS))

    expected = expected.fillna(_lookup_context(feature_base, short_table, short_keys, "median", MIN_CONTEXT_ROWS))
    matched_iqr = matched_iqr.fillna(_lookup_context(feature_base, short_table, short_keys, "iqr", MIN_CONTEXT_ROWS))

    expected = expected.fillna(global_median)
    matched_iqr = matched_iqr.where(np.isfinite(matched_iqr) & (matched_iqr > 0), global_iqr).fillna(global_iqr)

    signed_gap = (sleep - expected).clip(-SLEEP_GAP_CLIP, SLEEP_GAP_CLIP)
    absolute_gap = signed_gap.abs()
    deficit_gap = (-signed_gap).clip(lower=0.0)
    surplus_gap = signed_gap.clip(lower=0.0)
    robust_z_gap = signed_gap / matched_iqr.replace(0, global_iqr)
    cdc_shortfall = (CDC_ADULT_MIN_SLEEP_HOURS - sleep).clip(lower=0.0)

    state = pd.Series("context_typical", index=index, dtype="object")
    state.loc[sleep.isna()] = "missing_sleep"
    state.loc[sleep.notna() & (signed_gap < -TYPICAL_SLEEP_GAP_HOURS)] = "context_short"
    state.loc[sleep.notna() & (signed_gap > TYPICAL_SLEEP_GAP_HOURS)] = "context_long"

    new_features = pd.DataFrame(index=index)
    new_features["expected_context_sleep"] = expected.astype("float64")
    new_features["context_sleep_gap"] = signed_gap.astype("float64")
    new_features["context_sleep_abs_gap"] = absolute_gap.astype("float64")
    new_features["context_sleep_deficit_gap"] = deficit_gap.astype("float64")
    new_features["context_sleep_surplus_gap"] = surplus_gap.astype("float64")
    new_features["context_sleep_robust_z_gap"] = robust_z_gap.astype("float64")
    new_features["context_sleep_state"] = state
    new_features["cdc_adult_sleep_shortfall"] = cdc_shortfall.astype("float64")
    new_features["activity_demand_percentile_mean"] = activity_demand.astype("float64")
    new_features["activity_demand_bin"] = activity_bin.astype("object")

    return new_features


FEATURE_GROUPS = [
    {
        "name": "contextual_sleep_need_gap",
        "fn": add_contextual_sleep_need_gap,
        "depends_on": [],
        "description": "Contextual sleep residual features comparing observed sleep against stress, sleep-quality, activity, and smoking-alcohol peer expectations.",
    }
]