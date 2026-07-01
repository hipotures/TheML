import numpy as np
import pandas as pd


NUMERIC_COLUMNS = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
)

CATEGORICAL_COLUMNS = (
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
)

GROUP_KEY = "student_id"
NEAREST_K = 8
CHUNK_SIZE = 4096
MIN_OBSERVED_FIELDS = 3
IQR_FLOOR_FRACTION = 0.05
EPSILON = 1.0e-6


def _empty_external_personal_baseline_deviation(raw):
    return pd.DataFrame(
        {
            "nearest_mixed_distance": np.zeros(len(raw), dtype=np.float32),
            "weighted_mixed_distance": np.zeros(len(raw), dtype=np.float32),
            "outside_personal_iqr_rate": np.zeros(len(raw), dtype=np.float32),
            "high_side_outside_rate": np.zeros(len(raw), dtype=np.float32),
            "low_side_outside_rate": np.zeros(len(raw), dtype=np.float32),
            "categorical_off_mode_rate": np.zeros(len(raw), dtype=np.float32),
            "observed_field_coverage": np.zeros(len(raw), dtype=np.float32),
            "coverage_adjusted_anomaly": np.zeros(len(raw), dtype=np.float32),
            "low_coverage_flag": np.ones(len(raw), dtype=np.int8),
        },
        index=raw.index,
    )


def _safe_category_frame(df, columns):
    out = df.loc[:, columns].copy()
    for col in columns:
        out[col] = out[col].astype("object").where(out[col].notna(), "__missing__")
    return out


def _build_baselines(aux, numeric_cols, categorical_cols):
    grouped = aux.groupby(GROUP_KEY, sort=False)

    if numeric_cols:
        med = grouped[list(numeric_cols)].median()
        q25 = grouped[list(numeric_cols)].quantile(0.25)
        q75 = grouped[list(numeric_cols)].quantile(0.75)
        iqr = q75 - q25

        global_q25 = aux.loc[:, numeric_cols].quantile(0.25)
        global_q75 = aux.loc[:, numeric_cols].quantile(0.75)
        global_floor = ((global_q75 - global_q25).abs() * IQR_FLOOR_FRACTION).clip(lower=EPSILON)
        iqr = iqr.clip(lower=global_floor, axis=1).fillna(global_floor)
    else:
        med = pd.DataFrame(index=grouped.size().index)
        iqr = pd.DataFrame(index=med.index)

    baseline_index = med.index

    mode_values = {}
    mode_freqs = {}
    category_freq_maps = {}

    for col in categorical_cols:
        counts = aux.groupby([GROUP_KEY, col], sort=False).size()
        totals = counts.groupby(level=0).sum()
        freqs = counts / totals.reindex(counts.index.get_level_values(0)).to_numpy()

        mode_pos = freqs.groupby(level=0).idxmax()
        mode_values[col] = mode_pos.map(lambda item: item[1]).reindex(baseline_index).fillna("__missing__").to_numpy()
        mode_freqs[col] = freqs.loc[mode_pos.to_list()].to_numpy()
        category_freq_maps[col] = freqs.to_dict()

    return baseline_index, med, iqr, mode_values, mode_freqs, category_freq_maps


def _build_category_penalty_maps(baseline_ids, mode_values, category_freq_maps, categorical_cols):
    baseline_pos_by_id = {student_id: pos for pos, student_id in enumerate(baseline_ids)}
    penalty_maps = {}
    n_baselines = len(baseline_ids)

    for col in categorical_cols:
        mode_arr = mode_values[col]
        values = set(mode_arr.tolist())
        values.update(value for _student_id, value in category_freq_maps[col].keys())

        col_penalties = {}
        for value in values:
            penalties = np.ones(n_baselines, dtype=np.float32)
            penalties[mode_arr == value] = 0.0
            col_penalties[value] = penalties

        for (student_id, value), freq in category_freq_maps[col].items():
            baseline_pos = baseline_pos_by_id.get(student_id)
            if baseline_pos is None or mode_arr[baseline_pos] == value:
                continue
            col_penalties[value][baseline_pos] = 1.0 - float(freq)

        penalty_maps[col] = col_penalties

    return penalty_maps


def add_external_personal_baseline_deviation(raw, deps, aux):
    numeric_cols = tuple(col for col in NUMERIC_COLUMNS if col in raw.columns and col in aux.columns)
    categorical_cols = tuple(col for col in CATEGORICAL_COLUMNS if col in raw.columns and col in aux.columns)

    if aux is None or len(aux) == 0 or GROUP_KEY not in aux.columns or (not numeric_cols and not categorical_cols):
        return _empty_external_personal_baseline_deviation(raw)

    aux_work = aux.loc[:, (GROUP_KEY,) + numeric_cols + categorical_cols].copy()
    aux_work = aux_work.dropna(subset=[GROUP_KEY])
    if len(aux_work) == 0:
        return _empty_external_personal_baseline_deviation(raw)

    for col in numeric_cols:
        aux_work[col] = pd.to_numeric(aux_work[col], errors="coerce")
    if categorical_cols:
        aux_work.loc[:, categorical_cols] = _safe_category_frame(aux_work, categorical_cols)

    baseline_index, med, iqr, mode_values, mode_freqs, category_freq_maps = _build_baselines(
        aux_work, numeric_cols, categorical_cols
    )

    n_rows = len(raw)
    n_baselines = len(baseline_index)
    n_fields = len(numeric_cols) + len(categorical_cols)

    if n_baselines == 0 or n_fields == 0:
        return _empty_external_personal_baseline_deviation(raw)

    med_arr = med.loc[baseline_index, numeric_cols].to_numpy(dtype=np.float32, copy=True) if numeric_cols else None
    iqr_arr = iqr.loc[baseline_index, numeric_cols].to_numpy(dtype=np.float32, copy=True) if numeric_cols else None
    baseline_ids = baseline_index.to_numpy()
    category_penalty_maps = (
        _build_category_penalty_maps(baseline_ids, mode_values, category_freq_maps, categorical_cols)
        if categorical_cols
        else {}
    )

    result = {
        "nearest_mixed_distance": np.zeros(n_rows, dtype=np.float32),
        "weighted_mixed_distance": np.zeros(n_rows, dtype=np.float32),
        "outside_personal_iqr_rate": np.zeros(n_rows, dtype=np.float32),
        "high_side_outside_rate": np.zeros(n_rows, dtype=np.float32),
        "low_side_outside_rate": np.zeros(n_rows, dtype=np.float32),
        "categorical_off_mode_rate": np.zeros(n_rows, dtype=np.float32),
        "observed_field_coverage": np.zeros(n_rows, dtype=np.float32),
        "coverage_adjusted_anomaly": np.zeros(n_rows, dtype=np.float32),
        "low_coverage_flag": np.ones(n_rows, dtype=np.int8),
    }

    raw_num = raw.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else None
    raw_cat = _safe_category_frame(raw, categorical_cols) if categorical_cols else None

    for start in range(0, n_rows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_rows)
        chunk_len = end - start

        distance_sum = np.zeros((chunk_len, n_baselines), dtype=np.float32)
        observed_count = np.zeros(chunk_len, dtype=np.float32)

        x_num = None
        if numeric_cols:
            x_num = raw_num.iloc[start:end].to_numpy(dtype=np.float32, copy=True)
            num_obs = ~np.isnan(x_num)
            observed_count += num_obs.sum(axis=1).astype(np.float32)

            for j in range(len(numeric_cols)):
                obs = num_obs[:, j]
                if not np.any(obs):
                    continue
                diff = np.abs(x_num[obs, j:j + 1] - med_arr[:, j].reshape(1, -1)) / iqr_arr[:, j].reshape(1, -1)
                distance_sum[obs, :] += diff.astype(np.float32, copy=False)

        if categorical_cols:
            for col in categorical_cols:
                values = raw_cat.iloc[start:end][col].to_numpy(dtype=object, copy=True)
                obs = values != "__missing__"
                observed_count += obs.astype(np.float32)

                mode_arr = mode_values[col]
                penalty_map = category_penalty_maps[col]

                for value in pd.unique(values[obs]):
                    penalties = penalty_map.get(value)
                    if penalties is None:
                        penalties = np.ones(n_baselines, dtype=np.float32)
                        penalties[mode_arr == value] = 0.0
                        penalty_map[value] = penalties
                    distance_sum[obs & (values == value), :] += penalties

        valid = observed_count >= float(MIN_OBSERVED_FIELDS)
        coverage = observed_count / float(n_fields)
        result["observed_field_coverage"][start:end] = coverage.astype(np.float32)
        result["low_coverage_flag"][start:end] = (~valid).astype(np.int8)

        if not np.any(valid):
            continue

        denom = np.maximum(observed_count[valid].reshape(-1, 1), 1.0)
        mixed_distance = distance_sum[valid, :] / denom

        k = min(NEAREST_K, n_baselines)
        nearest_pos = np.argpartition(mixed_distance, kth=k - 1, axis=1)[:, :k]
        nearest_dist = np.take_along_axis(mixed_distance, nearest_pos, axis=1)
        order = np.argsort(nearest_dist, axis=1)
        nearest_pos = np.take_along_axis(nearest_pos, order, axis=1)
        nearest_dist = np.take_along_axis(nearest_dist, order, axis=1)

        weights = 1.0 / (nearest_dist + 0.05)
        weight_sum = weights.sum(axis=1)
        weighted_distance = (nearest_dist * weights).sum(axis=1) / weight_sum

        valid_global = np.where(valid)[0]
        result["nearest_mixed_distance"][start + valid_global] = nearest_dist[:, 0].astype(np.float32)
        result["weighted_mixed_distance"][start + valid_global] = weighted_distance.astype(np.float32)

        if numeric_cols:
            x_valid = x_num[valid]
            num_obs_valid = ~np.isnan(x_valid)
            numeric_den = np.maximum(num_obs_valid.sum(axis=1).astype(np.float32).reshape(-1, 1), 1.0)

            selected_med = med_arr[nearest_pos]
            selected_iqr = iqr_arr[nearest_pos]
            selected_vals = x_valid[:, None, :]
            observed_mask = num_obs_valid[:, None, :]

            high = (selected_vals > selected_med + selected_iqr) & observed_mask
            low = (selected_vals < selected_med - selected_iqr) & observed_mask
            outside = high | low

            outside_rate = outside.sum(axis=2).astype(np.float32) / numeric_den
            high_rate = high.sum(axis=2).astype(np.float32) / numeric_den
            low_rate = low.sum(axis=2).astype(np.float32) / numeric_den

            result["outside_personal_iqr_rate"][start + valid_global] = (
                (outside_rate * weights).sum(axis=1) / weight_sum
            ).astype(np.float32)
            result["high_side_outside_rate"][start + valid_global] = (
                (high_rate * weights).sum(axis=1) / weight_sum
            ).astype(np.float32)
            result["low_side_outside_rate"][start + valid_global] = (
                (low_rate * weights).sum(axis=1) / weight_sum
            ).astype(np.float32)

        if categorical_cols:
            cat_off_sum = np.zeros(len(valid_global), dtype=np.float32)
            cat_den = np.zeros(len(valid_global), dtype=np.float32)

            for col in categorical_cols:
                values = raw_cat.iloc[start:end][col].to_numpy(dtype=object, copy=True)[valid]
                obs = values != "__missing__"
                if not np.any(obs):
                    continue

                mode_arr = mode_values[col]
                obs_pos = np.where(obs)[0]
                selected_modes = mode_arr[nearest_pos[obs_pos]]
                off = selected_modes != values[obs_pos].reshape(-1, 1)
                cat_off_sum[obs_pos] += (
                    (off.astype(np.float32) * weights[obs_pos]).sum(axis=1) / weight_sum[obs_pos]
                )
                cat_den[obs_pos] += 1.0

            cat_rate = np.divide(cat_off_sum, cat_den, out=np.zeros_like(cat_off_sum), where=cat_den > 0.0)
            result["categorical_off_mode_rate"][start + valid_global] = cat_rate.astype(np.float32)

        anomaly = (
            result["weighted_mixed_distance"][start + valid_global]
            * np.sqrt(np.maximum(result["observed_field_coverage"][start + valid_global], 0.0))
        )
        result["coverage_adjusted_anomaly"][start + valid_global] = anomaly.astype(np.float32)

    return pd.DataFrame(result, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "external_personal_baseline_deviation",
        "fn": add_external_personal_baseline_deviation,
        "depends_on": [],
        "description": "Distance-weighted anomaly features comparing each row with repeated auxiliary student personal baselines.",
    }
]
