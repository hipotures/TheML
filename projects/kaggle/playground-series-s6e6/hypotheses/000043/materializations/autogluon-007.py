import numpy as np
import pandas as pd

BAND_COLUMNS = ("u", "g", "r", "i", "z")
REDSHIFT_COLUMN = "redshift"
ID_COLUMN = "id"
TRAIN_ID_MAX = 577346

DEFAULT_BIN_COUNT = 20
MIN_BIN_ROWS = 3000
MIN_BIN_FRACTION = 0.0025
PC_RETAINED = 3
MAG_LOW_PERCENTILE = 0.05
MAG_HIGH_PERCENTILE = 99.95
WINSOR_PERCENTILE = 99.75
STD_FLOOR = 1e-6
EPSILON = 1e-12

AUX_BOOL_TRAIN_COLUMNS = ("is_train", "_is_train", "train_mask", "_train_mask", "is_training", "in_train")
AUX_SOURCE_COLUMNS = ("split", "dataset", "source", "_split", "_dataset", "_source")
AUX_TRAIN_LABELS = ("train", "training", "tr")
AUX_TEST_LABELS = ("test", "testing", "te", "holdout", "validation", "valid", "val")

OUTPUT_COLUMNS = (
    "score_pc1",
    "score_pc2",
    "score_pc3",
    "recon_l2",
    "relative_recon_l2",
    "residual_energy_fraction",
    "omitted_pc4_coord",
    "omitted_pc5_coord",
    "signed_resid_u",
    "signed_resid_g",
    "signed_resid_r",
    "signed_resid_i",
    "signed_resid_z",
    "abs_resid_u",
    "abs_resid_g",
    "abs_resid_r",
    "abs_resid_i",
    "abs_resid_z",
    "max_abs_resid_band_idx",
    "max_abs_resid_value",
    "recon_l2_train_bin_percentile",
)

NONNEGATIVE_FEATURE_COLUMNS = (
    "recon_l2",
    "relative_recon_l2",
    "residual_energy_fraction",
    "abs_resid_u",
    "abs_resid_g",
    "abs_resid_r",
    "abs_resid_i",
    "abs_resid_z",
    "max_abs_resid_value",
)


def _empty_feature_frame(index):
    data = {}
    for column in OUTPUT_COLUMNS:
        if column == "max_abs_resid_band_idx":
            data[column] = np.empty(len(index), dtype=np.int16)
        else:
            data[column] = np.empty(len(index), dtype=np.float64)
    return pd.DataFrame(data, index=index)


def _coerce_numeric_array(frame, column, n, default_value):
    if column not in frame.columns:
        return np.full(n, default_value, dtype=np.float64)
    return np.asarray(pd.to_numeric(frame[column], errors="coerce"), dtype=np.float64).copy()


def _extract_numeric_matrix(frame, columns):
    n = len(frame)
    matrix = np.empty((n, len(columns)), dtype=np.float64)
    for j, column in enumerate(columns):
        matrix[:, j] = _coerce_numeric_array(frame, column, n, 0.0)
    return matrix


def _finite_median(values, default_value):
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(default_value)
    return float(np.median(finite))


def _finite_percentile(values, percentile, default_value):
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(default_value)
    return float(np.percentile(finite, percentile))


def _ensure_fit_mask(mask, n):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != n:
        mask = np.ones(n, dtype=bool)
    if n > 0 and not mask.any():
        mask = np.ones(n, dtype=bool)
    return mask


def _boolean_like_aux_mask(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool, copy=True)

    numeric = np.asarray(pd.to_numeric(series, errors="coerce"), dtype=np.float64)
    finite = np.isfinite(numeric)
    if finite.any():
        finite_values = numeric[finite]
        if np.all((finite_values == 0.0) | (finite_values == 1.0)):
            return np.where(finite, numeric == 1.0, False)

    return None


def _fit_mask_from_aux_or_id(raw, aux):
    n = len(raw)

    if isinstance(aux, pd.DataFrame) and len(aux) == n and not aux.empty:
        for column in AUX_BOOL_TRAIN_COLUMNS:
            if column in aux.columns:
                candidate = _boolean_like_aux_mask(aux[column])
                if candidate is not None and candidate.any():
                    return candidate

        for column in AUX_SOURCE_COLUMNS:
            if column in aux.columns:
                values = aux[column].astype(str).str.lower().to_numpy()
                train_mask = np.isin(values, AUX_TRAIN_LABELS)
                test_mask = np.isin(values, AUX_TEST_LABELS)
                if train_mask.any() and test_mask.any():
                    return train_mask

    if ID_COLUMN in raw.columns:
        ids = _coerce_numeric_array(raw, ID_COLUMN, n, np.nan)
        finite = np.isfinite(ids)
        candidate = finite & (ids <= TRAIN_ID_MAX)
        if candidate.any():
            return candidate

    return np.ones(n, dtype=bool)


def _prepare_magnitudes(magnitudes, fit_mask):
    prepared = magnitudes.copy()
    for j in range(prepared.shape[1]):
        column = prepared[:, j]
        fallback = _finite_median(column, 0.0)
        median = _finite_median(column[fit_mask], fallback)
        column[~np.isfinite(column)] = median

        low = _finite_percentile(column[fit_mask], MAG_LOW_PERCENTILE, median)
        high = _finite_percentile(column[fit_mask], MAG_HIGH_PERCENTILE, median)
        if high < low:
            low, high = high, low

        prepared[:, j] = np.clip(column, low, high)
    return prepared


def _prepare_redshift(raw, fit_mask):
    n = len(raw)
    redshift = _coerce_numeric_array(raw, REDSHIFT_COLUMN, n, 0.0)
    fallback = _finite_median(redshift, 0.0)
    median = _finite_median(redshift[fit_mask], fallback)
    redshift[~np.isfinite(redshift)] = median
    return redshift


def _magnitudes_to_shape(magnitudes):
    flux = 10.0 ** (-0.4 * magnitudes)
    flux = np.where(np.isfinite(flux) & (flux > 0.0), flux, 0.0)
    denom = flux.sum(axis=1, keepdims=True)
    shape = flux / (denom + EPSILON)

    bad_rows = denom[:, 0] <= EPSILON
    if bad_rows.any():
        shape[bad_rows, :] = 1.0 / float(shape.shape[1])

    return shape


def _build_redshift_bins(redshift_fit):
    finite = np.asarray(redshift_fit, dtype=np.float64)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        return np.asarray([0.0], dtype=np.float64), np.asarray([0.0], dtype=np.float64)

    z_min = float(np.min(finite))
    z_max = float(np.max(finite))
    if not z_max > z_min:
        return np.asarray([z_min], dtype=np.float64), np.asarray([z_max], dtype=np.float64)

    quantiles = np.linspace(0.0, 1.0, DEFAULT_BIN_COUNT + 1)
    raw_edges = np.quantile(finite, quantiles)
    raw_edges[0] = z_min
    raw_edges[-1] = z_max

    edges = []
    for edge in raw_edges:
        edge = float(edge)
        if not edges or edge > edges[-1] + EPSILON:
            edges.append(edge)

    if len(edges) < 2:
        edges = [z_min, z_max]
    if edges[0] > z_min:
        edges[0] = z_min
    if edges[-1] < z_max - EPSILON:
        edges.append(z_max)
    else:
        edges[-1] = z_max

    initial_left = np.asarray(edges[:-1], dtype=np.float64)
    initial_right = np.asarray(edges[1:], dtype=np.float64)
    n_initial = initial_right.shape[0]

    clipped = np.clip(finite, z_min, z_max)
    initial_bin = np.searchsorted(initial_right, clipped, side="left")
    initial_bin = np.clip(initial_bin, 0, n_initial - 1)
    counts = np.bincount(initial_bin, minlength=n_initial)

    min_rows = max(MIN_BIN_ROWS, int(np.ceil(MIN_BIN_FRACTION * finite.size)))
    min_rows = max(1, min_rows)

    merged_left = []
    merged_right = []
    current_left = float(initial_left[0])
    current_count = 0

    for i in range(n_initial):
        current_count += int(counts[i])
        is_last = i == n_initial - 1
        if current_count >= min_rows or is_last:
            if is_last and current_count < min_rows and merged_right:
                merged_right[-1] = float(initial_right[i])
            else:
                merged_left.append(current_left)
                merged_right.append(float(initial_right[i]))
            current_left = float(initial_right[i])
            current_count = 0

    if not merged_left:
        merged_left = [z_min]
        merged_right = [z_max]

    merged_left[0] = z_min
    merged_right[-1] = z_max

    return np.asarray(merged_left, dtype=np.float64), np.asarray(merged_right, dtype=np.float64)


def _assign_redshift_bins(redshift, lefts, rights):
    if lefts.size == 0:
        return np.zeros(redshift.shape[0], dtype=np.int16)

    z_min = float(lefts[0])
    z_max = float(rights[-1])
    midpoint = 0.5 * (z_min + z_max)

    z = np.asarray(redshift, dtype=np.float64).copy()
    z[~np.isfinite(z)] = midpoint
    z = np.clip(z, z_min, z_max)

    assigned = np.searchsorted(rights, z, side="left")
    assigned = np.clip(assigned, 0, lefts.size - 1)

    invalid = (z < lefts[assigned] - EPSILON) | (z > rights[assigned] + EPSILON)
    if invalid.any():
        midpoints = 0.5 * (lefts + rights)
        nearest = np.argmin(np.abs(z[invalid, None] - midpoints[None, :]), axis=1)
        assigned[invalid] = nearest

    return assigned.astype(np.int16, copy=False)


def _effective_rank(singular_values, matrix_shape):
    if singular_values.size == 0:
        return 0

    largest = float(singular_values[0])
    if not np.isfinite(largest) or largest <= EPSILON:
        return 0

    tolerance = max(matrix_shape[0], matrix_shape[1]) * np.finfo(np.float64).eps * largest
    tolerance = max(tolerance, EPSILON)
    return int(np.sum(singular_values > tolerance))


def _fit_pca_model(shape_values, row_mask):
    n_features = shape_values.shape[1]
    row_mask = np.asarray(row_mask, dtype=bool)
    subset = shape_values[row_mask]

    if subset.shape[0] == 0:
        subset = shape_values

    if subset.shape[0] == 0:
        mean = np.full(n_features, 1.0 / float(n_features), dtype=np.float64)
        std = np.full(n_features, STD_FLOOR, dtype=np.float64)
        components = np.empty((0, n_features), dtype=np.float64)
        return {
            "mean": mean,
            "std": std,
            "components": components,
            "rank": 0,
            "retain": 0,
            "error_sorted": np.empty(0, dtype=np.float64),
        }

    mean = np.mean(subset, axis=0)
    mean = np.where(np.isfinite(mean), mean, 1.0 / float(n_features))

    std = np.std(subset, axis=0)
    std = np.where(np.isfinite(std) & (std > STD_FLOOR), std, STD_FLOOR)

    x = (subset - mean) / std
    x = np.where(np.isfinite(x), x, 0.0)

    try:
        _, singular_values, components = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        singular_values = np.zeros(n_features, dtype=np.float64)
        components = np.eye(n_features, dtype=np.float64)

    rank = _effective_rank(singular_values, x.shape)
    retain = min(PC_RETAINED, rank)

    return {
        "mean": mean.astype(np.float64, copy=False),
        "std": std.astype(np.float64, copy=False),
        "components": components.astype(np.float64, copy=False),
        "rank": rank,
        "retain": retain,
        "error_sorted": np.empty(0, dtype=np.float64),
    }


def _model_error_l2(shape_values, model):
    if shape_values.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    x = (shape_values - model["mean"]) / model["std"]
    x = np.where(np.isfinite(x), x, 0.0)

    components = model["components"]
    retain = min(int(model["retain"]), components.shape[0])

    if retain > 0:
        retained = components[:retain]
        scores = x @ retained.T
        x_hat = scores @ retained
    else:
        x_hat = np.zeros_like(x)

    error = x - x_hat
    squared_error = np.sum(error * error, axis=1)
    return np.sqrt(np.maximum(squared_error, 0.0))


def _with_error_distribution(model, shape_values):
    model_copy = dict(model)
    errors = _model_error_l2(shape_values, model_copy)
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        model_copy["error_sorted"] = np.empty(0, dtype=np.float64)
    else:
        model_copy["error_sorted"] = np.sort(finite)
    return model_copy


def _transform_pca_model(shape_values, model):
    n = shape_values.shape[0]
    x = (shape_values - model["mean"]) / model["std"]
    x = np.where(np.isfinite(x), x, 0.0)

    components = model["components"]
    if components.size:
        scores_all = x @ components.T
    else:
        scores_all = np.empty((n, 0), dtype=np.float64)

    retain = min(int(model["retain"]), components.shape[0])
    if retain > 0:
        retained = components[:retain]
        x_hat = scores_all[:, :retain] @ retained
    else:
        x_hat = np.zeros_like(x)

    error = x - x_hat
    original_residual = error * model["std"]
    abs_residual = np.abs(original_residual)

    squared_error = np.sum(error * error, axis=1)
    x_squared = np.sum(x * x, axis=1)

    recon_l2 = np.sqrt(np.maximum(squared_error, 0.0))
    relative_recon_l2 = recon_l2 / (np.sqrt(np.maximum(x_squared, 0.0)) + EPSILON)
    residual_energy_fraction = squared_error / (x_squared + EPSILON)

    pc_scores = np.zeros((n, PC_RETAINED), dtype=np.float64)
    available_scores = min(PC_RETAINED, int(model["rank"]), scores_all.shape[1])
    if available_scores > 0:
        pc_scores[:, :available_scores] = scores_all[:, :available_scores]

    omitted_pc4 = np.zeros(n, dtype=np.float64)
    omitted_pc5 = np.zeros(n, dtype=np.float64)
    if int(model["rank"]) >= 4 and scores_all.shape[1] >= 4:
        omitted_pc4 = scores_all[:, 3]
    if int(model["rank"]) >= 5 and scores_all.shape[1] >= 5:
        omitted_pc5 = scores_all[:, 4]

    if n == 0:
        max_idx = np.empty(0, dtype=np.int16)
        max_value = np.empty(0, dtype=np.float64)
    else:
        max_idx = np.argmax(abs_residual, axis=1).astype(np.int16, copy=False)
        max_value = abs_residual[np.arange(n), max_idx]

    return {
        "pc_scores": pc_scores,
        "recon_l2": recon_l2,
        "relative_recon_l2": relative_recon_l2,
        "residual_energy_fraction": residual_energy_fraction,
        "omitted_pc4": omitted_pc4,
        "omitted_pc5": omitted_pc5,
        "signed_residual": original_residual,
        "abs_residual": abs_residual,
        "max_idx": max_idx,
        "max_value": max_value,
    }


def _percentile_rank(values, sorted_reference):
    reference = np.asarray(sorted_reference, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        return np.full(values.shape[0], 0.5, dtype=np.float64)

    ranks = np.searchsorted(reference, values, side="right").astype(np.float64)
    ranks = ranks / float(reference.size)
    return np.clip(ranks, 0.0, 1.0)


def _winsorize_features(features, fit_mask):
    frame = features.copy()
    fit_mask = _ensure_fit_mask(fit_mask, len(frame))

    for column in frame.columns:
        if column == "max_abs_resid_band_idx":
            values = np.asarray(frame[column], dtype=np.float64)
            values = np.nan_to_num(values, nan=0.0, posinf=4.0, neginf=0.0)
            frame[column] = np.clip(np.rint(values), 0.0, 4.0).astype(np.int16)
            continue

        if column == "recon_l2_train_bin_percentile":
            values = np.asarray(frame[column], dtype=np.float64)
            values = np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0)
            frame[column] = np.clip(values, 0.0, 1.0)
            continue

        values = np.asarray(frame[column], dtype=np.float64)
        values = np.where(np.isfinite(values), values, 0.0)

        fit_values = values[fit_mask & np.isfinite(values)]
        if fit_values.size == 0:
            fit_values = values[np.isfinite(values)]

        if fit_values.size == 0:
            frame[column] = np.zeros(len(frame), dtype=np.float64)
            continue

        if column in NONNEGATIVE_FEATURE_COLUMNS:
            cap = _finite_percentile(fit_values, WINSOR_PERCENTILE, 0.0)
            if not np.isfinite(cap) or cap < 0.0:
                cap = 0.0
            frame[column] = np.clip(values, 0.0, cap)
        else:
            cap = _finite_percentile(np.abs(fit_values), WINSOR_PERCENTILE, 0.0)
            if not np.isfinite(cap) or cap < 0.0:
                cap = 0.0
            frame[column] = np.clip(values, -cap, cap)

    return frame


def add_redshift_partitioned_sed_pca_residuals(raw, deps, aux):
    n = len(raw)
    if n == 0:
        return _empty_feature_frame(raw.index)

    fit_mask = _ensure_fit_mask(_fit_mask_from_aux_or_id(raw, aux), n)

    magnitudes = _extract_numeric_matrix(raw, BAND_COLUMNS)
    magnitudes = _prepare_magnitudes(magnitudes, fit_mask)
    shape = _magnitudes_to_shape(magnitudes)

    redshift = _prepare_redshift(raw, fit_mask)
    bin_lefts, bin_rights = _build_redshift_bins(redshift[fit_mask])
    bin_assignments = _assign_redshift_bins(redshift, bin_lefts, bin_rights)

    global_model = _fit_pca_model(shape, fit_mask)
    global_model = _with_error_distribution(global_model, shape[fit_mask])

    n_fit = int(np.sum(fit_mask))
    min_rows_for_local = max(2, max(MIN_BIN_ROWS, int(np.ceil(MIN_BIN_FRACTION * float(n_fit)))))

    models = []
    for bin_index in range(bin_lefts.shape[0]):
        bin_fit_mask = fit_mask & (bin_assignments == bin_index)
        bin_fit_count = int(np.sum(bin_fit_mask))

        if bin_fit_count >= min_rows_for_local or (bin_lefts.shape[0] == 1 and bin_fit_count > 0):
            local_model = _fit_pca_model(shape, bin_fit_mask)
            local_model = _with_error_distribution(local_model, shape[bin_fit_mask])
            models.append(local_model)
        else:
            models.append(global_model)

    score_pc = np.zeros((n, PC_RETAINED), dtype=np.float64)
    recon_l2 = np.zeros(n, dtype=np.float64)
    relative_recon_l2 = np.zeros(n, dtype=np.float64)
    residual_energy_fraction = np.zeros(n, dtype=np.float64)
    omitted_pc4 = np.zeros(n, dtype=np.float64)
    omitted_pc5 = np.zeros(n, dtype=np.float64)
    signed_residual = np.zeros((n, len(BAND_COLUMNS)), dtype=np.float64)
    abs_residual = np.zeros((n, len(BAND_COLUMNS)), dtype=np.float64)
    max_idx = np.zeros(n, dtype=np.int16)
    max_value = np.zeros(n, dtype=np.float64)
    percentile = np.zeros(n, dtype=np.float64)

    for bin_index, model in enumerate(models):
        row_mask = bin_assignments == bin_index
        if not row_mask.any():
            continue

        transformed = _transform_pca_model(shape[row_mask], model)

        score_pc[row_mask, :] = transformed["pc_scores"]
        recon_l2[row_mask] = transformed["recon_l2"]
        relative_recon_l2[row_mask] = transformed["relative_recon_l2"]
        residual_energy_fraction[row_mask] = transformed["residual_energy_fraction"]
        omitted_pc4[row_mask] = transformed["omitted_pc4"]
        omitted_pc5[row_mask] = transformed["omitted_pc5"]
        signed_residual[row_mask, :] = transformed["signed_residual"]
        abs_residual[row_mask, :] = transformed["abs_residual"]
        max_idx[row_mask] = transformed["max_idx"]
        max_value[row_mask] = transformed["max_value"]
        percentile[row_mask] = _percentile_rank(transformed["recon_l2"], model["error_sorted"])

    data = {
        "score_pc1": score_pc[:, 0],
        "score_pc2": score_pc[:, 1],
        "score_pc3": score_pc[:, 2],
        "recon_l2": recon_l2,
        "relative_recon_l2": relative_recon_l2,
        "residual_energy_fraction": residual_energy_fraction,
        "omitted_pc4_coord": omitted_pc4,
        "omitted_pc5_coord": omitted_pc5,
        "signed_resid_u": signed_residual[:, 0],
        "signed_resid_g": signed_residual[:, 1],
        "signed_resid_r": signed_residual[:, 2],
        "signed_resid_i": signed_residual[:, 3],
        "signed_resid_z": signed_residual[:, 4],
        "abs_resid_u": abs_residual[:, 0],
        "abs_resid_g": abs_residual[:, 1],
        "abs_resid_r": abs_residual[:, 2],
        "abs_resid_i": abs_residual[:, 3],
        "abs_resid_z": abs_residual[:, 4],
        "max_abs_resid_band_idx": max_idx,
        "max_abs_resid_value": max_value,
        "recon_l2_train_bin_percentile": percentile,
    }

    features = pd.DataFrame(data, index=raw.index)
    return _winsorize_features(features, fit_mask)


FEATURE_GROUPS = [
    {
        "name": "redshift_partitioned_sed_pca_residuals",
        "fn": add_redshift_partitioned_sed_pca_residuals,
        "depends_on": [],
        "description": "Redshift-local PCA residual features describing how each normalized ugriz spectral shape departs from its local photometric manifold.",
    }
]