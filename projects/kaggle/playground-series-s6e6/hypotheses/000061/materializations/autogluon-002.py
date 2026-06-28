import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

TRAIN_ID_MAX = 577346
C_LIGHT_KM_S = 299792.458
H0_KM_S_MPC = 70.0
OMEGA_M = 0.3
OMEGA_LAMBDA = 0.7
PHYSICAL_RADII_MPC = (0.5, 1.0, 3.0, 10.0)
LOW_REDSHIFT_RADII_DEG = (0.05, 0.2, 1.0, 2.0)
RADIUS_LABELS = ("r0p5", "r1", "r3", "r10")
COLOR_SCALE_LABELS = ("r1", "r3")
COLOR_SCALE_INDEXES = (1, 2)
NN_RANKS = (1, 5, 20)
K_NEIGHBORS = 128
CHUNK_SIZE = 20000
COSMOLOGY_GRID_SIZE = 768
TREE_LEAFSIZE = 40
TREE_WORKERS = -1
EPSILON = 0.000000001

def _numeric_array(frame, column, default):
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    return np.nan_to_num(values, nan=default, posinf=default, neginf=default)

def _infer_train_mask(raw, aux):
    n_rows = len(raw)

    if isinstance(aux, pd.DataFrame) and len(aux) == n_rows and not aux.empty:
        for column in ("is_train", "_is_train", "__is_train", "train"):
            if column in aux.columns:
                values = aux[column]
                if values.dtype == bool:
                    return values.to_numpy(dtype=bool, copy=True)
                numeric = pd.to_numeric(values, errors="coerce")
                if numeric.notna().any():
                    return numeric.fillna(0).to_numpy(dtype=np.float64) > 0

        for column in ("is_test", "_is_test", "__is_test", "test"):
            if column in aux.columns:
                values = aux[column]
                if values.dtype == bool:
                    return ~values.to_numpy(dtype=bool, copy=True)
                numeric = pd.to_numeric(values, errors="coerce")
                if numeric.notna().any():
                    return numeric.fillna(0).to_numpy(dtype=np.float64) <= 0

        for column in ("split", "dataset", "source"):
            if column in aux.columns:
                values = aux[column].astype(str).str.lower()
                if values.str.contains("train", regex=False).any():
                    return values.str.contains("train", regex=False).to_numpy(dtype=bool, copy=True)

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        finite = np.isfinite(ids)
        if finite.any() and np.nanmin(ids[finite]) <= TRAIN_ID_MAX and np.nanmax(ids[finite]) > TRAIN_ID_MAX:
            return ids <= TRAIN_ID_MAX

    return np.ones(n_rows, dtype=bool)

def _angular_diameter_distance_mpc(redshift):
    z_pos = np.maximum(redshift, 0.0)
    finite = np.isfinite(z_pos)
    if not finite.any():
        return np.zeros_like(z_pos, dtype=np.float64)

    z_max = float(np.max(z_pos[finite]))
    if z_max <= 0.0:
        return np.zeros_like(z_pos, dtype=np.float64)

    grid = np.linspace(0.0, z_max, COSMOLOGY_GRID_SIZE, dtype=np.float64)
    inv_e = 1.0 / np.sqrt(OMEGA_M * np.power(1.0 + grid, 3.0) + OMEGA_LAMBDA)
    dz_grid = np.diff(grid)
    integral = np.empty_like(grid)
    integral[0] = 0.0
    integral[1:] = np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * dz_grid)
    comoving = (C_LIGHT_KM_S / H0_KM_S_MPC) * np.interp(z_pos, grid, integral)
    return comoving / (1.0 + z_pos)

def _spherical_cap_area(radius_deg):
    radius_rad = np.deg2rad(np.maximum(radius_deg, 0.0))
    return 2.0 * np.pi * (1.0 - np.cos(radius_rad))

def _angular_deg_to_chord(radius_deg):
    radius_rad = np.deg2rad(np.maximum(radius_deg, 0.0))
    return 2.0 * np.sin(0.5 * radius_rad)

def _chord_to_deg(distance_chord):
    clipped = np.clip(distance_chord * 0.5, 0.0, 1.0)
    degrees = np.rad2deg(2.0 * np.arcsin(clipped))
    return np.where(np.isfinite(distance_chord), degrees, np.inf)

def _unit_sphere_xyz(alpha_deg, delta_deg):
    ra_rad = np.deg2rad(np.mod(alpha_deg, 360.0))
    dec_rad = np.deg2rad(np.clip(delta_deg, -90.0, 90.0))
    cos_dec = np.cos(dec_rad)
    return np.column_stack((cos_dec * np.cos(ra_rad), cos_dec * np.sin(ra_rad), np.sin(dec_rad))).astype(
        np.float64,
        copy=False,
    )

def _query_knn(tree, points, k):
    try:
        distances, indices = tree.query(points, k=k, workers=TREE_WORKERS)
    except TypeError:
        distances, indices = tree.query(points, k=k)

    if k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    return distances, indices

def _query_radius_count(tree, points, radius):
    try:
        counts = tree.query_ball_point(points, r=radius, return_length=True, workers=TREE_WORKERS)
    except TypeError:
        try:
            counts = tree.query_ball_point(points, r=radius, return_length=True)
        except TypeError:
            neighbors = tree.query_ball_point(points, r=radius)
            counts = np.fromiter((len(item) for item in neighbors), dtype=np.int64, count=len(points))
    except ValueError:
        counts = np.empty(len(points), dtype=np.int64)
        for row_idx in range(len(points)):
            counts[row_idx] = len(tree.query_ball_point(points[row_idx], r=float(radius[row_idx])))

    return np.asarray(counts, dtype=np.int64)

def _median_mad_by_row(diff_values, mask):
    valid_mask = mask & np.isfinite(diff_values)
    has_values = valid_mask.any(axis=1)

    medians = np.zeros(diff_values.shape[0], dtype=np.float32)
    mads = np.zeros(diff_values.shape[0], dtype=np.float32)
    empty = (~has_values).astype(np.int8)

    if has_values.any():
        selected = np.where(valid_mask[has_values], diff_values[has_values], np.nan)
        median_values = np.nanmedian(selected, axis=1)
        median_values = np.nan_to_num(median_values, nan=0.0, posinf=0.0, neginf=0.0)
        mad_values = np.nanmedian(np.abs(selected - median_values[:, None]), axis=1)
        mad_values = np.nan_to_num(mad_values, nan=0.0, posinf=0.0, neginf=0.0)
        medians[has_values] = median_values.astype(np.float32)
        mads[has_values] = mad_values.astype(np.float32)

    return medians, mads, empty

def add_redshift_slice_angular_environment(raw, deps, aux):
    n_rows = len(raw)
    index = raw.index

    if n_rows == 0:
        return pd.DataFrame(index=index)

    alpha = _numeric_array(raw, "alpha", 0.0)
    delta = _numeric_array(raw, "delta", 0.0)
    redshift = _numeric_array(raw, "redshift", 0.0)
    g_mag = _numeric_array(raw, "g", 0.0)
    r_mag = _numeric_array(raw, "r", 0.0)
    i_mag = _numeric_array(raw, "i", 0.0)

    redshift_nonnegative = np.maximum(redshift, 0.0)
    dz_window = np.maximum(0.01, 0.03 * (1.0 + redshift_nonnegative)).astype(np.float32)
    invalid_physical_scale = (redshift <= 0.003).astype(np.int8)

    angular_diameter_distance = _angular_diameter_distance_mpc(redshift)
    physical_radii = np.asarray(PHYSICAL_RADII_MPC, dtype=np.float64)
    low_redshift_radii = np.asarray(LOW_REDSHIFT_RADII_DEG, dtype=np.float64)

    theta_deg = np.empty((n_rows, len(PHYSICAL_RADII_MPC)), dtype=np.float32)
    valid_distance = (redshift > 0.003) & (angular_diameter_distance > EPSILON)
    physical_theta = np.rad2deg(physical_radii[None, :] / np.maximum(angular_diameter_distance[:, None], EPSILON))
    physical_theta = np.clip(physical_theta, 0.02, 2.0)
    theta_deg[:, :] = np.where(valid_distance[:, None], physical_theta, low_redshift_radii[None, :]).astype(np.float32)

    coords = _unit_sphere_xyz(alpha, delta)

    train_mask = _infer_train_mask(raw, aux)
    if train_mask.shape[0] != n_rows or not train_mask.any():
        train_mask = np.ones(n_rows, dtype=bool)

    ref_positions = np.flatnonzero(train_mask)
    ref_coords = coords[train_mask]
    ref_redshift = redshift[train_mask]
    ref_gr = (g_mag - r_mag)[train_mask]
    ref_ri = (r_mag - i_mag)[train_mask]
    n_ref = len(ref_positions)

    self_ref_position = np.full(n_rows, -1, dtype=np.int64)
    self_ref_position[ref_positions] = np.arange(n_ref, dtype=np.int64)

    if n_ref == 0:
        return pd.DataFrame(index=index)

    tree = cKDTree(ref_coords, leafsize=TREE_LEAFSIZE)
    k_query = min(n_ref, K_NEIGHBORS + 1)

    sorted_ref_redshift = np.sort(ref_redshift)
    left = np.searchsorted(sorted_ref_redshift, redshift - dz_window, side="left")
    right = np.searchsorted(sorted_ref_redshift, redshift + dz_window, side="right")
    redshift_ref_count = right - left
    redshift_ref_count = redshift_ref_count - (self_ref_position >= 0)
    redshift_ref_count = np.maximum(redshift_ref_count, 0)
    redshift_ref_denominator = np.maximum(n_ref - (self_ref_position >= 0).astype(np.int64), 1)
    redshift_ref_fraction = (redshift_ref_count / redshift_ref_denominator).astype(np.float32)

    slice_counts = np.zeros((n_rows, len(RADIUS_LABELS)), dtype=np.int32)
    annulus_counts = np.zeros((n_rows, len(RADIUS_LABELS)), dtype=np.int32)
    allz_counts = np.zeros((n_rows, len(RADIUS_LABELS)), dtype=np.int32)

    nn_distances = np.zeros((n_rows, len(NN_RANKS)), dtype=np.float32)
    nn_missing = np.ones((n_rows, len(NN_RANKS)), dtype=np.int8)
    nn_dist_ratios = np.zeros((n_rows, len(NN_RANKS)), dtype=np.float32)
    nn_rank_ratios = np.zeros((n_rows, len(NN_RANKS)), dtype=np.float32)

    gr_medians = np.zeros((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.float32)
    gr_mads = np.zeros((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.float32)
    gr_empty = np.ones((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.int8)
    ri_medians = np.zeros((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.float32)
    ri_mads = np.zeros((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.float32)
    ri_empty = np.ones((n_rows, len(COLOR_SCALE_INDEXES)), dtype=np.int8)

    candidate_truncated = np.zeros(n_rows, dtype=np.int8)

    for start in range(0, n_rows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_rows)
        chunk = slice(start, end)
        m_rows = end - start

        chunk_coords = coords[chunk]
        chunk_theta = theta_deg[chunk].astype(np.float64, copy=False)
        chunk_self_ref = self_ref_position[chunk]
        chunk_redshift = redshift[chunk]
        chunk_dz = dz_window[chunk].astype(np.float64, copy=False)

        for radius_idx in range(len(RADIUS_LABELS)):
            counts = _query_radius_count(
                tree,
                chunk_coords,
                _angular_deg_to_chord(chunk_theta[:, radius_idx]),
            ).astype(np.int32, copy=False)
            counts = counts - (chunk_self_ref >= 0).astype(np.int32)
            allz_counts[chunk, radius_idx] = np.maximum(counts, 0)

        distances_chord, indices = _query_knn(tree, chunk_coords, k_query)
        distances_deg = _chord_to_deg(distances_chord).astype(np.float32, copy=False)

        valid_all = np.isfinite(distances_chord) & (indices >= 0) & (indices < n_ref)
        valid_all &= indices != chunk_self_ref[:, None]

        safe_indices = np.clip(indices, 0, n_ref - 1)
        ref_z_neighbors = ref_redshift[safe_indices]
        same_redshift = valid_all & (np.abs(ref_z_neighbors - chunk_redshift[:, None]) <= chunk_dz[:, None])

        for radius_idx in range(len(RADIUS_LABELS)):
            radius = chunk_theta[:, radius_idx][:, None]
            slice_counts[chunk, radius_idx] = np.sum(same_redshift & (distances_deg <= radius), axis=1).astype(np.int32)
            annulus_counts[chunk, radius_idx] = np.sum(
                same_redshift & (distances_deg > 3.0 * radius) & (distances_deg <= 6.0 * radius),
                axis=1,
            ).astype(np.int32)

        distance_all = np.where(valid_all, distances_deg, np.inf)
        distance_same = np.where(same_redshift, distances_deg, np.inf)
        all_cumulative_rank = np.cumsum(valid_all, axis=1)
        same_cumulative_rank = np.cumsum(same_redshift, axis=1)

        rank_kths = [rank - 1 for rank in NN_RANKS if rank <= distance_same.shape[1]]
        all_partition = np.partition(distance_all, rank_kths, axis=1)
        same_partition = np.partition(distance_same, rank_kths, axis=1)
        row_numbers = np.arange(m_rows)

        for rank_idx, rank in enumerate(NN_RANKS):
            if rank <= distance_same.shape[1]:
                same_distance = same_partition[:, rank - 1]
                all_distance = all_partition[:, rank - 1]
                available = np.isfinite(same_distance)

                nn_distances[chunk, rank_idx] = np.where(available, same_distance, 0.0).astype(np.float32)
                nn_missing[chunk, rank_idx] = (~available).astype(np.int8)
                nn_dist_ratios[chunk, rank_idx] = np.where(
                    available & np.isfinite(all_distance),
                    same_distance / np.maximum(all_distance, EPSILON),
                    0.0,
                ).astype(np.float32)

                reached = same_cumulative_rank >= rank
                first_pos = np.argmax(reached, axis=1)
                any_reached = reached.any(axis=1)
                angular_rank = all_cumulative_rank[row_numbers, first_pos]
                nn_rank_ratios[chunk, rank_idx] = np.where(
                    any_reached,
                    angular_rank / float(rank),
                    0.0,
                ).astype(np.float32)

        if k_query < n_ref:
            farthest_candidate = np.max(np.where(valid_all, distances_deg, 0.0), axis=1)
            needed_outer_radius = 6.0 * np.max(chunk_theta, axis=1)
            candidate_truncated[chunk] = (farthest_candidate < needed_outer_radius).astype(np.int8)

        ref_gr_neighbors = ref_gr[safe_indices]
        ref_ri_neighbors = ref_ri[safe_indices]
        chunk_gr = (g_mag[chunk] - r_mag[chunk])[:, None]
        chunk_ri = (r_mag[chunk] - i_mag[chunk])[:, None]
        gr_diff = ref_gr_neighbors - chunk_gr
        ri_diff = ref_ri_neighbors - chunk_ri

        for scale_out_idx, radius_idx in enumerate(COLOR_SCALE_INDEXES):
            color_radius = chunk_theta[:, radius_idx][:, None]
            color_mask = same_redshift & (distances_deg <= color_radius)

            gr_med, gr_mad, gr_is_empty = _median_mad_by_row(gr_diff, color_mask)
            gr_medians[chunk, scale_out_idx] = gr_med
            gr_mads[chunk, scale_out_idx] = gr_mad
            gr_empty[chunk, scale_out_idx] = gr_is_empty

            ri_med, ri_mad, ri_is_empty = _median_mad_by_row(ri_diff, color_mask)
            ri_medians[chunk, scale_out_idx] = ri_med
            ri_mads[chunk, scale_out_idx] = ri_mad
            ri_empty[chunk, scale_out_idx] = ri_is_empty

    features = {
        "redshift_half_window": dz_window,
        "redshift_reference_fraction": redshift_ref_fraction,
        "invalid_physical_scale": invalid_physical_scale,
        "knn_background_truncated": candidate_truncated,
    }

    for radius_idx, label in enumerate(RADIUS_LABELS):
        radius = theta_deg[:, radius_idx].astype(np.float64, copy=False)
        inner_area = _spherical_cap_area(radius)
        annulus_area = np.maximum(_spherical_cap_area(6.0 * radius) - _spherical_cap_area(3.0 * radius), EPSILON)

        annulus_expected = annulus_counts[:, radius_idx].astype(np.float64) * inner_area / annulus_area
        allz_expected = allz_counts[:, radius_idx].astype(np.float64) * redshift_ref_fraction.astype(np.float64)

        features[f"theta_deg_{label}"] = theta_deg[:, radius_idx]
        features[f"slice_count_{label}"] = slice_counts[:, radius_idx]
        features[f"annulus_count_{label}"] = annulus_counts[:, radius_idx]
        features[f"allz_count_{label}"] = allz_counts[:, radius_idx]
        features[f"log_overdensity_annulus_{label}"] = (
            np.log1p(slice_counts[:, radius_idx].astype(np.float64)) - np.log1p(annulus_expected)
        ).astype(np.float32)
        features[f"log_overdensity_allz_{label}"] = (
            np.log1p(slice_counts[:, radius_idx].astype(np.float64)) - np.log1p(allz_expected)
        ).astype(np.float32)
        features[f"slice_fraction_allz_{label}"] = (
            slice_counts[:, radius_idx].astype(np.float64) / np.maximum(allz_counts[:, radius_idx].astype(np.float64), 1.0)
        ).astype(np.float32)

    for rank_idx, rank in enumerate(NN_RANKS):
        features[f"slice_nn{rank}_deg"] = nn_distances[:, rank_idx]
        features[f"slice_nn{rank}_missing"] = nn_missing[:, rank_idx]
        features[f"slice_nn{rank}_dist_ratio_allz"] = nn_dist_ratios[:, rank_idx]
        features[f"slice_nn{rank}_angular_rank_ratio"] = nn_rank_ratios[:, rank_idx]

    for scale_idx, scale_label in enumerate(COLOR_SCALE_LABELS):
        features[f"gr_diff_median_{scale_label}"] = gr_medians[:, scale_idx]
        features[f"gr_diff_mad_{scale_label}"] = gr_mads[:, scale_idx]
        features[f"gr_diff_empty_{scale_label}"] = gr_empty[:, scale_idx]
        features[f"ri_diff_median_{scale_label}"] = ri_medians[:, scale_idx]
        features[f"ri_diff_mad_{scale_label}"] = ri_mads[:, scale_idx]
        features[f"ri_diff_empty_{scale_label}"] = ri_empty[:, scale_idx]

    return pd.DataFrame(features, index=index)

FEATURE_GROUPS = [
    {
        "name": "redshift_slice_angular_environment",
        "fn": add_redshift_slice_angular_environment,
        "depends_on": [],
        "description": "Redshift-sliced spherical-neighborhood density, nearest-neighbor, and color-coherence features from the training sky reference.",
    }
]