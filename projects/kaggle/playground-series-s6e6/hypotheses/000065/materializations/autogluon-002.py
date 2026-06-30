import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

ARCMIN_PER_DEGREE = 60.0
ARCSEC_PER_DEGREE = 3600.0
ARCSEC_PER_RADIAN = 206264.80624709636
CROWDING_RADII_ARCSEC = (2.0, 3.0, 10.0, 55.0, 62.0, 120.0)
COLOR_PAIRS = (("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"))


def _as_numeric(raw, column, default=0.0):
    if column in raw.columns:
        return pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)
    return np.full(len(raw), default, dtype=float)


def _finite_quantile(values, q, fallback):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(fallback)
    return float(np.nanquantile(arr, q))


def _context_frame(raw, aux):
    needed = ("alpha", "delta", "redshift", "r", "u", "g", "i", "z")
    frames = [raw.loc[:, [c for c in needed if c in raw.columns]]]
    if isinstance(aux, pd.DataFrame) and len(aux) > 0 and "alpha" in aux.columns and "delta" in aux.columns:
        frames.append(aux.loc[:, [c for c in needed if c in aux.columns]])
    return pd.concat(frames, axis=0, ignore_index=True, copy=False)


def _unit_sphere_coords(alpha_deg, delta_deg):
    ra = np.deg2rad(np.asarray(alpha_deg, dtype=float) % 360.0)
    dec = np.deg2rad(np.clip(np.asarray(delta_deg, dtype=float), -90.0, 90.0))
    cos_dec = np.cos(dec)
    return np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))


def add_fiber_collision_crowding_context(raw, deps, aux):
    n = len(raw)
    out = pd.DataFrame(index=raw.index)

    if n == 0 or "alpha" not in raw.columns or "delta" not in raw.columns:
        return out

    context = _context_frame(raw, aux)
    alpha = pd.to_numeric(raw["alpha"], errors="coerce").to_numpy(dtype=float)
    delta = pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float)
    ctx_alpha = pd.to_numeric(context["alpha"], errors="coerce").to_numpy(dtype=float)
    ctx_delta = pd.to_numeric(context["delta"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(alpha) & np.isfinite(delta)
    ctx_valid = np.isfinite(ctx_alpha) & np.isfinite(ctx_delta)

    counts = {r: np.zeros(n, dtype=float) for r in CROWDING_RADII_ARCSEC}
    count_arrays = [counts[r] for r in CROWDING_RADII_ARCSEC]
    nearest = np.full(n, 120.0, dtype=float)
    second_nearest = np.full(n, 120.0, dtype=float)
    has_neighbor_120 = np.zeros(n, dtype=bool)

    redshift = _as_numeric(raw, "redshift")
    rmag = _as_numeric(raw, "r")
    color_values = {}
    for a, b in COLOR_PAIRS:
        color_values[a + "_" + b] = _as_numeric(raw, a) - _as_numeric(raw, b)

    redshift_center = _finite_quantile(redshift, 0.5, 0.0)
    rmag_center = _finite_quantile(rmag, 0.5, 0.0)
    dz_med = np.full(n, _finite_quantile(np.abs(redshift - redshift_center), 0.95, 1.0), dtype=float)
    dr_med = np.full(n, _finite_quantile(np.abs(rmag - rmag_center), 0.95, 5.0), dtype=float)
    color_med = {}
    for name, vals in color_values.items():
        center = _finite_quantile(vals, 0.5, 0.0)
        color_med[name] = np.full(n, _finite_quantile(np.abs(vals - center), 0.95, 2.0), dtype=float)

    if np.any(valid) and np.any(ctx_valid):
        query_coords = _unit_sphere_coords(alpha[valid], delta[valid])
        context_coords = _unit_sphere_coords(ctx_alpha[ctx_valid], ctx_delta[ctx_valid])

        radii_arcsec = np.asarray(CROWDING_RADII_ARCSEC, dtype=float)
        radius_chords = 2.0 * np.sin((radii_arcsec / ARCSEC_PER_RADIAN) / 2.0)
        max_radius_chord = float(radius_chords[-1])

        tree = KDTree(context_coords, metric="euclidean", leaf_size=64)
        valid_positions = np.flatnonzero(valid)

        ctx_raw_valid_positions = np.flatnonzero(ctx_valid[:n])
        raw_ctx_lookup = np.full(n, -1, dtype=int)
        raw_ctx_lookup[ctx_raw_valid_positions] = np.arange(ctx_raw_valid_positions.size)

        raw_counts_120 = tree.query_radius(query_coords, r=max_radius_chord, count_only=True).astype(float)
        self_mask = raw_ctx_lookup[valid_positions] >= 0
        raw_counts_120[self_mask] -= 1.0
        raw_counts_120 = np.maximum(raw_counts_120, 0.0)

        count_arrays[-1][valid_positions] = raw_counts_120
        has_neighbor_120[valid_positions] = raw_counts_120 > 0.0

        candidate_local = np.flatnonzero(raw_counts_120 > 0.0)
        if candidate_local.size > 0:
            candidate_positions = valid_positions[candidate_local]
            neighbor_indices, neighbor_dists = tree.query_radius(
                query_coords[candidate_local],
                r=max_radius_chord,
                return_distance=True,
                sort_results=False,
            )

            valid_context_original = np.flatnonzero(ctx_valid)
            default_series = pd.Series(np.nan, index=context.index)
            ctx_redshift = pd.to_numeric(context.get("redshift", default_series), errors="coerce").to_numpy(dtype=float)
            ctx_rmag = pd.to_numeric(context.get("r", default_series), errors="coerce").to_numpy(dtype=float)
            ctx_colors = {}
            for a, b in COLOR_PAIRS:
                ctx_a = pd.to_numeric(context.get(a, default_series), errors="coerce").to_numpy(dtype=float)
                ctx_b = pd.to_numeric(context.get(b, default_series), errors="coerce").to_numpy(dtype=float)
                ctx_colors[a + "_" + b] = ctx_a - ctx_b

            for local_i, raw_i in enumerate(candidate_positions):
                inds = neighbor_indices[local_i]
                dists = neighbor_dists[local_i]

                self_ctx = raw_ctx_lookup[raw_i]
                if self_ctx >= 0:
                    keep = inds != self_ctx
                    inds = inds[keep]
                    dists = dists[keep]

                if len(dists) == 0:
                    continue

                sorted_chord = np.sort(dists)
                count_values = np.searchsorted(sorted_chord, radius_chords, side="right")
                for idx, arr in enumerate(count_arrays):
                    arr[raw_i] = float(count_values[idx])

                nearest_chord = float(sorted_chord[0])
                nearest[raw_i] = min(2.0 * np.arcsin(np.clip(0.5 * nearest_chord, 0.0, 1.0)) * ARCSEC_PER_RADIAN, 120.0)
                if sorted_chord.size > 1:
                    second_chord = float(sorted_chord[1])
                    second_nearest[raw_i] = min(2.0 * np.arcsin(np.clip(0.5 * second_chord, 0.0, 1.0)) * ARCSEC_PER_RADIAN, 120.0)

                original_inds = valid_context_original[inds]
                dz = np.abs(ctx_redshift[original_inds] - redshift[raw_i])
                dr = np.abs(ctx_rmag[original_inds] - rmag[raw_i])
                if np.isfinite(dz).any():
                    dz_med[raw_i] = float(np.nanmedian(dz))
                if np.isfinite(dr).any():
                    dr_med[raw_i] = float(np.nanmedian(dr))
                for name, vals in color_values.items():
                    dc = np.abs(ctx_colors[name][original_inds] - vals[raw_i])
                    if np.isfinite(dc).any():
                        color_med[name][raw_i] = float(np.nanmedian(dc))

    for radius in CROWDING_RADII_ARCSEC:
        label = str(int(radius))
        out["log1p_neighbors_within_" + label + "arcsec"] = np.log1p(counts[radius])
        out["has_neighbor_within_" + label + "arcsec"] = counts[radius] > 0

    out["nearest_neighbor_arcsec_capped_120"] = nearest
    out["second_nearest_neighbor_arcsec_capped_120"] = second_nearest
    out["log_nearest_over_55arcsec"] = np.log((nearest + 0.25) / 55.0)
    out["log_second_nearest_over_55arcsec"] = np.log((second_nearest + 0.25) / 55.0)
    out["margin_to_3arcsec"] = 3.0 - nearest
    out["margin_to_55arcsec"] = 55.0 - nearest
    out["margin_to_62arcsec"] = 62.0 - nearest
    out["neighbor_count_55_to_120arcsec_log1p"] = np.log1p(np.maximum(counts[120.0] - counts[55.0], 0.0))
    out["neighbor_count_62_to_120arcsec_log1p"] = np.log1p(np.maximum(counts[120.0] - counts[62.0], 0.0))
    out["has_any_neighbor_within_120arcsec"] = has_neighbor_120
    out["median_abs_redshift_diff_neighbors_120arcsec"] = dz_med
    out["median_abs_rmag_diff_neighbors_120arcsec"] = dr_med

    for name in sorted(color_med):
        out["median_abs_" + name + "_diff_neighbors_120arcsec"] = color_med[name]

    return out


FEATURE_GROUPS = [
    {
        "name": "fiber_collision_crowding_context",
        "fn": add_fiber_collision_crowding_context,
        "depends_on": [],
        "description": "Fiber-scale angular neighbor counts, collision-threshold margins, and companion-similarity summaries from unlabeled sky-position context.",
    }
]