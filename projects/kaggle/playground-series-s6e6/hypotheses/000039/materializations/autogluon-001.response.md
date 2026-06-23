import numpy as np
import pandas as pd

REDSHIFT_BIN_EDGES = (-0.01, 0.2, 0.6, 1.2, 2.4, 4.0, 7.0)
N_ALPHA_BINS = 12
N_DELTA_BINS = 6
MIN_CONTEXT_SIZE = 300
MAD_SCALE = 1.4826
Z_SCORE_CLIP = 10.0
MAG_GUARD = 10000.0
EPSILON = 1e-6


def _get_series(frame, name):
    if name in frame.columns:
        return frame[name]
    return pd.Series(np.nan, index=frame.index)


def _to_numeric_series(frame, name):
    return pd.to_numeric(_get_series(frame, name), errors="coerce").pipe(lambda s: s.where(np.isfinite(s), np.nan))


def _clean_mag(frame, name):
    s = _to_numeric_series(frame, name)
    return s.where(s.between(-MAG_GUARD, MAG_GUARD), np.nan)


def _principal_colors(frame):
    idx = frame.index
    u = _clean_mag(frame, "u")
    g = _clean_mag(frame, "g")
    r = _clean_mag(frame, "r")
    i = _clean_mag(frame, "i")
    z = _clean_mag(frame, "z")

    out = pd.DataFrame(index=idx)
    out["s"] = -0.249 * u + 0.794 * g - 0.555 * r + 0.234
    out["w"] = -0.227 * g + 0.792 * r - 0.567 * i + 0.050
    out["x"] = 0.707 * g - 0.707 * r - 0.988
    out["y"] = -0.270 * r + 0.800 * i - 0.534 * z + 0.059
    out["l"] = -0.436 * u + 1.129 * g - 0.119 * r - 0.574 * i + 0.1984
    return out


def _assign_redshift_bin(frame):
    z = _to_numeric_series(frame, "redshift")
    z_arr = z.to_numpy(dtype=float)
    edges = np.array(REDSHIFT_BIN_EDGES, dtype=float)
    idx = np.zeros(len(frame), dtype=np.int16)
    valid = np.isfinite(z_arr)
    if valid.any():
        bins = np.searchsorted(edges, z_arr[valid], side="right") - 1
        bins = np.clip(bins, 0, len(edges) - 2)
        idx[valid] = bins
    return pd.Series(idx, index=frame.index, dtype=np.int16)


def _assign_sky_cell(frame):
    a = _to_numeric_series(frame, "alpha")
    d = _to_numeric_series(frame, "delta")
    a_arr = a.to_numpy(dtype=float)
    d_arr = d.to_numpy(dtype=float)

    a_bins = np.floor(np.mod(a_arr, 360.0) / 360.0 * float(N_ALPHA_BINS))
    a_bins = np.where(np.isfinite(a_arr), a_bins, 0.0)
    a_bins = np.clip(a_bins, 0, N_ALPHA_BINS - 1)

    d_bins = np.zeros(len(frame), dtype=float)
    valid_d = np.isfinite(d_arr)
    if valid_d.any():
        d_min = float(np.nanmin(d_arr[valid_d]))
        d_max = float(np.nanmax(d_arr[valid_d]))
        span = d_max - d_min
        if span > 0:
            d_bins[valid_d] = (d_arr[valid_d] - d_min) / span * float(N_DELTA_BINS)
    d_bins = np.where(valid_d, d_bins, 0.0)
    d_bins = np.floor(d_bins)
    d_bins = np.clip(d_bins, 0, N_DELTA_BINS - 1)

    return pd.Series((a_bins.astype(np.int16) * N_DELTA_BINS + d_bins.astype(np.int16)), index=frame.index, dtype=np.int16)


def _mad_frame(block):
    median = block.median()
    mad = (block - median).abs().median()
    return mad


def _group_stats(frame, keys, colors, suffix):
    grouped = frame.groupby(keys, sort=False, observed=True)
    med = grouped[colors].median().add_suffix(f"_med{suffix}")
    mad = grouped[colors].apply(_mad_frame).add_suffix(f"_mad{suffix}")
    counts = grouped.size().rename(f"count{suffix}")
    out = med.join(mad).join(counts)
    return out.reset_index()


def _global_reference_stats(base_colors, aux_colors):
    if aux_colors is None or aux_colors.empty:
        pool = [base_colors]
    else:
        pool = [base_colors, aux_colors]
    combined = pd.concat(pool, axis=0, ignore_index=True)
    med = combined.median()
    mad = (combined - med).abs().median()
    return med.fillna(0.0), mad.fillna(0.0)


def add_local_principal_color_residuals(raw, deps, aux):
    colors = ("s", "w", "x", "y", "l")

    principal = _principal_colors(raw)

    meta = pd.DataFrame(index=raw.index)
    meta["redshift_bin"] = _assign_redshift_bin(raw)
    meta["sky_cell"] = _assign_sky_cell(raw)

    st = _get_series(raw, "spectral_type").fillna("unknown").astype(object)
    gp = _get_series(raw, "galaxy_population").fillna("unknown").astype(object)
    meta["spectral_type"] = st
    meta["galaxy_population"] = gp

    feature_base = pd.concat([meta, principal], axis=1)

    full_keys = ["redshift_bin", "sky_cell", "spectral_type", "galaxy_population"]
    rz_tag_keys = ["redshift_bin", "spectral_type", "galaxy_population"]
    tag_keys = ["spectral_type", "galaxy_population"]

    full_stats = _group_stats(feature_base, full_keys, colors, "_full")
    rz_tag_stats = _group_stats(feature_base, rz_tag_keys, colors, "_zt")
    tag_stats = _group_stats(feature_base, tag_keys, colors, "_tag")

    merged = feature_base.merge(full_stats, on=full_keys, how="left", sort=False)
    merged = merged.merge(rz_tag_stats, on=rz_tag_keys, how="left", sort=False)
    merged = merged.merge(tag_stats, on=tag_keys, how="left", sort=False)

    aux_colors = None
    if isinstance(aux, pd.DataFrame) and not aux.empty:
        aux_colors = _principal_colors(aux)

    global_med, global_mad = _global_reference_stats(principal, aux_colors)

    out = pd.DataFrame(index=raw.index)

    abs_cols = []
    for c in colors:
        med = merged[f"{c}_med_full"].where(merged["count_full"] >= MIN_CONTEXT_SIZE)
        med = med.fillna(merged[f"{c}_med_zt"].where(merged["count_zt"] >= MIN_CONTEXT_SIZE))
        med = med.fillna(merged[f"{c}_med_tag"].where(merged["count_tag"] >= MIN_CONTEXT_SIZE))
        med = med.fillna(float(global_med[c]))

        mad = merged[f"{c}_mad_full"].where(merged["count_full"] >= MIN_CONTEXT_SIZE)
        mad = mad.fillna(merged[f"{c}_mad_zt"].where(merged["count_zt"] >= MIN_CONTEXT_SIZE))
        mad = mad.fillna(merged[f"{c}_mad_tag"].where(merged["count_tag"] >= MIN_CONTEXT_SIZE))
        mad = mad.fillna(float(global_mad[c]))

        z = (merged[c] - med) / (MAD_SCALE * mad + EPSILON)
        z = z.clip(-Z_SCORE_CLIP, Z_SCORE_CLIP)
        out[f"z_{c}"] = z
        out[f"abs_z_{c}"] = z.abs()
        abs_cols.append(f"abs_z_{c}")

    out["max_abs_z"] = out[abs_cols].max(axis=1)
    out["l2_z"] = np.sqrt(
        out["z_s"] ** 2 + out["z_w"] ** 2 + out["z_x"] ** 2 + out["z_y"] ** 2 + out["z_l"] ** 2
    )
    out["all_close"] = (out[abs_cols] < 0.8).all(axis=1)

    sign_s = (out["z_s"] >= 0).fillna(False)
    sign_w = (out["z_w"] >= 0).fillna(False)
    sign_x = (out["z_x"] >= 0).fillna(False)
    sign_y = (out["z_y"] >= 0).fillna(False)
    out["sign_bits_zwxy"] = (
        (sign_s.astype("uint8") << 3)
        | (sign_w.astype("uint8") << 2)
        | (sign_x.astype("uint8") << 1)
        | sign_y.astype("uint8")
    )
    out["sign_bit_s"] = sign_s.astype("uint8")
    out["sign_bit_w"] = sign_w.astype("uint8")
    out["sign_bit_x"] = sign_x.astype("uint8")
    out["sign_bit_y"] = sign_y.astype("uint8")

    return out


FEATURE_GROUPS = [
    {
        "name": "local_principal_color_residuals",
        "fn": add_local_principal_color_residuals,
        "depends_on": [],
        "description": "Build local principal-color residual z-scores over redshift/sky/context bins with hierarchical fallback to robust global statistics.",
    }
]