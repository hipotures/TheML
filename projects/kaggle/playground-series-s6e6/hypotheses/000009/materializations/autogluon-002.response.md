import numpy as np
import pandas as pd


LENS_STD_DIVISORS = (0.031, 0.025, 0.042, 0.023)
LENS_STD_CAP = 12.0


def add_canonical_locus_coordinates(raw, deps, aux):
    # Preserve required behavior; only local features are returned.
    _ = deps
    _ = aux

    u = raw["u"].to_numpy(dtype=float)
    g = raw["g"].to_numpy(dtype=float)
    r = raw["r"].to_numpy(dtype=float)
    i = raw["i"].to_numpy(dtype=float)
    z = raw["z"].to_numpy(dtype=float)

    # Color coordinates
    _ = u - g  # ug: computed per hypothesis (not retained as output)
    gr = g - r
    ri = r - i
    _ = i - z  # iz: computed per hypothesis (not retained as output)

    # Principal locus coordinates
    s = -0.249 * u + 0.794 * g - 0.555 * r + 0.234
    w = -0.227 * g + 0.792 * r - 0.567 * i + 0.050
    x = 0.707 * g - 0.707 * r - 0.988
    y = -0.270 * r + 0.800 * i - 0.534 * z + 0.054

    # SDSS axis-progress parameters
    p1_s = 0.910 * u - 0.495 * g - 0.415 * r - 1.280
    p1_w = 0.928 * g - 0.556 * r - 0.372 * i - 0.425
    p1_x = r - i
    p1_y = 0.895 * r - 0.448 * i - 0.447 * z - 0.600

    # Standardized distances
    ns = np.clip(s / LENS_STD_DIVISORS[0], -LENS_STD_CAP, LENS_STD_CAP)
    nw = np.clip(w / LENS_STD_DIVISORS[1], -LENS_STD_CAP, LENS_STD_CAP)
    nx = np.clip(x / LENS_STD_DIVISORS[2], -LENS_STD_CAP, LENS_STD_CAP)
    ny = np.clip(y / LENS_STD_DIVISORS[3], -LENS_STD_CAP, LENS_STD_CAP)

    # Ivezic-style axis validity windows
    ms = (r <= 19.0) & (p1_s >= -0.2) & (p1_s <= 0.8)
    mw = (r <= 20.0) & (p1_w >= -0.2) & (p1_w <= 0.6)
    mx = (r <= 19.0) & (p1_x >= 0.8) & (p1_x <= 1.6)
    my = (r <= 19.5) & (p1_y >= 0.1) & (p1_y <= 1.2)

    mask_matrix = np.column_stack((ms, mw, mx, my))
    nz_matrix = np.column_stack((ns, nw, nx, ny))
    active_count = mask_matrix.sum(axis=1)
    active_zero = active_count == 0

    nz_selected = np.where(mask_matrix, nz_matrix, np.nan)
    nz_effective = np.where(active_zero[:, None], nz_matrix, nz_selected)
    nz_abs = np.abs(nz_effective)

    feature_data = {
        "locus_ns": ns,
        "locus_nw": nw,
        "locus_nx": nx,
        "locus_ny": ny,
        "locus_mask_s": ms,
        "locus_mask_w": mw,
        "locus_mask_x": mx,
        "locus_mask_y": my,
        "locus_active_axes": active_count.astype(np.int8),
        "locus_active_ratio": active_count / 4.0,
        "locus_min_abs": np.nanmin(nz_abs, axis=1),
        "locus_mean_abs": np.nanmean(nz_abs, axis=1),
        "locus_max_abs": np.nanmax(nz_abs, axis=1),
        "locus_std_abs": np.nanstd(nz_abs, axis=1),
        "locus_signed_sum": np.nansum(nz_effective, axis=1),
        "locus_within1": np.sum(nz_abs <= 1.0, axis=1),
        "locus_within2": np.sum(nz_abs <= 2.0, axis=1),
        "locus_between2and4": np.sum((nz_abs > 2.0) & (nz_abs <= 4.0), axis=1),
        "locus_beyond4": np.sum(nz_abs > 4.0, axis=1),
    }

    c_perp = ri - gr / 4.0 - 0.18
    c_par = 0.7 * gr + 1.2 * (ri - 0.18)

    feature_data.update(
        {
            "c_perp": c_perp,
            "c_perp_abs": np.abs(c_perp),
            "c_perp_sign": np.sign(c_perp),
            "c_par": c_par,
        }
    )

    return pd.DataFrame(feature_data, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "canonical_locus_coordinates",
        "fn": add_canonical_locus_coordinates,
        "depends_on": [],
        "description": "Computes SDSS-based locus-projection distances, axis-validity aggregates, and red-galaxy track coordinates for astrophysical manifold-aware separation signals.",
    }
]