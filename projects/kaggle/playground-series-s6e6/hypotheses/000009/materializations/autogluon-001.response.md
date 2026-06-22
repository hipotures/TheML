import numpy as np
import pandas as pd


def add_canonical_locus_coordinates(raw, deps, aux):
    index = raw.index
    u = raw["u"].to_numpy(dtype="float64")
    g = raw["g"].to_numpy(dtype="float64")
    r = raw["r"].to_numpy(dtype="float64")
    i = raw["i"].to_numpy(dtype="float64")
    z = raw["z"].to_numpy(dtype="float64")

    ug = pd.Series(u - g, index=index, name="ug")
    gr = pd.Series(g - r, index=index, name="gr")
    ri = pd.Series(r - i, index=index, name="ri")
    iz = pd.Series(i - z, index=index, name="iz")

    s = pd.Series(-0.249 * u + 0.794 * g - 0.555 * r + 0.234, index=index, name="s")
    w = pd.Series(-0.227 * g + 0.792 * r - 0.567 * i + 0.050, index=index, name="w")
    x = pd.Series(0.707 * g - 0.707 * r - 0.988, index=index, name="x")
    y = pd.Series(-0.270 * r + 0.800 * i - 0.534 * z + 0.054, index=index, name="y")

    p1_s = pd.Series(0.910 * u - 0.495 * g - 0.415 * r - 1.280, index=index, name="p1_s")
    p1_w = pd.Series(0.928 * g - 0.556 * r - 0.372 * i - 0.425, index=index, name="p1_w")
    p1_x = pd.Series(r - i, index=index, name="p1_x")
    p1_y = pd.Series(0.895 * r - 0.448 * i - 0.447 * z - 0.600, index=index, name="p1_y")

    ns = (-0.249 * u + 0.794 * g - 0.555 * r + 0.234) / 0.031
    nw = (-0.227 * g + 0.792 * r - 0.567 * i + 0.050) / 0.025
    nx = (0.707 * g - 0.707 * r - 0.988) / 0.042
    ny = (-0.270 * r + 0.800 * i - 0.534 * z + 0.054) / 0.023

    ns = pd.Series(np.clip(ns, -12.0, 12.0), index=index, name="ns")
    nw = pd.Series(np.clip(nw, -12.0, 12.0), index=index, name="nw")
    nx = pd.Series(np.clip(nx, -12.0, 12.0), index=index, name="nx")
    ny = pd.Series(np.clip(ny, -12.0, 12.0), index=index, name="ny")

    active_s = (r <= 19.0) & (p1_s >= -0.2) & (p1_s <= 0.8)
    active_w = (r <= 20.0) & (p1_w >= -0.2) & (p1_w <= 0.6)
    active_x = (r <= 19.0) & (p1_x >= 0.8) & (p1_x <= 1.6)
    active_y = (r <= 19.5) & (p1_y >= 0.1) & (p1_y <= 1.2)

    active_count = (active_s.astype(int) + active_w.astype(int) + active_x.astype(int) + active_y.astype(int)).astype(int)
    active_axis_count = pd.Series(active_count, index=index, name="active_axis_count")

    deviations = np.column_stack([ns.to_numpy(), nw.to_numpy(), nx.to_numpy(), ny.to_numpy()])
    abs_devs = np.abs(deviations)

    active_mask = np.column_stack([active_s.to_numpy(), active_w.to_numpy(), active_x.to_numpy(), active_y.to_numpy()])
    selected = abs_devs.copy()
    selected[~active_mask] = np.nan

    fallback = active_count == 0
    if np.any(fallback):
        selected[fallback, :] = abs_devs[fallback, :]

    min_abs = pd.Series(np.nanmin(selected, axis=1), index=index, name="stellar_locus_min_abs_dev")
    mean_abs = pd.Series(np.nanmean(selected, axis=1), index=index, name="stellar_locus_mean_abs_dev")
    max_abs = pd.Series(np.nanmax(selected, axis=1), index=index, name="stellar_locus_max_abs_dev")

    within_1sigma = selected <= 1.0
    within_2sigma = selected <= 2.0
    beyond_4sigma = selected > 4.0

    count_within_1sigma = pd.Series(np.nansum(within_1sigma, axis=1), index=index, name="stellar_locus_count_within_1sigma").astype(int)
    count_within_2sigma = pd.Series(np.nansum(within_2sigma, axis=1), index=index, name="stellar_locus_count_within_2sigma").astype(int)
    count_beyond_4sigma = pd.Series(np.nansum(beyond_4sigma, axis=1), index=index, name="stellar_locus_count_beyond_4sigma").astype(int)

    c_perp = pd.Series(ri.to_numpy() - gr.to_numpy() / 4.0 - 0.18, index=index, name="c_perp")
    c_par = pd.Series(0.7 * gr.to_numpy() + 1.2 * (ri.to_numpy() - 0.18), index=index, name="c_par")
    abs_c_perp = pd.Series(np.abs(c_perp.to_numpy()), index=index, name="abs_c_perp")

    return pd.DataFrame(
        {
            "ug": ug,
            "gr": gr,
            "ri": ri,
            "iz": iz,
            "s": s,
            "w": w,
            "x": x,
            "y": y,
            "p1_s": p1_s,
            "p1_w": p1_w,
            "p1_x": p1_x,
            "p1_y": p1_y,
            "ns": ns,
            "nw": nw,
            "nx": nx,
            "ny": ny,
            "active_axis_count": active_axis_count,
            "stellar_locus_min_abs_dev": min_abs,
            "stellar_locus_mean_abs_dev": mean_abs,
            "stellar_locus_max_abs_dev": max_abs,
            "stellar_locus_count_within_1sigma": count_within_1sigma,
            "stellar_locus_count_within_2sigma": count_within_2sigma,
            "stellar_locus_count_beyond_4sigma": count_beyond_4sigma,
            "c_perp": c_perp,
            "c_par": c_par,
            "abs_c_perp": abs_c_perp,
        },
        index=index,
    )


FEATURE_GROUPS = [
    {
        "name": "canonical_locus_coordinates",
        "fn": add_canonical_locus_coordinates,
        "depends_on": [],
        "description": "Projects SDSS photometry into principal-color and red-galaxy-locus coordinates, computes normalized/active-deviation aggregates, and adds stellar- and galaxy-track proximity signals.",
    }
]