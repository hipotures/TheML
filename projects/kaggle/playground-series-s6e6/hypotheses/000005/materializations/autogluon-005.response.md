import numpy as np
import pandas as pd


COLOR_PAIRS = (("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"))
COLOR_NAMES = ("ug", "gr", "ri", "iz")
SPECTRAL_COL = "spectral_type"
POPULATION_COL = "galaxy_population"
SCALE_FLOOR = 0.02


def _robust_stats(frame, group_cols, color_cols):
    grouped = frame.groupby(group_cols, observed=True, sort=False)
    med = grouped[list(color_cols)].median()
    mad = grouped[list(color_cols)].agg(
        lambda x: float(np.median(np.abs(x.to_numpy(dtype=float) - np.median(x.to_numpy(dtype=float)))))
    )
    cnt = grouped.size().rename("count")
    return med, mad, cnt


def _aligned_group_values(keys, stats, count, color_cols, prefix):
    mu = stats.reindex(keys).reset_index(drop=True)
    scale = count.reindex(keys).reset_index(drop=True)
    mu.columns = [f"{prefix}_mu_{c}" for c in color_cols]
    return mu, scale


def add_catalog_template_residuals(raw, deps, aux):
    index = raw.index
    colors = pd.DataFrame(index=index)

    for name, (left, right) in zip(COLOR_NAMES, COLOR_PAIRS):
        colors[name] = raw[left].astype(float) - raw[right].astype(float)

    work = pd.concat(
        [
            raw[[SPECTRAL_COL, POPULATION_COL]].reset_index(drop=True),
            colors.reset_index(drop=True),
        ],
        axis=1,
    )

    global_mu = colors.median(axis=0)
    global_mad = colors.apply(
        lambda x: float(np.median(np.abs(x.to_numpy(dtype=float) - np.median(x.to_numpy(dtype=float))))),
        axis=0,
    ).clip(lower=SCALE_FLOOR)

    cell_med, cell_mad, cell_count = _robust_stats(work, [SPECTRAL_COL, POPULATION_COL], COLOR_NAMES)
    sp_med, sp_mad, sp_count = _robust_stats(work, [SPECTRAL_COL], COLOR_NAMES)
    pop_med, pop_mad, pop_count = _robust_stats(work, [POPULATION_COL], COLOR_NAMES)

    cell_keys = pd.MultiIndex.from_frame(work[[SPECTRAL_COL, POPULATION_COL]])
    sp_keys = pd.Index(work[SPECTRAL_COL])
    pop_keys = pd.Index(work[POPULATION_COL])

    cell_mu = cell_med.reindex(cell_keys).reset_index(drop=True)
    cell_s = cell_mad.reindex(cell_keys).reset_index(drop=True)
    n_cell = cell_count.reindex(cell_keys).reset_index(drop=True).fillna(0.0).to_numpy(dtype=float)

    sp_mu = sp_med.reindex(sp_keys).reset_index(drop=True)
    sp_s = sp_mad.reindex(sp_keys).reset_index(drop=True)
    n_sp = sp_count.reindex(sp_keys).reset_index(drop=True).fillna(0.0).to_numpy(dtype=float)

    pop_mu = pop_med.reindex(pop_keys).reset_index(drop=True)
    pop_s = pop_mad.reindex(pop_keys).reset_index(drop=True)

    global_mu_frame = pd.DataFrame(
        np.tile(global_mu.to_numpy(dtype=float), (len(raw), 1)),
        columns=COLOR_NAMES,
    )
    global_s_frame = pd.DataFrame(
        np.tile(global_mad.to_numpy(dtype=float), (len(raw), 1)),
        columns=COLOR_NAMES,
    )

    sp_mu = sp_mu.where(sp_mu.notna(), pop_mu).where(lambda x: x.notna(), global_mu_frame)
    sp_s = sp_s.where(sp_s.notna(), pop_s).where(lambda x: x.notna(), global_s_frame)

    cell_mu = cell_mu.where(cell_mu.notna(), sp_mu)
    cell_s = cell_s.where(cell_s.notna(), sp_s)

    w_cell = np.clip((n_cell - 500.0) / 1500.0, 0.0, 1.0).reshape(-1, 1)
    w_sp = np.clip((n_sp - 1000.0) / 3000.0, 0.0, 1.0).reshape(-1, 1)

    backed_mu = (w_sp * sp_mu.to_numpy(dtype=float)) + ((1.0 - w_sp) * global_mu_frame.to_numpy(dtype=float))
    backed_s = (w_sp * sp_s.to_numpy(dtype=float)) + ((1.0 - w_sp) * global_s_frame.to_numpy(dtype=float))

    template_mu = (w_cell * cell_mu.to_numpy(dtype=float)) + ((1.0 - w_cell) * backed_mu)
    template_s = (w_cell * cell_s.to_numpy(dtype=float)) + ((1.0 - w_cell) * backed_s)
    template_s = np.maximum(template_s, SCALE_FLOOR)

    z = (colors.to_numpy(dtype=float) - template_mu) / template_s
    z = np.clip(z, -10.0, 10.0)
    abs_z = np.abs(z)

    out = pd.DataFrame(index=index)
    for pos, name in enumerate(COLOR_NAMES):
        out[f"z_{name}"] = z[:, pos]

    out["mean_abs_z"] = abs_z.mean(axis=1)
    out["median_abs_z"] = np.median(abs_z, axis=1)
    out["max_abs_z"] = abs_z.max(axis=1)
    out["l1_abs_z"] = abs_z.sum(axis=1)
    out["l2_z"] = np.sqrt(np.square(z).sum(axis=1))
    out["residual_sum_squares"] = np.square(z).sum(axis=1)
    out["count_abs_z_gt_2"] = (abs_z > 2.0).sum(axis=1).astype(np.int8)
    out["count_abs_z_gt_3"] = (abs_z > 3.0).sum(axis=1).astype(np.int8)

    return out


FEATURE_GROUPS = [
    {
        "name": "catalog_template_residuals",
        "fn": add_catalog_template_residuals,
        "depends_on": [],
        "description": "Robust tag-conditioned ugriz color residuals and aggregate mismatch scores against shrunk catalog templates.",
    }
]