import numpy as np
import pandas as pd

_BAND_COLUMNS = ("u", "g", "r", "i", "z")
_WAVELENGTH_NM = (355.1, 468.6, 616.5, 748.1, 893.1)
_ALPHA = 1e-12
_EPS = 1e-12
_Z_LOW = -0.99
_Z_HIGH = 20.0


def add_flux_allocation_entropy(raw, deps, aux):
    n = raw.shape[0]
    mags = raw.loc[:, list(_BAND_COLUMNS)].to_numpy(dtype=float)
    shares = np.full((n, len(_BAND_COLUMNS)), 0.2, dtype=float)
    invalid_share = np.zeros(n, dtype=bool)

    finite_rows = np.isfinite(mags).all(axis=1)
    if finite_rows.any():
        valid_idx = np.nonzero(finite_rows)[0]
        l = -0.4 * mags[valid_idx]
        l = l - np.max(l, axis=1, keepdims=True)
        q = np.exp(np.log(10.0) * l)
        q_sum = np.sum(q, axis=1)
        calc_ok = np.isfinite(q_sum) & (q_sum > 0.0)

        if (~calc_ok).any():
            invalid_share[valid_idx[~calc_ok]] = True

        if calc_ok.any():
            ok_idx = valid_idx[calc_ok]
            shares[ok_idx] = (q[calc_ok] + _ALPHA) / (q_sum[calc_ok, None] + 5.0 * _ALPHA)

    if (~finite_rows).any():
        invalid_share[~finite_rows] = True

    share_u = shares[:, 0]
    share_g = shares[:, 1]
    share_r = shares[:, 2]
    share_i = shares[:, 3]
    share_z = shares[:, 4]

    sorted_shares = np.sort(shares, axis=1)[:, ::-1]
    p1 = sorted_shares[:, 0]
    p2 = sorted_shares[:, 1]

    entropy = -np.sum(np.where(shares > 0.0, shares * np.log(shares), 0.0), axis=1)
    entropy_norm = entropy / np.log(len(_BAND_COLUMNS))
    simpson = np.sum(shares * shares, axis=1)
    gini = 1.0 - simpson
    neff = 1.0 / (simpson + _EPS)
    top_ratio = p1 / (p2 + _EPS)
    top_gap = p1 - p2

    bucket_b = share_u + share_g
    bucket_m = share_g + share_r
    bucket_r = share_i + share_z
    delta_br = bucket_b - bucket_r
    delta_ui = share_u - share_i
    delta_gr = share_g - share_r
    delta_zi = share_z - share_g

    x = np.log(np.array(_WAVELENGTH_NM, dtype=float))
    mu = np.sum(shares * x, axis=1)
    dx = x[None, :] - mu[:, None]
    var = np.sum(shares * (dx ** 2), axis=1)
    sigma = np.maximum(np.sqrt(var), _EPS)
    skew = np.sum(shares * (dx ** 3), axis=1) / (sigma ** 3)
    kurt = np.sum(shares * (dx ** 4), axis=1) / (sigma ** 4) - 3.0
    skew = np.clip(skew, -8.0, 8.0)
    kurt = np.clip(kurt, -8.0, 8.0)

    x_min = x.min()
    x_max = x.max()
    x_span = x_max - x_min
    centroid_ratio = mu / x_max
    scale_ratio = var / (x_span ** 2)

    z = raw["redshift"].to_numpy(dtype=float)
    z_clamp = np.clip(z, _Z_LOW, _Z_HIGH)
    log_shift = np.log1p(z_clamp)
    x_rf = x[None, :] - log_shift[:, None]

    mu_rf = np.sum(shares * x_rf, axis=1)
    dx_rf = x_rf - mu_rf[:, None]
    var_rf = np.sum(shares * (dx_rf ** 2), axis=1)
    sigma_rf = np.maximum(np.sqrt(var_rf), _EPS)
    skew_rf = np.sum(shares * (dx_rf ** 3), axis=1) / (sigma_rf ** 3)
    kurt_rf = np.sum(shares * (dx_rf ** 4), axis=1) / (sigma_rf ** 4) - 3.0
    skew_rf = np.clip(skew_rf, -8.0, 8.0)
    kurt_rf = np.clip(kurt_rf, -8.0, 8.0)

    dmu = mu - mu_rf
    dv = var - var_rf

    def _clean(arr):
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    entropy = _clean(entropy)
    entropy_norm = _clean(entropy_norm)
    simpson = _clean(simpson)
    gini = _clean(gini)
    neff = _clean(neff)
    top_ratio = _clean(top_ratio)
    top_gap = _clean(top_gap)
    bucket_b = _clean(bucket_b)
    bucket_m = _clean(bucket_m)
    bucket_r = _clean(bucket_r)
    delta_br = _clean(delta_br)
    delta_ui = _clean(delta_ui)
    delta_gr = _clean(delta_gr)
    delta_zi = _clean(delta_zi)
    mu = _clean(mu)
    var = _clean(var)
    skew = _clean(skew)
    kurt = _clean(kurt)
    centroid_ratio = _clean(centroid_ratio)
    scale_ratio = _clean(scale_ratio)
    mu_rf = _clean(mu_rf)
    var_rf = _clean(var_rf)
    skew_rf = _clean(skew_rf)
    kurt_rf = _clean(kurt_rf)
    dmu = _clean(dmu)
    dv = _clean(dv)

    return pd.DataFrame(
        {
            "flux_share_u": share_u,
            "flux_share_g": share_g,
            "flux_share_r": share_r,
            "flux_share_i": share_i,
            "flux_share_z": share_z,
            "flux_entropy": entropy,
            "flux_entropy_norm": entropy_norm,
            "flux_simpson": simpson,
            "flux_gini": gini,
            "flux_neff": neff,
            "flux_top_ratio": top_ratio,
            "flux_top_gap": top_gap,
            "flux_bucket_b": bucket_b,
            "flux_bucket_m": bucket_m,
            "flux_bucket_r": bucket_r,
            "flux_delta_br": delta_br,
            "flux_delta_ui": delta_ui,
            "flux_delta_gr": delta_gr,
            "flux_delta_zi": delta_zi,
            "flux_log_lambda_mean": mu,
            "flux_log_lambda_var": var,
            "flux_log_lambda_skew": skew,
            "flux_log_lambda_kurtosis": kurt,
            "flux_log_lambda_centroid_ratio": centroid_ratio,
            "flux_log_lambda_scale_ratio": scale_ratio,
            "flux_log_lambda_mean_rf": mu_rf,
            "flux_log_lambda_var_rf": var_rf,
            "flux_log_lambda_skew_rf": skew_rf,
            "flux_log_lambda_kurtosis_rf": kurt_rf,
            "flux_delta_mu": dmu,
            "flux_delta_var": dv,
            "flux_invalid_share": invalid_share,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "flux_allocation_entropy",
        "fn": add_flux_allocation_entropy,
        "depends_on": [],
        "description": "Builds rest-frame-aware, five-band normalized flux-share descriptors with concentration, asymmetry, and moment features for SED-shape characterization.",
    }
]