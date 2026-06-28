import numpy as np
import pandas as pd


BANDS = ("u", "g", "r", "i", "z")
BETAS = {
    "u": 1.4e-10,
    "g": 0.9e-10,
    "r": 1.2e-10,
    "i": 1.8e-10,
    "z": 7.4e-10,
}
TRAIN_ID_MAX = 577346
EPS = 1e-12


def _training_mask(raw, aux):
    if aux is not None and len(aux) == len(raw):
        for col in ("is_train", "train", "train_mask"):
            if col in aux.columns:
                return aux[col].astype(bool).to_numpy()
        for col in ("is_test", "test", "test_mask"):
            if col in aux.columns:
                return ~aux[col].astype(bool).to_numpy()
        for col in ("split", "dataset", "fold_role"):
            if col in aux.columns:
                vals = aux[col].astype(str).str.lower()
                if vals.isin(("train", "training", "fit")).any():
                    return vals.isin(("train", "training", "fit")).to_numpy()

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        if ids.notna().all() and ids.max() > TRAIN_ID_MAX:
            return (ids <= TRAIN_ID_MAX).to_numpy()

    return np.ones(len(raw), dtype=bool)


def _longest_run(mask):
    best = 0
    cur = 0
    for val in mask:
        if val:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def add_asinh_censoring_regime_geometry(raw, deps, aux):
    out = pd.DataFrame(index=raw.index)
    n = len(raw)
    train_mask = _training_mask(raw, aux)

    flux = {}
    flux_w = {}
    regimes = {}
    m0 = {}
    m10 = {}

    log_factor = np.log(10.0) / 2.5

    for band in BANDS:
        beta = BETAS[band]
        mag = pd.to_numeric(raw[band], errors="coerce").astype(float).to_numpy()

        m0_b = -(2.5 / np.log(10.0)) * np.log(beta)
        m10_b = -(2.5 / np.log(10.0)) * (np.arcsinh(5.0) + np.log(beta))
        x_b = 2.0 * beta * np.sinh(-log_factor * mag - np.log(beta))
        q_b = x_b / (2.0 * beta)

        r_b = np.where(q_b >= 5.0, 2, np.where(q_b >= 0.0, 1, 0)).astype(np.int8)

        train_vals = x_b[train_mask & np.isfinite(x_b)]
        if train_vals.size:
            lo = np.nanquantile(train_vals, 0.001)
            hi = np.nanquantile(train_vals, 0.999)
            if not np.isfinite(lo):
                lo = np.nanmin(train_vals)
            if not np.isfinite(hi):
                hi = np.nanmax(train_vals)
            if lo > hi:
                lo, hi = hi, lo
        else:
            lo = np.nanmin(x_b) if np.isfinite(x_b).any() else 0.0
            hi = np.nanmax(x_b) if np.isfinite(x_b).any() else 0.0

        x_w = np.clip(x_b, lo, hi)

        flux[band] = x_b
        flux_w[band] = x_w
        regimes[band] = r_b
        m0[band] = m0_b
        m10[band] = m10_b

        out[f"{band}_regime"] = r_b
        out[f"{band}_regime_censored"] = (r_b == 0).astype(np.int8)
        out[f"{band}_regime_low"] = (r_b == 1).astype(np.int8)
        out[f"{band}_regime_reliable"] = (r_b == 2).astype(np.int8)
        out[f"{band}_m10_margin"] = m10_b - mag
        out[f"{band}_m0_margin"] = m0_b - mag
        out[f"{band}_boundary_softness"] = 1.0 / (1.0 + np.exp(-np.clip((m10_b - mag) / 0.10, -50.0, 50.0)))
        denom = m0_b - m10_b
        out[f"{band}_censor_depth_ratio"] = np.clip((mag - m10_b) / denom, 0.0, 1.0)

    rmat = np.column_stack([regimes[b] for b in BANDS])

    out["regime_count_censored"] = (rmat == 0).sum(axis=1)
    out["regime_count_low"] = (rmat == 1).sum(axis=1)
    out["regime_count_reliable"] = (rmat == 2).sum(axis=1)

    pattern_code = np.zeros(n, dtype=np.int16)
    power = 1
    for band in BANDS:
        pattern_code += regimes[band].astype(np.int16) * power
        power *= 3
    out["regime_pattern_code"] = pattern_code

    out["u_low_red_reliable"] = ((rmat[:, 0] <= 1) & (rmat[:, 1:] == 2).all(axis=1)).astype(np.int8)
    out["ug_low_riz_reliable"] = ((rmat[:, 0] <= 1) & (rmat[:, 1] <= 1) & (rmat[:, 2:] == 2).all(axis=1)).astype(np.int8)
    out["u_censored_red_positive"] = ((rmat[:, 0] == 0) & (rmat[:, 1:] > 0).all(axis=1)).astype(np.int8)
    out["ug_censored_riz_positive"] = ((rmat[:, 0] == 0) & (rmat[:, 1] == 0) & (rmat[:, 2:] > 0).all(axis=1)).astype(np.int8)
    out["z_low_blue_reliable"] = ((rmat[:, 4] <= 1) & (rmat[:, :4] == 2).all(axis=1)).astype(np.int8)
    out["iz_low_ugr_reliable"] = ((rmat[:, 3] <= 1) & (rmat[:, 4] <= 1) & (rmat[:, :3] == 2).all(axis=1)).astype(np.int8)
    out["z_censored_blue_positive"] = ((rmat[:, 4] == 0) & (rmat[:, :4] > 0).all(axis=1)).astype(np.int8)
    out["iz_censored_ugr_positive"] = ((rmat[:, 3] == 0) & (rmat[:, 4] == 0) & (rmat[:, :3] > 0).all(axis=1)).astype(np.int8)

    reliable = rmat == 2
    any_reliable = reliable.any(axis=1)
    out["first_reliable_band_index"] = np.where(any_reliable, reliable.argmax(axis=1), -1)
    out["last_reliable_band_index"] = np.where(any_reliable, 4 - reliable[:, ::-1].argmax(axis=1), -1)

    leading = np.zeros(n, dtype=np.int8)
    trailing = np.zeros(n, dtype=np.int8)
    longest_low_or_censored = np.zeros(n, dtype=np.int8)
    longest_censored = np.zeros(n, dtype=np.int8)
    for j in range(n):
        row = rmat[j]
        leading[j] = 0
        for val in row:
            if val <= 1:
                leading[j] += 1
            else:
                break
        trailing[j] = 0
        for val in row[::-1]:
            if val <= 1:
                trailing[j] += 1
            else:
                break
        longest_low_or_censored[j] = _longest_run(row <= 1)
        longest_censored[j] = _longest_run(row == 0)

    out["leading_censored_run_length"] = leading
    out["trailing_censored_run_length"] = trailing
    out["longest_low_or_censored_run"] = longest_low_or_censored
    out["longest_censored_run"] = longest_censored
    out["regime_transition_count"] = (rmat[:, 1:] != rmat[:, :-1]).sum(axis=1)

    bluer_censored_redder_detected = np.zeros(n, dtype=np.int8)
    bluer_detected_redder_censored = np.zeros(n, dtype=np.int8)
    for i in range(len(BANDS)):
        for j in range(i + 1, len(BANDS)):
            bluer_censored_redder_detected += ((rmat[:, i] <= 1) & (rmat[:, j] == 2)).astype(np.int8)
            bluer_detected_redder_censored += ((rmat[:, i] == 2) & (rmat[:, j] <= 1)).astype(np.int8)
    out["bluer_censored_redder_detected_pairs"] = bluer_censored_redder_detected
    out["bluer_detected_redder_censored_pairs"] = bluer_detected_redder_censored

    log_flux = {}
    valid = {}
    for band in BANDS:
        valid[band] = regimes[band] > 0
        log_flux[band] = np.log10(np.maximum(flux_w[band], 0.0) + EPS)

    slopes = {}
    for left, right in zip(BANDS[:-1], BANDS[1:]):
        ok = valid[left] & valid[right]
        slope = np.where(ok, log_flux[right] - log_flux[left], 0.0)
        name = f"log_flux_slope_{left}_{right}"
        slopes[name] = slope
        out[name] = slope
        out[f"{name}_valid"] = ok.astype(np.int8)

    slope_names = list(slopes)
    for left_name, right_name in zip(slope_names[:-1], slope_names[1:]):
        left_pair = left_name.replace("log_flux_slope_", "")
        right_pair = right_name.replace("log_flux_slope_", "")
        curv_name = f"log_flux_curvature_{left_pair}_{right_pair}"
        ok = (out[f"{left_name}_valid"].to_numpy() == 1) & (out[f"{right_name}_valid"].to_numpy() == 1)
        out[curv_name] = np.where(ok, slopes[right_name] - slopes[left_name], 0.0)
        out[f"{curv_name}_valid"] = ok.astype(np.int8)

    positive_flux = np.column_stack([
        np.where(regimes[b] > 0, np.maximum(flux_w[b], 0.0), 0.0)
        for b in BANDS
    ])
    total_flux = positive_flux.sum(axis=1)
    p = np.divide(
        positive_flux,
        total_flux[:, None],
        out=np.zeros_like(positive_flux, dtype=float),
        where=total_flux[:, None] > 0.0,
    )

    for idx, band in enumerate(BANDS):
        out[f"{band}_positive_flux_allocation"] = p[:, idx]

    entropy = -(p * np.log(np.clip(p, EPS, 1.0))).sum(axis=1) / np.log(float(len(BANDS)))
    indices = np.arange(len(BANDS), dtype=float)
    out["positive_flux_entropy"] = np.where(total_flux > 0.0, entropy, 0.0)
    out["positive_flux_centroid"] = np.where(total_flux > 0.0, (p * indices).sum(axis=1), -1.0)
    out["positive_flux_concentration"] = np.where(total_flux > 0.0, p.max(axis=1), 0.0)
    out["positive_flux_band_count"] = (positive_flux > 0.0).sum(axis=1)

    return out


FEATURE_GROUPS = [
    {
        "name": "asinh_censoring_regime_geometry",
        "fn": add_asinh_censoring_regime_geometry,
        "depends_on": [],
        "description": "Encodes asinh-photometry detection regimes, dropout topology, and censored SED geometry across ugriz bands.",
    }
]