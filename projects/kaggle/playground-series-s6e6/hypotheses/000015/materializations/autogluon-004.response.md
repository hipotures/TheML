import numpy as np
import pandas as pd


_TEMPLATE_DOMAINS = (
    ("star", -0.004002769142205858, 0.004002769142205858),
    ("galaxy", -0.01, 1.0),
    ("qso", 0.0333, 7.0),
)

_BREAKPOINTS = (-0.01, 0.0, 0.0333, 1.0, 2.2, 3.0, 3.5, 4.5, 5.0, 7.0)

_CLIP_LO = -10.0
_CLIP_HI = 10.0
_STELLAR_SCALE = 0.004002769142205858


def _finite_clipped(values):
    arr = np.asarray(values, dtype=np.float64)
    clipped = np.clip(arr, _CLIP_LO, _CLIP_HI)
    return np.where(np.isfinite(arr), clipped, clipped)


def _signed_log_abs(values):
    clipped = _finite_clipped(values)
    logged = np.sign(clipped) * np.log1p(np.abs(clipped))
    return np.where(np.isfinite(logged), logged, clipped)


def _add_continuous_pair(out, name, values):
    out[name] = _finite_clipped(values).astype(np.float32)
    out[name + "_log_abs_signed"] = _signed_log_abs(values).astype(np.float32)


def add_redshift_template_domain_margins(raw, deps, aux):
    z = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    z_finite = np.where(np.isfinite(z), z, 0.0)

    out = pd.DataFrame(index=raw.index)

    for domain_name, lo, hi in _TEMPLATE_DOMAINS:
        in_domain = (z_finite >= lo) & (z_finite <= hi)
        lower_margin = z_finite - lo
        upper_margin = hi - z_finite
        outside_signed = np.where(z_finite < lo, z_finite - lo, np.where(z_finite > hi, z_finite - hi, 0.0))
        outside_gap = np.maximum(lo - z_finite, 0.0) + np.maximum(z_finite - hi, 0.0)

        out[domain_name + "_in_domain"] = in_domain.astype(np.int8)
        _add_continuous_pair(out, domain_name + "_lower_margin", lower_margin)
        _add_continuous_pair(out, domain_name + "_upper_margin", upper_margin)
        _add_continuous_pair(out, domain_name + "_outside_signed_distance", outside_signed)
        _add_continuous_pair(out, domain_name + "_outside_abs_gap", outside_gap)

    _add_continuous_pair(out, "redshift_sign", np.sign(z_finite))
    _add_continuous_pair(out, "redshift_abs", np.abs(z_finite))
    _add_continuous_pair(out, "star_velocity_scaled", z_finite / _STELLAR_SCALE)

    breakpoints = np.asarray(_BREAKPOINTS, dtype=np.float64)
    interval_id = np.searchsorted(breakpoints, z_finite, side="right") - 1
    interval_id = np.clip(interval_id, 0, len(breakpoints) - 2)
    out["breakpoint_interval_id"] = interval_id.astype(np.int8)

    breakpoint_distances = np.abs(z_finite[:, None] - breakpoints[None, :])
    nearest_index = np.argmin(breakpoint_distances, axis=1)
    nearest_distance = breakpoint_distances[np.arange(len(z_finite)), nearest_index]

    _add_continuous_pair(out, "nearest_breakpoint_distance", nearest_distance)
    out["nearest_breakpoint_index"] = nearest_index.astype(np.int8)

    for idx, breakpoint in enumerate(_BREAKPOINTS):
        label = str(breakpoint).replace("-", "neg_").replace(".", "p")
        _add_continuous_pair(out, "breakpoint_" + label + "_lower_edge_distance", z_finite - breakpoint)
        _add_continuous_pair(out, "breakpoint_" + label + "_upper_edge_distance", breakpoint - z_finite)

    return out


FEATURE_GROUPS = [
    {
        "name": "redshift_template_domain_margins",
        "fn": add_redshift_template_domain_margins,
        "depends_on": [],
        "description": "Redshift feasibility margins and breakpoint geometry for stellar, galaxy, and quasar template domains.",
    }
]