from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_CACHE = {}


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_component_module(hypothesis_id: str, filename: str):
    cache_key = (hypothesis_id, filename)
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    source_path = _project_dir() / "hypotheses" / hypothesis_id / "materializations" / filename
    module_name = f"_tml_branch_b000022_{hypothesis_id}_{filename.replace('-', '_').replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load materialization {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[cache_key] = module
    return module


def _call_component(hypothesis_id: str, filename: str, fn_name: str, raw, aux):
    module = _load_component_module(hypothesis_id, filename)
    return getattr(module, fn_name)(raw, {}, aux)


def _branch_000001_broadband_color_shape(raw, deps, aux):
    return _call_component("000001", "autogluon-001.py", "add_broadband_color_shape", raw, aux)


def _branch_000021_aide_broadband_flux_ratios(raw, deps, aux):
    return _call_component("000021", "autogluon-001.py", "aide_broadband_flux_ratios", raw, aux)


def _branch_000022_aide_sky_cell_local_residuals(raw, deps, aux):
    return _call_component("000022", "autogluon-001.py", "aide_sky_cell_local_residuals", raw, aux)


def _branch_000023_aide_redshift_bin_color_residuals(raw, deps, aux):
    return _call_component("000023", "autogluon-001.py", "aide_redshift_bin_color_residuals", raw, aux)


def _branch_000024_aide_catalog_rank_frequency_context(raw, deps, aux):
    return _call_component("000024", "autogluon-001.py", "aide_catalog_rank_frequency_context", raw, aux)


def _branch_000025_aide_aux_reference_distribution_distance(raw, deps, aux):
    return _call_component("000025", "autogluon-001.py", "aide_aux_reference_distribution_distance", raw, aux)


def _branch_000026_aide_smooth_spline_sed_interactions(raw, deps, aux):
    return _call_component("000026", "autogluon-001.py", "aide_smooth_spline_sed_interactions", raw, aux)


def _branch_000027_aide_id_sequence_scan_context(raw, deps, aux):
    return _call_component("000027", "autogluon-001.py", "aide_id_sequence_scan_context", raw, aux)


FEATURE_GROUPS = [
    {
        "name": "broadband_color_shape",
        "fn": _branch_000001_broadband_color_shape,
        "depends_on": [],
        "description": "Create broadband color and curvature features from ugriz magnitudes.",
    },
    {
        "name": "aide_broadband_flux_ratios",
        "fn": _branch_000021_aide_broadband_flux_ratios,
        "depends_on": [],
        "description": "AIDE top-5 broadband ratios, absolute colors, pseudo-flux values, and redshift powers.",
    },
    {
        "name": "aide_sky_cell_local_residuals",
        "fn": _branch_000022_aide_sky_cell_local_residuals,
        "depends_on": [],
        "description": "AIDE half-degree sky-cell densities and local residual/z-score features.",
    },
    {
        "name": "aide_redshift_bin_color_residuals",
        "fn": _branch_000023_aide_redshift_bin_color_residuals,
        "depends_on": [],
        "description": "AIDE qcut-redshift color residuals and aggregate residual scores.",
    },
    {
        "name": "aide_catalog_rank_frequency_context",
        "fn": _branch_000024_aide_catalog_rank_frequency_context,
        "depends_on": [],
        "description": "AIDE category frequency, cross-frequency, global rank, qbin, and within-category rank context.",
    },
    {
        "name": "aide_aux_reference_distribution_distance",
        "fn": _branch_000025_aide_aux_reference_distribution_distance,
        "depends_on": [],
        "description": "AIDE robust auxiliary-reference z, CDF, joint distance, and Mahalanobis features.",
    },
    {
        "name": "aide_smooth_spline_sed_interactions",
        "fn": _branch_000026_aide_smooth_spline_sed_interactions,
        "depends_on": [],
        "description": "AIDE spline basis features and compact tensor interactions.",
    },
    {
        "name": "aide_id_sequence_scan_context",
        "fn": _branch_000027_aide_id_sequence_scan_context,
        "depends_on": [],
        "description": "AIDE id rank, block, modulo, parity, log, and neighbor-gap sequence features.",
    },
]
