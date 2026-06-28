import numpy as np
import pandas as pd


GALAXY_LINES = (3727.0, 4861.0, 4959.0, 5007.0, 6563.0)
QUASAR_LINES = (1216.0, 1549.0, 1909.0, 2798.0, 4861.0)
STELLAR_LINES = (3934.0, 3969.0, 4102.0, 4341.0, 4861.0, 6563.0)
SKY_ANCHORS = (5577.0, 6300.0, 6363.0)
WINDOW_BLUE = 3800.0
WINDOW_RED = 9200.0
WINDOW_EDGE_MARGIN = 50.0
OH_ZONE_START = 7000.0
DISTANCE_CAP = 500.0


def _family_features(redshift_factor, rest_lines, prefix, index):
    lines = np.asarray(rest_lines, dtype="float64")
    anchors = np.asarray(SKY_ANCHORS, dtype="float64")

    observed = redshift_factor[:, None] * lines[None, :]
    finite = np.isfinite(observed)

    inside_window = finite & (observed >= WINDOW_BLUE) & (observed <= WINDOW_RED)
    near_blue_edge = inside_window & (observed <= WINDOW_BLUE + WINDOW_EDGE_MARGIN)
    near_red_edge = inside_window & (observed >= WINDOW_RED - WINDOW_EDGE_MARGIN)
    oh_zone = inside_window & (observed >= OH_ZONE_START)

    anchor_dist = np.abs(observed[:, :, None] - anchors[None, None, :])
    min_line_dist = np.nanmin(anchor_dist, axis=2)
    clipped_line_dist = np.clip(min_line_dist, 0.0, DISTANCE_CAP)
    valid_dist = finite & inside_window

    clipped_for_min = np.where(valid_dist, clipped_line_dist, np.nan)
    min_dist = np.nanmin(clipped_for_min, axis=1)
    min_dist = np.where(np.isfinite(min_dist), min_dist, 0.0)

    narrow_score = np.where(valid_dist, np.exp(-((clipped_line_dist / 8.0) ** 2)), 0.0)
    broad_score = np.where(valid_dist, np.exp(-((clipped_line_dist / 25.0) ** 2)), 0.0)

    n_lines = float(len(rest_lines))
    features = pd.DataFrame(index=index)
    features[prefix + "_window_frac"] = inside_window.sum(axis=1) / n_lines
    features[prefix + "_blue_edge_frac"] = near_blue_edge.sum(axis=1) / n_lines
    features[prefix + "_red_edge_frac"] = near_red_edge.sum(axis=1) / n_lines
    features[prefix + "_sky_min_dist_clipped"] = min_dist
    features[prefix + "_sky_score_8a_mean"] = narrow_score.mean(axis=1)
    features[prefix + "_sky_score_25a_mean"] = broad_score.mean(axis=1)
    features[prefix + "_sky_score_8a_max"] = narrow_score.max(axis=1)
    features[prefix + "_sky_score_25a_max"] = broad_score.max(axis=1)
    features[prefix + "_sky_count_10a"] = (valid_dist & (clipped_line_dist <= 10.0)).sum(axis=1)
    features[prefix + "_sky_count_25a"] = (valid_dist & (clipped_line_dist <= 25.0)).sum(axis=1)
    features[prefix + "_oh_zone_frac"] = oh_zone.sum(axis=1) / n_lines
    return features


def add_spectral_skyline_interference(raw, deps, aux):
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype="float64", copy=True)
    redshift_factor = np.maximum(1.0 + redshift, 1e-6)
    redshift_factor = np.where(np.isfinite(redshift_factor), redshift_factor, 1e-6)

    galaxy = _family_features(redshift_factor, GALAXY_LINES, "galaxy_lines", raw.index)
    quasar = _family_features(redshift_factor, QUASAR_LINES, "quasar_lines", raw.index)
    stellar = _family_features(redshift_factor, STELLAR_LINES, "stellar_lines", raw.index)

    features = pd.concat((galaxy, quasar, stellar), axis=1)

    extragalactic_sky_25 = 0.5 * (
        features["galaxy_lines_sky_score_25a_mean"] + features["quasar_lines_sky_score_25a_mean"]
    )
    extragalactic_window = 0.5 * (
        features["galaxy_lines_window_frac"] + features["quasar_lines_window_frac"]
    )

    features["galaxy_minus_quasar_sky_score_25a"] = (
        features["galaxy_lines_sky_score_25a_mean"] - features["quasar_lines_sky_score_25a_mean"]
    )
    features["galaxy_minus_quasar_window_frac"] = (
        features["galaxy_lines_window_frac"] - features["quasar_lines_window_frac"]
    )
    features["stellar_minus_extragalactic_sky_score_25a"] = (
        features["stellar_lines_sky_score_25a_mean"] - extragalactic_sky_25
    )
    features["stellar_minus_extragalactic_window_frac"] = (
        features["stellar_lines_window_frac"] - extragalactic_window
    )
    features["galaxy_minus_quasar_oh_zone_frac"] = (
        features["galaxy_lines_oh_zone_frac"] - features["quasar_lines_oh_zone_frac"]
    )
    features["stellar_minus_extragalactic_oh_zone_frac"] = (
        features["stellar_lines_oh_zone_frac"]
        - 0.5 * (features["galaxy_lines_oh_zone_frac"] + features["quasar_lines_oh_zone_frac"])
    )

    return features


FEATURE_GROUPS = [
    {
        "name": "spectral_skyline_interference",
        "fn": add_spectral_skyline_interference,
        "depends_on": [],
        "description": "Redshift-dependent spectral line overlap with SDSS sky residuals, OH regions, and spectrograph window edges.",
    }
]