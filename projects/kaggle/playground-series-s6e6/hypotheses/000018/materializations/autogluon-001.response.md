import numpy as np
import pandas as pd

SPECTRAL_TAG_SCALE = (("O/B", 0.00), ("A/F", 0.33), ("G/K", 0.67), ("M", 1.00))
GALAXY_TAG_SCALE = (("Blue_Cloud", 0.00), ("Red_Sequence", 1.00))
LOW_REDNESS_BOUND = 0.33
HIGH_REDNESS_BOUND = 0.67
UNEXPECTED_TAG_EXPECTATION = 0.5


def add_catalog_tag_color_concordance(raw, deps, aux):
    spectral_type = raw["spectral_type"]
    galaxy_population = raw["galaxy_population"]

    spectral_map = dict(SPECTRAL_TAG_SCALE)
    galaxy_map = dict(GALAXY_TAG_SCALE)

    ug = (raw["u"].astype(float) - raw["g"].astype(float) + 0.5) / 3.5
    gr = (raw["g"].astype(float) - raw["r"].astype(float) + 0.5) / 2.0
    ri = (raw["r"].astype(float) - raw["i"].astype(float) + 0.5) / 1.5
    iz = (raw["i"].astype(float) - raw["z"].astype(float) + 0.5) / 1.5

    ug_clip = pd.Series(np.clip(ug, 0.0, 1.0), index=raw.index)
    gr_clip = pd.Series(np.clip(gr, 0.0, 1.0), index=raw.index)
    ri_clip = pd.Series(np.clip(ri, 0.0, 1.0), index=raw.index)
    iz_clip = pd.Series(np.clip(iz, 0.0, 1.0), index=raw.index)

    observed_redness = (ug_clip + gr_clip + ri_clip + iz_clip) / 4.0

    spectral_expectation = spectral_type.map(spectral_map).fillna(UNEXPECTED_TAG_EXPECTATION).astype(float)
    galaxy_expectation = galaxy_population.map(galaxy_map).fillna(UNEXPECTED_TAG_EXPECTATION).astype(float)

    diff_spectral = observed_redness - spectral_expectation
    diff_galaxy = observed_redness - galaxy_expectation
    diff_tag = spectral_expectation - galaxy_expectation

    coarse_redness_bin = pd.Series(
        np.where(observed_redness < LOW_REDNESS_BOUND, 0, np.where(observed_redness < HIGH_REDNESS_BOUND, 1, 2)),
        index=raw.index,
        dtype="int8",
    )

    is_hot_tag_red_color = (spectral_type == "O/B") | (spectral_type == "A/F")
    is_cool_tag_blue_color = (spectral_type == "M")
    is_red_sequence_blue_color = galaxy_population == "Red_Sequence"
    is_blue_cloud_red_color = galaxy_population == "Blue_Cloud"

    hot_tag_red_contradiction = (observed_redness >= HIGH_REDNESS_BOUND) & is_hot_tag_red_color
    cool_tag_blue_contradiction = (observed_redness <= LOW_REDNESS_BOUND) & is_cool_tag_blue_color
    red_sequence_blue_contradiction = (observed_redness <= LOW_REDNESS_BOUND) & is_red_sequence_blue_color
    blue_cloud_red_contradiction = (observed_redness >= HIGH_REDNESS_BOUND) & is_blue_cloud_red_color

    return pd.DataFrame(
        {
            "tag_color_ug_norm": ug_clip,
            "tag_color_gr_norm": gr_clip,
            "tag_color_ri_norm": ri_clip,
            "tag_color_iz_norm": iz_clip,
            "tag_color_observed_redness": observed_redness,
            "tag_color_spectral_expectation": spectral_expectation,
            "tag_color_galaxy_expectation": galaxy_expectation,
            "tag_color_spectral_diff": diff_spectral,
            "tag_color_spectral_abs_diff": diff_spectral.abs(),
            "tag_color_galaxy_diff": diff_galaxy,
            "tag_color_galaxy_abs_diff": diff_galaxy.abs(),
            "tag_color_tag_expectation_diff": diff_tag,
            "tag_color_tag_expectation_abs_diff": diff_tag.abs(),
            "tag_color_redness_bin": coarse_redness_bin,
            "tag_color_hot_tag_red_contradiction": hot_tag_red_contradiction,
            "tag_color_cool_tag_blue_contradiction": cool_tag_blue_contradiction,
            "tag_color_red_sequence_blue_contradiction": red_sequence_blue_contradiction,
            "tag_color_blue_cloud_red_contradiction": blue_cloud_red_contradiction,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "catalog_tag_color_concordance",
        "fn": add_catalog_tag_color_concordance,
        "depends_on": [],
        "description": "Compares clipped SDSS color redness against mapped spectral and galaxy tags to capture tag-color agreement and contradiction geometry.",
    },
]