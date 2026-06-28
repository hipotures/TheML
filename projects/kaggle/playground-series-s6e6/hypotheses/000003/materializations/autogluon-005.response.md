import numpy as np
import pandas as pd


_GALACTIC_ROTATION_MATRIX = (
    (-0.0548755604, -0.8734370902, -0.4838350155),
    (0.4941094279, -0.4448296300, 0.7469822445),
    (-0.8676661490, -0.1980763734, 0.4559837762),
)

_LATITUDE_BAND_LABELS = (
    "b00_05",
    "b05_10",
    "b10_20",
    "b20_30",
    "b30_45",
    "b45_60",
    "b60_75",
    "b75_90",
)


def add_galactic_sightline_context(raw, deps, aux):
    alpha_deg = pd.to_numeric(raw["alpha"], errors="coerce").to_numpy(dtype=float, copy=True)
    delta_deg = pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float, copy=True)

    alpha_rad = np.deg2rad(np.mod(alpha_deg, 360.0))
    delta_rad = np.deg2rad(delta_deg)

    cos_delta = np.cos(delta_rad)
    x_eq = cos_delta * np.cos(alpha_rad)
    y_eq = cos_delta * np.sin(alpha_rad)
    z_eq = np.sin(delta_rad)

    rotation = np.asarray(_GALACTIC_ROTATION_MATRIX, dtype=float)
    x_gal = rotation[0, 0] * x_eq + rotation[0, 1] * y_eq + rotation[0, 2] * z_eq
    y_gal = rotation[1, 0] * x_eq + rotation[1, 1] * y_eq + rotation[1, 2] * z_eq
    z_gal = rotation[2, 0] * x_eq + rotation[2, 1] * y_eq + rotation[2, 2] * z_eq

    z_gal_clipped = np.clip(z_gal, -1.0, 1.0)
    gal_l_deg = np.mod(np.rad2deg(np.arctan2(y_gal, x_gal)), 360.0)
    gal_b_deg = np.rad2deg(np.arcsin(z_gal_clipped))

    gal_l_rad = np.deg2rad(gal_l_deg)
    gal_b_rad = np.deg2rad(gal_b_deg)
    abs_b_deg = np.abs(gal_b_deg)
    abs_b_clipped = np.clip(abs_b_deg, 0.0, 90.0)

    latitude_band_id = np.searchsorted(
        np.asarray((5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0), dtype=float),
        abs_b_clipped,
        side="right",
    ).astype(np.int16)

    lon_sector_id = np.floor(gal_l_deg / 30.0).astype(np.int16)
    lon_sector_id = np.clip(lon_sector_id, 0, 11)

    sign_b = np.sign(gal_b_deg).astype(np.int8)
    hemisphere_id = (gal_b_deg >= 0.0).astype(np.int8)

    cos_b_cos_l = np.cos(gal_b_rad) * np.cos(gal_l_rad)
    distance_to_center_deg = np.rad2deg(np.arccos(np.clip(cos_b_cos_l, -1.0, 1.0)))
    distance_to_anticenter_deg = np.rad2deg(np.arccos(np.clip(-cos_b_cos_l, -1.0, 1.0)))

    latitude_band_labels = np.asarray(_LATITUDE_BAND_LABELS, dtype=object)[latitude_band_id]
    lon_sector_labels = np.asarray(
        (
            "l000_030",
            "l030_060",
            "l060_090",
            "l090_120",
            "l120_150",
            "l150_180",
            "l180_210",
            "l210_240",
            "l240_270",
            "l270_300",
            "l300_330",
            "l330_360",
        ),
        dtype=object,
    )[lon_sector_id]

    features = pd.DataFrame(
        {
            "gal_l_deg": gal_l_deg,
            "gal_b_deg": gal_b_deg,
            "gal_abs_b_deg": abs_b_deg,
            "gal_sign_b": sign_b,
            "gal_hemisphere_id": hemisphere_id,
            "gal_sin_l": np.sin(gal_l_rad),
            "gal_cos_l": np.cos(gal_l_rad),
            "gal_sin_b": np.sin(gal_b_rad),
            "gal_cos_b": np.cos(gal_b_rad),
            "gal_x": x_gal,
            "gal_y": y_gal,
            "gal_z": z_gal,
            "gal_distance_to_plane_deg": abs_b_deg,
            "gal_distance_to_center_deg": distance_to_center_deg,
            "gal_distance_to_anticenter_deg": distance_to_anticenter_deg,
            "gal_distance_to_north_pole_deg": 90.0 - gal_b_deg,
            "gal_distance_to_south_pole_deg": 90.0 + gal_b_deg,
            "gal_latitude_band_id": latitude_band_id,
            "gal_longitude_sector_id": lon_sector_id,
            "gal_band_sector_id": (latitude_band_id * 12 + lon_sector_id).astype(np.int16),
            "gal_hemisphere_latitude_band_id": (hemisphere_id * 8 + latitude_band_id).astype(np.int16),
            "gal_hemisphere_longitude_sector_id": (hemisphere_id * 12 + lon_sector_id).astype(np.int16),
            "gal_latitude_band": pd.Categorical(latitude_band_labels, categories=_LATITUDE_BAND_LABELS),
            "gal_longitude_sector": pd.Categorical(lon_sector_labels),
        },
        index=raw.index,
    )

    return features


FEATURE_GROUPS = [
    {
        "name": "galactic_sightline_context",
        "fn": add_galactic_sightline_context,
        "depends_on": [],
        "description": "Converts equatorial coordinates into Galactic line-of-sight geometry, angular distances, and coarse sky regimes.",
    }
]