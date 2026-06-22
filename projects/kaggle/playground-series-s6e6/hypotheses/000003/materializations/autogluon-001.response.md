import numpy as np
import pandas as pd


def add_galactic_sightline_context(raw, deps, aux):
    alpha_rad = np.deg2rad((pd.to_numeric(raw["alpha"], errors="coerce") % 360.0).to_numpy(dtype=float))
    delta_rad = np.deg2rad(pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float))

    # Equatorial unit-sphere coordinates (wrap-safe via alpha modulo 360)
    cos_delta = np.cos(delta_rad)
    x_eq = cos_delta * np.cos(alpha_rad)
    y_eq = cos_delta * np.sin(alpha_rad)
    z_eq = np.sin(delta_rad)

    # J2000 equatorial -> Galactic rotation matrix
    x_gal = (
        -0.0548755604 * x_eq
        + -0.8734370902 * y_eq
        + -0.4838350155 * z_eq
    )
    y_gal = (
        0.4941094279 * x_eq
        + -0.4448296300 * y_eq
        + 0.7469822445 * z_eq
    )
    z_gal = (
        -0.8676661490 * x_eq
        + -0.1980763734 * y_eq
        + 0.4559837762 * z_eq
    )

    z_clip = np.clip(z_gal, -1.0, 1.0)
    b_rad = np.arcsin(z_clip)
    b_deg = np.degrees(b_rad)
    abs_b_deg = np.abs(b_deg)

    l_rad = np.arctan2(y_gal, x_gal)
    l_deg = np.degrees(l_rad) % 360.0
    l_rad = np.deg2rad(l_deg)

    sin_l = np.sin(l_rad)
    cos_l = np.cos(l_rad)
    sin_b = np.sin(b_rad)
    cos_b = np.cos(b_rad)

    dist_to_galactic_center = np.degrees(np.arccos(np.clip(x_gal, -1.0, 1.0)))
    dist_to_galactic_anticenter = np.degrees(np.arccos(np.clip(-x_gal, -1.0, 1.0)))
    dist_to_north_galactic_pole = np.degrees(np.arccos(np.clip(z_gal, -1.0, 1.0)))
    dist_to_south_galactic_pole = np.degrees(np.arccos(np.clip(-z_gal, -1.0, 1.0)))

    b_band_edges = [0.0, 10.0, 20.0, 35.0, 50.0, 70.0, 90.0000001]
    b_band_labels = ["0-10", "10-20", "20-35", "35-50", "50-70", "70-90"]
    b_band = pd.cut(
        abs_b_deg,
        bins=b_band_edges,
        labels=b_band_labels,
        right=False,
        include_lowest=True,
    )

    l_sector_edges = np.arange(0.0, 390.0, 30.0)
    l_sector_labels = [f"{int(start):03d}-{int(start+30):03d}" for start in l_sector_edges[:-1]]
    l_sector = pd.cut(
        l_deg,
        bins=l_sector_edges,
        labels=l_sector_labels,
        right=False,
        include_lowest=True,
    )

    crossed_regime = b_band.astype("string").str.cat(l_sector.astype("string"), sep="|")

    return pd.DataFrame(
        {
            "eq_x": x_eq,
            "eq_y": y_eq,
            "eq_z": z_eq,
            "gal_l_deg": l_deg,
            "gal_b_deg": b_deg,
            "gal_abs_b_deg": abs_b_deg,
            "gal_sin_l": sin_l,
            "gal_cos_l": cos_l,
            "gal_sin_b": sin_b,
            "gal_cos_b": cos_b,
            "gal_plane_angular_distance_deg": abs_b_deg,
            "gal_dist_to_galactic_center_deg": dist_to_galactic_center,
            "gal_dist_to_galactic_anticenter_deg": dist_to_galactic_anticenter,
            "gal_dist_to_north_pole_deg": dist_to_north_galactic_pole,
            "gal_dist_to_south_pole_deg": dist_to_south_galactic_pole,
            "gal_b_band": b_band.astype("string"),
            "gal_l_sector": l_sector.astype("string"),
            "gal_b_band_l_sector": crossed_regime,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "galactic_sightline_context",
        "fn": add_galactic_sightline_context,
        "depends_on": [],
        "description": "Creates Galactic coordinate-derived spatial features from sky position, including smooth angular encodings, center/pole distances, and coarse latitude/longitude regime bins.",
    }
]