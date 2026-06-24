import numpy as np
import pandas as pd

ALPHA_NGP_DEG = 192.85948
DELTA_NGP_DEG = 27.12825
L_ASC_DEG = 122.93192
TWO_PI = 2.0 * np.pi

GAL_EQ2GAL_MATRIX = (
    (-0.0548755604, -0.8734370902, -0.4838350155),
    (0.4941094279, -0.4448296300, 0.7469822445),
    (-0.8676661490, -0.1980763734, 0.4559837762),
)

B_ABS_BAND_EDGES = (5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0)
L_SECTOR_EDGES = (30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0)


def add_galactic_sightline_context(raw, deps, aux):
    alpha_rad = np.deg2rad(np.mod(pd.to_numeric(raw["alpha"], errors="coerce").to_numpy(dtype=float), 360.0))
    delta_rad = np.deg2rad(pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float))

    alpha_ngp_rad = np.deg2rad(ALPHA_NGP_DEG)
    delta_ngp_rad = np.deg2rad(DELTA_NGP_DEG)
    l_asc_rad = np.deg2rad(L_ASC_DEG)

    d_alpha = alpha_rad - alpha_ngp_rad
    sin_delta = np.sin(delta_rad)
    cos_delta = np.cos(delta_rad)
    sin_delta_ngp = np.sin(delta_ngp_rad)
    cos_delta_ngp = np.cos(delta_ngp_rad)

    sinb_formula = sin_delta * sin_delta_ngp + cos_delta * cos_delta_ngp * np.cos(d_alpha)
    sinb_formula = np.clip(sinb_formula, -1.0, 1.0)
    b_formula_rad = np.arcsin(sinb_formula)

    l_formula_rad = (
        l_asc_rad
        - np.arctan2(
            cos_delta * np.sin(d_alpha),
            cos_delta_ngp * sin_delta - sin_delta_ngp * cos_delta * np.cos(d_alpha),
        )
    ) % TWO_PI

    # Parallel unit-vector path (equatorial -> galactic Cartesian, then back to l,b).
    eq_x = cos_delta * np.cos(alpha_rad)
    eq_y = cos_delta * np.sin(alpha_rad)
    eq_z = sin_delta

    rot = np.array(GAL_EQ2GAL_MATRIX, dtype=float)
    gal_x = rot[0][0] * eq_x + rot[0][1] * eq_y + rot[0][2] * eq_z
    gal_y = rot[1][0] * eq_x + rot[1][1] * eq_y + rot[1][2] * eq_z
    gal_z = rot[2][0] * eq_x + rot[2][1] * eq_y + rot[2][2] * eq_z

    b_vec_rad = np.arcsin(np.clip(gal_z, -1.0, 1.0))
    l_vec_rad = np.mod(np.arctan2(gal_y, gal_x), TWO_PI)

    needs_fallback = (~np.isfinite(l_formula_rad)) | (~np.isfinite(b_formula_rad))
    near_pole = np.abs(np.cos(b_formula_rad)) < 1e-10
    use_vec = needs_fallback | near_pole

    b_rad = np.where(use_vec, b_vec_rad, b_formula_rad)
    l_rad = np.where(use_vec, l_vec_rad, l_formula_rad)

    l_deg = np.mod(np.degrees(l_rad), 360.0)
    l_deg = np.where(np.isclose(l_deg, 360.0), 0.0, l_deg)
    b_deg = np.degrees(b_rad)
    b_abs_deg = np.abs(b_deg)

    b_sign = np.where(b_deg >= 0.0, 1, -1).astype(np.int8)
    b_sign_label = pd.Categorical(np.where(b_sign > 0, "north", "south"), categories=("south", "north"))

    sin_b = np.sin(b_rad)
    cos_b = np.cos(b_rad)
    sin_l = np.sin(l_rad)
    cos_l = np.cos(l_rad)

    cosb_cosl = np.clip(np.cos(b_rad) * np.cos(l_rad), -1.0, 1.0)
    d_gc_deg = np.degrees(np.arccos(cosb_cosl))
    d_ac_deg = np.degrees(np.arccos(np.clip(-cosb_cosl, -1.0, 1.0)))
    d_np_deg = np.degrees(np.arccos(np.clip(np.sin(b_rad), -1.0, 1.0)))
    d_sp_deg = np.degrees(np.arccos(np.clip(-np.sin(b_rad), -1.0, 1.0)))

    band_edges = np.array(B_ABS_BAND_EDGES, dtype=float)
    sector_edges = np.array(L_SECTOR_EDGES, dtype=float)

    b_band_id = np.searchsorted(band_edges, b_abs_deg, side="right")
    b_band_id = np.clip(b_band_id, 0, len(band_edges) - 1).astype(np.int16)

    l_sector_id = np.searchsorted(sector_edges, l_deg, side="right")
    l_sector_id = np.where(l_sector_id == sector_edges.size, 0, l_sector_id).astype(np.int16)

    band_sector_id = (b_band_id * sector_edges.size + l_sector_id).astype(np.int16)
    sign_x_band_id = (b_sign * (b_band_id + 1)).astype(np.int16)
    sign_x_sector_id = (b_sign * (l_sector_id + 1)).astype(np.int16)
    sign_x_band_label = pd.Categorical(
        np.where(b_sign > 0, "N_" + b_band_id.astype(str), "S_" + b_band_id.astype(str))
    )
    sign_x_sector_label = pd.Categorical(
        np.where(b_sign > 0, "N_" + l_sector_id.astype(str), "S_" + l_sector_id.astype(str))
    )

    return pd.DataFrame(
        {
            "gal_l_deg": l_deg,
            "gal_b_deg": b_deg,
            "gal_b_abs_deg": b_abs_deg,
            "gal_b_sign": b_sign,
            "gal_b_sign_label": b_sign_label,
            "gal_sin_b": sin_b,
            "gal_cos_b": cos_b,
            "gal_sin_l": sin_l,
            "gal_cos_l": cos_l,
            "gal_d_plane_deg": b_abs_deg,
            "gal_d_gc_deg": d_gc_deg,
            "gal_d_ac_deg": d_ac_deg,
            "gal_d_np_deg": d_np_deg,
            "gal_d_sp_deg": d_sp_deg,
            "gal_b_band_id": b_band_id,
            "gal_l_sector_id": l_sector_id,
            "gal_band_sector_id": band_sector_id,
            "gal_sign_x_band_id": sign_x_band_id,
            "gal_sign_x_sector_id": sign_x_sector_id,
            "gal_sign_x_band_label": sign_x_band_label,
            "gal_sign_x_sector_label": sign_x_sector_label,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "galactic_sightline_context",
        "fn": add_galactic_sightline_context,
        "depends_on": [],
        "description": "Compute robust Galactic-sky morphology descriptors from alpha/delta, including anisotropy-relevant angular distances, band/sector bins, and sign-aware interaction encodings.",
    }
]