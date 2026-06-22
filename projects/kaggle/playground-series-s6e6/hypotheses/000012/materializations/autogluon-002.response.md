import numpy as np
import pandas as pd


def _clean_nonfinite_dataframe(df):
    arr = df.to_numpy(dtype=np.float64, copy=True)
    arr[~np.isfinite(arr)] = 0.0
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def _clean_nonfinite_series(values, index):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = 0.0
    return pd.Series(arr, index=index)


def add_faint_blue_galaxy_wedge_margins(raw, deps, aux):
    u = pd.to_numeric(raw["u"], errors="coerce")
    g = pd.to_numeric(raw["g"], errors="coerce")
    r = pd.to_numeric(raw["r"], errors="coerce")
    i = pd.to_numeric(raw["i"], errors="coerce")
    z = pd.to_numeric(raw["z"], errors="coerce")

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z

    margin_1 = gr - (0.40 + 0.6 * ug)
    margin_2 = (1.7 - 0.1 * ug) - gr
    margin_3 = ug + 0.5
    margin_4 = 3.0 - ug
    margin_5 = gr - 0.0
    margin_6 = 1.8 - gr
    margin_7 = ri + 0.5
    margin_8 = 1.5 - ri
    margin_9 = iz + 1.0
    margin_10 = 1.5 - iz
    margin_11 = u - 18.0
    margin_12 = 24.0 - u
    margin_13 = g - 18.0
    margin_14 = 21.5 - g
    margin_15 = r - 17.8
    margin_16 = 19.5 - r
    margin_17 = i - 16.5
    margin_18 = 20.5 - i
    margin_19 = z - 16.0
    margin_20 = 20.0 - z

    margins = _clean_nonfinite_dataframe(
        pd.DataFrame(
            {
                "faint_blue_wedge_m_margin_1_ug_gr_upper": margin_1,
                "faint_blue_wedge_m_margin_2_ug_gr_lower": margin_2,
                "faint_blue_wedge_m_margin_3_ug_lower": margin_3,
                "faint_blue_wedge_m_margin_4_ug_upper": margin_4,
                "faint_blue_wedge_m_margin_5_gr_lower": margin_5,
                "faint_blue_wedge_m_margin_6_gr_upper": margin_6,
                "faint_blue_wedge_m_margin_7_ri_lower": margin_7,
                "faint_blue_wedge_m_margin_8_ri_upper": margin_8,
                "faint_blue_wedge_m_margin_9_iz_lower": margin_9,
                "faint_blue_wedge_m_margin_10_iz_upper": margin_10,
                "faint_blue_wedge_m_margin_11_u_lower": margin_11,
                "faint_blue_wedge_m_margin_12_u_upper": margin_12,
                "faint_blue_wedge_m_margin_13_g_lower": margin_13,
                "faint_blue_wedge_m_margin_14_g_upper": margin_14,
                "faint_blue_wedge_m_margin_15_r_lower": margin_15,
                "faint_blue_wedge_m_margin_16_r_upper": margin_16,
                "faint_blue_wedge_m_margin_17_i_lower": margin_17,
                "faint_blue_wedge_m_margin_18_i_upper": margin_18,
                "faint_blue_wedge_m_margin_19_z_lower": margin_19,
                "faint_blue_wedge_m_margin_20_z_upper": margin_20,
            },
            index=raw.index,
        )
    )

    color_cols = [
        "faint_blue_wedge_m_margin_1_ug_gr_upper",
        "faint_blue_wedge_m_margin_2_ug_gr_lower",
        "faint_blue_wedge_m_margin_3_ug_lower",
        "faint_blue_wedge_m_margin_4_ug_upper",
        "faint_blue_wedge_m_margin_5_gr_lower",
        "faint_blue_wedge_m_margin_6_gr_upper",
        "faint_blue_wedge_m_margin_7_ri_lower",
        "faint_blue_wedge_m_margin_8_ri_upper",
        "faint_blue_wedge_m_margin_9_iz_lower",
        "faint_blue_wedge_m_margin_10_iz_upper",
    ]
    mag_cols = [
        "faint_blue_wedge_m_margin_11_u_lower",
        "faint_blue_wedge_m_margin_12_u_upper",
        "faint_blue_wedge_m_margin_13_g_lower",
        "faint_blue_wedge_m_margin_14_g_upper",
        "faint_blue_wedge_m_margin_15_r_lower",
        "faint_blue_wedge_m_margin_16_r_upper",
        "faint_blue_wedge_m_margin_17_i_lower",
        "faint_blue_wedge_m_margin_18_i_upper",
        "faint_blue_wedge_m_margin_19_z_lower",
        "faint_blue_wedge_m_margin_20_z_upper",
    ]

    all_margin_min = margins.min(axis=1)
    color_margin_min = margins[color_cols].min(axis=1)
    mag_margin_min = margins[mag_cols].min(axis=1)
    violation_count = (margins < 0.0).sum(axis=1).astype("int64")

    sampling_input = margins["faint_blue_wedge_m_margin_1_ug_gr_upper"].to_numpy(dtype=np.float64)
    sampling_intensity = np.exp(0.1411 * sampling_input)
    sampling_intensity = np.clip(sampling_intensity, 0.0, 10.0)
    sampling_intensity = _clean_nonfinite_series(sampling_intensity, raw.index)

    return pd.DataFrame(
        {
            "faint_blue_wedge_m_margin_1_ug_gr_upper": margins["faint_blue_wedge_m_margin_1_ug_gr_upper"],
            "faint_blue_wedge_m_margin_2_ug_gr_lower": margins["faint_blue_wedge_m_margin_2_ug_gr_lower"],
            "faint_blue_wedge_m_margin_3_ug_lower": margins["faint_blue_wedge_m_margin_3_ug_lower"],
            "faint_blue_wedge_m_margin_4_ug_upper": margins["faint_blue_wedge_m_margin_4_ug_upper"],
            "faint_blue_wedge_m_margin_5_gr_lower": margins["faint_blue_wedge_m_margin_5_gr_lower"],
            "faint_blue_wedge_m_margin_6_gr_upper": margins["faint_blue_wedge_m_margin_6_gr_upper"],
            "faint_blue_wedge_m_margin_7_ri_lower": margins["faint_blue_wedge_m_margin_7_ri_lower"],
            "faint_blue_wedge_m_margin_8_ri_upper": margins["faint_blue_wedge_m_margin_8_ri_upper"],
            "faint_blue_wedge_m_margin_9_iz_lower": margins["faint_blue_wedge_m_margin_9_iz_lower"],
            "faint_blue_wedge_m_margin_10_iz_upper": margins["faint_blue_wedge_m_margin_10_iz_upper"],
            "faint_blue_wedge_m_margin_11_u_lower": margins["faint_blue_wedge_m_margin_11_u_lower"],
            "faint_blue_wedge_m_margin_12_u_upper": margins["faint_blue_wedge_m_margin_12_u_upper"],
            "faint_blue_wedge_m_margin_13_g_lower": margins["faint_blue_wedge_m_margin_13_g_lower"],
            "faint_blue_wedge_m_margin_14_g_upper": margins["faint_blue_wedge_m_margin_14_g_upper"],
            "faint_blue_wedge_m_margin_15_r_lower": margins["faint_blue_wedge_m_margin_15_r_lower"],
            "faint_blue_wedge_m_margin_16_r_upper": margins["faint_blue_wedge_m_margin_16_r_upper"],
            "faint_blue_wedge_m_margin_17_i_lower": margins["faint_blue_wedge_m_margin_17_i_lower"],
            "faint_blue_wedge_m_margin_18_i_upper": margins["faint_blue_wedge_m_margin_18_i_upper"],
            "faint_blue_wedge_m_margin_19_z_lower": margins["faint_blue_wedge_m_margin_19_z_lower"],
            "faint_blue_wedge_m_margin_20_z_upper": margins["faint_blue_wedge_m_margin_20_z_upper"],
            "faint_blue_wedge_m_margin_min_all": all_margin_min,
            "faint_blue_wedge_m_margin_min_color": color_margin_min,
            "faint_blue_wedge_m_margin_min_magnitude": mag_margin_min,
            "faint_blue_wedge_m_violation_count": violation_count,
            "faint_blue_wedge_sampling_intensity": sampling_intensity,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "faint_blue_galaxy_wedge_margins",
        "fn": add_faint_blue_galaxy_wedge_margins,
        "depends_on": [],
        "description": "Build signed faint-blue wedge margins in color-magnitude space and summarize them with minimum scores, violation counts, and a capped SDSS-style sampling intensity proxy.",
    }
]