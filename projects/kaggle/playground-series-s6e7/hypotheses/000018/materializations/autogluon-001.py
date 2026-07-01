import pandas as pd


POLARITY_MAPPINGS = {
    "stress_level": {"low": 1, "medium": 0, "high": -1},
    "sleep_quality": {"good": 1, "average": 0, "poor": -1},
    "physical_activity_level": {"active": 1, "moderate": 0, "sedentary": -1},
    "smoking_alcohol": {"no": 1, "occasional": 0, "yes": -1},
    "diet_type": {"balanced": 1, "veg": 0, "non-veg": -1},
}


def add_self_report_health_polarity(raw, deps, aux):
    mapped_parts = []

    for col, mapping in POLARITY_MAPPINGS.items():
        if col in raw.columns:
            values = raw[col].map(mapping)
        else:
            values = pd.Series(pd.NA, index=raw.index)
        mapped_parts.append(values.rename(col))

    mapped = pd.concat(mapped_parts, axis=1)

    observed_count = mapped.notna().sum(axis=1).astype("int16")
    signed_sum = mapped.sum(axis=1, skipna=True).fillna(0).astype("int16")

    protective_count = mapped.eq(1).sum(axis=1).astype("int16")
    adverse_count = mapped.eq(-1).sum(axis=1).astype("int16")
    neutral_count = mapped.eq(0).sum(axis=1).astype("int16")

    observed_safe = observed_count.where(observed_count > 0, 1)
    signed_mean = (signed_sum / observed_safe).where(observed_count > 0, 0.0)
    consensus_margin = (
        (protective_count - adverse_count).abs() / observed_safe
    ).where(observed_count > 0, 0.0)

    mixed_polarity_count = pd.concat(
        [protective_count, adverse_count], axis=1
    ).min(axis=1).astype("int16")

    all_protective = (
        (observed_count >= 3) & (protective_count > 0) & (adverse_count == 0)
    ).astype("int8")
    all_adverse = (
        (observed_count >= 3) & (adverse_count > 0) & (protective_count == 0)
    ).astype("int8")

    return pd.DataFrame(
        {
            "signed_sum": signed_sum,
            "signed_mean": signed_mean.astype("float32"),
            "protective_count": protective_count,
            "adverse_count": adverse_count,
            "neutral_count": neutral_count,
            "observed_count": observed_count,
            "consensus_margin": consensus_margin.astype("float32"),
            "mixed_polarity_count": mixed_polarity_count,
            "all_protective": all_protective,
            "all_adverse": all_adverse,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "self_report_health_polarity",
        "fn": add_self_report_health_polarity,
        "depends_on": [],
        "description": "Compact ordinal polarity features from self-reported lifestyle categories.",
    }
]