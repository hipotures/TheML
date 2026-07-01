import numpy as np
import pandas as pd

SUBSTANCE_BURDEN_MAP = {
    "no": 0,
    "occasional": 1,
    "yes": 2,
}

def _column(raw, name):
    if name in raw.columns:
        return raw[name]
    return pd.Series(pd.NA, index=raw.index)

def _normalized_text(raw, name):
    return _column(raw, name).astype("string").str.strip().str.lower()

def _int8_flag(mask):
    return mask.fillna(False).astype("int8")

def add_substance_use_context_buffer(raw, deps, aux):
    index = raw.index

    smoking_alcohol = _normalized_text(raw, "smoking_alcohol")
    diet_type = _normalized_text(raw, "diet_type")
    stress_level = _normalized_text(raw, "stress_level")
    sleep_quality = _normalized_text(raw, "sleep_quality")
    physical_activity_level = _normalized_text(raw, "physical_activity_level")

    sleep_duration = pd.to_numeric(_column(raw, "sleep_duration"), errors="coerce")
    heart_rate = pd.to_numeric(_column(raw, "heart_rate"), errors="coerce")
    bmi = pd.to_numeric(_column(raw, "bmi"), errors="coerce")
    step_count = pd.to_numeric(_column(raw, "step_count"), errors="coerce")
    exercise_duration = pd.to_numeric(_column(raw, "exercise_duration"), errors="coerce")

    substance_burden = smoking_alcohol.map(SUBSTANCE_BURDEN_MAP).fillna(-1).astype("int8")

    protective_context_score = (
        _int8_flag(diet_type.isin(("balanced", "veg")))
        + _int8_flag(physical_activity_level.isin(("moderate", "active")))
        + _int8_flag(sleep_quality.eq("good"))
        + _int8_flag(stress_level.eq("low"))
        + _int8_flag(sleep_duration.between(7.0, 9.0, inclusive="both"))
        + _int8_flag(step_count.ge(8000.0))
        + _int8_flag(exercise_duration.ge(30.0))
        + _int8_flag(bmi.ge(18.5) & bmi.lt(25.0))
    ).astype("int8")

    adverse_context_score = (
        _int8_flag(stress_level.eq("high"))
        + _int8_flag(sleep_quality.eq("poor"))
        + _int8_flag(physical_activity_level.eq("sedentary"))
        + _int8_flag(diet_type.eq("non-veg"))
        + _int8_flag(sleep_duration.lt(6.0) | sleep_duration.gt(9.5))
        + _int8_flag(step_count.lt(5000.0))
        + _int8_flag(exercise_duration.lt(10.0))
        + _int8_flag(bmi.lt(18.5) | bmi.ge(30.0))
        + _int8_flag(heart_rate.ge(90.0))
    ).astype("int8")

    has_substance_value = substance_burden.ge(0)
    uses_substance = substance_burden.ge(1)
    no_substance = substance_burden.eq(0)

    substance_buffer_gap = pd.Series(np.nan, index=index, dtype="float32")
    substance_buffer_gap.loc[has_substance_value] = (
        adverse_context_score.loc[has_substance_value].astype("float32")
        + substance_burden.loc[has_substance_value].astype("float32")
        - protective_context_score.loc[has_substance_value].astype("float32")
    )

    substance_amplification = pd.Series(np.nan, index=index, dtype="float32")
    substance_amplification.loc[has_substance_value] = (
        substance_burden.loc[has_substance_value].astype("float32")
        * adverse_context_score.loc[has_substance_value].astype("float32")
    )

    substance_buffering = pd.Series(np.nan, index=index, dtype="float32")
    substance_buffering.loc[has_substance_value] = (
        substance_burden.loc[has_substance_value].astype("float32")
        * protective_context_score.loc[has_substance_value].astype("float32")
    )

    context_bucket = pd.Series("mixed_context", index=index, dtype="object")
    context_bucket.loc[~has_substance_value] = "missing_substance"
    context_bucket.loc[
        no_substance & protective_context_score.ge(5) & adverse_context_score.le(1)
    ] = "clean_supported"
    context_bucket.loc[
        no_substance & adverse_context_score.ge(4)
    ] = "non_substance_lifestyle_risk"
    context_bucket.loc[
        uses_substance & protective_context_score.ge(4) & adverse_context_score.le(2)
    ] = "buffered_use"
    context_bucket.loc[
        uses_substance & adverse_context_score.ge(4)
    ] = "compounding_use"

    return pd.DataFrame(
        {
            "substance_burden": substance_burden,
            "protective_context_score": protective_context_score,
            "adverse_context_score": adverse_context_score,
            "substance_buffer_gap": substance_buffer_gap,
            "substance_amplification": substance_amplification,
            "substance_buffering": substance_buffering,
            "context_bucket": context_bucket,
        },
        index=index,
    )

FEATURE_GROUPS = [
    {
        "name": "substance_use_context_buffer",
        "fn": add_substance_use_context_buffer,
        "depends_on": [],
        "description": "Encodes whether smoking and alcohol burden is buffered or amplified by protective and adverse lifestyle context.",
    }
]