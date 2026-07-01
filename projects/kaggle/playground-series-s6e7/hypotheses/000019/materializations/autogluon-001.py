import numpy as np
import pandas as pd


SLEEP_QUALITY_SCORE = {
    "good": 1.0,
    "average": 0.5,
    "poor": 0.0,
}

ACTIVITY_LEVEL_SCORE = {
    "active": 1.0,
    "moderate": 0.5,
    "sedentary": 0.0,
}

DIET_TYPE_SCORE = {
    "balanced": 1.0,
    "veg": 0.5,
    "non-veg": 0.5,
}

STRESS_LEVEL_SCORE = {
    "low": 1.0,
    "medium": 0.5,
    "high": 0.0,
}

SMOKING_ALCOHOL_SCORE = {
    "no": 1.0,
    "occasional": 0.5,
    "yes": 0.0,
}

PILLAR_NAME_TO_ID = {
    "sleep": 0,
    "movement": 1,
    "nutrition": 2,
    "stress": 3,
    "substance": 4,
    "cardiometabolic": 5,
}


def _numeric(raw, column):
    if column not in raw:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce")


def _categorical_score(raw, column, mapping):
    if column not in raw:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    values = raw[column].astype("string").str.lower().str.strip()
    return values.map(mapping).astype("float64")


def _average_available(index, components):
    block = pd.concat(components, axis=1)
    score = block.mean(axis=1, skipna=True)
    return score.fillna(0.5).astype("float64").reindex(index)


def _sleep_duration_score(raw):
    duration = _numeric(raw, "sleep_duration")
    score = pd.Series(np.nan, index=raw.index, dtype="float64")
    score.loc[duration.between(7.0, 9.0, inclusive="both")] = 1.0
    score.loc[
        duration.between(6.0, 7.0, inclusive="left")
        | duration.between(9.0, 10.0, inclusive="right")
    ] = 0.5
    score.loc[duration.notna() & score.isna()] = 0.0
    return score


def _step_count_score(raw):
    steps = _numeric(raw, "step_count")
    score = pd.Series(np.nan, index=raw.index, dtype="float64")
    score.loc[steps >= 10000.0] = 1.0
    score.loc[steps.between(5000.0, 10000.0, inclusive="left")] = 0.5
    score.loc[steps.notna() & score.isna()] = 0.0
    return score


def _exercise_duration_score(raw):
    minutes = _numeric(raw, "exercise_duration")
    score = pd.Series(np.nan, index=raw.index, dtype="float64")
    score.loc[minutes >= 30.0] = 1.0
    score.loc[minutes.between(10.0, 30.0, inclusive="left")] = 0.5
    score.loc[minutes.notna() & score.isna()] = 0.0
    return score


def _water_intake_score(raw):
    water = _numeric(raw, "water_intake")
    score = pd.Series(np.nan, index=raw.index, dtype="float64")
    score.loc[water.between(2.0, 4.0, inclusive="both")] = 1.0
    score.loc[
        water.between(1.5, 2.0, inclusive="left")
        | water.between(4.0, 4.5, inclusive="right")
    ] = 0.5
    score.loc[water.notna() & score.isna()] = 0.0
    return score


def _bmi_score(raw):
    bmi = _numeric(raw, "bmi")
    score = pd.Series(np.nan, index=raw.index, dtype="float64")
    score.loc[bmi.between(18.5, 24.9, inclusive="both")] = 1.0
    score.loc[
        bmi.between(17.0, 18.5, inclusive="left")
        | bmi.between(24.9, 29.9, inclusive="right")
    ] = 0.5
    score.loc[bmi.notna() & score.isna()] = 0.0
    return score


def _heart_rate_score(raw):
    heart_rate = _numeric(raw, "heart_rate")
    score = pd.Series(np.nan, index=raw.index, dtype="float64")
    score.loc[heart_rate.between(60.0, 90.0, inclusive="both")] = 1.0
    score.loc[
        heart_rate.between(50.0, 60.0, inclusive="left")
        | heart_rate.between(90.0, 100.0, inclusive="right")
    ] = 0.5
    score.loc[heart_rate.notna() & score.isna()] = 0.0
    return score


def add_lifestyle_pillar_bottleneck(raw, deps, aux):
    sleep = _average_available(
        raw.index,
        [
            _sleep_duration_score(raw),
            _categorical_score(raw, "sleep_quality", SLEEP_QUALITY_SCORE),
        ],
    )
    movement = _average_available(
        raw.index,
        [
            _step_count_score(raw),
            _exercise_duration_score(raw),
            _categorical_score(raw, "physical_activity_level", ACTIVITY_LEVEL_SCORE),
        ],
    )
    nutrition = _average_available(
        raw.index,
        [
            _categorical_score(raw, "diet_type", DIET_TYPE_SCORE),
            _water_intake_score(raw),
        ],
    )
    stress = _average_available(
        raw.index,
        [
            _categorical_score(raw, "stress_level", STRESS_LEVEL_SCORE),
        ],
    )
    substance = _average_available(
        raw.index,
        [
            _categorical_score(raw, "smoking_alcohol", SMOKING_ALCOHOL_SCORE),
        ],
    )
    cardiometabolic = _average_available(
        raw.index,
        [
            _bmi_score(raw),
            _heart_rate_score(raw),
        ],
    )

    pillars = pd.DataFrame(
        {
            "sleep": sleep,
            "movement": movement,
            "nutrition": nutrition,
            "stress": stress,
            "substance": substance,
            "cardiometabolic": cardiometabolic,
        },
        index=raw.index,
    )

    values = pillars.to_numpy(dtype="float64", copy=True)
    sorted_values = np.sort(values, axis=1)
    weakest_names = pillars.idxmin(axis=1)

    features = pd.DataFrame(index=raw.index)
    features["sleep_pillar_score"] = pillars["sleep"]
    features["movement_pillar_score"] = pillars["movement"]
    features["nutrition_pillar_score"] = pillars["nutrition"]
    features["stress_pillar_score"] = pillars["stress"]
    features["substance_pillar_score"] = pillars["substance"]
    features["cardiometabolic_pillar_score"] = pillars["cardiometabolic"]
    features["weakest_pillar_score"] = sorted_values[:, 0]
    features["weakest_pillar_id"] = weakest_names.map(PILLAR_NAME_TO_ID).astype("int8")
    features["weakest_pillar_name"] = weakest_names.astype("string")
    features["second_weakest_pillar_score"] = sorted_values[:, 1]
    features["low_pillar_count"] = (pillars < 0.34).sum(axis=1).astype("int8")
    features["high_pillar_count"] = (pillars > 0.66).sum(axis=1).astype("int8")
    features["pillar_score_spread"] = sorted_values[:, -1] - sorted_values[:, 0]
    features["two_weakest_pillar_mean"] = sorted_values[:, :2].mean(axis=1)
    features["two_strongest_pillar_mean"] = sorted_values[:, -2:].mean(axis=1)
    features["strongest_minus_weakest_gap"] = sorted_values[:, -1] - sorted_values[:, 0]
    features["strongest_two_minus_weakest_two_gap"] = (
        features["two_strongest_pillar_mean"] - features["two_weakest_pillar_mean"]
    )
    features["pillar_balance_ratio"] = (
        features["weakest_pillar_score"] / (sorted_values[:, -1] + 0.001)
    )

    return features


FEATURE_GROUPS = [
    {
        "name": "lifestyle_pillar_bottleneck",
        "fn": add_lifestyle_pillar_bottleneck,
        "depends_on": [],
        "description": "Creates domain-informed lifestyle pillar scores and bottleneck balance features from sleep, movement, nutrition, stress, substance, and cardiometabolic signals.",
    }
]