import numpy as np
import pandas as pd


ORDERED_RESPONSE_SPECS = {
    "stress_level": {
        "protective": "low",
        "midpoint": "medium",
        "adverse": "high",
    },
    "sleep_quality": {
        "protective": "good",
        "midpoint": "average",
        "adverse": "poor",
    },
    "physical_activity_level": {
        "protective": "active",
        "midpoint": "moderate",
        "adverse": "sedentary",
    },
    "smoking_alcohol": {
        "protective": "no",
        "midpoint": "occasional",
        "adverse": "yes",
    },
}


def add_ordinal_response_tree_decomposition(raw, deps, aux):
    new_features = pd.DataFrame(index=raw.index)

    midpoint_columns = []
    direction_columns = []

    for column_name, spec in ORDERED_RESPONSE_SPECS.items():
        midpoint_col = column_name + "_midpoint_flag"
        direction_col = column_name + "_endpoint_risk_direction"

        if column_name in raw.columns:
            values = raw[column_name].astype("string").str.strip().str.lower()

            midpoint_map = {
                spec["protective"]: 0.0,
                spec["midpoint"]: 1.0,
                spec["adverse"]: 0.0,
            }
            direction_map = {
                spec["protective"]: -1.0,
                spec["midpoint"]: 0.0,
                spec["adverse"]: 1.0,
            }

            new_features[midpoint_col] = values.map(midpoint_map).astype("float64")
            new_features[direction_col] = values.map(direction_map).astype("float64")
        else:
            new_features[midpoint_col] = np.nan
            new_features[direction_col] = np.nan

        midpoint_columns.append(midpoint_col)
        direction_columns.append(direction_col)

    midpoint_matrix = new_features[midpoint_columns]
    direction_matrix = new_features[direction_columns]

    observed_count = direction_matrix.notna().sum(axis=1).astype("float64")
    midpoint_count = midpoint_matrix.sum(axis=1, min_count=1).fillna(0.0)
    endpoint_count = ((direction_matrix == -1.0) | (direction_matrix == 1.0)).sum(axis=1).astype("float64")
    adverse_endpoint_count = (direction_matrix == 1.0).sum(axis=1).astype("float64")
    protective_endpoint_count = (direction_matrix == -1.0).sum(axis=1).astype("float64")

    has_observed = observed_count > 0.0
    has_endpoint = endpoint_count > 0.0

    new_features["observed_ordered_count"] = observed_count
    new_features["midpoint_count"] = midpoint_count
    new_features["endpoint_count"] = endpoint_count
    new_features["adverse_endpoint_count"] = adverse_endpoint_count
    new_features["protective_endpoint_count"] = protective_endpoint_count

    new_features["midpoint_fraction"] = np.where(has_observed, midpoint_count / observed_count, np.nan)
    new_features["endpoint_fraction"] = np.where(has_observed, endpoint_count / observed_count, np.nan)
    new_features["adverse_endpoint_fraction"] = np.where(
        has_observed,
        adverse_endpoint_count / observed_count,
        np.nan,
    )
    new_features["protective_endpoint_fraction"] = np.where(
        has_observed,
        protective_endpoint_count / observed_count,
        np.nan,
    )
    new_features["net_endpoint_risk"] = np.where(
        has_endpoint,
        (adverse_endpoint_count - protective_endpoint_count) / endpoint_count,
        0.0,
    )

    return new_features


FEATURE_GROUPS = [
    {
        "name": "ordinal_response_tree_decomposition",
        "fn": add_ordinal_response_tree_decomposition,
        "depends_on": [],
        "description": "Decomposes ordered self-reported lifestyle categories into midpoint response behavior and endpoint health-risk direction features.",
    }
]