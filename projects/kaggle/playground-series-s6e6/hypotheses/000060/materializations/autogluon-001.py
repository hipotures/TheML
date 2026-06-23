import numpy as np
import pandas as pd

_CLASS_LABELS = ("GALAXY", "QSO", "STAR")
_REDSHIFT_BINS = (-0.01, 0.1, 0.4, 1.0, 2.0, 2.8, 3.5, 4.5, 7.02)
_I_BANDS = (17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0)
_ALPHA_SMOOTHING = 30.0
_BACKOFF_STRENGTH = (80.0, 40.0, 20.0)
_EPSILON = 1e-6
_UNKNOWN_TOKEN = "__UNKNOWN__"
_LABEL_CANDIDATES = ("class", "target", "label", "y", "labels")
_ID_COL = "id"


def _normalize_class_series(values):
    if values is None:
        return pd.Series(dtype="string")
    s = pd.Series(values).astype("string").str.strip().str.upper()
    s = s.str.replace(r"[^A-Z/]", "", regex=True)
    s = s.replace(
        {
            "Q": "QSO",
            "QUASAR": "QSO",
            "QSO": "QSO",
            "G": "GALAXY",
            "GAL": "GALAXY",
            "GALAXY": "GALAXY",
            "S": "STAR",
            "STAR": "STAR",
        }
    )
    return s.where(s.isin(_CLASS_LABELS), pd.NA)


def _build_global_prior(label_series):
    cleaned = _normalize_class_series(label_series).dropna()
    if cleaned.empty:
        return pd.Series(
            {
                "GALAXY": 1.0 / len(_CLASS_LABELS),
                "QSO": 1.0 / len(_CLASS_LABELS),
                "STAR": 1.0 / len(_CLASS_LABELS),
            }
        )
    counts = cleaned.value_counts().reindex(_CLASS_LABELS, fill_value=0).astype(float)
    total = float(counts.sum())
    if total <= 0:
        return pd.Series(
            {
                "GALAXY": 1.0 / len(_CLASS_LABELS),
                "QSO": 1.0 / len(_CLASS_LABELS),
                "STAR": 1.0 / len(_CLASS_LABELS),
            }
        )
    return counts / total


def _extract_labels(raw, aux):
    raw_index = raw.index
    for label_col in _LABEL_CANDIDATES:
        if label_col in raw.columns:
            raw_norm = _normalize_class_series(raw[label_col])
            if raw_norm.notna().any():
                return raw_norm, pd.Series(dtype="string")

    if aux is None or not isinstance(aux, pd.DataFrame) or aux.empty:
        return pd.Series(pd.NA, index=raw_index, dtype="string"), pd.Series(dtype="string")

    aux_pool = pd.Series(dtype="string")

    for label_col in _LABEL_CANDIDATES:
        if label_col not in aux.columns:
            continue

        aux_norm = _normalize_class_series(aux[label_col])
        if not aux_norm.notna().any():
            continue

        if aux_pool.empty:
            aux_pool = aux_norm.dropna()

        if _ID_COL in raw.columns and _ID_COL in aux.columns:
            aux_map_src = aux[[_ID_COL, label_col]].dropna(subset=[label_col])
            if not aux_map_src.empty:
                aux_map = pd.Series(
                    _normalize_class_series(aux_map_src[label_col]).to_numpy(),
                    index=aux_map_src[_ID_COL].to_numpy(),
                    dtype="string",
                )
                mapped = _normalize_class_series(raw[_ID_COL].map(aux_map))
                if mapped.notna().any():
                    return mapped, aux_pool

        if len(aux) == len(raw):
            candidate = pd.Series(aux_norm.to_numpy(), index=raw_index, dtype="string")
            if candidate.notna().any():
                return candidate, aux_pool

    return pd.Series(pd.NA, index=raw_index, dtype="string"), aux_pool


def _build_level_posteriors(source_df, group_cols, global_prior):
    if source_df.empty:
        return pd.DataFrame(columns=tuple(_CLASS_LABELS) + ("n",))
    counts = source_df.groupby(group_cols + ["class"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=_CLASS_LABELS, fill_value=0).astype(float)
    n = counts.sum(axis=1).astype(float)
    posterior = counts.add(_ALPHA_SMOOTHING * global_prior, axis=1)
    posterior = posterior.div(n + _ALPHA_SMOOTHING, axis=0)
    posterior["n"] = n
    return posterior


def _lookup_level(level_df, row_keys, raw_index, global_prior):
    if level_df.empty:
        base = np.tile(global_prior.to_numpy(), (len(raw_index), 1))
        return pd.DataFrame(base, columns=_CLASS_LABELS, index=raw_index), pd.Series(
            0.0, index=raw_index
        )

    aligned = level_df.reindex(row_keys)
    probs = aligned.reindex(columns=_CLASS_LABELS).copy()
    probs = probs.fillna({label: float(global_prior[label]) for label in _CLASS_LABELS})
    counts = aligned.get("n")
    if not isinstance(counts, pd.Series):
        counts = pd.Series(0.0, index=row_keys)
    counts = counts.fillna(0.0).astype(float)
    probs.index = raw_index
    counts.index = raw_index
    return probs, counts


def _global_features(index, global_prior):
    p = np.tile(global_prior.to_numpy(dtype=float), (len(index), 1))
    p = np.clip(p, _EPSILON, 1.0 - _EPSILON)
    logits = np.log((p + _EPSILON) / (1.0 - p + 3.0 * _EPSILON))
    entropy = -(p * np.log(p)).sum(axis=1)
    top2 = np.sort(p, axis=1)
    top2_gap = top2[:, 2] - top2[:, 1]

    return pd.DataFrame(
        {
            "class_prior_galaxy": p[:, 0],
            "class_prior_qso": p[:, 1],
            "class_prior_star": p[:, 2],
            "class_logit_margin_galaxy": logits[:, 0],
            "class_logit_margin_qso": logits[:, 1],
            "class_logit_margin_star": logits[:, 2],
            "class_prior_entropy": entropy,
            "class_top2_gap": top2_gap,
        },
        index=index,
    )


def add_metadata_redshift_magnitude_class_priors(raw, deps, aux):
    frame = raw
    index = frame.index

    raw_labels, aux_label_pool = _extract_labels(frame, aux)
    global_pool = pd.concat([raw_labels.dropna(), aux_label_pool], ignore_index=True)
    global_prior = _build_global_prior(global_pool)

    required = ("spectral_type", "galaxy_population", "redshift", "i")
    if any(col not in frame.columns for col in required):
        return _global_features(index, global_prior)

    spectral = frame["spectral_type"].fillna(_UNKNOWN_TOKEN).astype("string").astype("object")
    galaxy_population = frame["galaxy_population"].fillna(_UNKNOWN_TOKEN).astype("string").astype("object")

    redshift = pd.to_numeric(frame["redshift"], errors="coerce")
    i_mag = pd.to_numeric(frame["i"], errors="coerce")
    redshift = redshift.mask(~redshift.between(-0.05, 7.5))
    i_mag = i_mag.mask(~i_mag.between(8.0, 31.0))

    red_bins = (float("-inf"),) + _REDSHIFT_BINS + (float("inf"),)
    i_bins = (float("-inf"),) + _I_BANDS + (float("inf"),)

    red_bin = pd.cut(redshift, bins=red_bins, labels=False, include_lowest=True, right=True).fillna(-1).astype(int)
    i_bin = pd.cut(i_mag, bins=i_bins, labels=False, include_lowest=True, right=True).fillna(-1).astype(int)

    labeled_mask = raw_labels.notna()
    train_df = pd.DataFrame(
        {
            "spectral_type": spectral.loc[labeled_mask],
            "galaxy_population": galaxy_population.loc[labeled_mask],
            "redshift_bin": red_bin.loc[labeled_mask],
            "i_bin": i_bin.loc[labeled_mask],
            "class": raw_labels.loc[labeled_mask],
        },
        index=index[labeled_mask],
    )

    level0 = _build_level_posteriors(train_df, ["spectral_type", "galaxy_population", "redshift_bin", "i_bin"], global_prior)
    level1 = _build_level_posteriors(train_df, ["spectral_type", "galaxy_population", "redshift_bin"], global_prior)
    level2 = _build_level_posteriors(train_df, ["spectral_type", "galaxy_population"], global_prior)

    keys0 = pd.MultiIndex.from_arrays(
        [spectral.to_numpy(), galaxy_population.to_numpy(), red_bin.to_numpy(), i_bin.to_numpy()],
        names=("spectral_type", "galaxy_population", "redshift_bin", "i_bin"),
    )
    keys1 = pd.MultiIndex.from_arrays(
        [spectral.to_numpy(), galaxy_population.to_numpy(), red_bin.to_numpy()],
        names=("spectral_type", "galaxy_population", "redshift_bin"),
    )
    keys2 = pd.MultiIndex.from_arrays(
        [spectral.to_numpy(), galaxy_population.to_numpy()],
        names=("spectral_type", "galaxy_population"),
    )

    p0, n0 = _lookup_level(level0, keys0, index, global_prior)
    p1, n1 = _lookup_level(level1, keys1, index, global_prior)
    p2, n2 = _lookup_level(level2, keys2, index, global_prior)

    p0_arr = p0.to_numpy(dtype=float)
    p1_arr = p1.to_numpy(dtype=float)
    p2_arr = p2.to_numpy(dtype=float)
    g_arr = np.tile(global_prior.to_numpy(dtype=float), (len(index), 1))

    n0_arr = n0.to_numpy(dtype=float)
    n1_arr = n1.to_numpy(dtype=float)
    n2_arr = n2.to_numpy(dtype=float)

    w0 = np.divide(n0_arr, n0_arr + _BACKOFF_STRENGTH[0], out=np.zeros_like(n0_arr, dtype=float), where=(n0_arr + _BACKOFF_STRENGTH[0]) != 0)
    w1 = np.divide(n1_arr, n1_arr + _BACKOFF_STRENGTH[1], out=np.zeros_like(n1_arr, dtype=float), where=(n1_arr + _BACKOFF_STRENGTH[1]) != 0)
    w2 = np.divide(n2_arr, n2_arr + _BACKOFF_STRENGTH[2], out=np.zeros_like(n2_arr, dtype=float), where=(n2_arr + _BACKOFF_STRENGTH[2]) != 0)

    p2_blend = p2_arr * w2[:, None] + g_arr * (1.0 - w2[:, None])
    p1_blend = p1_arr * w1[:, None] + p2_blend * (1.0 - w1[:, None])
    p = p0_arr * w0[:, None] + p1_blend * (1.0 - w0[:, None])

    p = np.clip(p, _EPSILON, 1.0 - _EPSILON)
    margins = np.log((p + _EPSILON) / (1.0 - p + 3.0 * _EPSILON))
    entropy = -(p * np.log(np.clip(p, _EPSILON, 1.0))).sum(axis=1)
    sorted_p = np.sort(p, axis=1)
    top2_gap = sorted_p[:, 2] - sorted_p[:, 1]

    features = pd.DataFrame(
        {
            "class_prior_galaxy": p[:, 0],
            "class_prior_qso": p[:, 1],
            "class_prior_star": p[:, 2],
            "class_logit_margin_galaxy": margins[:, 0],
            "class_logit_margin_qso": margins[:, 1],
            "class_logit_margin_star": margins[:, 2],
            "class_prior_entropy": entropy,
            "class_top2_gap": top2_gap,
        },
        index=index,
    )
    return features


FEATURE_GROUPS = [
    {
        "name": "metadata_redshift_magnitude_class_priors",
        "fn": add_metadata_redshift_magnitude_class_priors,
        "depends_on": [],
        "description": "Context anchored class priors in spectral/population/redshift/i-band strata with hierarchical backoff and shrinkage.",
    }
]