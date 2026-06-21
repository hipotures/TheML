from __future__ import annotations

import pandas as pd


def add_numeric_missingness_indicators(raw, deps, aux, ctx):
    _ = (deps, aux, ctx)
    out = pd.DataFrame(index=raw.index)
    for column in raw.select_dtypes(include='number').columns:
        if str(column).lower() == 'id':
            continue
        out[f'{column}_is_missing'] = raw[column].isna().astype('int8')
    return out


FEATURE_GROUPS = [
    {
        'name': 'numeric_missingness_indicators',
        'fn': add_numeric_missingness_indicators,
        'depends_on': [],
        'description': 'Binary missingness indicators for numeric raw columns.',
    }
]
