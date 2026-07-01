"""Raw AutoGluon baseline.

This materialization intentionally defines no feature groups. The fixed
AutoGluon wrapper will train on the raw project columns, with profile-level
ignored columns still applied.
"""

FEATURE_GROUPS = []
