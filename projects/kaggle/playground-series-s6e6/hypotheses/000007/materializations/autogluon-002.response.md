import pandas as pd
import numpy as np

TOLERANCE = 0.02
BAND_COLUMNS = ("u", "g", "r", "i", "z")
TEMPLATE_FORWARD = (0, 1, 2, 3, 4)
TEMPLATE_REVERSE = (4, 3, 2, 1, 0)
INTERIOR_BANDS = (1, 2, 3)

def _ordered_band_indices(mags):
    order = np.argsort(mags, kind="mergesort")
    sorted_vals = mags[order]
    sorted_idx = order.tolist()
    final_order = []
    tie_blocks = 0
    start = 0
    n = len(sorted_idx)

    while start < n:
        end = start + 1
        while end < n and abs(sorted_vals[end] - sorted_vals[end - 1]) <= TOLERANCE:
            end += 1

        block = sorted_idx[start:end]
        block.sort()
        final_order.extend(block)
        tie_blocks += 1
        start = end

    return final_order, tie_blocks

def _count_pairwise_inversions(mags, template):
    template_pos = np.full(5, -1, dtype=np.int8)
    for rank, band_idx in enumerate(template):
        template_pos[band_idx] = rank

    inv = 0
    for a in range(5):
        ma = mags[a]
        pa = template_pos[a]
        for b in range(a + 1, 5):
            mb = mags[b]
            if abs(ma - mb) <= TOLERANCE:
                continue
            expected = pa < template_pos[b]
            actual = ma < mb
            if expected != actual:
                inv += 1
    return inv

def _initial_signs(mags):
    d1 = mags[1] - mags[0]
    d2 = mags[2] - mags[1]
    d3 = mags[3] - mags[2]
    d4 = mags[4] - mags[3]

    return np.array([
        1 if d1 > TOLERANCE else -1 if d1 < -TOLERANCE else 0,
        1 if d2 > TOLERANCE else -1 if d2 < -TOLERANCE else 0,
        1 if d3 > TOLERANCE else -1 if d3 < -TOLERANCE else 0,
        1 if d4 > TOLERANCE else -1 if d4 < -TOLERANCE else 0,
    ], dtype=np.int8)

def _impute_signs(initial_signs):
    q = initial_signs.copy()
    n = len(q)

    for i in range(n):
        if q[i] != 0:
            continue

        left_idx = None
        right_idx = None

        for li in range(i - 1, -1, -1):
            if q[li] != 0:
                left_idx = li
                break
        for ri in range(i + 1, n):
            if q[ri] != 0:
                right_idx = ri
                break

        if left_idx is None and right_idx is None:
            continue
        if left_idx is None:
            q[i] = q[right_idx]
            continue
        if right_idx is None:
            q[i] = q[left_idx]
            continue

        dl = i - left_idx
        dr = right_idx - i
        ls = q[left_idx]
        rs = q[right_idx]

        if dl < dr:
            q[i] = ls
        elif dr < dl:
            q[i] = rs
        else:
            q[i] = ls if ls == rs else 0

    return q

def add_band_order_topology(raw, deps, aux):
    mags = raw[list(BAND_COLUMNS)].to_numpy(dtype=np.float64, copy=False)
    n_rows = len(raw)

    flat_topology = np.zeros(n_rows, dtype=np.int8)
    brightest_pos = np.zeros(n_rows, dtype=np.int8)
    faintest_pos = np.zeros(n_rows, dtype=np.int8)
    brightest_interior = np.zeros(n_rows, dtype=np.int8)
    faintest_interior = np.zeros(n_rows, dtype=np.int8)
    tie_block_count = np.zeros(n_rows, dtype=np.int8)

    inv_tpl = np.zeros(n_rows, dtype=np.int8)
    inv_rev = np.zeros(n_rows, dtype=np.int8)

    monotone = np.zeros(n_rows, dtype=np.int8)
    sign_changes = np.zeros(n_rows, dtype=np.int8)
    one_turn = np.zeros(n_rows, dtype=np.int8)
    first_change = np.zeros(n_rows, dtype=np.int8)
    second_change = np.zeros(n_rows, dtype=np.int8)

    bucket_strict = np.zeros(n_rows, dtype=np.int8)
    bucket_single = np.zeros(n_rows, dtype=np.int8)
    bucket_multi = np.zeros(n_rows, dtype=np.int8)
    bucket_flat = np.zeros(n_rows, dtype=np.int8)
    topology_bucket = np.array(["flat_or_ambiguous"] * n_rows, dtype=object)

    for i in range(n_rows):
        row = mags[i]
        spread = row.max() - row.min()

        if spread <= 2.0 * TOLERANCE:
            flat_topology[i] = 1
            brightest_pos[i] = 0
            faintest_pos[i] = 0
            brightest_interior[i] = 0
            faintest_interior[i] = 0
            tie_block_count[i] = _ordered_band_indices(row)[1]
            inv_tpl[i] = 0
            inv_rev[i] = 0
            monotone[i] = 1
            sign_changes[i] = 0
            one_turn[i] = 0
            first_change[i] = 0
            second_change[i] = 0
            bucket_flat[i] = 1
            topology_bucket[i] = "flat_or_ambiguous"
            continue

        order, blocks = _ordered_band_indices(row)
        tie_block_count[i] = blocks

        bright_idx = order[0]
        faint_idx = order[-1]
        brightest_pos[i] = order.index(bright_idx) + 1
        faintest_pos[i] = order.index(faint_idx) + 1
        brightest_interior[i] = int(bright_idx in INTERIOR_BANDS)
        faintest_interior[i] = int(faint_idx in INTERIOR_BANDS)

        inv_tpl[i] = _count_pairwise_inversions(row, TEMPLATE_FORWARD)
        inv_rev[i] = _count_pairwise_inversions(row, TEMPLATE_REVERSE)

        q0 = _initial_signs(row)
        q = _impute_signs(q0)

        changes = []
        for j in range(1, len(q)):
            if q[j] != q[j - 1]:
                changes.append(j)

        sign_changes[i] = len(changes)
        first_change[i] = changes[0] if changes else 0
        second_change[i] = changes[1] if len(changes) > 1 else 0
        one_turn[i] = 1 if len(changes) == 1 else 0

        nz = q[q != 0]
        monotone[i] = 1 if nz.size > 0 and np.all(nz == nz[0]) else 0

        if monotone[i] == 1 and sign_changes[i] == 0:
            bucket_strict[i] = 1
            topology_bucket[i] = "strictly_monotone"
        elif sign_changes[i] == 1:
            bucket_single[i] = 1
            topology_bucket[i] = "single_turn"
        elif sign_changes[i] >= 2:
            bucket_multi[i] = 1
            topology_bucket[i] = "multi_turn"
        else:
            bucket_flat[i] = 1
            topology_bucket[i] = "flat_or_ambiguous"

    return pd.DataFrame(
        {
            "flat_topology": flat_topology,
            "brightest_band_pos": brightest_pos,
            "faintest_band_pos": faintest_pos,
            "brightest_band_is_interior": brightest_interior,
            "faintest_band_is_interior": faintest_interior,
            "band_tie_block_count": tie_block_count,
            "inversion_count_template": inv_tpl,
            "inversion_count_reverse_template": inv_rev,
            "monotone_flag": monotone,
            "sign_change_count": sign_changes,
            "one_turn_flag": one_turn,
            "first_change": first_change,
            "second_change": second_change,
            "topology_bucket": topology_bucket,
            "bucket_strictly_monotone": bucket_strict,
            "bucket_single_turn": bucket_single,
            "bucket_multi_turn": bucket_multi,
            "bucket_flat_or_ambiguous": bucket_flat,
        },
        index=raw.index,
    )

FEATURE_GROUPS = [
    {
        "name": "band_order_topology",
        "fn": add_band_order_topology,
        "depends_on": [],
        "description": "Creates tolerance-aware topology features for ugriz ordering and local sign-turn structure.",
    }
]