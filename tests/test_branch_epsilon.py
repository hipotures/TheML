from __future__ import annotations

from pathlib import Path

import pytest

from tml.branches import grow
from tml.branches.algorithms import (
    BranchAlgorithmConfig,
    BranchAlgorithmItem,
    ScoreEpsilon,
    epsilon_delta,
    parse_score_epsilon,
    score_improves,
)
from tml.branches.grow import BranchGrowPlan
from tml.branches.run import BranchRunItem
from tml.core.errors import TmlError


def test_parse_numeric_absolute_epsilon():
    epsilon = parse_score_epsilon(0.0001)

    assert epsilon == ScoreEpsilon(raw=0.0001, kind="absolute", value=0.0001)


def test_parse_string_absolute_epsilon():
    epsilon = parse_score_epsilon("0.0001")

    assert epsilon == ScoreEpsilon(raw="0.0001", kind="absolute", value=0.0001)


def test_parse_percentage_epsilon():
    epsilon = parse_score_epsilon("0.1%")

    assert epsilon == ScoreEpsilon(raw="0.1%", kind="percent", value=0.1)
    assert epsilon_delta(0.968, epsilon) == pytest.approx(0.000968)


def test_rejects_malformed_percentage_epsilon():
    with pytest.raises(TmlError):
        parse_score_epsilon("abc%")


def test_rejects_negative_epsilon():
    with pytest.raises(TmlError):
        parse_score_epsilon("-0.1%")


def test_score_improves_with_absolute_epsilon_in_maximize_mode():
    epsilon = ScoreEpsilon(raw=0.0001, kind="absolute", value=0.0001)

    assert score_improves(0.9681, 0.9680, epsilon)
    assert not score_improves(0.96809, 0.9680, epsilon)


def test_percentage_epsilon_with_zero_reference_score():
    epsilon = ScoreEpsilon(raw="0.1%", kind="percent", value=0.1)

    assert epsilon_delta(0.0, epsilon) == 0.0
    assert score_improves(0.0, 0.0, epsilon)


def test_branch_grow_does_not_switch_accepted_parent_below_epsilon(monkeypatch, tmp_path):
    _patch_grow_plan(monkeypatch, epsilon=ScoreEpsilon(raw=0.1, kind="absolute", value=0.1))
    preferred_refs: list[str | None] = []
    next_branch = 1

    def fake_add(project_dir, *, algo_id, mode=None, preferred_parent_ref=None):
        nonlocal next_branch
        preferred_refs.append(preferred_parent_ref)
        parent_ref = preferred_parent_ref or "B000100"
        branch_id = f"B{next_branch:06d}"
        next_branch += 1
        return _algorithm_item(branch_id, parent_ref=parent_ref, parent_score=100.0)

    def fake_run(project_dir, mode=None, *, branch_id, profile_overrides=None, progress=None):
        return BranchRunItem(branch_id=branch_id, run_status="complete", metric=100.05, node_id=f"node-{branch_id}", run_seconds=1)

    monkeypatch.setattr(grow, "next_pending_branch", lambda project_dir, *, mode, profile_id: None)
    monkeypatch.setattr(grow, "branch_add_algorithmic_one", fake_add)
    monkeypatch.setattr(grow, "run_one_branch", fake_run)

    result = grow.branch_grow(tmp_path, steps=2, algo_id="default")

    assert result.processed_steps == 2
    assert preferred_refs == [None, "B000100"]
    assert result.items[1].parent_ref == "B000100"


def test_branch_grow_switches_accepted_parent_above_epsilon(monkeypatch, tmp_path):
    _patch_grow_plan(monkeypatch, epsilon=ScoreEpsilon(raw=0.1, kind="absolute", value=0.1))
    preferred_refs: list[str | None] = []
    next_branch = 1

    def fake_add(project_dir, *, algo_id, mode=None, preferred_parent_ref=None):
        nonlocal next_branch
        preferred_refs.append(preferred_parent_ref)
        parent_ref = preferred_parent_ref or "B000100"
        parent_score = 100.2 if parent_ref == "B000001" else 100.0
        branch_id = f"B{next_branch:06d}"
        next_branch += 1
        return _algorithm_item(branch_id, parent_ref=parent_ref, parent_score=parent_score)

    def fake_run(project_dir, mode=None, *, branch_id, profile_overrides=None, progress=None):
        return BranchRunItem(branch_id=branch_id, run_status="complete", metric=100.2, node_id=f"node-{branch_id}", run_seconds=1)

    monkeypatch.setattr(grow, "next_pending_branch", lambda project_dir, *, mode, profile_id: None)
    monkeypatch.setattr(grow, "branch_add_algorithmic_one", fake_add)
    monkeypatch.setattr(grow, "run_one_branch", fake_run)

    result = grow.branch_grow(tmp_path, steps=2, algo_id="default")

    assert result.processed_steps == 2
    assert preferred_refs == [None, "B000001"]
    assert result.items[1].parent_ref == "B000001"


def _patch_grow_plan(monkeypatch, *, epsilon: ScoreEpsilon) -> None:
    algorithm = BranchAlgorithmConfig(
        algorithm_id="default",
        path=Path("profiles/branch/default.yaml"),
        parent_kinds=["root", "branch"],
        source_kinds=["root"],
        max_children=10,
        epsilon=epsilon,
    )
    monkeypatch.setattr(grow, "load_branch_algorithm", lambda project_dir, algo_id: algorithm)
    monkeypatch.setattr(grow, "branch_algorithm_top_parent", lambda project_dir, *, algo_id, mode=None: None)
    monkeypatch.setattr(
        grow,
        "branch_grow_plan",
        lambda project_dir, *, steps, algo_id, mode=None, profile_overrides=None: BranchGrowPlan(
            mode=mode or "autogluon",
            profile_id="profile-a",
            algorithm_id=algo_id,
            requested_steps=steps,
            pending_branch_runs=0,
            execution_timeout_seconds=123,
            config_path=Path("profiles/branch/default.yaml"),
        ),
    )


def _algorithm_item(branch_id: str, *, parent_ref: str, parent_score: float) -> BranchAlgorithmItem:
    return BranchAlgorithmItem(
        branch_id=branch_id,
        parent_ref=parent_ref,
        source_ref="000001",
        parent_score=parent_score,
        source_score=99.0,
        composition_hash=f"hash-{branch_id}",
        materialization_path=Path(f"branches/{branch_id}/materializations/autogluon-001.py"),
    )
