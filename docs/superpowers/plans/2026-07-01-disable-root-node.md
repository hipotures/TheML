# Disable ROOT Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep submitted ROOT run nodes as history while excluding disabled nodes from active ROOT best-score selection.

**Architecture:** Store a `node.disabled.yaml` marker beside the node artifacts. Reindex maps that marker to `nodes.status = disabled` and `evaluations.status = disabled`; active ranking queries continue to use only `status='complete'`.

**Tech Stack:** Python, Typer CLI, SQLite state database, project-local YAML artifacts.

---

### Task 1: Marker And CLI

**Files:**
- Modify: `src/tml/cli/main.py`

- [ ] Add `uv run tml root disable node=<node_id> reason=<text>` to write `node.disabled.yaml`.
- [ ] Resolve the node under `projects/.../runs/*/artifacts/<node_id>`.
- [ ] Reject non-ROOT nodes using `node.start.yaml`.
- [ ] Show `⊘` for disabled rows in ROOT run status.

### Task 2: Reindex And Active Ranking

**Files:**
- Modify: `src/tml/db/reindex.py`
- Modify: `src/tml/db/state.py`

- [ ] Make `classify_node()` return `disabled` when `node.disabled.yaml` exists.
- [ ] Keep disabled nodes out of `best_score()` and incomplete counts.
- [ ] Keep disabled evaluations out of active ROOT tree selection.

### Task 3: Apply To Bad Node

**Commands:**
- `uv run tml root disable node=20260701T143825-143f83dd-4 reason="audit score too optimistic; public/CV alignment favors step 6" yes=true`
- `uv run tml reindex yes=true`
- `uv run tml root run status`
- `uv run tml root status`
