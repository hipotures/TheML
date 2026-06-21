## Python environment and package management

Use `uv` as the project environment manager, but install Python packages through `uv pip`, not through plain `pip`, `poetry`, `conda`, or other package managers.

Rules:

* Use the existing `.venv` managed by `uv` whenever possible.
* Install dependencies with:

```bash
uv pip install <package>
```

* Install from requirement files with:

```bash
uv pip install -r requirements.txt
```

* Do not use:

```bash
pip install <package>
python -m pip install <package>
conda install <package>
poetry add <package>
```

unless the user explicitly asks for it.

* When running Python scripts, prefer:

```bash
uv run python <script.py>
```

* When running tools installed in the environment, prefer:

```bash
uv run <command>
```

* Keep dependency changes minimal and explicit. If a new dependency is required, explain why it is needed and add it to the appropriate requirements file if the project uses one.

## When checking command output:
- Run commands directly without pipes when possible
- If you need to limit output, use command-specific flags (e.g., `git log -n 10` instead of `git log | head -10`)
- Avoid chained pipes that can cause output to buffer indefinitely

## CLI output conventions
- Use Rich consistently for user-facing CLI output.
- Prefer Rich tables for structured lists, summaries, and status reports.
- Prefer Rich trees for hierarchical filesystem/project layouts.
- Use Rich progress bars or spinners for long-running operations such as downloads, extraction, compression, indexing, training, and bulk artifact generation.
- Keep plain text only when machine-readable output is explicitly requested.
- Flush or live-update long-running output so the terminal does not appear idle while work is in progress.
- Do not tune Rich table column widths based on Codex tool output screenshots/logs; the user's real terminal renders differently. Only change column layout or widths when explicitly requested.
- When verifying CLI data for yourself, prefer JSON output flags such as `json=true`/`json_output=true` over Rich table output whenever available.

## Testing policy
- Do not create, expand, or modify test files unless the user explicitly asks for tests in that specific request.
- Do not add tests as a default safety step for code, prompt, documentation, or configuration changes.
- Do not run test suites unless the user explicitly asks to run tests.

## Git workflow
- If you modify files in a Git repository, do not finish the task with uncommitted changes unless the user explicitly says not to commit.
- Any task that changes files must end in one of two states: changes committed, or an explicit explanation why they were not committed.
- Before committing, run relevant verification and inspect `git status --short`.
- If the user asks for `git push`, first ask whether they want to check test coverage and add missing tests before pushing.
- Commit only changes made for the current task.
- Generated hypothesis artifacts under `projects/*/*/hypotheses/` are source artifacts and must be committed.
- If changes in `research_hypotheses/` are detected or created, you must commit those `research_hypotheses/` changes in a separate commit from code, script, or documentation changes outside `research_hypotheses/`.
- Never commit unrelated user changes.
- Use concise commit messages.
