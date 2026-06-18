# TheML

TheML (`tml`) is a filesystem-first ML experimentation CLI. Canonical state is
stored as small YAML, Markdown, JSON, and Python artifacts; SQLite is a local
rebuildable index.

Use `uv` for all environment work:

```bash
uv pip install -e .
uv run tml --help
```

Initialize the first Kaggle project:

```bash
uv run tml init project playground-series-s6e6
uv run tml project use playground-series-s6e6
uv run tml root status
```

AutoGluon is optional at install time. Without it or without Kaggle data, ROOT
runs write structured failure artifacts instead of crashing.
