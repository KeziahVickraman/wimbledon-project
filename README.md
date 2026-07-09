# wimbledon

## Setup (once)
1. Unzip wimbledon-datasets.zip contents into `data/` (so `data/atp_matches/`, `data/mcp/`)
2. `uv sync` — creates .venv, installs everything incl. the package in editable mode
3. VS Code: Python: Select Interpreter -> .venv

## The loop
- Explore in `notebooks/` (first cell: `%load_ext autoreload` / `%autoreload 2`)
- Stable code moves to `src/wimbledon/`
- `uv run pytest` — data loaders are covered; add tests before new src code (iron law)
- `uv run python -m scripts.fit_tendency` — fits + diagnostics gate + saves artifacts/*.nc
- `uv run uvicorn api.main:app --reload` -> http://localhost:8000/docs

## Status
- data.py, features.surface_elo, reshape, models.build_tendency_model: written and
  smoke-tested against the real bundle (loaders + reshape verified; fit not yet run)
- features.randomize_winner_framing: TODO (leakage guard for the predictive model)
- decisions.py: stub — Phase P
- DESIGN.md has open decisions to make BEFORE building further
