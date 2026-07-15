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
- `uv run python -m scripts.fit_predictive` — GBM baseline, chronological holdout, saves artifacts/*.joblib
- `uv run python -m scripts.calibration_report` — Brier + reliability curve vs closing odds, saves artifacts/calibration_*
- `uv run uvicorn api.main:app --reload` -> http://localhost:8000/docs
- `cd web && npm run dev` -> http://localhost:3000 (dashboard; expects the API on :8000)
- No real bundle yet? `uv run python -m scripts.make_synthetic_data` fills `data/` with a
  schema-matching stand-in (including synthetic closing odds) so the whole loop runs end to end.

## Status (Phase P, see DESIGN_v2.md)
- Tendency model (PyMC), predictive GBM baseline, and calibration report all fit + pass
  their gates against synthetic data. Re-run `fit_tendency` and `calibration_report`
  against the real MCP/TML bundle + tennis-data.co.uk odds once that's unzipped into `data/`.
- Dashboard: single view (server search -> side/serve-number -> `/tendency` -> bar chart
  with 80% CI) in `web/`.
- decisions.py: stub — v1.1, not v1 (see DESIGN_v2.md's definition of done)
- DESIGN_v2.md has the closed Phase D decisions and the Phase P entry checklist
