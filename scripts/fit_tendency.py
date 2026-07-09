"""Fit the serve-tendency model and save a versioned posterior artifact.

Run from repo root:  uv run python -m scripts.fit_tendency
"""

import datetime as dt
import json
from pathlib import Path

import arviz as az

from wimbledon.data import load_serve_direction
from wimbledon.features import reshape_serve_direction
from wimbledon.models import build_tendency_model, diagnostics_gate, fit

ARTIFACTS = Path("artifacts")
VERSION = 1

def main() -> None:
    raw = load_serve_direction("m")
    grass = raw[raw["Surface"] == "Grass"]  # start grass-only; widen later with a surface dim
    obs = reshape_serve_direction(grass)
    print(f"{obs['server'].nunique()} servers, {len(obs)} rows")

    model = build_tendency_model(obs)
    idata = fit(model)

    gate = diagnostics_gate(idata)
    print(json.dumps(gate, indent=2))
    if not gate["pass"]:
        raise SystemExit("Diagnostics gate failed — do not ship this posterior.")

    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / f"posterior_tendency_v{VERSION}_{dt.date.today():%Y%m%d}.nc"
    az.to_netcdf(idata, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
