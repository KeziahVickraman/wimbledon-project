"""Serving layer. Loads the posterior artifact once; request-time work is indexing only.

Run:  uv run uvicorn api.main:app --reload   ->  http://localhost:8000/docs
"""

import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

import arviz as az
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

ARTIFACT = sorted(Path("artifacts").glob("posterior_tendency_v*.nc"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ARTIFACT:
        raise RuntimeError("No posterior artifact found — run scripts/fit_tendency.py first.")
    path = ARTIFACT[-1]
    idata = az.from_netcdf(path)
    post = idata.posterior
    # Precompute per-server direction probabilities: softmax(mu + z*sigma), draws collapsed
    theta = post["mu"] + post["z"] * post["sigma_server"]
    e = np.exp(theta - theta.max("dir"))
    p = e / e.sum("dir")
    state["p_mean"] = p.mean(("chain", "draw"))
    state["p_lo"] = p.quantile(0.10, ("chain", "draw"))
    state["p_hi"] = p.quantile(0.90, ("chain", "draw"))
    state["servers"] = list(post["server"].values)
    state["hash"] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    state["version"] = path.stem
    yield
    state.clear()


app = FastAPI(title="wimbledon-serve", lifespan=lifespan)


class TendencyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server: str
    side: str  # "deuce" | "ad"
    serve_n: str = "first"  # "first" | "second"


class DirEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: str
    p_mean: float
    p_ci80: tuple[float, float]


class TendencyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server: str
    side: str
    serve_n: str
    estimates: list[DirEstimate]


@app.get("/health")
def health() -> dict:
    return {"model": state["version"], "artifact_hash": state["hash"], "servers": len(state["servers"])}


@app.post("/tendency", response_model=TendencyResponse)
def tendency(req: TendencyRequest) -> TendencyResponse:
    if req.server not in state["servers"]:
        raise HTTPException(404, f"unknown server '{req.server}'")
    sel = dict(server=req.server, side=req.side, serve_n=req.serve_n)
    try:
        mean = state["p_mean"].sel(**sel)
        lo, hi = state["p_lo"].sel(**sel), state["p_hi"].sel(**sel)
    except KeyError as e:
        raise HTTPException(422, str(e)) from e
    ests = [
        DirEstimate(
            direction=str(d),
            p_mean=float(mean.sel(dir=d)),
            p_ci80=(float(lo.sel(dir=d)), float(hi.sel(dir=d))),
        )
        for d in mean["dir"].values
    ]
    return TendencyResponse(server=req.server, side=req.side, serve_n=req.serve_n, estimates=ests)
