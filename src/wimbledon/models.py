"""PyMC model definitions. Sampling happens ONLY in scripts/, never in api/."""

import pandas as pd
import pymc as pm

DIRS = ["wide", "body", "T"]


def build_tendency_model(df: pd.DataFrame) -> pm.Model:
    """Hierarchical multinomial over serve direction.

    Expects one row per (server, side, serve_n) with columns wide/body/T (counts).
    Uses the aggregate ServeDirection counts — reshape upstream:
        deuce_wide/deuce_middle/deuce_t -> side='deuce', wide/body/T   (middle == body)
    """
    servers = df["server"].unique().tolist()
    coords = {
        "dir": DIRS,
        "server": servers,
        "side": ["deuce", "ad"],
        "serve_n": ["first", "second"],
    }
    s_idx = pd.Categorical(df["server"], categories=servers).codes
    side_idx = pd.Categorical(df["side"], categories=coords["side"]).codes
    sn_idx = pd.Categorical(df["serve_n"], categories=coords["serve_n"]).codes
    counts = df[DIRS].to_numpy()
    n = counts.sum(axis=1)

    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", 0.0, 1.5, dims=("side", "serve_n", "dir"))
        sigma_server = pm.HalfNormal("sigma_server", 1.0)
        z = pm.Normal("z", 0.0, 1.0, dims=("server", "side", "serve_n", "dir"))
        theta = mu[side_idx, sn_idx] + z[s_idx, side_idx, sn_idx] * sigma_server
        p = pm.math.softmax(theta, axis=-1)
        pm.Multinomial("y", n=n, p=p, observed=counts)
    return model


def fit(model: pm.Model, draws: int = 1000, tune: int = 1000, seed: int = 65):
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=4, target_accept=0.9, random_seed=seed,
            progressbar=False,
        )
    return idata


def diagnostics_gate(idata) -> dict:
    """R-hat / ESS / divergence checks. Ship nothing that fails these."""
    import arviz as az

    summ = az.summary(idata, var_names=["mu", "sigma_server"], round_to="none")
    div = int(idata.sample_stats["diverging"].sum())
    max_rhat = float(summ["r_hat"].max())
    min_ess_bulk = float(summ["ess_bulk"].min())
    return {
        "divergences": div,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "pass": div == 0 and max_rhat < 1.01 and min_ess_bulk > 400,
    }
