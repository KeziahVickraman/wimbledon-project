"""Fit the GBM predictive baseline; gate on beating a base-rate baseline; save artifact.

Run from repo root:  uv run python -m scripts.fit_predictive
"""

import datetime as dt
import json
from pathlib import Path

import joblib
from sklearn.metrics import brier_score_loss

from wimbledon.data import load_atp_matches
from wimbledon.features import build_predictive_features
from wimbledon.predictive import FEATURE_COLS, build_predictive_model, chronological_split

ARTIFACTS = Path("artifacts")
VERSION = 1


def main() -> None:
    matches = load_atp_matches(range(2018, 2025))
    feats = build_predictive_features(matches)
    feats["surface"] = feats["surface"].astype("category")
    train, test = chronological_split(feats)
    print(f"{len(train)} train matches, {len(test)} test matches (chronological holdout)")

    model = build_predictive_model()
    model.fit(train[FEATURE_COLS], train["y"])

    proba = model.predict_proba(test[FEATURE_COLS])[:, 1]
    brier = brier_score_loss(test["y"], proba)
    naive_proba = [train["y"].mean()] * len(test)
    naive_brier = brier_score_loss(test["y"], naive_proba)

    gate = {"brier": brier, "naive_brier": naive_brier, "pass": brier < naive_brier}
    print(json.dumps(gate, indent=2))
    if not gate["pass"]:
        raise SystemExit("GBM does not beat the naive base-rate baseline — do not ship.")

    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / f"predictive_v{VERSION}_{dt.date.today():%Y%m%d}.joblib"
    joblib.dump(model, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
