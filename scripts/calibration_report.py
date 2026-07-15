"""Calibration report for the predictive model vs tennis-data.co.uk closing odds.

Refits on the same chronological split as fit_predictive.py, joins the holdout
to closing odds by (Winner, Loser, Date), and reports Brier score + reliability
curve for the model against the de-vigged market probability. Saves a JSON
summary and a reliability-diagram PNG.

Run from repo root:  uv run python -m scripts.calibration_report
"""

import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from wimbledon.calibration import implied_probabilities, load_closing_odds
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

    model = build_predictive_model()
    model.fit(train[FEATURE_COLS], train["y"])
    test = test.copy()
    test["model_p1_win"] = model.predict_proba(test[FEATURE_COLS])[:, 1]

    # p1/p2 are randomly permuted (winner isn't always p1), so join on an
    # order-independent pair key rather than assuming p1_name == Winner.
    test["pair_a"] = test[["p1_name", "p2_name"]].min(axis=1)
    test["pair_b"] = test[["p1_name", "p2_name"]].max(axis=1)

    odds = implied_probabilities(load_closing_odds())
    odds["pair_a"] = odds[["Winner", "Loser"]].min(axis=1)
    odds["pair_b"] = odds[["Winner", "Loser"]].max(axis=1)

    joined = test.merge(
        odds[["Date", "pair_a", "pair_b", "p_market_winner"]],
        left_on=["tourney_date", "pair_a", "pair_b"],
        right_on=["Date", "pair_a", "pair_b"],
        how="inner",
    )
    if joined.empty:
        raise SystemExit("No holdout matches joined to closing odds — check the odds schema.")

    # Re-express the market's winner-relative probability in p1's frame so it lines
    # up with the model's p1-win probability and with y (both vary 0/1 — unlike the
    # winner's-perspective framing, where y is trivially always 1).
    joined["market_p1_win"] = joined["p_market_winner"].where(
        joined["y"] == 1, 1 - joined["p_market_winner"]
    )

    model_brier = brier_score_loss(joined["y"], joined["model_p1_win"])
    market_brier = brier_score_loss(joined["y"], joined["market_p1_win"])

    report = {
        "n_matches": len(joined),
        "model_brier": model_brier,
        "market_brier": market_brier,
        "model_beats_market": model_brier < market_brier,
    }
    print(json.dumps(report, indent=2))

    ARTIFACTS.mkdir(exist_ok=True)
    stamp = f"{VERSION}_{dt.date.today():%Y%m%d}"
    (ARTIFACTS / f"calibration_report_v{stamp}.json").write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(5, 5))
    for label, col in [("model", "model_p1_win"), ("market", "market_p1_win")]:
        frac_pos, mean_pred = calibration_curve(
            joined["y"], joined[col], n_bins=10, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.set_xlabel("mean predicted P(actual winner)")
    ax.set_ylabel("empirical win rate")
    ax.set_title("Reliability: predictive model vs closing-odds market")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / f"calibration_reliability_v{stamp}.png", dpi=150)
    print(f"saved artifacts/calibration_report_v{stamp}.json and _reliability_v{stamp}.png")


if __name__ == "__main__":
    main()
