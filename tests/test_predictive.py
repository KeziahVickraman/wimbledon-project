import numpy as np
import pandas as pd

from wimbledon.predictive import FEATURE_COLS, build_predictive_model, chronological_split


def test_chronological_split_holds_out_the_most_recent_matches():
    feats = pd.DataFrame(
        {
            "tourney_date": pd.to_datetime(
                ["2024-01-01", "2024-03-01", "2024-02-01", "2024-04-01", "2024-05-01"]
            ),
        }
    )

    train, test = chronological_split(feats, test_frac=0.4)

    assert len(train) == 3
    assert len(test) == 2
    assert train["tourney_date"].max() < test["tourney_date"].min()


def test_build_predictive_model_fits_and_predicts_probabilities():
    rng = np.random.default_rng(0)
    n = 200
    elo_diff = rng.normal(0, 200, n)
    df = pd.DataFrame(
        {
            "surface": pd.Categorical(rng.choice(["Hard", "Clay", "Grass"], n)),
            "elo_diff": elo_diff,
            "rank_points_diff": rng.normal(0, 500, n),
            "age_diff": rng.normal(0, 3, n),
            "ht_diff": rng.normal(0, 5, n),
            "p1_left": rng.integers(0, 2, n),
            "p2_left": rng.integers(0, 2, n),
        }
    )
    y = (elo_diff + rng.normal(0, 50, n) > 0).astype(int)

    model = build_predictive_model()
    model.fit(df[FEATURE_COLS], y)
    proba = model.predict_proba(df[FEATURE_COLS])[:, 1]

    assert proba.shape == (n,)
    assert ((proba >= 0) & (proba <= 1)).all()
    # elo_diff is the dominant signal: higher elo_diff should mean higher P(p1 wins)
    assert np.corrcoef(proba, elo_diff)[0, 1] > 0.5
