"""Predictive match-winner model: GBM baseline (Phase P decision, DESIGN_v2.md).

Sampling-free — unlike the tendency model this fits in milliseconds, but the
same rule applies: fitting happens in scripts/, never in api/.
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURE_COLS = [
    "surface", "elo_diff", "rank_points_diff", "age_diff", "ht_diff", "p1_left", "p2_left",
]


def build_predictive_model(random_state: int = 65) -> HistGradientBoostingClassifier:
    """HistGradientBoosting baseline. ``surface`` must be a pandas category dtype."""
    return HistGradientBoostingClassifier(random_state=random_state)


def chronological_split(
    feats: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split strictly by time: the holdout is the most recent ``test_frac`` of matches."""
    ordered = feats.sort_values("tourney_date")
    cut = int(len(ordered) * (1 - test_frac))
    return ordered.iloc[:cut], ordered.iloc[cut:]
