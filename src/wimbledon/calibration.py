"""Calibration of the predictive model against bookmaker closing odds.

Odds schema follows tennis-data.co.uk: one row per match, columns are
relative to the actual outcome (``AvgW``/``AvgL`` = average closing odds on
the winner/loser). That means market Brier scores are computed the same
retrospective way as model Brier scores, so the comparison is apples-to-apples.
"""

from pathlib import Path

import pandas as pd

from wimbledon.data import DATA_DIR


def load_closing_odds(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """tennis-data.co.uk-style closing odds. Expects Date, Winner, Loser, AvgW, AvgL."""
    df = pd.read_csv(data_dir / "odds" / "closing_odds.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def implied_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    """De-vig the two-way market: normalize 1/odds so the pair sums to 1."""
    inv_w = 1.0 / odds["AvgW"]
    inv_l = 1.0 / odds["AvgL"]
    out = odds.copy()
    out["p_market_winner"] = inv_w / (inv_w + inv_l)
    return out
