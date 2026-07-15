import pandas as pd

from wimbledon.calibration import implied_probabilities


def test_implied_probabilities_removes_the_overround():
    odds = pd.DataFrame({"AvgW": [1.5, 2.0], "AvgL": [2.75, 2.0]})

    out = implied_probabilities(odds)

    # shorter-priced favorite (1.5 vs 2.75) should have de-vigged prob > 0.5
    assert out.loc[0, "p_market_winner"] > 0.5
    # a symmetric 2.0/2.0 book should split exactly 50/50 once de-vigged
    assert abs(out.loc[1, "p_market_winner"] - 0.5) < 1e-9
