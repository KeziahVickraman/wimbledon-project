import pandas as pd
import pytest

from wimbledon.features import (
    build_predictive_features,
    randomize_winner_framing,
    reshape_serve_direction,
)


def test_randomize_winner_framing_removes_result_order_and_is_reproducible():
    matches = pd.DataFrame(
        {
            "tourney_name": ["Wimbledon"] * 8,
            "winner_id": list(range(10, 18)),
            "loser_id": list(range(20, 28)),
            "winner_rank": list(range(1, 9)),
            "loser_rank": list(range(11, 19)),
        }
    )

    framed = randomize_winner_framing(matches, seed=8)

    assert not any(c.startswith(("winner_", "loser_")) for c in framed)
    assert {"p1_id", "p2_id", "p1_rank", "p2_rank", "y"} <= set(framed)
    assert set(framed["y"]) == {0, 1}
    for original, result in zip(matches.itertuples(), framed.itertuples()):
        if result.y:
            assert (result.p1_id, result.p2_id) == (original.winner_id, original.loser_id)
        else:
            assert (result.p1_id, result.p2_id) == (original.loser_id, original.winner_id)
    pd.testing.assert_frame_equal(framed, randomize_winner_framing(matches, seed=8))


def test_reshape_serve_direction_aggregates_sides_and_filters_low_volume_players():
    raw = pd.DataFrame(
        {
            "row": ["1", "2", "1"],
            "player": ["A", "A", "B"],
            "deuce_wide": [4, 1, 1], "deuce_middle": [2, 1, 1], "deuce_t": [4, 2, 1],
            "ad_wide": [3, 2, 1], "ad_middle": [1, 1, 1], "ad_t": [6, 3, 1],
        }
    )

    result = reshape_serve_direction(raw, min_points=20)

    assert len(result) == 4
    assert set(result["server"]) == {"A"}
    assert result.loc[(result.server == "A") & (result.side == "deuce") & (result.serve_n == "first"), "body"].item() == 2


def test_reshape_serve_direction_reports_missing_schema_columns():
    with pytest.raises(ValueError, match="ad_t"):
        reshape_serve_direction(pd.DataFrame({"row": [], "player": []}))


def test_build_predictive_features_diffs_are_signed_from_p1s_perspective():
    matches = pd.DataFrame(
        {
            "tourney_date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
            "surface": ["Hard", "Hard", "Grass"],
            "winner_id": [1, 2, 1],
            "loser_id": [2, 1, 3],
            "winner_name": ["Alice", "Bob", "Alice"],
            "loser_name": ["Bob", "Alice", "Cara"],
            "winner_hand": ["R", "L", "R"],
            "loser_hand": ["L", "R", "R"],
            "winner_ht": [185, 178, 185],
            "loser_ht": [178, 185, 190],
            "winner_age": [24.0, 30.0, 24.5],
            "loser_age": [30.0, 24.0, 22.0],
            "winner_rank_points": [3000, 1200, 3100],
            "loser_rank_points": [1200, 3000, 800],
        }
    )

    feats = build_predictive_features(matches, seed=1)

    expected_cols = {
        "tourney_date", "surface", "p1_name", "p2_name", "elo_diff", "rank_points_diff",
        "age_diff", "ht_diff", "p1_left", "p2_left", "y",
    }
    assert expected_cols <= set(feats.columns)
    assert set(feats["y"]) <= {0, 1}
    assert len(feats) == len(matches)
    # y=1 means p1 was the actual winner, so the winner's rank_points minus the
    # loser's should carry the same sign as (p1 - p2) once un-anonymized.
    for feat, orig in zip(feats.itertuples(), matches.itertuples()):
        winner_minus_loser = orig.winner_rank_points - orig.loser_rank_points
        signed = feat.rank_points_diff if feat.y else -feat.rank_points_diff
        assert signed == winner_minus_loser
