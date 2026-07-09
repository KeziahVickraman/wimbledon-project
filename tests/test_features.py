import pandas as pd
import pytest

from wimbledon.features import randomize_winner_framing, reshape_serve_direction


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
