"""Data loader tests; integration checks skip until the bundle is installed."""
import pytest

from wimbledon.data import DATA_DIR, SERVE_DIR_CODES, load_points_with_direction, load_serve_direction


requires_data = pytest.mark.skipif(
    not (DATA_DIR / "mcp").exists(), reason="wimbledon dataset bundle is not installed"
)


def test_serve_codes_complete():
    assert set(SERVE_DIR_CODES.values()) == {"wide", "body", "T"}


@requires_data
def test_points_decode():
    pts = load_points_with_direction()
    assert pts["serve_dir"].isin(["wide", "body", "T"]).all()
    # sanity: server should win 60-70% of points on tour
    assert 0.55 < pts["server_won"].mean() < 0.72


@requires_data
def test_serve_direction_has_surface():
    sd = load_serve_direction("m")
    assert (sd["Surface"] == "Grass").sum() > 1000
