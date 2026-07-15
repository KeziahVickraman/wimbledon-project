"""Feature construction for the predictive model.

Leakage rule: every feature for match i may use only matches strictly before i.
The Elo below respects this by updating sequentially.
"""

from collections import defaultdict

import numpy as np
import pandas as pd


def surface_elo(matches: pd.DataFrame, k: float = 32.0, start: float = 1500.0) -> pd.DataFrame:
    """Sequential surface-specific Elo. Returns matches with pre-match Elo columns.

    Requires matches sorted chronologically (load_atp_matches guarantees this).
    """
    ratings: dict[tuple[str, str], float] = defaultdict(lambda: start)
    w_elo, l_elo = [], []
    for row in matches.itertuples():
        s = row.surface
        rw, rl = ratings[(row.winner_id, s)], ratings[(row.loser_id, s)]
        w_elo.append(rw)
        l_elo.append(rl)
        expected_w = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
        ratings[(row.winner_id, s)] = rw + k * (1.0 - expected_w)
        ratings[(row.loser_id, s)] = rl - k * (1.0 - expected_w)
    out = matches.copy()
    out["winner_elo_pre"] = w_elo
    out["loser_elo_pre"] = l_elo
    return out


def randomize_winner_framing(matches: pd.DataFrame, seed: int = 65) -> pd.DataFrame:
    """Create randomly ordered player features and a match-winner target.

    Every matched ``winner_*``/``loser_*`` pair becomes ``p1_*``/``p2_*`` and
    the original pair is removed, so column position cannot reveal the label.
    Non-player columns (for example, tournament metadata) are retained.

    This is a framing guard, not a feature-selection policy: downstream code
    must still select only features known before the match.
    """
    winner_cols = {c.removeprefix("winner_"): c for c in matches if c.startswith("winner_")}
    loser_cols = {c.removeprefix("loser_"): c for c in matches if c.startswith("loser_")}
    suffixes = sorted(winner_cols.keys() & loser_cols.keys())
    if not suffixes:
        raise ValueError("matches must include at least one winner_*/loser_* column pair")

    winner_first = np.random.default_rng(seed).integers(0, 2, len(matches)).astype(bool)
    paired_cols = {winner_cols[s] for s in suffixes} | {loser_cols[s] for s in suffixes}
    out = matches.drop(columns=paired_cols).copy()
    for suffix in suffixes:
        winner = matches[winner_cols[suffix]]
        loser = matches[loser_cols[suffix]]
        out[f"p1_{suffix}"] = winner.where(winner_first, loser)
        out[f"p2_{suffix}"] = loser.where(winner_first, winner)
    out["y"] = winner_first.astype("int8")
    return out


def build_predictive_features(matches: pd.DataFrame, seed: int = 65) -> pd.DataFrame:
    """Elo + framing -> model-ready feature frame for the GBM predictive baseline.

    One row per match: tourney_date, surface, signed diffs (p1 minus p2) of
    pre-match surface Elo / rank points / age / height, a left-handedness
    indicator per player, and ``y`` (1 if p1 is the actual match winner).
    """
    framed = randomize_winner_framing(surface_elo(matches), seed=seed)
    return pd.DataFrame(
        {
            "tourney_date": framed["tourney_date"],
            "surface": framed["surface"],
            "p1_name": framed["p1_name"],
            "p2_name": framed["p2_name"],
            "elo_diff": framed["p1_elo_pre"] - framed["p2_elo_pre"],
            "rank_points_diff": framed["p1_rank_points"] - framed["p2_rank_points"],
            "age_diff": framed["p1_age"] - framed["p2_age"],
            "ht_diff": framed["p1_ht"] - framed["p2_ht"],
            "p1_left": (framed["p1_hand"] == "L").astype("int8"),
            "p2_left": (framed["p2_hand"] == "L").astype("int8"),
            "y": framed["y"],
        }
    )


def reshape_serve_direction(raw: pd.DataFrame, min_points: int = 30) -> pd.DataFrame:
    """Aggregate MCP ServeDirection rows to (server, side, serve number) counts.

    MCP labels the body serve as ``middle``.  Players with fewer than
    ``min_points`` observed serves across all four cells are excluded.
    """
    required = {
        "row", "player", "deuce_wide", "deuce_middle", "deuce_t",
        "ad_wide", "ad_middle", "ad_t",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"ServeDirection data is missing columns: {', '.join(missing)}")
    if min_points < 0:
        raise ValueError("min_points must be non-negative")

    df = raw[raw["row"].astype(str).isin(["1", "2"])].copy()
    df["serve_n"] = df["row"].astype(str).map({"1": "first", "2": "second"})
    frames = []
    for side in ["deuce", "ad"]:
        part = df[["player", "serve_n", f"{side}_wide", f"{side}_middle", f"{side}_t"]].copy()
        part.columns = ["server", "serve_n", "wide", "body", "T"]
        part["side"] = side
        frames.append(part)
    long = pd.concat(frames, ignore_index=True)
    agg = long.groupby(["server", "side", "serve_n"], as_index=False)[["wide", "body", "T"]].sum()
    totals = agg.groupby("server")[["wide", "body", "T"]].transform("sum").sum(axis=1)
    return agg.loc[totals >= min_points].reset_index(drop=True)
