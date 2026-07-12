"""Generate a synthetic dataset matching the TML/MCP schemas that data.py expects.

This is a stand-in for the real wimbledon-datasets.zip bundle (see README setup
step 1), which is CC BY-NC-SA licensed and not distributed with this repo. It
lets the rest of the pipeline (features, models, scripts, api) be built and
exercised end-to-end before the real bundle is installed. Swap it out by
unzipping the real bundle into data/ — no code changes needed elsewhere.

Run from repo root:  uv run python -m scripts.make_synthetic_data
"""

import numpy as np
import pandas as pd

from wimbledon.data import DATA_DIR

RNG = np.random.default_rng(65)
N_PLAYERS = 48
PLAYERS = [f"Player{i:02d}" for i in range(1, N_PLAYERS + 1)]
SKILL = dict(zip(PLAYERS, RNG.normal(1500, 150, N_PLAYERS)))


def make_atp_matches() -> None:
    years = range(2018, 2025)
    surfaces = ["Hard", "Clay", "Grass"]
    surface_p = [0.58, 0.30, 0.12]
    for year in years:
        n_matches = RNG.integers(180, 220)
        rows = []
        for i in range(n_matches):
            p1, p2 = RNG.choice(PLAYERS, size=2, replace=False)
            surface = RNG.choice(surfaces, p=surface_p)
            expected_p1 = 1.0 / (1.0 + 10 ** ((SKILL[p2] - SKILL[p1]) / 400.0))
            winner, loser = (p1, p2) if RNG.random() < expected_p1 else (p2, p1)
            month = RNG.integers(1, 13)
            day = RNG.integers(1, 29)
            rows.append(
                {
                    "tourney_id": f"{year}-{RNG.integers(100, 999)}",
                    "tourney_name": "Synthetic Open",
                    "surface": surface,
                    "tourney_date": f"{year}{month:02d}{day:02d}",
                    "match_num": i + 1,
                    "winner_id": 200000 + PLAYERS.index(winner) + 1,
                    "winner_name": winner,
                    "winner_hand": RNG.choice(["R", "L"], p=[0.85, 0.15]),
                    "winner_ht": int(RNG.integers(175, 205)),
                    "winner_ioc": "XXX",
                    "winner_age": round(float(RNG.uniform(18, 35)), 1),
                    "winner_rank": int(max(1, RNG.normal(50, 40))),
                    "winner_rank_points": int(max(0, RNG.normal(1500, 800))),
                    "loser_id": 200000 + PLAYERS.index(loser) + 1,
                    "loser_name": loser,
                    "loser_hand": RNG.choice(["R", "L"], p=[0.85, 0.15]),
                    "loser_ht": int(RNG.integers(175, 205)),
                    "loser_ioc": "XXX",
                    "loser_age": round(float(RNG.uniform(18, 35)), 1),
                    "loser_rank": int(max(1, RNG.normal(70, 50))),
                    "loser_rank_points": int(max(0, RNG.normal(1000, 700))),
                    "score": "6-4 6-3",
                    "best_of": 3,
                    "round": RNG.choice(["R32", "R16", "QF", "SF", "F"]),
                    "minutes": int(RNG.integers(60, 200)),
                }
            )
        pd.DataFrame(rows).to_csv(DATA_DIR / "atp_matches" / f"{year}.csv", index=False)


def make_mcp() -> None:
    matches_rows, sd_rows, points_rows = [], [], []

    # Per-player serve-direction tendencies: dirichlet draw per (side, serve_n)
    tendency = {
        p: {
            (side, sn): RNG.dirichlet([4, 3, 4])
            for side in ("deuce", "ad")
            for sn in ("first", "second")
        }
        for p in PLAYERS
    }

    n_grass, n_other = 320, 140
    match_specs = [("Grass", "Wimbledon", n_grass)] + [
        (s, "Masters", n_other) for s in ("Hard", "Clay")
    ]

    for surface, tourney, n in match_specs:
        for i in range(n):
            p1, p2 = RNG.choice(PLAYERS, size=2, replace=False)
            date = f"202{RNG.integers(0, 5)}{RNG.integers(1, 13):02d}{RNG.integers(1, 28):02d}"
            match_id = f"{date}-M-{tourney}-{p1}-{p2}-{i}"
            matches_rows.append({"match_id": match_id, "Date": date, "Surface": surface})

            for server, returner in ((p1, p2), (p2, p1)):
                for serve_n, row_label in (("first", "1"), ("second", "2")):
                    counts = {}
                    for side in ("deuce", "ad"):
                        n_pts = int(RNG.integers(8, 22))
                        wide, body, t = RNG.multinomial(n_pts, tendency[server][(side, serve_n)])
                        counts[f"{side}_wide"] = wide
                        counts[f"{side}_middle"] = body
                        counts[f"{side}_t"] = t
                    sd_rows.append(
                        {"match_id": match_id, "row": row_label, "player": server, **counts}
                    )

            # point-level rows for the 2020s points file (server_won ~ tour-average rate)
            n_rally = int(RNG.integers(60, 130))
            servers = RNG.choice([1, 2], size=n_rally)
            server_wins = RNG.random(n_rally) < 0.63
            winners = np.where(server_wins, servers, 3 - servers)
            for svr, ptw in zip(servers, winners):
                side = RNG.choice(["deuce", "ad"])
                server_name = p1 if svr == 1 else p2
                code = RNG.choice(list("456"), p=list(tendency[server_name][(side, "first")]))
                is_second = RNG.random() < 0.35
                points_rows.append(
                    {
                        "match_id": match_id,
                        "Svr": int(svr),
                        "PtWinner": int(ptw),
                        "1st": "n" if is_second else f"{code}f8",
                        "2nd": f"{code}f8" if is_second else "",
                    }
                )

    pd.DataFrame(matches_rows).to_csv(DATA_DIR / "mcp" / "charting-m-matches.csv", index=False)
    pd.DataFrame(sd_rows).to_csv(
        DATA_DIR / "mcp" / "charting-m-stats-ServeDirection.csv", index=False
    )
    pd.DataFrame(points_rows).to_csv(
        DATA_DIR / "mcp" / "charting-m-points-2020s.csv", index=False
    )


def main() -> None:
    (DATA_DIR / "atp_matches").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "mcp").mkdir(parents=True, exist_ok=True)
    make_atp_matches()
    make_mcp()
    print(f"synthetic data written to {DATA_DIR}")


if __name__ == "__main__":
    main()
