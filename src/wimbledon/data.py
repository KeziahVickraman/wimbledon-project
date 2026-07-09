"""Loaders for the two datasets in the bundle.

Expects the wimbledon-datasets bundle unzipped so that:
    data/atp_matches/   <- TML yearly CSVs (Sackmann schema)
    data/mcp/           <- Match Charting Project files
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# MCP serve notation: first char of the rally string
SERVE_DIR_CODES = {"4": "wide", "5": "body", "6": "T"}


def load_atp_matches(years: range, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Stack yearly TML match files; parse dates; sort chronologically."""
    frames = [pd.read_csv(data_dir / "atp_matches" / f"{y}.csv") for y in years]
    df = pd.concat(frames, ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    return df.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)


def load_serve_direction(tour: str = "m", data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """MCP per-match serve-direction counts. row: 'Total', '1', '2' (serve number)."""
    df = pd.read_csv(data_dir / "mcp" / f"charting-{tour}-stats-ServeDirection.csv")
    matches = pd.read_csv(
        data_dir / "mcp" / f"charting-{tour}-matches.csv",
        usecols=["match_id", "Surface", "Date"],
        encoding_errors="replace",
    )
    return df.merge(matches, on="match_id", how="left")


def load_points_with_direction(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """MCP 2020s points; decode serve direction; keep only decodable serves.

    Adds: serve_dir (wide/body/T), serve_n (1/2), server_won (bool).
    """
    pts = pd.read_csv(data_dir / "mcp" / "charting-m-points-2020s.csv", low_memory=False)
    first = pts["1st"].astype(str).str.get(0)
    second = pts["2nd"].astype(str).str.get(0)
    # If the 2nd-serve column has a decodable direction, the point was on 2nd serve
    pts["serve_n"] = (second.isin(SERVE_DIR_CODES)).map({True: 2, False: 1})
    code = second.where(pts["serve_n"] == 2, first)
    pts["serve_dir"] = code.map(SERVE_DIR_CODES)
    pts = pts.dropna(subset=["serve_dir"])
    pts["server_won"] = pts["PtWinner"] == pts["Svr"]
    return pts
