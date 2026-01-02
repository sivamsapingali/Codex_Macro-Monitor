#!/usr/bin/env python3
"""
Export start/end dates for cached FRED + market series.
"""
import sqlite3
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import INDICATORS, SECTOR_ETFS


def build_series_meta() -> dict:
    meta = {}
    for category, info in INDICATORS.items():
        for series_id, series_meta in info.get("series", {}).items():
            meta[series_id] = {
                "name": series_meta.get("name", series_id),
                "category": category,
                "freq": series_meta.get("freq"),
            }
    return meta


def main() -> None:
    db_path = Path("data/timeseries.db")
    if not db_path.exists():
        print("Missing data/timeseries.db")
        return

    series_meta = build_series_meta()

    with sqlite3.connect(db_path) as conn:
        fred_rows = conn.execute(
            """
            SELECT series_id, MIN(date) AS start_date, MAX(date) AS end_date, COUNT(*) AS rows
            FROM fred_observations
            GROUP BY series_id
            ORDER BY series_id;
            """
        ).fetchall()
        market_rows = conn.execute(
            """
            SELECT symbol, MIN(date) AS start_date, MAX(date) AS end_date, COUNT(*) AS rows
            FROM market_observations
            GROUP BY symbol
            ORDER BY symbol;
            """
        ).fetchall()

    fred_df = pd.DataFrame(fred_rows, columns=["series_id", "start_date", "end_date", "rows"])
    if not fred_df.empty:
        fred_df["name"] = fred_df["series_id"].map(lambda s: series_meta.get(s, {}).get("name", s))
        fred_df["category"] = fred_df["series_id"].map(lambda s: series_meta.get(s, {}).get("category", ""))
        fred_df["freq"] = fred_df["series_id"].map(lambda s: series_meta.get(s, {}).get("freq", ""))
        fred_df = fred_df[["series_id", "name", "category", "freq", "start_date", "end_date", "rows"]]
        fred_df.to_csv("data/series_coverage.csv", index=False)

    market_df = pd.DataFrame(market_rows, columns=["symbol", "start_date", "end_date", "rows"])
    if not market_df.empty:
        market_df["name"] = market_df["symbol"].map(lambda s: SECTOR_ETFS.get(s, {}).get("name", s))
        market_df = market_df[["symbol", "name", "start_date", "end_date", "rows"]]
        market_df.to_csv("data/market_coverage.csv", index=False)

    print("Exported coverage to data/series_coverage.csv and data/market_coverage.csv")


if __name__ == "__main__":
    main()
