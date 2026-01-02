"""
Ingest existing CSV caches into the SQLite time series database.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_store import SQLiteStore


def _load_series(path: Path, column: str) -> pd.Series:
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return df[column].sort_index()


def ingest_fred(data_dir: Path, store: SQLiteStore) -> tuple[int, int]:
    series_count = 0
    row_count = 0
    for path in sorted(data_dir.glob("*.csv")):
        if path.name == "metadata.csv":
            continue
        series_id = path.stem
        series = _load_series(path, "value")
        if series.empty:
            continue
        rows = store.upsert_fred_series(series_id, series)
        series_count += 1
        row_count += rows
    return series_count, row_count


def ingest_market(market_dir: Path, store: SQLiteStore) -> tuple[int, int]:
    series_count = 0
    row_count = 0
    for path in sorted(market_dir.glob("*.csv")):
        if path.name == "metadata.csv":
            continue
        symbol = path.stem
        series = _load_series(path, "close")
        if series.empty:
            continue
        rows = store.upsert_market_series(symbol, series)
        series_count += 1
        row_count += rows
    return series_count, row_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate CSV caches into SQLite DB.")
    parser.add_argument("--data-dir", default="data", help="FRED CSV cache directory")
    parser.add_argument("--market-dir", default="data/market", help="Market CSV cache directory")
    parser.add_argument("--db-path", default="data/timeseries.db", help="SQLite DB path")
    args = parser.parse_args()

    store = SQLiteStore(args.db_path)

    fred_count, fred_rows = ingest_fred(Path(args.data_dir), store)
    market_count, market_rows = ingest_market(Path(args.market_dir), store)

    print(f"FRED series ingested: {fred_count} | rows: {fred_rows}")
    print(f"Market series ingested: {market_count} | rows: {market_rows}")
    print(f"DB path: {args.db_path}")


if __name__ == "__main__":
    main()
