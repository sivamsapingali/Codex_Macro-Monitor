"""
SQLite-backed time series store for FRED + market data.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


class SQLiteStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fred_observations (
                    series_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    value REAL,
                    PRIMARY KEY (series_id, date)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_observations (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    close REAL,
                    PRIMARY KEY (symbol, date)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fred_series ON fred_observations(series_id, date);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_symbol ON market_observations(symbol, date);"
            )

    def _series_rows(self, key: str, series: pd.Series) -> list[tuple]:
        if series.empty:
            return []
        idx = pd.to_datetime(series.index, errors="coerce")
        rows = []
        for date, value in zip(idx, series.values):
            if pd.isna(date):
                continue
            date_str = date.strftime("%Y-%m-%d")
            val = None if pd.isna(value) else float(value)
            rows.append((key, date_str, val))
        return rows

    def has_fred_series(self, series_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM fred_observations WHERE series_id = ? LIMIT 1;",
                (series_id,),
            ).fetchone()
        return row is not None

    def has_market_series(self, symbol: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM market_observations WHERE symbol = ? LIMIT 1;",
                (symbol,),
            ).fetchone()
        return row is not None

    def get_latest_fred_date(self, series_id: str) -> Optional[pd.Timestamp]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM fred_observations WHERE series_id = ?;",
                (series_id,),
            ).fetchone()
        if not row or row[0] is None:
            return None
        return pd.to_datetime(row[0], errors="coerce")

    def get_latest_market_date(self, symbol: str) -> Optional[pd.Timestamp]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM market_observations WHERE symbol = ?;",
                (symbol,),
            ).fetchone()
        if not row or row[0] is None:
            return None
        return pd.to_datetime(row[0], errors="coerce")

    def fetch_fred_series(self, series_id: str) -> pd.Series:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, value FROM fred_observations WHERE series_id = ? ORDER BY date;",
                (series_id,),
            ).fetchall()
        if not rows:
            return pd.Series(dtype=float)
        df = pd.DataFrame(rows, columns=["date", "value"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
        return df["value"].sort_index()

    def fetch_market_series(self, symbol: str) -> pd.Series:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, close FROM market_observations WHERE symbol = ? ORDER BY date;",
                (symbol,),
            ).fetchall()
        if not rows:
            return pd.Series(dtype=float)
        df = pd.DataFrame(rows, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
        return df["close"].sort_index()

    def upsert_fred_series(self, series_id: str, series: pd.Series) -> int:
        rows = self._series_rows(series_id, series)
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO fred_observations (series_id, date, value) VALUES (?, ?, ?);",
                rows,
            )
        return len(rows)

    def upsert_market_series(self, symbol: str, series: pd.Series) -> int:
        rows = self._series_rows(symbol, series)
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO market_observations (symbol, date, close) VALUES (?, ?, ?);",
                rows,
            )
        return len(rows)
