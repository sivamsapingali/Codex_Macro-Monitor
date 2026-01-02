"""
Market Data Engine
Fetches and caches sector ETF price series for prediction targets.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from api_usage import ApiUsageTracker
from config import MARKET_DATA_PROVIDER, ALPHA_VANTAGE_API_KEY, SECTOR_ETFS
from data_store import SQLiteStore


class MarketDataEngine:
    def __init__(
        self,
        data_dir: str = "data/market",
        provider: Optional[str] = None,
        usage_tracker: Optional[ApiUsageTracker] = None,
        db_path: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else (self.data_dir.parent / "timeseries.db")
        self.store = SQLiteStore(str(self.db_path)) if self.db_path else None
        self.metadata_file = self.data_dir / "metadata.csv"
        self.provider = (provider or MARKET_DATA_PROVIDER).lower()
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.usage = usage_tracker or ApiUsageTracker()
        self.last_request_ts: dict[str, float] = {}
        self.alpha_vantage_unavailable = False
        self.metadata = self._load_metadata()
        if self.provider == "alpha_vantage" and not self.api_key:
            print("⚠️ ALPHA_VANTAGE_API_KEY not set. Falling back to Stooq.")
            self.provider = "stooq"

    def _load_metadata(self) -> dict:
        if not self.metadata_file.exists():
            return {}
        try:
            df = pd.read_csv(self.metadata_file)
            meta = {}
            for row in df.itertuples(index=False):
                symbol = getattr(row, "symbol", None)
                if not symbol:
                    continue
                last_updated = pd.to_datetime(getattr(row, "last_updated", None), errors="coerce")
                last_obs = pd.to_datetime(getattr(row, "last_observation", None), errors="coerce")
                last_checked = pd.to_datetime(getattr(row, "last_checked", None), errors="coerce")
                meta[symbol] = {
                    "last_updated": last_updated if pd.notna(last_updated) else None,
                    "last_observation": last_obs if pd.notna(last_obs) else None,
                    "last_checked": last_checked if pd.notna(last_checked) else None,
                }
            return meta
        except Exception as e:
            print(f"⚠️ Failed to read market metadata: {e}")
            return {}

    def _save_metadata(self) -> None:
        rows = []
        for symbol, meta in self.metadata.items():
            rows.append({
                "symbol": symbol,
                "last_updated": meta.get("last_updated"),
                "last_observation": meta.get("last_observation"),
                "last_checked": meta.get("last_checked"),
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.metadata_file, index=False)

    def initialize_data(self, refresh_existing: bool = False):
        print(f"📈 Initializing Market Data Engine ({self.provider})...")
        for symbol in SECTOR_ETFS.keys():
            if not refresh_existing:
                if self.store:
                    try:
                        if self.store.has_market_series(symbol):
                            continue
                    except Exception:
                        pass
                file_path = self._file_path(symbol)
                if file_path.exists():
                    try:
                        existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        if not existing.empty and "close" in existing.columns:
                            self._cache_series_to_store(symbol, existing["close"])
                            continue
                    except Exception:
                        pass
            self.update_series(symbol, force=refresh_existing)

    def refresh_stale(self):
        """Refresh only stale market series, respecting check intervals."""
        print("Refreshing stale market series...")
        for symbol in SECTOR_ETFS.keys():
            self.update_series(symbol, force=False)

    def _file_path(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol}.csv"

    def _check_interval_days(self) -> int:
        return 1

    def _is_stale(self, last_date: pd.Timestamp | None) -> bool:
        if last_date is None or pd.isna(last_date):
            return True
        return (datetime.now() - last_date).days > 2

    def _mark_checked(
        self,
        symbol: str,
        last_observation: pd.Timestamp | None,
        updated: bool,
    ) -> None:
        now = datetime.now()
        meta = self.metadata.get(symbol, {})
        meta["last_checked"] = now
        if updated:
            meta["last_updated"] = now
        if last_observation is not None and not pd.isna(last_observation):
            meta["last_observation"] = last_observation
        self.metadata[symbol] = meta
        self._save_metadata()

    def _throttle(self, provider: str) -> None:
        min_interval = {"alpha_vantage": 12}.get(provider)
        if not min_interval:
            return
        last_ts = self.last_request_ts.get(provider)
        if last_ts is None:
            return
        elapsed = time.time() - last_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def get_series(self, symbol: str) -> pd.Series:
        series = self._get_series_from_store(symbol)
        if not series.empty:
            return series
        series = self._get_series_from_csv(symbol)
        if not series.empty:
            self._cache_series_to_store(symbol, series)
        return series

    def _get_series_from_store(self, symbol: str) -> pd.Series:
        if not self.store:
            return pd.Series(dtype=float)
        try:
            return self.store.fetch_market_series(symbol)
        except Exception:
            return pd.Series(dtype=float)

    def _get_series_from_csv(self, symbol: str) -> pd.Series:
        file_path = self._file_path(symbol)
        if not file_path.exists():
            return pd.Series(dtype=float)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if "close" not in df.columns:
            return pd.Series(dtype=float)
        return df["close"].sort_index()

    def _cache_series_to_store(self, symbol: str, series: pd.Series) -> None:
        if not self.store or series.empty:
            return
        try:
            self.store.upsert_market_series(symbol, series)
        except Exception:
            return

    def _ensure_store_series(self, symbol: str) -> None:
        if not self.store:
            return
        try:
            if self.store.has_market_series(symbol):
                return
        except Exception:
            return
        series = self._get_series_from_csv(symbol)
        if not series.empty:
            self._cache_series_to_store(symbol, series)

    def _latest_local_date(self, symbol: str) -> pd.Timestamp | None:
        if self.store:
            try:
                last = self.store.get_latest_market_date(symbol)
                if last is not None:
                    return last
            except Exception:
                pass
        series = self._get_series_from_csv(symbol)
        if series.empty:
            return None
        return series.index.max()

    def update_series(self, symbol: str, force: bool = False) -> bool:
        file_path = self._file_path(symbol)
        last_observation = None
        last_date = None
        existing = None
        if not force:
            self._ensure_store_series(symbol)
            last_date = self._latest_local_date(symbol)
            if last_date is not None:
                last_observation = last_date
                if not self._is_stale(last_date):
                    return False

                last_checked = self.metadata.get(symbol, {}).get("last_checked")
                if last_checked is not None:
                    days_since = (datetime.now() - last_checked).days
                    if days_since < self._check_interval_days():
                        return False

        if file_path.exists():
            try:
                existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
            except Exception:
                existing = None

        provider = self.provider
        if provider == "alpha_vantage" and self.alpha_vantage_unavailable:
            provider = "stooq"
        if not self.usage.can_request(provider):
            print(f"⚠️ {provider} daily limit reached. Using cached data.")
            return False

        self._throttle(provider)
        self.last_request_ts[provider] = time.time()
        self.usage.record(provider)

        if provider == "alpha_vantage" and not self.alpha_vantage_unavailable:
            df = self._fetch_alpha_vantage(symbol)
            if df is None or df.empty:
                df = self._fetch_stooq(symbol)
        else:
            df = self._fetch_stooq(symbol)

        if df is None or df.empty:
            self._mark_checked(symbol, last_observation, updated=False)
            return False
        df = df.sort_index()
        new_df = df
        if last_date is not None:
            new_df = df[df.index > last_date]

        if new_df.empty:
            self._mark_checked(symbol, last_observation, updated=False)
            return False

        if self.store:
            try:
                self.store.upsert_market_series(symbol, new_df["close"])
            except Exception:
                pass

        final_df = None
        if existing is not None and not existing.empty:
            final_df = pd.concat([existing, new_df])
            final_df = final_df[~final_df.index.duplicated(keep='last')]
        elif file_path.exists() or force:
            if self.store:
                try:
                    final_df = self.store.fetch_market_series(symbol).to_frame("close")
                except Exception:
                    final_df = df
            else:
                final_df = df

        if final_df is not None and not final_df.empty:
            final_df.sort_index().to_csv(file_path)

        last_obs = final_df.index.max() if final_df is not None and not final_df.empty else new_df.index.max()
        self._mark_checked(symbol, last_obs, updated=True)
        return True

    def _fetch_stooq(self, symbol: str) -> pd.DataFrame | None:
        stooq_symbol = f"{symbol.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        try:
            df = pd.read_csv(url)
        except Exception as e:
            print(f"⚠️ Stooq fetch failed for {symbol}: {e}")
            return None

        if "Date" not in df.columns or "Close" not in df.columns:
            return None

        df = df.rename(columns={"Date": "date", "Close": "close"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")
        df = df[["close"]]
        return df

    def _fetch_alpha_vantage(self, symbol: str) -> pd.DataFrame | None:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self.api_key,
        }
        url = f"https://www.alphavantage.co/query?{urlencode(params)}"
        try:
            with urlopen(url) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"⚠️ Alpha Vantage fetch failed for {symbol}: {e}")
            return None

        series = payload.get("Time Series (Daily)")
        if not series:
            note = payload.get("Note") or payload.get("Information")
            if note:
                print(f"⚠️ Alpha Vantage response for {symbol}: {note}")
                if "premium" in note.lower() or "frequency" in note.lower():
                    self.alpha_vantage_unavailable = True
            return None

        rows = []
        for date_str, values in series.items():
            try:
                close = float(values.get("4. close", "nan"))
            except ValueError:
                continue
            rows.append((date_str, close))

        df = pd.DataFrame(rows, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")
        df = df.sort_index()
        return df
