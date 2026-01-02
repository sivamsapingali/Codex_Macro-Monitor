
"""
FRED Data Engine
Handles incremental fetching, CSV persistence, and data management for hundreds of economic series.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from fredapi import Fred
from api_usage import ApiUsageTracker
from config import FRED_API_KEY, INDICATORS
from derivatives import compute_all_derivatives
import time

class FredDataEngine:
    def __init__(self, data_dir: str = "data", usage_tracker: ApiUsageTracker | None = None):
        self.fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
        self.can_fetch = self.fred is not None
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_file = self.data_dir / "metadata.csv"
        self.metrics_cache = {}
        self.metadata = self._load_metadata()
        self.api_usage = usage_tracker or ApiUsageTracker()
        
        # Flatten structure for easy lookup
        self.series_map = {}
        for cat_id, cat in INDICATORS.items():
            for series_id, meta in cat['series'].items():
                self.series_map[series_id] = meta
                self.series_map[series_id]['category'] = cat_id
        if not self.can_fetch:
            print("⚠️ FRED_API_KEY not set. Running in cached-data mode only.")

    def _load_metadata(self) -> dict:
        if not self.metadata_file.exists():
            return {}
        try:
            df = pd.read_csv(self.metadata_file)
            meta = {}
            for row in df.itertuples(index=False):
                last_updated = pd.to_datetime(getattr(row, "last_updated", None), errors="coerce")
                last_obs = pd.to_datetime(getattr(row, "last_observation", None), errors="coerce")
                last_checked = pd.to_datetime(getattr(row, "last_checked", None), errors="coerce")
                meta[row.series_id] = {
                    "last_updated": last_updated if pd.notna(last_updated) else None,
                    "last_observation": last_obs if pd.notna(last_obs) else None,
                    "last_checked": last_checked if pd.notna(last_checked) else None,
                    "freq": getattr(row, "freq", None),
                }
            return meta
        except Exception as e:
            print(f"⚠️ Failed to read metadata: {e}")
            return {}

    def _save_metadata(self) -> None:
        rows = []
        for series_id, meta in self.metadata.items():
            rows.append({
                "series_id": series_id,
                "last_updated": meta.get("last_updated"),
                "last_observation": meta.get("last_observation"),
                "last_checked": meta.get("last_checked"),
                "freq": meta.get("freq"),
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.metadata_file, index=False)

    def _max_age_days(self, freq: str) -> int:
        freq = (freq or "M").upper()
        return {
            "D": 2,
            "W": 10,
            "M": 45,
            "Q": 140,
        }.get(freq, 30)

    def _check_interval_days(self, freq: str) -> int:
        freq = (freq or "M").upper()
        return {
            "D": 1,
            "W": 3,
            "M": 10,
            "Q": 30,
        }.get(freq, 7)

    def _periods_map(self, freq: str) -> dict:
        freq = (freq or "M").upper()
        if freq == "D":
            return {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
        if freq == "W":
            return {"1m": 4, "3m": 13, "6m": 26, "12m": 52}
        if freq == "Q":
            return {"1m": 1, "3m": 1, "6m": 2, "12m": 4}
        return {"1m": 1, "3m": 3, "6m": 6, "12m": 12}

    def _lookback_window(self, freq: str) -> int:
        periods_map = self._periods_map(freq)
        return periods_map["12m"] * 5

    def _is_stale(self, last_date: pd.Timestamp, freq: str) -> bool:
        if last_date is None or pd.isna(last_date):
            return True
        age_days = (datetime.now() - last_date).days
        return age_days > self._max_age_days(freq)

    def _mark_checked(self, series_id: str, freq: str, last_observation: pd.Timestamp | None, updated: bool) -> None:
        now = datetime.now()
        meta = self.metadata.get(series_id, {})
        meta["freq"] = freq
        meta["last_checked"] = now
        if updated:
            meta["last_updated"] = now
        if last_observation is not None and not pd.isna(last_observation):
            meta["last_observation"] = last_observation
        self.metadata[series_id] = meta
        self._save_metadata()

    def initialize_data(self, refresh_existing: bool = False):
        """Initialize data; when refresh_existing is False, only fetch missing series."""
        print(f"🚀 Initializing FRED Data Engine with {len(self.series_map)} series...")
        if not self.can_fetch:
            print("✅ Initialization skipped (no API key). Using cached data only.")
            return
        updated_count = 0

        for series_id in self.series_map:
            file_path = self.data_dir / f"{series_id}.csv"
            if not refresh_existing and file_path.exists():
                try:
                    existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    if not existing.empty:
                        continue
                except Exception:
                    pass

            updated, requested = self._update_series(series_id, force=refresh_existing)
            if updated:
                updated_count += 1
            if requested:
                # Respect FRED rate limits (120/min)
                time.sleep(0.6)

        print(f"✅ Initialization complete. Updated {updated_count} series.")

    def get_series(self, series_id: str) -> pd.Series:
        """Get full history for a series from local CSV"""
        file_path = self.data_dir / f"{series_id}.csv"
        
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df['value'].sort_index()
        return pd.Series(dtype=float)

    def _latest_value(self, series: pd.Series):
        if series is None or series.empty:
            return None
        cleaned = series.dropna()
        if cleaned.empty:
            return None
        return cleaned.iloc[-1]

    def get_series_metrics(self, series_id: str) -> dict:
        file_path = self.data_dir / f"{series_id}.csv"
        if not file_path.exists():
            return {}
        mtime = file_path.stat().st_mtime
        cached = self.metrics_cache.get(series_id)
        if cached and cached.get("mtime") == mtime:
            return cached.get("metrics", {})

        series = self.get_series(series_id)
        if series.empty:
            return {}

        meta = self.series_map.get(series_id, {})
        freq = meta.get("freq", "M")
        periods_map = self._periods_map(freq)
        lookback = self._lookback_window(freq)
        transform = meta.get("transform", "pct")

        metrics = compute_all_derivatives(
            series,
            transform=transform,
            lookback=lookback,
            periods_map=periods_map,
        )

        latest_metrics = {
            "id": series_id,
            "name": meta.get("name", series_id),
            "freq": freq,
            "unit": meta.get("unit"),
            "transform": transform,
            "last_date": series.index[-1].strftime("%Y-%m-%d"),
            "value": self._latest_value(metrics.get("value")),
            "transformed": self._latest_value(metrics.get("transformed")),
            "roc_1m": self._latest_value(metrics.get("roc_1m")),
            "roc_3m": self._latest_value(metrics.get("roc_3m")),
            "roc_6m": self._latest_value(metrics.get("roc_6m")),
            "roc_12m": self._latest_value(metrics.get("roc_12m")),
            "acceleration": self._latest_value(metrics.get("acceleration")),
            "z_score": self._latest_value(metrics.get("z_score")),
            "percentile": self._latest_value(metrics.get("percentile")),
            "inflection": self._latest_value(metrics.get("inflection")),
            "signal": self._latest_value(metrics.get("signal")),
        }

        self.metrics_cache[series_id] = {"mtime": mtime, "metrics": latest_metrics}
        return latest_metrics

    def force_update_all(self):
        """Trigger a full update regardless of standard checks"""
        print("Force updating all series...")
        if not self.can_fetch:
            print("⚠️ Cannot force update without FRED_API_KEY.")
            return
        for series_id in self.series_map:
            _, requested = self._update_series(series_id, force=True)
            if requested:
                time.sleep(0.6)

    def refresh_stale(self):
        """Refresh only stale series, respecting check intervals."""
        print("Refreshing stale FRED series...")
        if not self.can_fetch:
            print("⚠️ Cannot refresh without FRED_API_KEY.")
            return
        updated_count = 0
        for series_id in self.series_map:
            updated, requested = self._update_series(series_id, force=False)
            if updated:
                updated_count += 1
            if requested:
                time.sleep(0.6)
        print(f"✅ Refresh complete. Updated {updated_count} series.")

    def _update_series(self, series_id: str, force: bool = False) -> tuple[bool, bool]:
        """
        Update a single series. 
        Returns (updated, requested) where requested indicates an API call was made.
        """
        if not self.can_fetch:
            return False, False
        if not self.api_usage.can_request("fred"):
            print("⚠️ FRED daily limit reached. Using cached data.")
            return False, False
        file_path = self.data_dir / f"{series_id}.csv"
        start_date = "1980-01-01" # Default start for macro history
        existing_data = None
        meta = self.series_map.get(series_id, {})
        freq = meta.get("freq", "M")
        last_observation = None
        
        # Check existing data
        if file_path.exists() and not force:
            try:
                existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if not existing_data.empty:
                    last_date = existing_data.index.max()
                    last_observation = last_date
                    if not self._is_stale(last_date, freq):
                        return False, False # Up to date

                    last_checked = self.metadata.get(series_id, {}).get("last_checked")
                    if last_checked is not None:
                        days_since = (datetime.now() - last_checked).days
                        if days_since < self._check_interval_days(freq):
                            return False, False

                    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            except Exception as e:
                print(f"Error reading {series_id}: {e}")
                existing_data = None
        
        try:
            # Fetch from FRED
            # print(f"Fetching {series_id} from {start_date}...")
            self.api_usage.record("fred")
            new_data = self.fred.get_series(series_id, observation_start=start_date)
            
            if new_data.empty:
                self._mark_checked(series_id, freq, last_observation, updated=False)
                return False, True
                
            new_df = pd.DataFrame(new_data, columns=['value'])
            new_df.index.name = 'date'
            
            # Combine
            if existing_data is not None and not existing_data.empty:
                final_df = pd.concat([existing_data, new_df])
                final_df = final_df[~final_df.index.duplicated(keep='last')] # Dedup
            else:
                final_df = new_df
                
            final_df.sort_index().to_csv(file_path)
            last_obs = final_df.index.max()
            self._mark_checked(series_id, freq, last_obs, updated=True)
            self.metrics_cache.pop(series_id, None)
            return True, True
            
        except Exception as e:
            print(f"⚠️ Failed to update {series_id}: {e}")
            return False, True

    def get_category_data(self, category: str) -> dict:
        """Get latest values and trends for all series in a category"""
        results = {}
        if category not in INDICATORS:
            return {}
            
        for series_id, meta in INDICATORS[category]['series'].items():
            series = self.get_series(series_id)
            if series.empty:
                continue
                
            # Calculate stats
            latest = series.iloc[-1]
            prev = series.iloc[-2] if len(series) > 1 else latest
            freq = meta.get("freq", "M")
            periods_map = self._periods_map(freq)
            year_periods = periods_map["12m"]
            transform = meta.get("transform", "pct")
            last_date = series.index[-1]
            
            # Calculate Rate of Change (1Y)
            if len(series) > year_periods:
                if transform in ["level", "diff"]:
                    roc_1y = series.diff(year_periods).iloc[-1]
                else:
                    roc_1y = series.pct_change(year_periods).iloc[-1]
            else:
                roc_1y = 0

            metrics = self.get_series_metrics(series_id)
            last_updated = self.metadata.get(series_id, {}).get("last_updated")
            
            results[series_id] = {
                "name": meta['name'],
                "value": latest,
                "change": latest - prev,
                "change_pct": (latest - prev) / prev if prev != 0 else 0,
                "roc_1y": roc_1y,
                "unit": meta['unit'],
                "date": last_date.strftime('%Y-%m-%d'),
                "stale": self._is_stale(last_date, freq),
                "last_updated": last_updated.strftime("%Y-%m-%d %H:%M") if last_updated else None,
                "transform": transform,
                "value_transformed": metrics.get("transformed"),
                "signal": metrics.get("signal"),
            }
            
        return results
