"""
API Usage Tracker
Tracks daily API usage and exposes utilization data.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

DEFAULT_LIMITS = {
    "fred": {"daily": None, "per_minute": 120},
    "alpha_vantage": {"daily": 500, "per_minute": 5},
    "stooq": {"daily": None, "per_minute": None},
}


class ApiUsageTracker:
    def __init__(self, path: str = "data/api_usage.json", limits: Optional[Dict] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = limits or DEFAULT_LIMITS
        self.usage = self._load()

    def _load(self) -> Dict:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.usage, f, ensure_ascii=True, indent=2, sort_keys=True, default=str)

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _init_provider(self, provider: str) -> Dict:
        if provider not in self.usage:
            self.usage[provider] = {}
        record = self.usage[provider]
        if record.get("date") != self._today():
            record["date"] = self._today()
            record["count"] = 0
        limits = self.limits.get(provider, {})
        record["daily_limit"] = limits.get("daily")
        record["per_minute_limit"] = limits.get("per_minute")
        return record

    def can_request(self, provider: str) -> bool:
        record = self._init_provider(provider)
        daily_limit = record.get("daily_limit")
        if daily_limit is None:
            return True
        return record.get("count", 0) < daily_limit

    def record(self, provider: str, count: int = 1) -> None:
        record = self._init_provider(provider)
        record["count"] = record.get("count", 0) + count
        record["last_request"] = datetime.now().isoformat(timespec="seconds")
        self.usage[provider] = record
        self._save()

    def get_usage(self, provider: str) -> Dict:
        record = self._init_provider(provider)
        daily_limit = record.get("daily_limit")
        count = record.get("count", 0)
        remaining = None if daily_limit is None else max(daily_limit - count, 0)
        return {
            "provider": provider,
            "date": record.get("date"),
            "count": count,
            "daily_limit": daily_limit,
            "remaining": remaining,
            "per_minute_limit": record.get("per_minute_limit"),
            "last_request": record.get("last_request"),
        }

    def summary(self) -> Dict:
        providers = set(self.limits.keys()) | set(self.usage.keys())
        return {provider: self.get_usage(provider) for provider in sorted(providers)}
