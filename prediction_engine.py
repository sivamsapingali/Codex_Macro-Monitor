"""
Acceleration/Deceleration Prediction Engine
Builds multi-horizon sector ETF direction forecasts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from config import PREDICTOR_SERIES, PREDICTION_HORIZONS_WEEKS, SECTOR_ETFS
from derivatives import compute_all_derivatives

WEEKLY_PERIODS = {"1m": 4, "3m": 13, "6m": 26, "12m": 52}


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window // 2).mean()
    std = series.rolling(window=window, min_periods=window // 2).std()
    return (series - mean) / std


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


@dataclass
class ModelResult:
    expected_return: float
    direction: str
    confidence: float
    metrics: Dict[str, float]
    top_drivers: List[Dict[str, float]]


class PredictionEngine:
    def __init__(self, fred_engine, market_engine, cache_seconds: int = 300):
        self.fred_engine = fred_engine
        self.market_engine = market_engine
        self.cache_seconds = cache_seconds
        self._cache = {"timestamp": None, "data": None}

    def _to_weekly(self, series: pd.Series, freq: str) -> pd.Series:
        if series.empty:
            return series
        series = series.sort_index()
        if freq in {"D", "W"}:
            weekly = series.resample("W-FRI").last()
        else:
            weekly = series.resample("W-FRI").ffill()
        return weekly

    def _build_feature_frame(self) -> pd.DataFrame:
        features = {}
        weekly_index = None
        for series_id in PREDICTOR_SERIES:
            meta = self.fred_engine.series_map.get(series_id)
            if not meta:
                continue
            series = self.fred_engine.get_series(series_id)
            if series.empty:
                continue
            freq = meta.get("freq", "M")
            weekly = self._to_weekly(series, freq)
            if weekly_index is None:
                weekly_index = weekly.index
            else:
                weekly = weekly.reindex(weekly_index).ffill()

            metrics = compute_all_derivatives(
                weekly,
                transform=meta.get("transform", "pct"),
                lookback=WEEKLY_PERIODS["12m"] * 5,
                periods_map=WEEKLY_PERIODS,
            )

            accel = metrics.get("acceleration", pd.Series(dtype=float))
            roc_3m = metrics.get("roc_3m", pd.Series(dtype=float))
            roc_6m = metrics.get("roc_6m", pd.Series(dtype=float))
            value_z = metrics.get("z_score", pd.Series(dtype=float))

            features[f"{series_id}__accel_z"] = _rolling_zscore(accel, 52)
            features[f"{series_id}__roc3_z"] = _rolling_zscore(roc_3m, 52)
            features[f"{series_id}__roc6_z"] = _rolling_zscore(roc_6m, 52)
            features[f"{series_id}__value_z"] = value_z

        if not features:
            return pd.DataFrame()

        frame = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(how="all")
        return frame.fillna(0)

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
        X = np.asarray(X)
        y = np.asarray(y)
        intercept = np.ones((X.shape[0], 1))
        Xb = np.hstack([intercept, X])
        identity = np.eye(Xb.shape[1])
        identity[0, 0] = 0
        return np.linalg.solve(Xb.T @ Xb + l2 * identity, Xb.T @ y)

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> ModelResult | None:
        if X.empty or y.empty:
            return None

        df = X.join(y.rename("target"), how="inner")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 120:
            return None

        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        X_train = train.drop(columns=["target"]).values
        y_train = train["target"].values
        X_test = test.drop(columns=["target"]).values
        y_test = test["target"].values

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        weights = self._fit_ridge(X_train, y_train, l2=1.0)
        intercept = weights[0]
        coeffs = weights[1:]

        pred_test = intercept + X_test @ coeffs
        pred_latest = intercept + ((df.drop(columns=["target"]).iloc[-1].values - mean) / std) @ coeffs

        pred_up = pred_test >= 0
        actual_up = y_test >= 0
        accuracy = float((pred_up == actual_up).mean())
        precision = float(((pred_up) & (actual_up)).sum() / max(pred_up.sum(), 1))
        recall = float(((pred_up) & (actual_up)).sum() / max(actual_up.sum(), 1))
        rmse = float(np.sqrt(np.mean((pred_test - y_test) ** 2)))

        vol = np.std(y_train) or 1
        score = pred_latest / vol
        prob_up = float(_sigmoid(score))
        direction = "UP" if pred_latest >= 0 else "DOWN"
        confidence = prob_up if direction == "UP" else 1 - prob_up

        feature_names = df.drop(columns=["target"]).columns
        top_indices = np.argsort(np.abs(coeffs))[-5:][::-1]
        top_drivers = []
        for idx in top_indices:
            feature = feature_names[idx]
            series_id = feature.split("__", 1)[0]
            meta = self.fred_engine.series_map.get(series_id, {})
            top_drivers.append({
                "feature": feature,
                "series": meta.get("name", series_id),
                "weight": float(coeffs[idx]),
            })

        return ModelResult(
            expected_return=float(pred_latest),
            direction=direction,
            confidence=float(confidence),
            metrics={
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "rmse": rmse,
                "train_size": int(len(train)),
                "test_size": int(len(test)),
            },
            top_drivers=top_drivers,
        )

    def _get_target_series(self, symbol: str) -> pd.Series:
        series = self.market_engine.get_series(symbol)
        if series.empty:
            return series
        return self._to_weekly(series, "D")

    def get_predictions(self) -> Dict:
        now = datetime.now()
        cached_time = self._cache["timestamp"]
        if cached_time and (now - cached_time).total_seconds() < self.cache_seconds:
            return self._cache["data"]

        feature_frame = self._build_feature_frame()
        if feature_frame.empty:
            data = {"error": "Insufficient feature data", "as_of": now.strftime("%Y-%m-%d")}
            self._cache = {"timestamp": now, "data": data}
            return data

        predictions = {}
        ranked = {horizon: [] for horizon in PREDICTION_HORIZONS_WEEKS.keys()}

        for symbol, meta in SECTOR_ETFS.items():
            price = self._get_target_series(symbol)
            if price.empty:
                continue

            horizons = {}
            for horizon_name, horizon_weeks in PREDICTION_HORIZONS_WEEKS.items():
                forward_return = price.pct_change(horizon_weeks).shift(-horizon_weeks)
                model_result = self._fit_model(feature_frame, forward_return)
                if model_result is None:
                    continue

                horizons[horizon_name] = {
                    "expected_return": model_result.expected_return,
                    "direction": model_result.direction,
                    "confidence": model_result.confidence,
                    "metrics": model_result.metrics,
                    "top_drivers": model_result.top_drivers,
                }
                ranked[horizon_name].append({
                    "symbol": symbol,
                    "name": meta["name"],
                    "expected_return": model_result.expected_return,
                    "direction": model_result.direction,
                    "confidence": model_result.confidence,
                })

            if horizons:
                predictions[symbol] = {
                    "symbol": symbol,
                    "name": meta["name"],
                    "icon": meta["icon"],
                    "last_price": float(price.dropna().iloc[-1]),
                    "last_date": price.dropna().index[-1].strftime("%Y-%m-%d"),
                    "horizons": horizons,
                }

        ranked_summary = {}
        for horizon, items in ranked.items():
            sorted_items = sorted(items, key=lambda x: x["expected_return"], reverse=True)
            ranked_summary[horizon] = {
                "bullish": sorted_items[:3],
                "bearish": sorted_items[-3:][::-1],
            }

        data = {
            "as_of": now.strftime("%Y-%m-%d %H:%M"),
            "horizons": PREDICTION_HORIZONS_WEEKS,
            "predictions": predictions,
            "ranked": ranked_summary,
        }
        self._cache = {"timestamp": now, "data": data}
        return data
