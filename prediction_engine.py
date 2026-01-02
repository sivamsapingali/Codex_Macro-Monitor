"""
Acceleration/Deceleration Prediction Engine
Builds multi-horizon sector ETF direction forecasts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple

import hashlib
import math
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    BACKTEST_CACHE_SECONDS,
    BACKTEST_MAX_SPLITS,
    BACKTEST_MIN_TRAIN_WEEKS,
    BACKTEST_TEST_WEEKS,
    BACKTEST_WINDOW_YEARS,
    CORE_MACRO_SERIES,
    DIRECTION_DEADBAND,
    DIRECTION_MODEL_MODE,
    INDICATORS,
    MAX_POSITION_PCT,
    MIN_VOL,
    MODEL_SELECTION_MAX_SPLITS,
    MODEL_SELECTION_MIN_SAMPLES,
    MODEL_SELECTION_GROUPS,
    MODEL_SELECTION_HORIZONS,
    MODEL_SELECTION_MODELS,
    MODEL_SELECTION_PATH,
    MODEL_SELECTION_TEST_WEEKS,
    PORTFOLIO_NOTIONAL,
    PREDICTOR_SERIES,
    PREDICTION_HORIZONS_WEEKS,
    SECTOR_ETFS,
    SECTOR_MACRO_MAP,
    TARGET_RETURN,
    TARGET_RISK_PCT,
)
from derivatives import compute_all_derivatives

WEEKLY_PERIODS = {"1m": 4, "3m": 13, "6m": 26, "12m": 52}
TRADING_DAYS = 252
DERIVED_SERIES_META = {
    "REAL10Y": {"name": "10Y Real Yield", "transform": "level"},
    "REAL2Y": {"name": "2Y Real Yield", "transform": "level"},
    "CREDIT_SPREAD": {"name": "HY-IG Credit Spread", "transform": "level"},
    "NET_LIQUIDITY": {"name": "Net Liquidity (WALCL - RRP)", "transform": "level"},
}
FEATURE_GROUPS = [
    "macro_core",
    "macro_sector",
    "macro_all",
    "sector_tech",
    "macro_core_plus_tech",
    "macro_sector_plus_tech",
    "macro_all_plus_tech",
]
DEFAULT_FEATURE_GROUP = "macro_sector_plus_tech"
THRESHOLD_GRID = np.linspace(0.35, 0.65, 13)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window // 2).mean()
    std = series.rolling(window=window, min_periods=window // 2).std()
    return (series - mean) / std


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100.0


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs_prices(spot: float, strike: float, t_years: float, rate: float, sigma: float) -> tuple[float, float]:
    if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        call = max(spot - strike, 0.0)
        put = max(strike - spot, 0.0)
        return call, put
    vol_sqrt = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma ** 2) * t_years) / vol_sqrt
    d2 = d1 - vol_sqrt
    call = spot * _norm_cdf(d1) - strike * math.exp(-rate * t_years) * _norm_cdf(d2)
    put = strike * math.exp(-rate * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    return call, put


@dataclass
class ModelResult:
    expected_return: float
    direction: str
    confidence: float
    metrics: Dict[str, float]
    top_drivers: List[Dict[str, float]]


class PredictionEngine:
    def __init__(
        self,
        fred_engine,
        market_engine,
        cache_seconds: int = 300,
        direction_mode: str | None = None,
    ):
        self.fred_engine = fred_engine
        self.market_engine = market_engine
        self.cache_seconds = cache_seconds
        self._cache = {"timestamp": None, "data": None}
        self._options_cache = {"timestamp": None, "data": None}
        self.direction_mode = (direction_mode or DIRECTION_MODEL_MODE).lower()
        self.direction_deadband = float(DIRECTION_DEADBAND)
        self.portfolio_notional = float(PORTFOLIO_NOTIONAL)
        self.target_risk_pct = float(TARGET_RISK_PCT)
        self.max_position_pct = float(MAX_POSITION_PCT)
        self.min_vol = float(MIN_VOL)
        self.target_return = float(TARGET_RETURN)
        self.feature_meta: Dict[str, Dict[str, str]] = {}
        self.model_selection_path = Path(MODEL_SELECTION_PATH)
        self.model_selection_min_samples = int(MODEL_SELECTION_MIN_SAMPLES)
        self.model_selection_max_splits = int(MODEL_SELECTION_MAX_SPLITS)
        self.model_selection_test_weeks = int(MODEL_SELECTION_TEST_WEEKS)
        self.backtest_cache_seconds = int(BACKTEST_CACHE_SECONDS)
        self.backtest_min_train = int(BACKTEST_MIN_TRAIN_WEEKS)
        self.backtest_test_weeks = int(BACKTEST_TEST_WEEKS)
        self.backtest_max_splits = int(BACKTEST_MAX_SPLITS)
        self.backtest_window_years = int(BACKTEST_WINDOW_YEARS)
        self._backtest_cache = {"timestamp": None, "data": None}
        self.model_selection_meta: Dict[str, str] = {}
        self.model_selection = self._load_model_selection()

    def _to_weekly(self, series: pd.Series, freq: str) -> pd.Series:
        if series.empty:
            return series
        series = series.sort_index()
        if freq in {"D", "W"}:
            weekly = series.resample("W-FRI").last()
        else:
            weekly = series.resample("W-FRI").ffill()
        return weekly

    def _build_derived_series(self, series_map: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        derived: Dict[str, pd.Series] = {}

        def align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
            idx = a.index.union(b.index)
            return a.reindex(idx).ffill(), b.reindex(idx).ffill()

        dgs10 = series_map.get("DGS10")
        t10yie = series_map.get("T10YIE")
        if dgs10 is not None and t10yie is not None and not dgs10.empty and not t10yie.empty:
            a, b = align(dgs10, t10yie)
            derived["REAL10Y"] = a - b

        dgs2 = series_map.get("DGS2")
        t5yie = series_map.get("T5YIE")
        if dgs2 is not None and t5yie is not None and not dgs2.empty and not t5yie.empty:
            a, b = align(dgs2, t5yie)
            derived["REAL2Y"] = a - b

        hy = series_map.get("BAMLH0A0HYM2")
        ig = series_map.get("BAMLC0A0CM")
        if hy is not None and ig is not None and not hy.empty and not ig.empty:
            a, b = align(hy, ig)
            derived["CREDIT_SPREAD"] = a - b

        walcl = series_map.get("WALCL")
        rrp = series_map.get("RRPONTSYD")
        if walcl is not None and rrp is not None and not walcl.empty and not rrp.empty:
            a, b = align(walcl, rrp)
            derived["NET_LIQUIDITY"] = a - b

        return derived

    def _build_macro_feature_frame(self) -> pd.DataFrame:
        series_map: Dict[str, pd.Series] = {}
        series_meta: Dict[str, Dict[str, str]] = {}

        for series_id in PREDICTOR_SERIES:
            meta = self.fred_engine.series_map.get(series_id)
            if not meta:
                continue
            series = self.fred_engine.get_series(series_id)
            if series.empty:
                continue
            freq = meta.get("freq", "M")
            weekly = self._to_weekly(series, freq)
            series_map[series_id] = weekly
            series_meta[series_id] = {
                "name": meta.get("name", series_id),
                "transform": meta.get("transform", "pct"),
            }

        if not series_map:
            return pd.DataFrame()

        derived = self._build_derived_series(series_map)
        for series_id, weekly in derived.items():
            series_map[series_id] = weekly
            series_meta[series_id] = DERIVED_SERIES_META.get(series_id, {"name": series_id, "transform": "level"})

        weekly_index = None
        for weekly in series_map.values():
            weekly_index = weekly.index if weekly_index is None else weekly_index.union(weekly.index)
        weekly_index = weekly_index.sort_values()

        features = {}
        for series_id, weekly in series_map.items():
            weekly = weekly.reindex(weekly_index).ffill()
            transform = series_meta.get(series_id, {}).get("transform", "pct")

            metrics = compute_all_derivatives(
                weekly,
                transform=transform,
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

        self.feature_meta = series_meta

        if not features:
            return pd.DataFrame()

        frame = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(how="all")
        return frame.fillna(0)

    def _build_sector_features(
        self,
        symbol: str,
        price: pd.Series,
        sp500_weekly: pd.Series,
    ) -> pd.DataFrame:
        weekly = self._to_weekly(price, "D")
        returns = weekly.pct_change()
        ret_1w = weekly.pct_change(1)
        ret_4w = weekly.pct_change(4)
        ret_13w = weekly.pct_change(13)
        ret_26w = weekly.pct_change(26)
        ret_52w = weekly.pct_change(52)
        vol_4w = returns.rolling(4).std()
        vol_13w = returns.rolling(13).std()

        sp500_weekly = sp500_weekly.reindex(weekly.index).ffill()
        sp500_returns = sp500_weekly.pct_change()
        sp500_ret_1w = sp500_returns
        sp500_ret_4w = sp500_weekly.pct_change(4)
        sp500_ret_13w = sp500_weekly.pct_change(13)
        sp500_ret_26w = sp500_weekly.pct_change(26)
        sp500_vol_4w = sp500_returns.rolling(4).std()
        sp500_vol_13w = sp500_returns.rolling(13).std()

        rel_1w = ret_1w - sp500_ret_1w
        rel_4w = ret_4w - sp500_ret_4w
        rel_13w = ret_13w - sp500_ret_13w
        rel_26w = ret_26w - sp500_ret_26w

        ma_13w = weekly.rolling(13).mean()
        ma_26w = weekly.rolling(26).mean()
        ma_ratio_13w = weekly / ma_13w - 1
        ma_ratio_26w = weekly / ma_26w - 1
        trend_13_26 = ma_13w / ma_26w - 1
        drawdown_26w = weekly / weekly.rolling(26).max() - 1
        vol_ratio = vol_4w / vol_13w.replace(0, np.nan)
        vol_spread = vol_4w - sp500_vol_4w
        rsi_14w = _rsi(weekly, 14)
        beta_13w = returns.rolling(13).cov(sp500_returns) / sp500_returns.rolling(13).var()
        corr_13w = returns.rolling(13).corr(sp500_returns)

        data = {
            f"{symbol}__ret_1w": ret_1w,
            f"{symbol}__ret_4w": ret_4w,
            f"{symbol}__ret_13w": ret_13w,
            f"{symbol}__ret_26w": ret_26w,
            f"{symbol}__ret_52w": ret_52w,
            f"{symbol}__vol_4w": vol_4w,
            f"{symbol}__vol_13w": vol_13w,
            f"{symbol}__vol_ratio": vol_ratio,
            f"{symbol}__vol_spread": vol_spread,
            f"{symbol}__ma_ratio_13w": ma_ratio_13w,
            f"{symbol}__ma_ratio_26w": ma_ratio_26w,
            f"{symbol}__trend_13_26": trend_13_26,
            f"{symbol}__drawdown_26w": drawdown_26w,
            f"{symbol}__rsi_14w": rsi_14w,
            f"{symbol}__rel_4w": rel_4w,
            f"{symbol}__rel_13w": rel_13w,
            f"{symbol}__rel_1w": rel_1w,
            f"{symbol}__rel_26w": rel_26w,
            f"{symbol}__beta_13w": beta_13w,
            f"{symbol}__corr_13w": corr_13w,
        }
        frame = pd.DataFrame(data).replace([np.inf, -np.inf], np.nan)
        return frame.dropna(how="all").fillna(0)

    def _load_model_selection(self) -> Dict[str, Dict]:
        if not self.model_selection_path.exists():
            self.model_selection_meta = {}
            return {}
        try:
            with self.model_selection_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                self.model_selection_meta = data.get("meta", {}) or {}
                selections = data.get("selections")
                if isinstance(selections, dict):
                    return selections
                return data
        except Exception:
            pass
        self.model_selection_meta = {}
        return {}

    def _save_model_selection(self, selections: Dict[str, Dict], meta: Dict[str, str]) -> None:
        payload = {"meta": meta, "selections": selections}
        try:
            self.model_selection_path.parent.mkdir(parents=True, exist_ok=True)
            with self.model_selection_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        except Exception:
            return

    def _subset_macro_features(self, macro_features: pd.DataFrame, series_ids: List[str]) -> pd.DataFrame:
        if macro_features.empty:
            return macro_features
        if not series_ids:
            return macro_features
        cols = [c for c in macro_features.columns if c.split("__", 1)[0] in series_ids]
        if not cols:
            return macro_features
        return macro_features[cols]

    def _combine_feature_frames(self, macro_features: pd.DataFrame, sector_features: pd.DataFrame) -> pd.DataFrame:
        if macro_features.empty and sector_features.empty:
            return pd.DataFrame()
        if sector_features.empty:
            return macro_features.copy()
        if macro_features.empty:
            return sector_features.copy()
        combined = macro_features.reindex(sector_features.index).ffill()
        combined = combined.join(sector_features, how="inner")
        return combined.replace([np.inf, -np.inf], np.nan).dropna()

    def _build_feature_groups(
        self,
        symbol: str,
        price: pd.Series,
        macro_features: pd.DataFrame,
        sp500_weekly: pd.Series,
    ) -> Dict[str, pd.DataFrame]:
        sector_features = self._build_sector_features(symbol, price, sp500_weekly)
        core_series = list(CORE_MACRO_SERIES)
        sector_series = list(CORE_MACRO_SERIES) + list(SECTOR_MACRO_MAP.get(symbol, []))

        macro_core = self._subset_macro_features(macro_features, core_series)
        macro_sector = self._subset_macro_features(macro_features, sector_series)
        macro_all = macro_features.copy()

        groups = {
            "macro_core": macro_core,
            "macro_sector": macro_sector,
            "macro_all": macro_all,
            "sector_tech": sector_features,
            "macro_core_plus_tech": self._combine_feature_frames(macro_core, sector_features),
            "macro_sector_plus_tech": self._combine_feature_frames(macro_sector, sector_features),
            "macro_all_plus_tech": self._combine_feature_frames(macro_all, sector_features),
        }

        return groups

    def _select_macro_features(self, macro_features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        allowed = list(CORE_MACRO_SERIES) + list(SECTOR_MACRO_MAP.get(symbol, []))
        return self._subset_macro_features(macro_features, allowed)

    def _assemble_features(
        self,
        symbol: str,
        price: pd.Series,
        macro_features: pd.DataFrame,
        sp500_weekly: pd.Series,
    ) -> pd.DataFrame:
        groups = self._build_feature_groups(symbol, price, macro_features, sp500_weekly)
        preferred = groups.get(DEFAULT_FEATURE_GROUP)
        if preferred is not None and not preferred.empty:
            return preferred
        for group_name in FEATURE_GROUPS:
            candidate = groups.get(group_name)
            if candidate is not None and not candidate.empty:
                return candidate
        return pd.DataFrame()

    def _feature_hash(self, features: pd.DataFrame, target: pd.Series) -> str:
        if features.empty or target.empty:
            return ""
        tail = features.tail(260).copy()
        aligned_target = target.reindex(tail.index).fillna(0.0)
        tail["__target__"] = aligned_target
        cleaned = tail.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        arr = np.nan_to_num(cleaned.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        hasher = hashlib.sha256()
        hasher.update(",".join(cleaned.columns).encode("utf-8"))
        hasher.update(arr.tobytes())
        return hasher.hexdigest()

    def _direction_model_catalog(self) -> Dict[str, object]:
        catalog = {
            "logistic": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
                ]
            ),
            "mlp": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(48, 24),
                            max_iter=400,
                            early_stopping=True,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "gboost": GradientBoostingClassifier(
                random_state=42,
                n_estimators=110,
                learning_rate=0.05,
                max_depth=2,
            ),
            "hist_gb": HistGradientBoostingClassifier(
                random_state=42,
                max_iter=200,
                learning_rate=0.05,
                max_depth=3,
                l2_regularization=0.2,
            ),
        }

        if self.direction_mode in catalog:
            return {self.direction_mode: catalog[self.direction_mode]}
        if self.direction_mode == "fast":
            allowed = ["logistic"]
        elif self.direction_mode == "auto":
            allowed = ["logistic", "mlp", "gboost", "hist_gb"]
        else:
            allowed = ["logistic", "gboost", "hist_gb"]
        return {name: catalog[name] for name in allowed if name in catalog}

    def _direction_model_candidates(self, preferred_model: str | None = None) -> List[Tuple[str, object]]:
        catalog = self._direction_model_catalog()
        if preferred_model:
            model = catalog.get(preferred_model)
            return [(preferred_model, model)] if model is not None else []
        return list(catalog.items())

    def _fit_model(self, model: object, X: pd.DataFrame, y: pd.Series, weights: pd.Series) -> None:
        try:
            if isinstance(model, Pipeline):
                model.fit(X, y, clf__sample_weight=weights)
            else:
                model.fit(X, y, sample_weight=weights)
        except TypeError:
            model.fit(X, y)

    def _predict_scores(self, model: object, X: pd.DataFrame) -> np.ndarray | None:
        if hasattr(model, "predict_proba"):
            try:
                return model.predict_proba(X)[:, 1]
            except Exception:
                pass
        if hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X)
                return 1 / (1 + np.exp(-scores))
            except Exception:
                pass
        try:
            return model.predict(X)
        except Exception:
            return None

    def _choose_threshold(self, scores: np.ndarray | None, target: pd.Series) -> float:
        if scores is None or len(scores) != len(target):
            return 0.5
        best_threshold = 0.5
        best_score = -1.0
        for threshold in THRESHOLD_GRID:
            preds = scores >= threshold
            score = balanced_accuracy_score(target, preds)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return best_threshold

    def _build_splits(
        self,
        n_samples: int,
        min_train: int,
        test_window: int,
        max_splits: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        if n_samples <= min_train + 1:
            return splits
        train_end = min_train
        while train_end + test_window <= n_samples and len(splits) < max_splits:
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, train_end + test_window)
            splits.append((train_idx, test_idx))
            train_end += test_window
        if not splits:
            split_idx = int(n_samples * 0.8)
            if split_idx > 0 and split_idx < n_samples:
                splits = [(np.arange(split_idx), np.arange(split_idx, n_samples))]
        return splits

    def _build_direction_dataset(
        self,
        X: pd.DataFrame,
        forward_return: pd.Series,
        vol_horizon: pd.Series | None,
        min_samples: int,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series | None] | None:
        if X.empty or forward_return.empty:
            return None
        df = X.join(forward_return.rename("forward_return"), how="inner")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < min_samples:
            return None
        if df["forward_return"].nunique() < 2:
            return None

        features = df.drop(columns=["forward_return"])
        target = (df["forward_return"] > 0).astype(int)
        weights = pd.Series(1.0, index=df.index)
        vol = None

        if vol_horizon is not None and self.direction_deadband > 0:
            vol = vol_horizon.reindex(df.index)
            vol = vol.fillna(vol.median())
            if not vol.isna().all():
                threshold = self.direction_deadband * vol
                mask = df["forward_return"].abs() >= threshold
                filtered = df.loc[mask]
                if len(filtered) >= min_samples and filtered["forward_return"].nunique() > 1:
                    df = filtered
                    features = df.drop(columns=["forward_return"])
                    target = (df["forward_return"] > 0).astype(int)
                    vol = vol.reindex(df.index).fillna(vol.median())
                    scale = (self.direction_deadband * vol).replace(0, np.nan)
                    scale = scale.fillna(scale.median())
                    scale = scale.fillna(df["forward_return"].abs().median())
                    scale = scale.replace(0, 1e-6)
                    weights = 1 + (df["forward_return"].abs() / scale)
                    weights = weights.clip(1.0, 3.0)

        if vol is None and vol_horizon is not None:
            vol = vol_horizon.reindex(df.index).fillna(vol_horizon.median())

        return features, target, weights, df["forward_return"], vol

    def _evaluate_direction_candidate(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        weights: pd.Series,
        model: object,
    ) -> Dict[str, float] | None:
        n_samples = len(features)
        min_train = max(120, self.model_selection_min_samples - self.model_selection_test_weeks)
        splits = self._build_splits(
            n_samples=n_samples,
            min_train=min_train,
            test_window=self.model_selection_test_weeks,
            max_splits=self.model_selection_max_splits,
        )
        if not splits:
            return None

        metrics = []
        thresholds = []
        aucs = []

        for train_idx, test_idx in splits:
            X_train = features.iloc[train_idx]
            y_train = target.iloc[train_idx]
            w_train = weights.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_test = target.iloc[test_idx]

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue

            model_clone = clone(model)
            self._fit_model(model_clone, X_train, y_train, w_train)
            train_scores = self._predict_scores(model_clone, X_train)
            threshold = self._choose_threshold(train_scores, y_train)
            test_scores = self._predict_scores(model_clone, X_test)
            if test_scores is None:
                preds = model_clone.predict(X_test)
                threshold = 0.5
            else:
                preds = test_scores >= threshold

            accuracy = accuracy_score(y_test, preds)
            balanced = balanced_accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)
            metrics.append((accuracy, balanced, precision, recall))
            thresholds.append(threshold)

            if test_scores is not None and y_test.nunique() > 1:
                try:
                    aucs.append(roc_auc_score(y_test, test_scores))
                except Exception:
                    pass

        if not metrics:
            return None

        accuracy = float(np.mean([m[0] for m in metrics]))
        balanced = float(np.mean([m[1] for m in metrics]))
        precision = float(np.mean([m[2] for m in metrics]))
        recall = float(np.mean([m[3] for m in metrics]))
        threshold = float(np.median(thresholds)) if thresholds else 0.5
        auc = float(np.mean(aucs)) if aucs else None

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "threshold": threshold,
            "samples": int(n_samples),
            "splits": int(len(metrics)),
        }

    def _fit_direction_model(
        self,
        X: pd.DataFrame,
        forward_return: pd.Series,
        vol_horizon: pd.Series | None,
        preferred_model: str | None = None,
        threshold_override: float | None = None,
        min_samples: int = 160,
    ) -> tuple[str, float, float, Dict[str, float], object] | None:
        dataset = self._build_direction_dataset(X, forward_return, vol_horizon, min_samples=min_samples)
        if dataset is None:
            return None
        features, target, weights, _, _ = dataset

        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx]
        y_train = target.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = target.iloc[split_idx:]
        w_train = weights.iloc[:split_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            return None

        best = None
        for name, model in self._direction_model_candidates(preferred_model):
            if model is None:
                continue
            self._fit_model(model, X_train, y_train, w_train)
            probs = self._predict_scores(model, X_test)
            auc = None

            if probs is None:
                preds = model.predict(X_test)
                best_threshold = 0.5
            else:
                if threshold_override is None:
                    best_threshold = self._choose_threshold(probs, y_test)
                else:
                    best_threshold = float(threshold_override)
                preds = probs >= best_threshold
                if y_test.nunique() > 1:
                    try:
                        auc = roc_auc_score(y_test, probs)
                    except Exception:
                        auc = None

            accuracy = accuracy_score(y_test, preds)
            balanced = balanced_accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)

            metrics = {
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced),
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc) if auc is not None else None,
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "deadband": float(self.direction_deadband),
                "threshold": float(best_threshold),
            }
            score = metrics["balanced_accuracy"]
            if best is None or score > best[1]:
                best = (name, score, best_threshold, metrics, model)

        if best is None:
            return None

        name, _, threshold, metrics, model = best
        self._fit_model(model, features, target, weights)

        latest = features.iloc[[-1]]
        probs_latest = self._predict_scores(model, latest)
        if probs_latest is None:
            prob_up = 0.5
        else:
            prob_up = float(np.atleast_1d(probs_latest)[0])

        return name, prob_up, threshold, metrics, model

    def _fit_return_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[float, float, List[Dict[str, float]]] | None:
        if X.empty or y.empty:
            return None
        df = X.join(y.rename("target"), how="inner")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 160:
            return None

        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        X_train = train.drop(columns=["target"])
        y_train = train["target"]
        X_test = test.drop(columns=["target"])
        y_test = test["target"]

        model = self._fit_return_pipeline(X_train, y_train)

        pred_test = model.predict(X_test)
        pred_latest = float(model.predict(df.drop(columns=["target"]).iloc[[-1]])[0])
        rmse = float(np.sqrt(np.mean((pred_test - y_test) ** 2)))

        coeffs = model.named_steps["ridge"].coef_
        feature_names = X_train.columns
        top_indices = np.argsort(np.abs(coeffs))[-5:][::-1]
        top_drivers = []
        for idx in top_indices:
            feature = feature_names[idx]
            series_id = feature.split("__", 1)[0]
            top_drivers.append({
                "feature": feature,
                "series": self._series_label(series_id),
                "weight": float(coeffs[idx]),
            })

        return pred_latest, rmse, top_drivers

    def _fit_return_pipeline(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        model.fit(X, y)
        return model

    def _size_position(self, expected_return: float, confidence: float, vol_horizon: float | None) -> float:
        if vol_horizon is None or not math.isfinite(vol_horizon):
            return 0.0
        if not math.isfinite(expected_return):
            return 0.0
        signal = (confidence - 0.5) * 2
        if self.target_return > 0:
            signal *= min(1.0, abs(expected_return) / self.target_return)
        vol = max(vol_horizon, self.min_vol)
        weight = signal * (self.target_risk_pct / vol)
        weight = max(-self.max_position_pct, min(self.max_position_pct, weight))
        return float(weight)

    def _max_drawdown(self, equity: np.ndarray) -> float:
        if equity.size == 0:
            return 0.0
        peaks = np.maximum.accumulate(equity)
        drawdown = equity - peaks
        return float(drawdown.min())

    def _backtest_symbol(
        self,
        features: pd.DataFrame,
        forward_return: pd.Series,
        vol_horizon: pd.Series | None,
        model_name: str,
        threshold: float | None,
        horizon_weeks: int,
        cutoff_date: pd.Timestamp | None,
    ) -> tuple[Dict[pd.Timestamp, float], int, int, List[float]]:
        min_samples = max(self.backtest_min_train + self.backtest_test_weeks, 160)
        dataset = self._build_direction_dataset(
            features,
            forward_return,
            vol_horizon,
            min_samples=min_samples,
        )
        if dataset is None:
            return {}, 0, 0, []

        features_df, target, weights, forward_filtered, vol_filtered = dataset
        splits = self._build_splits(
            n_samples=len(features_df),
            min_train=self.backtest_min_train,
            test_window=self.backtest_test_weeks,
            max_splits=self.backtest_max_splits,
        )
        if not splits:
            return {}, 0, 0, []

        pnl_by_date: Dict[pd.Timestamp, float] = {}
        trade_returns: List[float] = []
        trade_count = 0
        hit_count = 0

        model_catalog = self._direction_model_catalog()
        model_template = model_catalog.get(model_name)
        if model_template is None:
            model_template = next(iter(model_catalog.values()))

        for train_idx, test_idx in splits:
            X_train = features_df.iloc[train_idx]
            y_train = target.iloc[train_idx]
            w_train = weights.iloc[train_idx]
            X_test = features_df.iloc[test_idx]
            y_test = target.iloc[test_idx]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue

            direction_model = clone(model_template)
            self._fit_model(direction_model, X_train, y_train, w_train)
            train_scores = self._predict_scores(direction_model, X_train)
            chosen_threshold = float(threshold) if threshold is not None else self._choose_threshold(train_scores, y_train)
            test_scores = self._predict_scores(direction_model, X_test)

            return_model = self._fit_return_pipeline(X_train, forward_filtered.iloc[train_idx])
            expected_returns = return_model.predict(X_test)

            if test_scores is None:
                preds = direction_model.predict(X_test)
                probs = np.array(preds, dtype=float)
                chosen_threshold = 0.5
            else:
                probs = np.asarray(test_scores, dtype=float)
                preds = probs >= chosen_threshold

            for idx, date in enumerate(X_test.index):
                expected_return = float(expected_returns[idx])
                prob_up = float(probs[idx])
                direction = "UP" if prob_up >= chosen_threshold else "DOWN"
                confidence = prob_up if direction == "UP" else 1 - prob_up
                vol_value = None
                if vol_filtered is not None and date in vol_filtered.index:
                    vol_value = float(vol_filtered.loc[date])
                weight = self._size_position(expected_return, confidence, vol_value)
                realized = float(forward_filtered.loc[date])
                pnl = weight * self.portfolio_notional * realized
                pnl_date = date + pd.Timedelta(weeks=horizon_weeks)
                if cutoff_date is None or pnl_date >= cutoff_date:
                    pnl_by_date[pnl_date] = pnl_by_date.get(pnl_date, 0.0) + pnl
                    trade_return = weight * realized
                    trade_returns.append(trade_return)
                    trade_count += 1
                    if pnl > 0:
                        hit_count += 1

        return pnl_by_date, trade_count, hit_count, trade_returns

    def _series_label(self, series_id: str) -> str:
        meta = self.fred_engine.series_map.get(series_id)
        if meta and meta.get("name"):
            return meta["name"]
        meta = self.feature_meta.get(series_id)
        if meta and meta.get("name"):
            return meta["name"]
        return series_id

    def _realized_vol_annual(self, series: pd.Series, lookback: int = 60) -> float | None:
        returns = series.pct_change().dropna()
        if returns.empty:
            return None
        window = returns.tail(lookback)
        vol_daily = float(window.std()) if len(window) >= 5 else float(returns.std())
        if not math.isfinite(vol_daily) or vol_daily <= 0:
            return None
        return vol_daily * math.sqrt(TRADING_DAYS)

    def _risk_free_rate(self) -> float:
        series = self.fred_engine.get_series("FEDFUNDS")
        if series.empty:
            return 0.0
        latest = series.dropna()
        if latest.empty:
            return 0.0
        rate = float(latest.iloc[-1]) / 100.0
        return rate if math.isfinite(rate) else 0.0

    def _rng_for(self, key: str) -> np.random.Generator:
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")
        return np.random.default_rng(seed)

    def _simulate_terminal_prices(
        self,
        spot: float,
        expected_return: float,
        sigma_annual: float,
        t_years: float,
        simulations: int,
        seed_key: str,
    ) -> np.ndarray:
        if spot <= 0 or t_years <= 0 or simulations <= 0:
            return np.array([])
        sigma_t = sigma_annual * math.sqrt(t_years)
        mu = max(expected_return, -0.99)
        mu_log = math.log1p(mu) - 0.5 * sigma_t ** 2
        rng = self._rng_for(seed_key)
        z = rng.standard_normal(simulations)
        return spot * np.exp(mu_log + sigma_t * z)

    def _get_target_series(self, symbol: str) -> pd.Series:
        series = self.market_engine.get_series(symbol)
        if series.empty:
            return series
        return self._to_weekly(series, "D")

    def get_model_selection(self) -> Dict:
        return {"meta": self.model_selection_meta, "selections": self.model_selection}

    def optimize_models(self) -> Dict:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        macro_features = self._build_macro_feature_frame()
        if macro_features.empty:
            return {"error": "Insufficient feature data", "as_of": now}

        sp500_series = self.fred_engine.get_series("SP500")
        sp500_weekly = self._to_weekly(sp500_series, "D") if not sp500_series.empty else pd.Series(dtype=float)

        full_catalog = self._direction_model_catalog()
        model_catalog = list(full_catalog.keys())
        selection_groups = FEATURE_GROUPS
        if MODEL_SELECTION_GROUPS:
            requested = [g.strip() for g in MODEL_SELECTION_GROUPS.split(",") if g.strip()]
            selection_groups = [g for g in requested if g in FEATURE_GROUPS] or FEATURE_GROUPS
        selection_models = full_catalog
        if MODEL_SELECTION_MODELS:
            requested = [m.strip() for m in MODEL_SELECTION_MODELS.split(",") if m.strip()]
            selection_models = {name: full_catalog[name] for name in requested if name in full_catalog} or full_catalog
        selection_horizons = list(PREDICTION_HORIZONS_WEEKS.keys())
        if MODEL_SELECTION_HORIZONS:
            requested = [h.strip() for h in MODEL_SELECTION_HORIZONS.split(",") if h.strip()]
            selection_horizons = [h for h in requested if h in PREDICTION_HORIZONS_WEEKS] or selection_horizons

        selections: Dict[str, Dict] = {
            symbol: {h: data for h, data in horizons.items() if h in selection_horizons}
            for symbol, horizons in self.model_selection.items()
            if isinstance(horizons, dict)
        }

        selection_meta = {
            "generated_at": now,
            "metric": "balanced_accuracy",
            "deadband": float(self.direction_deadband),
            "feature_groups": selection_groups,
            "models": list(selection_models.keys()),
            "min_samples": int(self.model_selection_min_samples),
            "test_window_weeks": int(self.model_selection_test_weeks),
        }

        total_targets = len(SECTOR_ETFS) * len(selection_horizons)
        attempted = 0

        for symbol, meta in SECTOR_ETFS.items():
            price = self._get_target_series(symbol)
            if price.empty:
                continue

            feature_groups = self._build_feature_groups(symbol, price, macro_features, sp500_weekly)
            weekly_returns = price.pct_change(1)
            vol_1w = weekly_returns.rolling(52).std()

            existing_symbol = selections.get(symbol, {})
            if symbol not in selections:
                selections[symbol] = existing_symbol

            for horizon_name in selection_horizons:
                horizon_weeks = PREDICTION_HORIZONS_WEEKS[horizon_name]
                forward_return = price.pct_change(horizon_weeks).shift(-horizon_weeks)
                vol_horizon = vol_1w * math.sqrt(horizon_weeks)

                existing = existing_symbol.get(horizon_name) if isinstance(existing_symbol, dict) else None
                if existing:
                    existing_group = existing.get("feature_group")
                    existing_features = feature_groups.get(existing_group)
                    if existing_features is not None and not existing_features.empty:
                        dataset = self._build_direction_dataset(
                            existing_features,
                            forward_return,
                            vol_horizon,
                            min_samples=self.model_selection_min_samples,
                        )
                        if dataset is not None:
                            features_df, _, _, forward_filtered, _ = dataset
                            current_hash = self._feature_hash(features_df, forward_filtered)
                            if current_hash == existing.get("feature_hash"):
                                selections[symbol][horizon_name] = existing
                                attempted += 1
                                if attempted % 5 == 0:
                                    print(f"Model selection progress: {attempted}/{total_targets}")
                                self.model_selection = selections
                                self.model_selection_meta = selection_meta
                                self._save_model_selection(selections, selection_meta)
                                continue

                candidate_results = []
                best = None

                for group_name in selection_groups:
                    features = feature_groups.get(group_name)
                    if features is None or features.empty:
                        continue
                    dataset = self._build_direction_dataset(
                        features,
                        forward_return,
                        vol_horizon,
                        min_samples=self.model_selection_min_samples,
                    )
                    if dataset is None:
                        continue
                    features_df, target, weights, forward_filtered, _ = dataset
                    feature_hash = self._feature_hash(features_df, forward_filtered)

                    for model_name, model in selection_models.items():
                        metrics = self._evaluate_direction_candidate(features_df, target, weights, model)
                        if not metrics:
                            continue
                        result = {
                            "feature_group": group_name,
                            "model": model_name,
                            "threshold": metrics["threshold"],
                            "score": metrics["balanced_accuracy"],
                            "metrics": metrics,
                            "feature_hash": feature_hash,
                            "feature_cols": int(features_df.shape[1]),
                            "samples": int(metrics.get("samples", 0)),
                        }
                        candidate_results.append(result)

                        if best is None:
                            best = result
                            continue
                        best_score = best.get("score", 0)
                        best_auc = (best.get("metrics") or {}).get("auc") or 0
                        cand_score = result.get("score", 0)
                        cand_auc = (result.get("metrics") or {}).get("auc") or 0
                        if cand_score > best_score or (cand_score == best_score and cand_auc > best_auc):
                            best = result

                if best:
                    top_candidates = sorted(
                        candidate_results,
                        key=lambda item: (item.get("score", 0), (item.get("metrics") or {}).get("auc") or 0),
                        reverse=True,
                    )[:3]
                    selections[symbol][horizon_name] = {
                        "symbol": symbol,
                        "name": meta["name"],
                        "feature_group": best["feature_group"],
                        "model": best["model"],
                        "threshold": best["threshold"],
                        "score": best["score"],
                        "metrics": best["metrics"],
                        "feature_hash": best["feature_hash"],
                        "feature_cols": best["feature_cols"],
                        "samples": best["samples"],
                        "updated_at": now,
                        "candidates": top_candidates,
                    }
                attempted += 1
                if attempted % 5 == 0:
                    print(f"Model selection progress: {attempted}/{total_targets}")
                self.model_selection = selections
                self.model_selection_meta = selection_meta
                self._save_model_selection(selections, selection_meta)

        self.model_selection = selections
        self.model_selection_meta = selection_meta
        self._save_model_selection(selections, selection_meta)

        return {"meta": selection_meta, "selections": selections}

    def get_predictions(self) -> Dict:
        now = datetime.now()
        cached_time = self._cache["timestamp"]
        if cached_time and (now - cached_time).total_seconds() < self.cache_seconds:
            return self._cache["data"]

        macro_features = self._build_macro_feature_frame()
        if macro_features.empty:
            data = {"error": "Insufficient feature data", "as_of": now.strftime("%Y-%m-%d")}
            self._cache = {"timestamp": now, "data": data}
            return data

        sp500_series = self.fred_engine.get_series("SP500")
        sp500_weekly = self._to_weekly(sp500_series, "D") if not sp500_series.empty else pd.Series(dtype=float)

        predictions = {}
        ranked = {horizon: [] for horizon in PREDICTION_HORIZONS_WEEKS.keys()}

        for symbol, meta in SECTOR_ETFS.items():
            price = self._get_target_series(symbol)
            if price.empty:
                continue
            feature_groups = self._build_feature_groups(symbol, price, macro_features, sp500_weekly)
            weekly_returns = price.pct_change(1)
            vol_1w = weekly_returns.rolling(52).std()
            selection_map = self.model_selection.get(symbol, {})

            horizons = {}
            for horizon_name, horizon_weeks in PREDICTION_HORIZONS_WEEKS.items():
                forward_return = price.pct_change(horizon_weeks).shift(-horizon_weeks)
                vol_horizon = vol_1w * math.sqrt(horizon_weeks)

                selection = selection_map.get(horizon_name) if isinstance(selection_map, dict) else None
                selection_valid = False
                feature_group_used = DEFAULT_FEATURE_GROUP
                preferred_model = None
                threshold_override = None
                features = None

                if selection:
                    group_name = selection.get("feature_group")
                    group_features = feature_groups.get(group_name)
                    if group_features is not None and not group_features.empty:
                        feature_group_used = group_name
                        features = group_features
                        dataset = self._build_direction_dataset(
                            group_features,
                            forward_return,
                            vol_horizon,
                            min_samples=self.model_selection_min_samples,
                        )
                        if dataset is not None:
                            features_df, _, _, forward_filtered, _ = dataset
                            current_hash = self._feature_hash(features_df, forward_filtered)
                            selection_valid = current_hash == selection.get("feature_hash")
                        if selection_valid:
                            preferred_model = selection.get("model")
                            threshold_override = selection.get("threshold")

                if features is None or features.empty:
                    features = feature_groups.get(DEFAULT_FEATURE_GROUP)
                    feature_group_used = DEFAULT_FEATURE_GROUP
                if features is None or features.empty:
                    for group_name in FEATURE_GROUPS:
                        candidate = feature_groups.get(group_name)
                        if candidate is not None and not candidate.empty:
                            features = candidate
                            feature_group_used = group_name
                            break
                if features is None or features.empty:
                    continue

                if preferred_model and preferred_model not in self._direction_model_catalog():
                    preferred_model = None
                if threshold_override is not None and not math.isfinite(float(threshold_override)):
                    threshold_override = None

                direction_result = self._fit_direction_model(
                    features,
                    forward_return,
                    vol_horizon,
                    preferred_model=preferred_model,
                    threshold_override=threshold_override,
                    min_samples=160,
                )
                return_result = self._fit_return_model(features, forward_return)
                if direction_result is None or return_result is None:
                    continue

                model_name, prob_up, threshold, metrics, _ = direction_result
                expected_return, rmse, top_drivers = return_result
                direction = "UP" if prob_up >= threshold else "DOWN"
                confidence = prob_up if direction == "UP" else 1 - prob_up
                metrics = dict(metrics)
                metrics["rmse"] = rmse
                metrics["model"] = model_name
                metrics["feature_group"] = feature_group_used
                metrics["selection_valid"] = selection_valid

                vol_value = None
                if vol_horizon is not None and not vol_horizon.dropna().empty:
                    vol_value = float(vol_horizon.dropna().iloc[-1])

                position_weight = self._size_position(expected_return, confidence, vol_value)
                position_dollars = position_weight * self.portfolio_notional
                expected_pnl = position_dollars * expected_return
                expected_pnl_pct = expected_pnl / self.portfolio_notional if self.portfolio_notional else 0.0

                horizons[horizon_name] = {
                    "expected_return": expected_return,
                    "direction": direction,
                    "confidence": confidence,
                    "position_weight": position_weight,
                    "position_dollars": position_dollars,
                    "expected_pnl": expected_pnl,
                    "expected_pnl_pct": expected_pnl_pct,
                    "vol_horizon": vol_value,
                    "metrics": metrics,
                    "top_drivers": top_drivers,
                }
                ranked[horizon_name].append({
                    "symbol": symbol,
                    "name": meta["name"],
                    "expected_return": expected_return,
                    "direction": direction,
                    "confidence": confidence,
                    "position_weight": position_weight,
                    "expected_pnl": expected_pnl,
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
            def rank_score(item):
                return item.get("expected_pnl", 0.0)

            long_items = [i for i in items if i.get("direction") == "UP"]
            short_items = [i for i in items if i.get("direction") == "DOWN"]
            long_sorted = sorted(long_items, key=rank_score, reverse=True)
            short_sorted = sorted(short_items, key=rank_score, reverse=True)

            ranked_summary[horizon] = {
                "long": long_sorted[:3],
                "short": short_sorted[:3],
                "bullish": long_sorted[:3],
                "bearish": short_sorted[:3],
            }

        data = {
            "as_of": now.strftime("%Y-%m-%d %H:%M"),
            "horizons": PREDICTION_HORIZONS_WEEKS,
            "portfolio": {
                "notional": self.portfolio_notional,
                "target_risk_pct": self.target_risk_pct,
                "max_position_pct": self.max_position_pct,
                "target_return": self.target_return,
            },
            "predictions": predictions,
            "ranked": ranked_summary,
        }
        self._cache = {"timestamp": now, "data": data}
        return data

    def get_backtest(self) -> Dict:
        now = datetime.now()
        cached_time = self._backtest_cache["timestamp"]
        if cached_time and (now - cached_time).total_seconds() < self.backtest_cache_seconds:
            return self._backtest_cache["data"]

        cutoff_date = None
        if self.backtest_window_years > 0:
            cutoff_date = (pd.Timestamp(now) - pd.DateOffset(years=self.backtest_window_years)).normalize()

        macro_features = self._build_macro_feature_frame()
        if macro_features.empty:
            data = {"error": "Insufficient feature data", "as_of": now.strftime("%Y-%m-%d")}
            self._backtest_cache = {"timestamp": now, "data": data}
            return data

        sp500_series = self.fred_engine.get_series("SP500")
        sp500_weekly = self._to_weekly(sp500_series, "D") if not sp500_series.empty else pd.Series(dtype=float)

        pnl_by_horizon: Dict[str, Dict[pd.Timestamp, float]] = {h: {} for h in PREDICTION_HORIZONS_WEEKS}
        stats_by_horizon = {
            h: {"trades": 0, "hits": 0, "trade_returns": []} for h in PREDICTION_HORIZONS_WEEKS
        }

        model_catalog = self._direction_model_catalog()

        for symbol in SECTOR_ETFS.keys():
            price = self._get_target_series(symbol)
            if price.empty:
                continue
            feature_groups = self._build_feature_groups(symbol, price, macro_features, sp500_weekly)
            weekly_returns = price.pct_change(1)
            vol_1w = weekly_returns.rolling(52).std()
            selection_map = self.model_selection.get(symbol, {})

            for horizon_name, horizon_weeks in PREDICTION_HORIZONS_WEEKS.items():
                forward_return = price.pct_change(horizon_weeks).shift(-horizon_weeks)
                vol_horizon = vol_1w * math.sqrt(horizon_weeks)

                selection = selection_map.get(horizon_name) if isinstance(selection_map, dict) else None
                feature_group = DEFAULT_FEATURE_GROUP
                model_name = None
                threshold = None

                if selection:
                    feature_group = selection.get("feature_group", feature_group)
                    model_name = selection.get("model")
                    threshold = selection.get("threshold")

                features = feature_groups.get(feature_group)
                if features is None or features.empty:
                    features = feature_groups.get(DEFAULT_FEATURE_GROUP)
                if features is None or features.empty:
                    for group_name in FEATURE_GROUPS:
                        candidate = feature_groups.get(group_name)
                        if candidate is not None and not candidate.empty:
                            features = candidate
                            break
                if features is None or features.empty:
                    continue

                if not model_name or model_name not in model_catalog:
                    model_name = next(iter(model_catalog.keys()))

                pnl_series, trade_count, hit_count, trade_returns = self._backtest_symbol(
                    features,
                    forward_return,
                    vol_horizon,
                    model_name=model_name,
                    threshold=threshold,
                    horizon_weeks=horizon_weeks,
                    cutoff_date=cutoff_date,
                )

                if pnl_series:
                    horizon_pnl = pnl_by_horizon[horizon_name]
                    for date, pnl in pnl_series.items():
                        horizon_pnl[date] = horizon_pnl.get(date, 0.0) + pnl

                stats = stats_by_horizon[horizon_name]
                stats["trades"] += trade_count
                stats["hits"] += hit_count
                stats["trade_returns"].extend(trade_returns)

        backtests = {}
        for horizon_name, horizon_weeks in PREDICTION_HORIZONS_WEEKS.items():
            pnl_series = pnl_by_horizon.get(horizon_name, {})
            if pnl_series:
                pnl_series = dict(sorted(pnl_series.items()))
                dates = list(pnl_series.keys())
                pnl_values = np.array(list(pnl_series.values()), dtype=float)
                equity = np.cumsum(pnl_values)
                equity_curve = [
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "pnl": float(pnl),
                        "equity": float(equity[idx]),
                    }
                    for idx, (date, pnl) in enumerate(zip(dates, pnl_values))
                ]
                total_pnl = float(equity[-1]) if equity.size else 0.0
                max_drawdown = self._max_drawdown(equity)
            else:
                equity_curve = []
                total_pnl = 0.0
                max_drawdown = 0.0

            stats = stats_by_horizon[horizon_name]
            trades = stats["trades"]
            hits = stats["hits"]
            trade_returns = stats["trade_returns"]
            hit_rate = float(hits / trades) if trades else 0.0
            avg_trade_return = float(np.mean(trade_returns)) if trade_returns else 0.0
            total_return = total_pnl / self.portfolio_notional if self.portfolio_notional else 0.0

            backtests[horizon_name] = {
                "horizon_weeks": horizon_weeks,
                "equity_curve": equity_curve,
                "stats": {
                    "total_pnl": total_pnl,
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                    "hit_rate": hit_rate,
                    "avg_trade_return": avg_trade_return,
                    "trades": trades,
                },
            }

        data = {
            "as_of": now.strftime("%Y-%m-%d %H:%M"),
            "window": {
                "years": self.backtest_window_years,
                "start": cutoff_date.strftime("%Y-%m-%d") if cutoff_date is not None else None,
                "end": now.strftime("%Y-%m-%d"),
            },
            "portfolio": {
                "notional": self.portfolio_notional,
                "target_risk_pct": self.target_risk_pct,
                "max_position_pct": self.max_position_pct,
                "target_return": self.target_return,
            },
            "backtests": backtests,
        }
        self._backtest_cache = {"timestamp": now, "data": data}
        return data

    def get_options_ideas(
        self,
        simulations: int = 2000,
        strike_steps: int = 2,
    ) -> Dict:
        now = datetime.now()
        cached_time = self._options_cache["timestamp"]
        if cached_time and (now - cached_time).total_seconds() < self.cache_seconds:
            return self._options_cache["data"]

        predictions = self.get_predictions()
        if "error" in predictions:
            data = {
                "error": predictions["error"],
                "as_of": now.strftime("%Y-%m-%d %H:%M"),
            }
            self._options_cache = {"timestamp": now, "data": data}
            return data

        ideas = {}
        rate = self._risk_free_rate()
        strike_offsets = [0.0, 0.05, 0.10][: max(strike_steps, 1) + 1]

        for symbol, pred in predictions.get("predictions", {}).items():
            price_series = self.market_engine.get_series(symbol)
            if price_series.empty:
                continue
            spot = float(price_series.dropna().iloc[-1])
            vol_annual = self._realized_vol_annual(price_series)
            if vol_annual is None:
                continue

            horizons = pred.get("horizons", {})
            symbol_ideas = {}
            for horizon_name, horizon_weeks in PREDICTION_HORIZONS_WEEKS.items():
                horizon = horizons.get(horizon_name)
                if not horizon:
                    continue
                expected_return = float(horizon.get("expected_return", 0.0))
                direction = horizon.get("direction", "UP")
                t_years = horizon_weeks / 52.0
                sigma_t = vol_annual * math.sqrt(t_years)

                if direction == "DOWN":
                    option_type = "put"
                    strike_multipliers = [1.0 - o for o in strike_offsets]
                else:
                    option_type = "call"
                    strike_multipliers = [1.0 + o for o in strike_offsets]

                terminal = self._simulate_terminal_prices(
                    spot,
                    expected_return,
                    vol_annual,
                    t_years,
                    simulations,
                    seed_key=f"{symbol}-{horizon_name}-{option_type}",
                )
                if terminal.size == 0:
                    continue

                best = None
                for mult in strike_multipliers:
                    strike = spot * mult
                    call_price, put_price = _bs_prices(spot, strike, t_years, rate, vol_annual)
                    premium = call_price if option_type == "call" else put_price
                    if premium <= 0:
                        continue

                    if option_type == "call":
                        payoffs = np.maximum(terminal - strike, 0.0)
                        prob_itm = float((terminal > strike).mean())
                    else:
                        payoffs = np.maximum(strike - terminal, 0.0)
                        prob_itm = float((terminal < strike).mean())

                    expected_payoff = float(payoffs.mean())
                    expected_edge = (expected_payoff - premium) / premium

                    candidate = {
                        "option_type": option_type,
                        "strike": float(strike),
                        "strike_pct": float(mult),
                        "premium_est": float(premium),
                        "expected_payoff": expected_payoff,
                        "expected_edge": float(expected_edge),
                        "prob_itm": prob_itm,
                        "expected_return": expected_return,
                        "vol_annual": float(vol_annual),
                        "vol_horizon": float(sigma_t),
                        "direction": direction,
                        "simulations": simulations,
                    }

                    if best is None or candidate["expected_edge"] > best["expected_edge"]:
                        best = candidate

                if best:
                    symbol_ideas[horizon_name] = best

            if symbol_ideas:
                ideas[symbol] = {
                    "symbol": symbol,
                    "name": pred.get("name"),
                    "spot": spot,
                    "ideas": symbol_ideas,
                }

        data = {
            "as_of": now.strftime("%Y-%m-%d %H:%M"),
            "assumptions": {
                "vol_source": "realized_60d",
                "pricing_model": "black_scholes_realized_vol",
                "options_only": "long_calls_or_puts",
                "risk_free_rate": rate,
                "simulations": simulations,
            },
            "ideas": ideas,
        }
        self._options_cache = {"timestamp": now, "data": data}
        return data
