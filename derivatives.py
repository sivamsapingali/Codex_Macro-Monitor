"""
Derivative Calculation Engine
Computes rate of change (1st derivative) and acceleration (2nd derivative) for macro indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


def calculate_rate_of_change(
    series: pd.Series,
    periods: int = 1,
    method: str = 'pct',
    periods_per_year: int = 12
) -> pd.Series:
    """
    Calculate rate of change (1st derivative)
    
    Args:
        series: Time series data
        periods: Number of periods for change calculation
        method: 'pct' for percentage change, 'diff' for absolute difference
    
    Returns:
        Series of rate of change values
    """
    if method == 'pct':
        return series.pct_change(periods=periods) * 100
    elif method == 'diff':
        return series.diff(periods=periods)
    elif method == 'pct_yoy':
        # Year-over-year percentage change (frequency-aware)
        return series.pct_change(periods=periods_per_year) * 100
    else:
        return series.pct_change(periods=periods) * 100


def calculate_acceleration(
    series: pd.Series,
    periods: int = 1,
    method: str = 'pct',
    periods_per_year: int = 12
) -> pd.Series:
    """
    Calculate acceleration (2nd derivative) - change in rate of change
    
    Args:
        series: Time series data
        periods: Number of periods for acceleration calculation
    
    Returns:
        Series of acceleration values
    """
    roc = calculate_rate_of_change(series, periods, method, periods_per_year)
    acceleration = roc.diff(periods=periods)
    return acceleration


def calculate_z_score(series: pd.Series, lookback: int = 60) -> pd.Series:
    """
    Calculate rolling Z-score to determine how extreme current values are
    
    Args:
        series: Time series data
        lookback: Rolling window size (e.g., 60 months = 5 years)
    
    Returns:
        Series of Z-score values
    """
    rolling_mean = series.rolling(window=lookback, min_periods=lookback//2).mean()
    rolling_std = series.rolling(window=lookback, min_periods=lookback//2).std()
    z_score = (series - rolling_mean) / rolling_std
    return z_score


def calculate_percentile(series: pd.Series, lookback: int = 60) -> pd.Series:
    """
    Calculate historical percentile ranking
    
    Args:
        series: Time series data
        lookback: Rolling window size
    
    Returns:
        Series of percentile values (0-100)
    """
    def rolling_percentile(window):
        if len(window) < 2:
            return np.nan
        return (window.rank().iloc[-1] - 1) / (len(window) - 1) * 100
    
    return series.rolling(window=lookback, min_periods=lookback//2).apply(rolling_percentile)


def detect_inflection(roc_series: pd.Series, threshold: float = 0.5) -> pd.Series:
    """
    Detect inflection points where rate of change changes direction
    
    Args:
        roc_series: Rate of change series
        threshold: Minimum change to count as inflection
    
    Returns:
        Series with inflection signals (-1: negative inflection, 0: none, 1: positive inflection)
    """
    acceleration = roc_series.diff()
    inflection = pd.Series(0, index=roc_series.index)
    
    # Positive inflection: rate of change was negative but acceleration is positive
    pos_inflection = (roc_series.shift(1) < 0) & (acceleration > threshold)
    inflection[pos_inflection] = 1
    
    # Negative inflection: rate of change was positive but acceleration is negative
    neg_inflection = (roc_series.shift(1) > 0) & (acceleration < -threshold)
    inflection[neg_inflection] = -1
    
    return inflection


def compute_all_derivatives(
    series: pd.Series,
    transform: str = 'pct',
    lookback: int = 60,
    periods_map: Optional[Dict[str, int]] = None
) -> Dict[str, pd.Series]:
    """
    Compute all derivative metrics for a time series
    
    Args:
        series: Raw time series data
        transform: Transformation method ('pct', 'diff', 'pct_yoy', 'level')
        lookback: Window for Z-score and percentile calculations
    
    Returns:
        Dictionary containing all computed metrics
    """
    if periods_map is None:
        periods_map = {"1m": 1, "3m": 3, "6m": 6, "12m": 12}
    p1 = periods_map["1m"]
    p3 = periods_map["3m"]
    p6 = periods_map["6m"]
    p12 = periods_map["12m"]

    results = {
        'value': series,
        'transformed': pd.Series(dtype=float),
        'roc_1m': pd.Series(dtype=float),
        'roc_3m': pd.Series(dtype=float),
        'roc_6m': pd.Series(dtype=float),
        'roc_12m': pd.Series(dtype=float),
        'acceleration': pd.Series(dtype=float),
        'z_score': pd.Series(dtype=float),
        'percentile': pd.Series(dtype=float),
        'inflection': pd.Series(dtype=float),
        'signal': pd.Series(dtype=str),
    }
    
    if series.empty:
        return results
    
    periods_per_year = p12
    if transform == 'pct_yoy':
        base = series.pct_change(periods=periods_per_year) * 100
        results['transformed'] = base
        results['roc_1m'] = base.diff(p1)
        results['roc_3m'] = base.diff(p3)
        results['roc_6m'] = base.diff(p6)
        results['roc_12m'] = base.diff(p12)
    else:
        if transform == 'level':
            results['transformed'] = series
            results['roc_1m'] = series.diff(p1)
            results['roc_3m'] = series.diff(p3)
            results['roc_6m'] = series.diff(p6)
            results['roc_12m'] = series.diff(p12)
        elif transform == 'diff':
            results['transformed'] = series.diff(p1)
            results['roc_1m'] = series.diff(p1)
            results['roc_3m'] = series.diff(p3)
            results['roc_6m'] = series.diff(p6)
            results['roc_12m'] = series.diff(p12)
        else:
            results['transformed'] = series.pct_change(p1) * 100
            results['roc_1m'] = calculate_rate_of_change(series, p1, transform, periods_per_year)
            results['roc_3m'] = calculate_rate_of_change(series, p3, transform, periods_per_year)
            results['roc_6m'] = calculate_rate_of_change(series, p6, transform, periods_per_year)
            results['roc_12m'] = calculate_rate_of_change(series, p12, transform, periods_per_year)

    # Calculate acceleration (2nd derivative)
    results['acceleration'] = results['roc_1m'].diff(p1)

    # Z-score and percentile on transformed series
    results['z_score'] = calculate_z_score(results['transformed'], lookback)
    results['percentile'] = calculate_percentile(results['transformed'], lookback)
    
    # Inflection detection
    results['inflection'] = detect_inflection(results['roc_1m'])
    
    # Generate signal based on rate of change and acceleration
    results['signal'] = generate_signal(results['roc_1m'], results['acceleration'])
    
    return results


def generate_signal(roc: pd.Series, acceleration: pd.Series) -> pd.Series:
    """
    Generate trading signals based on rate of change and acceleration
    
    Signal logic:
    - 🟢 ACCELERATING_UP: Positive RoC and positive acceleration
    - 🟡 DECELERATING_UP: Positive RoC but negative acceleration
    - 🟠 DECELERATING_DOWN: Negative RoC but positive acceleration (improving)
    - 🔴 ACCELERATING_DOWN: Negative RoC and negative acceleration
    - ⚪ STABLE: Near-zero RoC and acceleration
    """
    signal = pd.Series('STABLE', index=roc.index)
    
    # Define thresholds
    roc_threshold = 0.1
    accel_threshold = 0.05
    
    # Classify each point
    accel_up = (roc > roc_threshold) & (acceleration > accel_threshold)
    decel_up = (roc > roc_threshold) & (acceleration < -accel_threshold)
    decel_down = (roc < -roc_threshold) & (acceleration > accel_threshold)
    accel_down = (roc < -roc_threshold) & (acceleration < -accel_threshold)
    
    signal[accel_up] = 'ACCELERATING_UP'
    signal[decel_up] = 'DECELERATING_UP'
    signal[decel_down] = 'DECELERATING_DOWN'
    signal[accel_down] = 'ACCELERATING_DOWN'
    
    return signal


def get_signal_emoji(signal: str) -> str:
    """Convert signal string to emoji"""
    emoji_map = {
        'ACCELERATING_UP': '🟢',
        'DECELERATING_UP': '🟡',
        'DECELERATING_DOWN': '🟠',
        'ACCELERATING_DOWN': '🔴',
        'STABLE': '⚪'
    }
    return emoji_map.get(signal, '⚪')


def get_signal_description(signal: str) -> str:
    """Get human-readable description of signal"""
    desc_map = {
        'ACCELERATING_UP': 'Rising and accelerating',
        'DECELERATING_UP': 'Rising but slowing',
        'DECELERATING_DOWN': 'Falling but improving',
        'ACCELERATING_DOWN': 'Falling and worsening',
        'STABLE': 'Stable/unchanged'
    }
    return desc_map.get(signal, 'Unknown')


def summarize_derivatives(derivatives: Dict[str, pd.Series]) -> Dict:
    """
    Create a summary of the latest derivative values
    
    Args:
        derivatives: Output from compute_all_derivatives
    
    Returns:
        Dictionary with latest values for dashboard display
    """
    summary = {}
    
    for key, series in derivatives.items():
        if isinstance(series, pd.Series) and not series.empty:
            latest = series.dropna().iloc[-1] if len(series.dropna()) > 0 else None
            # Convert numpy types to native Python types for JSON serialization
            if latest is not None:
                if hasattr(latest, 'item'):
                    latest = latest.item()  # Convert numpy scalar to Python scalar
            summary[key] = latest
        else:
            summary[key] = None
    
    # Add signal emoji
    if summary.get('signal'):
        summary['signal_emoji'] = get_signal_emoji(summary['signal'])
        summary['signal_description'] = get_signal_description(summary['signal'])
    
    return summary
