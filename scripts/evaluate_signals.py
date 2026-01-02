"""
Lightweight evaluation harness for macro signals.
Runs on cached data; no network calls required.
"""
import argparse
import pandas as pd

from fred_engine import FredDataEngine


def _to_monthly_last(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return series.resample("ME").last()


def _forward_window_max(series: pd.Series, window: int) -> pd.Series:
    # Compute forward-looking max over the next `window` periods.
    rev = series.iloc[::-1]
    fwd = rev.rolling(window, min_periods=1).max().iloc[::-1]
    return fwd


def _summarize_signal(name: str, signal: pd.Series, recession_future: pd.Series) -> None:
    signal = signal.reindex(recession_future.index).fillna(False)
    recession_future = recession_future.fillna(False)

    alerts = int(signal.sum())
    hits = int((signal & recession_future).sum())
    false_pos = int((signal & ~recession_future).sum())
    recession_months = int(recession_future.sum())

    hit_rate = hits / alerts if alerts else 0
    coverage = hits / recession_months if recession_months else 0
    false_rate = false_pos / alerts if alerts else 0

    print(f"\n{name}:")
    print(f"  alerts: {alerts}")
    print(f"  hits: {hits}")
    print(f"  hit_rate: {hit_rate:.2%}")
    print(f"  coverage: {coverage:.2%}")
    print(f"  false_rate: {false_rate:.2%}")


def evaluate(window_months: int) -> None:
    engine = FredDataEngine()

    unrate = engine.get_series("UNRATE")
    curve = engine.get_series("T10Y2Y")
    usrec = engine.get_series("USREC")

    if unrate.empty or curve.empty or usrec.empty:
        print("Missing required series (UNRATE, T10Y2Y, USREC).")
        return

    unrate_m = _to_monthly_last(unrate)
    curve_m = _to_monthly_last(curve)
    usrec_m = _to_monthly_last(usrec)

    idx = unrate_m.index.intersection(curve_m.index).intersection(usrec_m.index)
    unrate_m = unrate_m.loc[idx]
    curve_m = curve_m.loc[idx]
    usrec_m = usrec_m.loc[idx]

    sahm = (unrate_m.rolling(3).mean() - unrate_m.rolling(12).min()) >= 0.50
    inversion = curve_m < 0
    combined = sahm | inversion

    recession_future = _forward_window_max(usrec_m, window_months).astype(bool)

    print(f"Evaluating {idx.min().date()} to {idx.max().date()} | window={window_months} months")
    _summarize_signal("Sahm Rule", sahm, recession_future)
    _summarize_signal("Yield Curve Inversion", inversion, recession_future)
    _summarize_signal("Combined", combined, recession_future)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate macro signal hit rates.")
    parser.add_argument("--window-months", type=int, default=12, help="Forward window for recession hits")
    args = parser.parse_args()
    evaluate(args.window_months)
