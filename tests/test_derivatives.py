import unittest
import pandas as pd

from derivatives import compute_all_derivatives


class TestDerivatives(unittest.TestCase):
    def test_pct_yoy_uses_periods_map(self):
        dates = pd.date_range("2020-01-31", periods=5, freq="ME")
        series = pd.Series([10, 20, 30, 40, 50], index=dates)
        periods_map = {"1m": 1, "3m": 3, "6m": 6, "12m": 4}

        metrics = compute_all_derivatives(series, transform="pct_yoy", lookback=3, periods_map=periods_map)
        transformed = metrics["transformed"].dropna()

        # With periods_per_year=4, last value compares 50 vs 10
        expected = (50 / 10 - 1) * 100
        self.assertAlmostEqual(transformed.iloc[-1], expected, places=6)

    def test_level_transform_keeps_raw_series(self):
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        series = pd.Series([1.0, 2.0, 3.0], index=dates)

        metrics = compute_all_derivatives(series, transform="level", lookback=3)
        transformed = metrics["transformed"]

        self.assertTrue(transformed.equals(series))


if __name__ == "__main__":
    unittest.main()
