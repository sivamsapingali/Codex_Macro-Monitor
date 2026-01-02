import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fred_engine import FredDataEngine


class TestFredEngine(unittest.TestCase):
    def _write_series(self, root: Path, series_id: str, values, dates):
        df = pd.DataFrame({"value": values}, index=dates)
        df.index.name = "date"
        df.to_csv(root / f"{series_id}.csv")

    def test_get_series_metrics_basic(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.date_range("2020-01-31", periods=13, freq="ME")
            values = list(range(100, 113))
            self._write_series(root, "INDPRO", values, dates)

            engine = FredDataEngine(data_dir=str(root), db_path=str(root / "timeseries.db"))
            metrics = engine.get_series_metrics("INDPRO")

            self.assertEqual(metrics.get("last_date"), dates[-1].strftime("%Y-%m-%d"))
            self.assertIsNotNone(metrics.get("roc_12m"))

    def test_category_data_includes_transform_fields(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.date_range("2021-01-31", periods=13, freq="ME")
            values = list(range(200, 213))
            self._write_series(root, "CPIAUCSL", values, dates)

            engine = FredDataEngine(data_dir=str(root), db_path=str(root / "timeseries.db"))
            data = engine.get_category_data("inflation")

            self.assertIn("CPIAUCSL", data)
            self.assertIn("transform", data["CPIAUCSL"])
            self.assertIn("value_transformed", data["CPIAUCSL"])


if __name__ == "__main__":
    unittest.main()
