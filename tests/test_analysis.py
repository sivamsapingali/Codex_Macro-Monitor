import unittest
import pandas as pd

from analysis import MacroAnalyzer


class StubEngine:
    def __init__(self, metrics_map):
        self.metrics_map = metrics_map
        self.series_map = {}

    def get_series_metrics(self, series_id):
        return self.metrics_map.get(series_id, {})

    def get_series(self, series_id):
        return pd.Series(dtype=float)


class TestMacroAnalyzer(unittest.TestCase):
    def test_determine_regime_reflation(self):
        metrics = {
            "INDPRO": {"roc_6m": 1.2},
            "PAYEMS": {"roc_3m": 450},
            "RSXFS": {"roc_6m": 0.5},
            "CPIAUCSL": {"roc_3m": 0.2},
            "PCEPILFE": {"roc_3m": 0.1},
        }
        engine = StubEngine(metrics)
        analyzer = MacroAnalyzer(engine)

        regime = analyzer._determine_regime()
        self.assertEqual(regime["name"], "Reflation")

    def test_determine_regime_unknown_without_inflation(self):
        metrics = {
            "INDPRO": {"roc_6m": 1.2},
            "PAYEMS": {"roc_3m": 450},
            "RSXFS": {"roc_6m": 0.5},
        }
        engine = StubEngine(metrics)
        analyzer = MacroAnalyzer(engine)

        regime = analyzer._determine_regime()
        self.assertEqual(regime["name"], "Unknown")


if __name__ == "__main__":
    unittest.main()
