#!/usr/bin/env python3
"""
Run model selection across feature groups + models per sector/horizon.
Uses cached DB/CSV data only.
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from prediction_engine import PredictionEngine
from fred_engine import FredDataEngine
from market_data import MarketDataEngine


def main() -> None:
    fred_engine = FredDataEngine()
    market_engine = MarketDataEngine()
    predictor = PredictionEngine(fred_engine, market_engine)
    results = predictor.optimize_models()
    if "error" in results:
        print(f"Model selection failed: {results['error']}")
        return
    selections = results.get("selections", {})
    total = sum(len(horizons) for horizons in selections.values())
    print(f"Model selection complete. Stored {total} selections.")


if __name__ == "__main__":
    main()
