
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import math

from config import INDICATORS, FRED_API_KEY
from api_usage import ApiUsageTracker
from fred_engine import FredDataEngine
from market_data import MarketDataEngine
from prediction_engine import PredictionEngine
from analysis import MacroAnalyzer
from config import SECTOR_ETFS

# Initialize FastAPI
app = FastAPI(title="Macro Intelligence Dashboard", description="AI-Powered Macro Analytics")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engines
usage_tracker = ApiUsageTracker()
engine = FredDataEngine(usage_tracker=usage_tracker)
market_engine = MarketDataEngine(usage_tracker=usage_tracker)
predictor = PredictionEngine(engine, market_engine)
analyzer = MacroAnalyzer(engine)

# Helpers
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup"""
    import threading
    threading.Thread(target=engine.refresh_stale).start()
    threading.Thread(target=market_engine.refresh_stale).start()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

@app.get("/")
async def read_root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "engine": "active",
        "data_points": len(engine.series_map),
        "market_symbols": len(SECTOR_ETFS),
        "data_mode": "cached" if not engine.can_fetch else "live",
        "market_provider": market_engine.provider,
    }

@app.get("/api/intelligence")
async def get_intelligence():
    """Get the calculated macro regime, signals, and anomalies"""
    try:
        data = analyzer.get_full_analysis()
        return convert_numpy_types(data)
    except Exception as e:
        print(f"Analysis Error: {e}")
        # Return empty safe defaults on error
        return {
            "regime": {"name": "Loading...", "description": "Insufficient data"},
            "signals": [],
            "anomalies": []
        }

@app.get("/api/dashboard")
async def get_dashboard_summary():
    """Get high-level summary of all categories"""
    summary = {}
    
    for cat_id, cat_meta in INDICATORS.items():
        data = engine.get_category_data(cat_id)
        summary[cat_id] = {
            "meta": cat_meta,
            "data": data
        }
    
    return convert_numpy_types(summary)

@app.get("/api/series/{series_id}")
async def get_series_data(series_id: str, range: str = "5y"):
    """Get chart data for a specific series"""
    series = engine.get_series(series_id)
    if series.empty:
        raise HTTPException(status_code=404, detail="Series not found or no data")
    
    # Filter range
    if range == "1y":
        start_date = datetime.now() - pd.DateOffset(years=1)
    elif range == "5y":
        start_date = datetime.now() - pd.DateOffset(years=5)
    elif range == "10y":
        start_date = datetime.now() - pd.DateOffset(years=10)
    elif range == "max":
        start_date = pd.Timestamp.min
    else:
        start_date = datetime.now() - pd.DateOffset(years=5)
        
    filtered = series[series.index >= start_date]
    
    payload = {
        "id": series_id,
        "name": engine.series_map.get(series_id, {}).get('name', series_id),
        "data": [{"date": d.strftime("%Y-%m-%d"), "value": v} for d, v in filtered.items()]
    }
    return convert_numpy_types(payload)

@app.get("/api/admin/force-update")
async def force_update():
    """Trigger a full data refresh from FRED"""
    import threading
    threading.Thread(target=engine.force_update_all).start()
    return {"message": "Force update started in background"}

@app.get("/api/admin/refresh-stale")
async def refresh_stale():
    """Refresh stale FRED data only"""
    import threading
    threading.Thread(target=engine.refresh_stale).start()
    return {"message": "Stale refresh started in background"}

@app.get("/api/admin/force-update-market")
async def force_update_market():
    """Trigger a full market data refresh"""
    import threading
    threading.Thread(target=lambda: market_engine.initialize_data(refresh_existing=True)).start()
    return {"message": "Market update started in background"}

@app.get("/api/admin/refresh-stale-market")
async def refresh_stale_market():
    """Refresh stale market data only"""
    import threading
    threading.Thread(target=market_engine.refresh_stale).start()
    return {"message": "Market stale refresh started in background"}

@app.get("/api/predictions")
async def get_predictions():
    try:
        data = predictor.get_predictions()
        return convert_numpy_types(data)
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"error": "Prediction engine failed"}

@app.get("/api/options")
async def get_options():
    try:
        data = predictor.get_options_ideas()
        return convert_numpy_types(data)
    except Exception as e:
        print(f"Options Error: {e}")
        return {"error": "Options engine failed"}

@app.get("/api/backtest")
async def get_backtest():
    try:
        data = predictor.get_backtest()
        return convert_numpy_types(data)
    except Exception as e:
        print(f"Backtest Error: {e}")
        return {"error": "Backtest engine failed"}

@app.get("/api/model-selection")
async def get_model_selection():
    try:
        data = predictor.get_model_selection()
        return convert_numpy_types(data)
    except Exception as e:
        print(f"Model selection error: {e}")
        return {"error": "Model selection load failed"}

@app.get("/api/admin/optimize-models")
async def optimize_models():
    import threading
    threading.Thread(target=predictor.optimize_models).start()
    return {"message": "Model optimization started in background"}

@app.get("/api/usage")
async def get_usage():
    return usage_tracker.summary()

# Serve static files from a dedicated directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Macro Intelligence System...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
