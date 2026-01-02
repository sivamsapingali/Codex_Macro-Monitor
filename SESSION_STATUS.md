# Session Status
Last updated: 2026-01-02 14:23

## Goal
Build a macro + market prediction dashboard with a real acceleration/deceleration prediction engine for sector ETF direction, using highest-frequency inputs. Priorities: data > logic > UI.

## Current State
- App serves from `server.py` and static assets live in `static/`.
- Server currently running on port `8006` (log: `server_8006.log`).
- FRED data uses env var `FRED_API_KEY` and is cached in `data/`.
- Market data engine (`market_data.py`) ingests sector ETF prices with Alpha Vantage (if available) and falls back to Stooq; cached in `data/market/`. Rate-limit tracking enforced.
- API usage tracking (`api_usage.py`) logs daily usage to `data/api_usage.json` and enforces daily limits; `/api/usage` exposes utilization.
- Prediction engine (`prediction_engine.py`) builds weekly features from macro series, fits ridge regression per horizon, and predicts ETF direction/expected return.
- Startup now uses cached data only; refresh happens only via the Sync button or admin endpoints.
- Sync button now triggers stale-only refresh (avoids repeated API calls within check intervals).
- UI refactored to the terminal-style layout; actionable items added; methodology now visualizes backtest accuracy vs baseline.

## What Was Changed
- `config.py`: env-based FRED key; added market provider + Alpha Vantage key fields, sector ETF list, prediction horizons, predictor series, and a new `markets` category.
- `server.py`: now initializes `MarketDataEngine`, `PredictionEngine`, `ApiUsageTracker`; added `/api/predictions`, `/api/usage`, `/api/admin/force-update-market` endpoints.
- `fred_engine.py`: added API usage tracking; frequency-aware staleness; metrics cache; `get_series_metrics` for standardized derivatives.
- `fred_engine.py`: updated update delay to respect FRED 120/min rate limits.
- `derivatives.py`: frequency-aware derivative calculations via `periods_map`.
- `analysis.py`: regime/signals/anomalies now use standardized metrics; sector heuristics updated to use metrics.
- `static/index.html`: added Actionable Items panel and methodology chart container.
- `static/styles.css`: added actionable card styles, overflow fixes for charts, methodology chart styling.
- `static/app.js`: added actionable items rendering, horizon accuracy chart, and backtest summary text.
- `api_usage.py`: new usage tracking helper.
- `prediction_engine.py`: new acceleration/deceleration model with multi-horizon outputs.
- `market_data.py`: new market data engine with rate-limit awareness and fallback behavior; Alpha Vantage uses `TIME_SERIES_DAILY` and auto-fallback to Stooq when blocked/limited.
- `fred_engine.py`/`market_data.py`: startup now skips refreshing existing cached data.
- `fred_engine.py`: added `last_checked` metadata + per-frequency check intervals to avoid repeated stale fetches; update calls now record check attempts.
- `market_data.py`: added metadata and check intervals; stale refresh respects cached timestamps.
- `server.py`: added `/api/admin/refresh-stale` and `/api/admin/refresh-stale-market` endpoints.
- `static/app.js`: Sync button now calls stale-only refresh endpoints.
- `scripts/evaluate_signals.py`: updated resample freq to `ME`.
- `static/app.js`/`static/styles.css`: prediction UI rows and utility colors.
- Tests added in `tests/` for derivatives, engine metrics, and regime logic.

## Known Issues / To-Do
- Alpha Vantage key provided appears to be premium-blocked for TIME_SERIES_DAILY_ADJUSTED; engine falls back to Stooq automatically.
- `/api/predictions` can be slow if market data missing or features sparse; consider caching longer or precomputing.
- Consider fine-tuning layout spacing after final data tuning.

## Next Steps
1. Start server with env vars:
   - `FRED_API_KEY=...`
   - `MARKET_DATA_PROVIDER=stooq` (or `alpha_vantage` if premium works)
   - `ALPHA_VANTAGE_API_KEY=...`
2. Refresh stale data only: `GET /api/admin/refresh-stale` and `/api/admin/refresh-stale-market`
3. Force update market data (only if needed): `GET /api/admin/force-update-market`
4. Verify predictions: `GET /api/predictions`
5. If predictions empty, confirm `data/market/*.csv` exists and `PREDICTOR_SERIES` in `config.py` are populated in `data/`.

## Commands Used
- `nohup python3 -m uvicorn server:app --host 127.0.0.1 --port 8006 > server_8006.log 2>&1 &`
- `python3 -m unittest discover -s tests`

## Open Questions
- Confirm which market data provider to use long-term (Stooq vs Alpha Vantage vs other).
- Decide on model enhancements (feature selection, walk-forward validation, confidence calibration).
