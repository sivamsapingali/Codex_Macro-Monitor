# Session Status
Last updated: 2026-01-03 02:25

## Goal
Build a macro + market prediction dashboard that produces directional sector ETF trades with sizing and expected P&L, using highest-frequency inputs. Priorities: data > logic > UI.

## Current State
- App serves from `server.py` and static assets live in `static/`.
- Project paused; uvicorn server stopped (last used port `8012`).
- FRED data uses env var `FRED_API_KEY` and is cached in `data/`.
- Market data engine (`market_data.py`) ingests sector ETF prices with Alpha Vantage (if available) and falls back to Stooq; cached in `data/market/`. Rate-limit tracking enforced.
- SQLite time series store at `data/timeseries.db` now serves as primary cache for FRED + market data (CSV remains as fallback).
- API usage tracking (`api_usage.py`) logs daily usage to `data/api_usage.json` and enforces daily limits; `/api/usage` exposes utilization.
- Prediction engine now uses macro + sector technical features (momentum/volatility/RSI/trend/beta/drawdown), sector-specific macro inputs, a classifier for direction, and ridge regression for expected return.
- Direction model uses a deadband filter + weighting to focus on larger moves (configurable via `DIRECTION_DEADBAND`), plus threshold tuning.
- Direction model mode is configurable via `DIRECTION_MODEL_MODE` (`auto` = logistic + gradient boosting, `fast` = logistic only).
- Model selection cache stored in `data/model_selection.json` with per-sector/horizon feature group + model + threshold choices (cached and validated by feature hash).
- Horizons scoped to 1w + 1m for predictions/backtests.
- Backtests are limited to the last 3 years (configurable via `BACKTEST_WINDOW_YEARS`).
- Backtest API (`/api/backtest`) computes walk-forward P&L curves; Methodology view renders the equity curves.
- Coverage exports written to `data/series_coverage.csv` and `data/market_coverage.csv`.
- Trade ideas now render as a visual grid with expected P&L + sizing bars in the Sector view.
- Predictions include recommended position sizing and expected P&L using configurable portfolio/risk settings.
- Options ideas endpoint (`/api/options`) produces approximate long-call/long-put ideas using realized volatility and Black-Scholes pricing (no option chain data).
- Startup now refreshes stale data in the background (if API keys exist); Sync still triggers stale-only refresh.
- Sync button now triggers stale-only refresh (avoids repeated API calls within check intervals).
- UI refactored to the terminal-style layout; actionable items added; methodology now visualizes backtest accuracy vs baseline.

## What Was Changed
- `config.py`: env-based FRED key; added market provider + Alpha Vantage key fields, sector ETF list, prediction horizons, predictor series, and a new `markets` category.
- `server.py`: now initializes `MarketDataEngine`, `PredictionEngine`, `ApiUsageTracker`; added `/api/predictions`, `/api/usage`, `/api/admin/force-update-market` endpoints.
- `server.py`: added `/api/options` endpoint for options ideas.
- `server.py`: added `/api/backtest`, `/api/model-selection`, `/api/admin/optimize-models` endpoints.
- `server.py`: startup now calls stale refresh; `/api/status` reports data mode + market provider.
- `fred_engine.py`: added API usage tracking; frequency-aware staleness; metrics cache; `get_series_metrics` for standardized derivatives.
- `fred_engine.py`: updated update delay to respect FRED 120/min rate limits.
- `derivatives.py`: frequency-aware derivative calculations via `periods_map`.
- `analysis.py`: regime/signals/anomalies now use standardized metrics; sector heuristics updated to use metrics.
- `static/index.html`: added Actionable Items panel and methodology chart container.
- `static/styles.css`: added actionable card styles, overflow fixes for charts, methodology chart styling.
- `static/app.js`: added actionable items rendering, horizon accuracy chart, and backtest summary text.
- `api_usage.py`: new usage tracking helper.
- `prediction_engine.py`: sector-specific macro feature filtering + derived series (real yields, credit spread, net liquidity).
- `prediction_engine.py`: direction classifier uses deadband weighting + tuned threshold; includes MLP candidate; added position sizing + expected P&L.
- `prediction_engine.py`: model-selection optimizer with feature-group/model search, hash validation, and cached selections.
- `prediction_engine.py`: walk-forward backtest with equity curves + stats for each horizon.
- `prediction_engine.py`: expanded sector technical feature set (RSI, drawdown, moving-average ratios, beta/correlation).
- `prediction_engine.py`: ranking now uses expected P&L and provides long/short lists.
- `config.py`: added sizing + risk env flags (`PORTFOLIO_NOTIONAL`, `TARGET_RISK_PCT`, `MAX_POSITION_PCT`, `MIN_VOL`, `TARGET_RETURN`) and sector macro maps.
- `config.py`: added model selection + backtest tuning env flags (`MODEL_SELECTION_*`, `BACKTEST_*`) and 3-year backtest window.
- `config.py`: horizons trimmed to 1w + 1m for focused modeling.
- `static/index.html`: methodology updated to reflect sizing + expected P&L ranking.
- `static/index.html`: added backtest panel + chart container.
- `static/index.html`: Actionable Items tag updated to 1w–1m focus.
- `static/app.js`: trade ideas now show expected P&L + position size; methodology shows balanced accuracy + AUC.
- `static/app.js`: backtest loading + chart rendering; backtest summary in methodology.
- `static/app.js`: horizons now follow backend order; status shows data mode + provider.
- `static/styles.css`: backtest panel sizing.
- `market_data.py`: new market data engine with rate-limit awareness and fallback behavior; Alpha Vantage uses `TIME_SERIES_DAILY` and auto-fallback to Stooq when blocked/limited.
- `fred_engine.py`/`market_data.py`: stale refresh now used on startup via `server.py`.
- `fred_engine.py`: added `last_checked` metadata + per-frequency check intervals to avoid repeated stale fetches; update calls now record check attempts.
- `market_data.py`: added metadata and check intervals; stale refresh respects cached timestamps.
- `server.py`: added `/api/admin/refresh-stale` and `/api/admin/refresh-stale-market` endpoints.
- `static/app.js`: Sync button now calls stale-only refresh endpoints.
- `scripts/evaluate_signals.py`: updated resample freq to `ME`.
- `static/app.js`/`static/styles.css`: prediction UI rows and utility colors.
- `data_store.py`: SQLite-backed time series store for FRED + market data.
- `scripts/migrate_to_db.py`: migration script to ingest existing CSV caches into SQLite.
- `scripts/optimize_models.py`: run model selection and cache to `data/model_selection.json`.
- `scripts/export_coverage.py`: export start/end coverage for all cached series.
- `fred_engine.py`/`market_data.py`: DB-first reads with CSV fallback; incremental updates append only new observations.
- `prediction_engine.py`: options ideas generation using realized vol + Black-Scholes + Monte Carlo.
- Tests added in `tests/` for derivatives, engine metrics, and regime logic.

## Known Issues / To-Do
- Alpha Vantage key provided appears to be premium-blocked for TIME_SERIES_DAILY_ADJUSTED; engine falls back to Stooq automatically.
- `/api/predictions` can be slow if market data missing or features sparse; consider caching longer or precomputing.
- Direction model in `auto` mode may be slower on first run; use `DIRECTION_MODEL_MODE=fast` if needed.
- Deadband filtering reduces training samples; adjust `DIRECTION_DEADBAND` if signals feel sparse.
- Options ideas are approximate (realized vol proxy; no option chain or implied vol data).
- Model selection runs can be long; current cache covers 1w/1m for all sectors. Increase `MODEL_SELECTION_MAX_SPLITS` if you want deeper validation.
- Data will remain stale if `FRED_API_KEY` is not set; the status bar will show `cached` until refreshed with a key.
- Attempted server start on `127.0.0.1:8006` failed with `Errno 48` (address already in use).
- Consider fine-tuning layout spacing after final data tuning.

## Next Steps
1. Restart server with env vars:
   - `FRED_API_KEY=...`
   - `MARKET_DATA_PROVIDER=stooq` (or `alpha_vantage` if premium works)
   - `ALPHA_VANTAGE_API_KEY=...`
2. Refresh stale data only: `GET /api/admin/refresh-stale` and `/api/admin/refresh-stale-market`
3. Force update market data (only if needed): `GET /api/admin/force-update-market`
4. Run model selection (optional): `python3 scripts/optimize_models.py` or `GET /api/admin/optimize-models`
5. Verify predictions: `GET /api/predictions`
6. Pull backtest curves: `GET /api/backtest`
7. Check options ideas: `GET /api/options`
8. If predictions empty, confirm `data/market/*.csv` exists and `PREDICTOR_SERIES` in `config.py` are populated in `data/`.

## Commands Used
- `nohup python3 -m uvicorn server:app --host 127.0.0.1 --port 8006 > server_8006.log 2>&1 &`
- `nohup env DIRECTION_MODEL_MODE=auto python3 -m uvicorn server:app --host 127.0.0.1 --port 8006 > server_8006.log 2>&1 &`
- `nohup env DIRECTION_MODEL_MODE=auto python3 -m uvicorn server:app --host 127.0.0.1 --port 8011 > server_8011.log 2>&1 &`
- `nohup env DIRECTION_MODEL_MODE=auto python3 -m uvicorn server:app --host 0.0.0.0 --port 8011 > server_8011.log 2>&1 &`
- `nohup env DIRECTION_MODEL_MODE=auto python3 -m uvicorn server:app --host 127.0.0.1 --port 8012 > server_8012.log 2>&1 &`
- `pkill -f "uvicorn server:app --host 127.0.0.1 --port 8012"`
- `nohup env DIRECTION_MODEL_MODE=auto python3 -m uvicorn server:app --host 127.0.0.1 --port 8012 > server_8012.log 2>&1 &`
- `python3 -m unittest discover -s tests`
- `MODEL_SELECTION_HORIZONS=1m MODEL_SELECTION_GROUPS=macro_core,macro_sector,sector_tech,macro_sector_plus_tech MODEL_SELECTION_MAX_SPLITS=2 MODEL_SELECTION_TEST_WEEKS=13 MODEL_SELECTION_MODELS=logistic,gboost,hist_gb python3 scripts/optimize_models.py`
- `MODEL_SELECTION_HORIZONS=1w MODEL_SELECTION_GROUPS=macro_core,macro_sector,sector_tech,macro_sector_plus_tech MODEL_SELECTION_MAX_SPLITS=2 MODEL_SELECTION_TEST_WEEKS=13 MODEL_SELECTION_MODELS=logistic,gboost,hist_gb python3 scripts/optimize_models.py`
- `MODEL_SELECTION_HORIZONS=3m MODEL_SELECTION_GROUPS=macro_core,macro_sector,sector_tech,macro_sector_plus_tech MODEL_SELECTION_MAX_SPLITS=2 MODEL_SELECTION_TEST_WEEKS=13 MODEL_SELECTION_MODELS=logistic,gboost,hist_gb python3 scripts/optimize_models.py`
- `MODEL_SELECTION_HORIZONS=6m MODEL_SELECTION_GROUPS=macro_core,macro_sector,sector_tech,macro_sector_plus_tech MODEL_SELECTION_MAX_SPLITS=2 MODEL_SELECTION_TEST_WEEKS=13 MODEL_SELECTION_MODELS=logistic,gboost,hist_gb python3 scripts/optimize_models.py`
- `MODEL_SELECTION_HORIZONS=12m MODEL_SELECTION_GROUPS=macro_core,macro_sector,sector_tech,macro_sector_plus_tech MODEL_SELECTION_MAX_SPLITS=2 MODEL_SELECTION_TEST_WEEKS=13 MODEL_SELECTION_MODELS=logistic,gboost,hist_gb python3 scripts/optimize_models.py`
- `python3 scripts/export_coverage.py`
- `pkill -f "uvicorn server:app"`
- `nohup env DIRECTION_MODEL_MODE=auto python3 -m uvicorn server:app --host 127.0.0.1 --port 8012 > server_8012.log 2>&1 &`
- `MODEL_SELECTION_TEST_WEEKS=13 MODEL_SELECTION_MAX_SPLITS=3 python3 scripts/optimize_models.py`

## Open Questions
- Confirm which market data provider to use long-term (Stooq vs Alpha Vantage vs other).
- Decide on model enhancements (feature selection, walk-forward validation, confidence calibration).
- Pending request: switch model selection objective to optimize hit-rate or expected P&L (especially 1w) and potentially use FreqAI/HF.
