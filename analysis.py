
"""
Macro Analysis Engine
Derives insights, regimes, and signals from raw FRED data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from derivatives import compute_all_derivatives

class MacroAnalyzer:
    def __init__(self, engine):
        self.engine = engine
        self.sector_gen = SectorSignalGenerator(engine)

    def _metric_value(self, metrics: dict, key: str):
        val = metrics.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return val

    def get_full_analysis(self):
        """Main entry point for dashboard intelligence"""
        return {
            "regime": self._determine_regime(),
            "signals": self._generate_signals(),
            "anomalies": self._detect_anomalies(),
            "sectors": self.sector_gen.get_all_signals(),
            "summary": self._generate_summary_stats()
        }

    def _determine_regime(self):
        """
        Determine the current macro economic regime (Growth vs Inflation).
        Returns a quadrant: 'Goldilocks', 'Reflation', 'Stagflation', 'Deflation'.
        """
        # 1. Growth Score (Composite of INDPRO, PAYEMS, Retail Sales)
        growth_score = 0
        growth_signals = 0
        
        # INDPRO (6m momentum)
        indpro = self.engine.get_series_metrics("INDPRO")
        indpro_roc = self._metric_value(indpro, "roc_6m")
        if indpro_roc is not None:
            growth_score += 1 if indpro_roc > 0 else -1
            growth_signals += 1
            
        # Payrolls (3m trend)
        payems = self.engine.get_series_metrics("PAYEMS")
        payems_roc = self._metric_value(payems, "roc_3m")
        if payems_roc is not None:
            monthly_avg = payems_roc / 3
            growth_score += 1 if monthly_avg > 100 else -1 # >100k jobs per month
            growth_signals += 1

        # Retail Sales (6m trend)
        retail = self.engine.get_series_metrics("RSXFS")
        retail_roc = self._metric_value(retail, "roc_6m")
        if retail_roc is not None:
            growth_score += 1 if retail_roc > 0 else -1
            growth_signals += 1

        # 2. Inflation Score (CPI, PCE)
        inf_score = 0
        inf_signals = 0
        
        cpi = self.engine.get_series_metrics("CPIAUCSL")
        cpi_accel = self._metric_value(cpi, "roc_3m")
        if cpi_accel is not None:
            inf_score += 1 if cpi_accel > 0 else -1
            inf_signals += 1

        pce = self.engine.get_series_metrics("PCEPILFE")
        pce_accel = self._metric_value(pce, "roc_3m")
        if pce_accel is not None:
            inf_score += 1 if pce_accel > 0 else -1
            inf_signals += 1

        if growth_signals == 0 or inf_signals == 0:
            return {
                "name": "Unknown",
                "growth_trend": "Unknown",
                "inflation_trend": "Unknown",
                "description": "Insufficient data to determine regime."
            }

        # Determine Quadrant
        # Growth > 0, Inf > 0 -> Reflation
        # Growth > 0, Inf < 0 -> Goldilocks
        # Growth < 0, Inf > 0 -> Stagflation
        # Growth < 0, Inf < 0 -> Deflation
        
        g_final = growth_score > 0
        i_final = inf_score > 0
        
        regime = "Unknown"
        if g_final and i_final: regime = "Reflation"
        elif g_final and not i_final: regime = "Goldilocks"
        elif not g_final and i_final: regime = "Stagflation"
        elif not g_final and not i_final: regime = "Deflation"
        
        return {
            "name": regime,
            "growth_trend": "Rising" if g_final else "Falling",
            "inflation_trend": "Rising" if i_final else "Falling",
            "description": self._get_regime_desc(regime)
        }

    def _get_regime_desc(self, regime):
        map = {
            "Goldilocks": "Ideal state: Growth is positive while inflation cools. Risk assets usually thrive.",
            "Reflation": "Economy is heating up. Growth and inflation both rising. Commodities and Cyclicals often outperform.",
            "Stagflation": "The difficult zone: Growth slowing while inflation stays stubborn. Cash and defensive assets preferred.",
            "Deflation": "Recessionary forces. Growth and prices falling. Bonds and Gold often act as hedges."
        }
        return map.get(regime, "Insufficient data to determine regime.")

    def _generate_signals(self):
        """Check for specific macro triggers"""
        signals = []
        
        # 1. Sahm Rule (Recession Indicator)
        # Avg of last 3 months unemployment rate vs min of last 12 months
        unrate = self.engine.get_series("UNRATE")
        if not unrate.empty:
            current_3m_avg = unrate.rolling(3).mean().iloc[-1]
            min_12m = unrate.rolling(12).min().iloc[-1]
            if (current_3m_avg - min_12m) >= 0.50:
                signals.append({
                    "type": "danger", 
                    "name": "Sahm Rule Triggered", 
                    "desc": "Recession indicator active (>0.5% rise in unemployment)."
                })
        
        # 2. Yield Curve Inversion (10Y-2Y)
        curve = self.engine.get_series_metrics("T10Y2Y")
        curve_val = self._metric_value(curve, "value")
        if curve_val is not None and curve_val < 0:
            signals.append({
                "type": "warning",
                "name": "Yield Curve Inverted",
                "desc": f"10Y-2Y Spread is negative ({curve_val:.2f}%), signaling potential recession."
            })
                
        # 3. Inflation Target
        pce = self.engine.get_series_metrics("PCEPILFE") # Core PCE
        pce_val = self._metric_value(pce, "transformed")
        if pce_val is not None and pce_val > 3.0:
            signals.append({
                "type": "warning",
                "name": "High Inflation",
                "desc": f"Core PCE is {pce_val:.1f}%, significantly above Fed target (2%)."
            })
        
        return signals

    def _detect_anomalies(self):
        """Find series that are > 2 std devs from mean (Z-Score)"""
        anomalies = []
        # Check a subset of key indicators
        keys = ["UNRATE", "INDPRO", "CPIAUCSL", "M2SL", "HOUST", "DGS10"]
        
        for k in keys:
            s = self.engine.get_series(k)
            if s.empty:
                continue

            meta = self.engine.series_map.get(k, {})
            freq = meta.get("freq", "M")
            periods_map = self.engine._periods_map(freq)
            lookback = self.engine._lookback_window(freq)
            metrics = compute_all_derivatives(
                s,
                transform=meta.get("transform", "pct"),
                lookback=lookback,
                periods_map=periods_map,
            )
            z_series = metrics.get("z_score", pd.Series(dtype=float))
            z_val = z_series.dropna().iloc[-1] if not z_series.dropna().empty else None
            if z_val is None or abs(z_val) <= 2.0:
                continue

            transformed = metrics.get("transformed", pd.Series(dtype=float))
            window = transformed.dropna().iloc[-lookback:]
            if window.empty:
                continue

            curr = window.iloc[-1]
            mean = window.mean()

            anomalies.append({
                "id": k,
                "name": meta.get('name', k),
                "z_score": round(float(z_val), 2),
                "current": round(float(curr), 2),
                "mean_5y": round(float(mean), 2)
            })
                
        return sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)


    def _generate_summary_stats(self):
        # Quick aggregation for the header
        return {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

class SectorSignalGenerator:
    """
    Generates trading signals for Sector ETFs based on Macro Regimes and Driver Correlations.
    Since we don't have ETF price data in FRED, we use 'Macro Proxies' to determine likely performance.
    
    Model:
    1. Identify Key Drivers for each Sector (e.g. Energy <- Oil Prices)
    2. Calculate Z-Score and Trend of Drivers
    3. Output: Sentiment (Bullish/Bearish) and Conviction (0-100)
    """
    def __init__(self, engine):
        self.engine = engine

    def _metric_value(self, metrics: dict, key: str):
        val = metrics.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return val
        
    def get_all_signals(self):
        return [
            self._analyze_energy(),
            self._analyze_tech(),
            self._analyze_financials(),
            self._analyze_real_estate(),
            self._analyze_defensives()
        ]
        
    def _analyze_energy(self):
        score = 50
        reasons = []
        raw_data = {}
        
        oil = self.engine.get_series_metrics("DCOILWTICO")
        oil_price = self._metric_value(oil, "value")
        oil_change = self._metric_value(oil, "roc_1m")
        if oil_price is not None:
            raw_data["Oil Price"] = f"${oil_price:.2f}"
            if oil_change is not None and oil_price != 0:
                trend = oil_change / oil_price
                raw_data["Oil 1m Chg"] = f"{oil_change:+.2f}"
                raw_data["Oil 1m %"] = f"{trend*100:+.1f}%"
                if trend > 0.05:
                    score += 20
                    reasons.append("Strong Oil Momentum")
                elif trend < -0.05:
                    score -= 20
                    reasons.append("Weak Oil Momentum")

        inf = self.engine.get_series_metrics("T5YIE")
        inf_val = self._metric_value(inf, "value")
        inf_change = self._metric_value(inf, "roc_1m")
        if inf_val is not None:
            raw_data["5Y Breakeven"] = f"{inf_val:.2f}%"
            if inf_change is not None:
                raw_data["Breakeven Chg (1m)"] = f"{inf_change:+.2f}%"
                if inf_change > 0.1:
                    score += 10
                    reasons.append("Rising Inflation Exp")
                
        return self._build_signal("Energy (XLE)", score, reasons, "Oil Price & Inflation Exp", raw_data)

    def _analyze_tech(self):
        score = 50
        reasons = []
        raw_data = {}
        
        rates = self.engine.get_series_metrics("DGS10")
        curr = self._metric_value(rates, "value")
        change_1m = self._metric_value(rates, "roc_1m")
        if curr is not None:
            raw_data["10Y Yield"] = f"{curr:.2f}%"
            if change_1m is not None:
                raw_data["Yield Chg (1m)"] = f"{change_1m:+.2f}%"
                if change_1m > 0.2: 
                    score -= 25
                    reasons.append("Rates Spiking")
                elif change_1m < -0.1:
                    score += 15
                    reasons.append("Rates Cooling")
                
        liq = self.engine.get_series_metrics("WALCL")
        liq_yoy = self._metric_value(liq, "transformed")
        if liq_yoy is not None:
            raw_data["Fed Assets YoY"] = f"{liq_yoy:+.2f}%"
            if liq_yoy > 0: 
                score += 15
                reasons.append("Liquidity Expanding")
            
        return self._build_signal("Technology (XLK)", score, reasons, "Real Rates & Fed Liquidity", raw_data)

    def _analyze_financials(self):
        score = 50
        reasons = []
        raw_data = {}
        
        curve = self.engine.get_series_metrics("T10Y2Y")
        val = self._metric_value(curve, "value")
        if val is not None:
            raw_data["10Y-2Y Curve"] = f"{val:.2f}%"
            if val < 0:
                score -= 30
                reasons.append("Inverted Curve")
            elif val > 0.5:
                score += 20
                reasons.append("Steep Curve")
                
        spread = self.engine.get_series("BAMLH0A0HYM2")
        if not spread.empty:
            curr = spread.iloc[-1]
            avg_6m = spread.rolling(120).mean().iloc[-1]
            raw_data["HY Credit Spread"] = f"{curr:.2f}%"
            raw_data["Spread vs 6m Avg"] = f"{(curr - avg_6m):+.2f}%"
            
            if curr < avg_6m:
                score += 10
                reasons.append("Tight Spreads")
            elif curr > (avg_6m + 1.0):
                score -= 20
                reasons.append("Stress Rising")

        return self._build_signal("Financials (XLF)", score, reasons, "Yield Curve & Credit Spreads", raw_data)

    def _analyze_real_estate(self):
         score = 50
         reasons = []
         raw_data = {}
         
         mtg = self.engine.get_series_metrics("MORTGAGE30US")
         val = self._metric_value(mtg, "value")
         if val is not None:
             raw_data["30Y Mortgage"] = f"{val:.2f}%"
             
             if val > 7.0:
                 score -= 30
                 reasons.append("Restrictive Rates")
             elif val < 6.0:
                 score += 20
                 reasons.append("Accommodative Rates")
                 
         return self._build_signal("Real Estate (XLRE)", score, reasons, "Mortgage Rates", raw_data)

    def _analyze_defensives(self):
        score = 50
        reasons = []
        raw_data = {}
        
        ind = self.engine.get_series_metrics("INDPRO")
        trend = self._metric_value(ind, "roc_6m")
        if trend is not None:
            raw_data["Ind. Production (6m)"] = f"{trend:+.2f}%"
            
            if trend < 0:
                score += 25
                reasons.append("Activity Slowing")
            else:
                score -= 10
                reasons.append("Growth Accelerating")
                
        return self._build_signal("Defensives (XLU/XLP)", score, reasons, "Industrial Production", raw_data)

    def _build_signal(self, name, score, reasons, drivers, raw_data):
        score = max(0, min(100, score))
        
        sentiment = "Neutral"
        if score >= 70: sentiment = "Bullish"
        if score >= 85: sentiment = "Strong Bullish"
        if score <= 30: sentiment = "Bearish"
        if score <= 15: sentiment = "Strong Bearish"
        
        return {
            "sector": name,
            "sentiment": sentiment,
            "score": score,
            "drivers": drivers,
            "rationale": reasons,
            "raw_data": raw_data # Transparency Layer
        }
