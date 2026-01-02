"""
Macro Indicator Configuration
50+ FRED series organized by category for comprehensive macro analysis
"""
import os

# FRED API Key - configure via environment variable for deployment safety
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
MARKET_DATA_PROVIDER = os.getenv("MARKET_DATA_PROVIDER", "stooq")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

SECTOR_ETFS = {
    "XLE": {"name": "Energy", "icon": "🛢️"},
    "XLF": {"name": "Financials", "icon": "🏦"},
    "XLK": {"name": "Technology", "icon": "💻"},
    "XLY": {"name": "Consumer Discretionary", "icon": "🛍️"},
    "XLP": {"name": "Consumer Staples", "icon": "🧃"},
    "XLI": {"name": "Industrials", "icon": "🏭"},
    "XLV": {"name": "Health Care", "icon": "🧬"},
    "XLU": {"name": "Utilities", "icon": "⚡"},
    "XLB": {"name": "Materials", "icon": "🧱"},
    "XLC": {"name": "Communication Services", "icon": "📡"},
    "XLRE": {"name": "Real Estate", "icon": "🏠"},
}

PREDICTION_HORIZONS_WEEKS = {
    "1w": 1,
    "1m": 4,
    "3m": 13,
    "6m": 26,
    "12m": 52,
}

PREDICTOR_SERIES = [
    "INDPRO",
    "PAYEMS",
    "RSXFS",
    "CFNAI",
    "CPIAUCSL",
    "PCEPILFE",
    "T5YIE",
    "T10YIE",
    "UNRATE",
    "ICSA",
    "FEDFUNDS",
    "DGS2",
    "DGS10",
    "DGS30",
    "T10Y2Y",
    "BAMLC0A0CM",
    "BAMLH0A0HYM2",
    "MORTGAGE30US",
    "WALCL",
    "RRPONTSYD",
    "ANFCI",
    "BUSLOANS",
    "DCOILWTICO",
    "DCOILBRENTEU",
    "PCOPPUSDM",
    "PALLFNFINDEXM",
    "SP500",
    "VIXCLS",
    "DTWEXBGS",
    "STLFSI4",
]

# Macro indicator categories with FRED series IDs
INDICATORS = {
    "growth": {
        "name": "Growth & Output",
        "description": "Economic growth and production metrics",
        "series": {
            "GDP": {"name": "Nominal GDP", "freq": "Q", "unit": "$B", "transform": "pct"},
            "GDPC1": {"name": "Real GDP", "freq": "Q", "unit": "$B", "transform": "pct"},
            "INDPRO": {"name": "Industrial Production", "freq": "M", "unit": "Index", "transform": "pct"},
            "PAYEMS": {"name": "Nonfarm Payrolls", "freq": "M", "unit": "K", "transform": "diff"},
            "UMCSENT": {"name": "Consumer Sentiment", "freq": "M", "unit": "Index", "transform": "pct"},
            "CFNAI": {"name": "Chicago Fed Activity", "freq": "M", "unit": "Index", "transform": "level"},
            "USREC": {"name": "Recession Prob", "freq": "M", "unit": "Binary", "transform": "level"},
            "RSXFS": {"name": "Retail Sales (ex Food/Svc)", "freq": "M", "unit": "$M", "transform": "pct"},
            "DGORDER": {"name": "Durable Goods Orders", "freq": "M", "unit": "$M", "transform": "pct"}
        }
    },
    "inflation": {
        "name": "Inflation & Prices",
        "description": "Price level measures (Consumer & Producer)",
        "series": {
            "CPIAUCSL": {"name": "CPI (Headline)", "freq": "M", "unit": "Index", "transform": "pct_yoy"},
            "CPILFESL": {"name": "Core CPI", "freq": "M", "unit": "Index", "transform": "pct_yoy"},
            "PCEPILFE": {"name": "Core PCE (Fed Target)", "freq": "M", "unit": "Index", "transform": "pct_yoy"},
            "PPIFIS": {"name": "PPI Final Demand", "freq": "M", "unit": "Index", "transform": "pct_yoy"},
            "MICH": {"name": "1Y Inflation Exp", "freq": "M", "unit": "%", "transform": "level"},
            "T5YIE": {"name": "5Y Breakeven", "freq": "D", "unit": "%", "transform": "level"},
            "T10YIE": {"name": "10Y Breakeven", "freq": "D", "unit": "%", "transform": "level"},
            "OILPRICE": {"name": "WTI Crude (Proxy)", "freq": "M", "unit": "$", "transform": "level"} 
        }
    },
    "labor": { 
        "name": "Labor Market",
        "description": "Employment, wages, and openings",
        "series": {
            "UNRATE": {"name": "Unemployment Rate", "freq": "M", "unit": "%", "transform": "level"},
            "JTSJOL": {"name": "Job Openings (JOLTS)", "freq": "M", "unit": "K", "transform": "level"},
            "ICSA": {"name": "Initial Claims", "freq": "W", "unit": "K", "transform": "level"},
            "AHETPI": {"name": "Avg Hourly Earnings", "freq": "M", "unit": "$", "transform": "pct_yoy"},
            "CIVPART": {"name": "Participation Rate", "freq": "M", "unit": "%", "transform": "level"},
            "U6RATE": {"name": "Underemployment (U6)", "freq": "M", "unit": "%", "transform": "level"}
        }
    },
    "rates": {
        "name": "Yields & Bonds",
        "description": "Treasury yield curve and spreads",
        "series": {
            "FEDFUNDS": {"name": "Fed Funds Rate", "freq": "M", "unit": "%", "transform": "level"},
            "DGS2": {"name": "2Y Treasury", "freq": "D", "unit": "%", "transform": "level"},
            "DGS10": {"name": "10Y Treasury", "freq": "D", "unit": "%", "transform": "level"},
            "DGS30": {"name": "30Y Treasury", "freq": "D", "unit": "%", "transform": "level"},
            "T10Y2Y": {"name": "10Y-2Y Curve", "freq": "D", "unit": "%", "transform": "level"},
            "BAMLC0A0CM": {"name": "US Corp OAS (IG)", "freq": "D", "unit": "%", "transform": "level"},
            "BAMLH0A0HYM2": {"name": "US High Yield OAS", "freq": "D", "unit": "%", "transform": "level"},
            "MORTGAGE30US": {"name": "30Y Mortgage", "freq": "W", "unit": "%", "transform": "level"}
        }
    },
    "money": {
        "name": "Liquidity & Credit",
        "description": "Money supply and financial conditions",
        "series": {
            "M2SL": {"name": "M2 Money Supply", "freq": "M", "unit": "$B", "transform": "pct_yoy"},
            "WM2NS": {"name": "M2 (Weekly)", "freq": "W", "unit": "$B", "transform": "pct_yoy"}, 
            "WALCL": {"name": "Fed Assets", "freq": "W", "unit": "$M", "transform": "pct_yoy"},
            "RRPONTSYD": {"name": "Reverse Repo", "freq": "D", "unit": "$B", "transform": "level"},
            "ANFCI": {"name": "Chi Fed Financial Cond", "freq": "W", "unit": "Index", "transform": "level"},
            "BUSLOANS": {"name": "C&I Loans", "freq": "W", "unit": "$B", "transform": "pct_yoy"}
        }
    },
    "housing": {
        "name": "Real Estate",
        "description": "Residential and commercial real estate",
        "series": {
             "HOUST": {"name": "Housing Starts", "freq": "M", "unit": "K", "transform": "level"},
             "CSUSHPINSA": {"name": "Case-Shiller Price", "freq": "M", "unit": "Index", "transform": "pct_yoy"},
             "HSN1F": {"name": "New Home Sales", "freq": "M", "unit": "K", "transform": "level"}
        }
    },
    "commodities": {
        "name": "Commodities",
        "description": "Direct commodity series available from FRED",
        "series": {
             "DCOILWTICO": {"name": "WTI Crude Oil", "freq": "D", "unit": "$", "transform": "level"},
             "DCOILBRENTEU": {"name": "Brent Crude Oil", "freq": "D", "unit": "$", "transform": "level"},
             "GASREGW": {"name": "Regular Gas Price", "freq": "W", "unit": "$", "transform": "level"},
             "PCOPPUSDM": {"name": "Global Copper Price", "freq": "M", "unit": "$", "transform": "level"},
             "PALLFNFINDEXM": {"name": "Global Commodities Index", "freq": "M", "unit": "Index", "transform": "level"}
        }
    },
    "markets": {
        "name": "Markets & Risk",
        "description": "High-frequency market and risk indicators",
        "series": {
             "SP500": {"name": "S&P 500", "freq": "D", "unit": "Index", "transform": "pct"},
             "VIXCLS": {"name": "VIX Volatility", "freq": "D", "unit": "Index", "transform": "level"},
             "DTWEXBGS": {"name": "USD Broad Index", "freq": "D", "unit": "Index", "transform": "pct"},
             "STLFSI4": {"name": "St. Louis FSI", "freq": "W", "unit": "Index", "transform": "level"}
        }
    }
}
