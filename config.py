"""
Macro Indicator Configuration
50+ FRED series organized by category for comprehensive macro analysis
"""
import os

# FRED API Key - configure via environment variable for deployment safety
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
MARKET_DATA_PROVIDER = os.getenv("MARKET_DATA_PROVIDER", "stooq")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
DIRECTION_MODEL_MODE = os.getenv("DIRECTION_MODEL_MODE", "auto")
DIRECTION_DEADBAND = float(os.getenv("DIRECTION_DEADBAND", "0.25"))
PORTFOLIO_NOTIONAL = float(os.getenv("PORTFOLIO_NOTIONAL", "100000"))
TARGET_RISK_PCT = float(os.getenv("TARGET_RISK_PCT", "0.015"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.2"))
MIN_VOL = float(os.getenv("MIN_VOL", "0.01"))
TARGET_RETURN = float(os.getenv("TARGET_RETURN", "0.03"))
MODEL_SELECTION_PATH = os.getenv("MODEL_SELECTION_PATH", "data/model_selection.json")
MODEL_SELECTION_MIN_SAMPLES = int(os.getenv("MODEL_SELECTION_MIN_SAMPLES", "180"))
MODEL_SELECTION_MAX_SPLITS = int(os.getenv("MODEL_SELECTION_MAX_SPLITS", "4"))
MODEL_SELECTION_TEST_WEEKS = int(os.getenv("MODEL_SELECTION_TEST_WEEKS", "26"))
MODEL_SELECTION_GROUPS = os.getenv("MODEL_SELECTION_GROUPS", "")
MODEL_SELECTION_MODELS = os.getenv("MODEL_SELECTION_MODELS", "")
MODEL_SELECTION_HORIZONS = os.getenv("MODEL_SELECTION_HORIZONS", "")
BACKTEST_CACHE_SECONDS = int(os.getenv("BACKTEST_CACHE_SECONDS", "3600"))
BACKTEST_MIN_TRAIN_WEEKS = int(os.getenv("BACKTEST_MIN_TRAIN_WEEKS", "156"))
BACKTEST_TEST_WEEKS = int(os.getenv("BACKTEST_TEST_WEEKS", "26"))
BACKTEST_MAX_SPLITS = int(os.getenv("BACKTEST_MAX_SPLITS", "6"))
BACKTEST_WINDOW_YEARS = int(os.getenv("BACKTEST_WINDOW_YEARS", "3"))

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
}

PREDICTOR_SERIES = [
    "GDP",
    "GDPC1",
    "INDPRO",
    "PAYEMS",
    "RSXFS",
    "CFNAI",
    "CPIAUCSL",
    "CPILFESL",
    "PCEPILFE",
    "UMCSENT",
    "M2SL",
    "WM2NS",
    "HOUST",
    "CSUSHPINSA",
    "T5YIE",
    "T10YIE",
    "UNRATE",
    "U6RATE",
    "ICSA",
    "JTSJOL",
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
    "DGORDER",
    "DCOILWTICO",
    "DCOILBRENTEU",
    "GASREGW",
    "PCOPPUSDM",
    "PALLFNFINDEXM",
    "SP500",
    "VIXCLS",
    "DTWEXBGS",
    "STLFSI4",
]

CORE_MACRO_SERIES = [
    "CFNAI",
    "SP500",
    "VIXCLS",
    "DTWEXBGS",
    "STLFSI4",
]

SECTOR_MACRO_MAP = {
    "XLE": ["DCOILWTICO", "DCOILBRENTEU", "T5YIE", "T10YIE", "CREDIT_SPREAD", "NET_LIQUIDITY"],
    "XLF": ["DGS2", "DGS10", "T10Y2Y", "FEDFUNDS", "CREDIT_SPREAD", "NET_LIQUIDITY"],
    "XLK": ["REAL10Y", "REAL2Y", "NET_LIQUIDITY", "VIXCLS"],
    "XLY": ["PAYEMS", "RSXFS", "UMCSENT", "CPIAUCSL", "UNRATE"],
    "XLP": ["PAYEMS", "CPIAUCSL", "UNRATE", "M2SL"],
    "XLI": ["INDPRO", "DGORDER", "GDP", "CFNAI", "DCOILWTICO"],
    "XLV": ["PAYEMS", "UNRATE", "CPIAUCSL", "PCEPILFE"],
    "XLU": ["DGS10", "DGS30", "REAL10Y", "CPIAUCSL", "NET_LIQUIDITY"],
    "XLB": ["PCOPPUSDM", "PALLFNFINDEXM", "INDPRO", "DTWEXBGS", "DCOILWTICO"],
    "XLC": ["REAL10Y", "RSXFS", "UMCSENT", "VIXCLS"],
    "XLRE": ["MORTGAGE30US", "DGS10", "REAL10Y", "CSUSHPINSA", "HOUST"],
}

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
