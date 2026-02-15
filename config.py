"""
Project configuration for the Polymarket weather trading stack.
"""
from __future__ import annotations

import os

# Runtime
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "6.0"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "20"))
DEFAULT_ZSCORE_THRESHOLD = float(os.getenv("DEFAULT_ZSCORE_THRESHOLD", "2.5"))
MIN_SOURCES_FOR_SIGNAL = int(os.getenv("MIN_SOURCES_FOR_SIGNAL", "4"))
MONITOR_INTERVAL_SECONDS = int(os.getenv("MONITOR_INTERVAL_SECONDS", "120"))
SAFETY_MARGIN_C = 0.50
MIN_EDGE_TO_TRADE = 0.02
KELLY_FRACTION = 0.40
MAX_KELLY_FRACTION = float(os.getenv("MAX_KELLY_FRACTION", "0.10"))
BANKROLL_USD = float(os.getenv("BANKROLL_USD", "1000"))
MIN_MARKET_LIQUIDITY = float(os.getenv("MIN_MARKET_LIQUIDITY", "1000"))
OPPORTUNITY_LOG_FILE = os.getenv("OPPORTUNITY_LOG_FILE", "opportunities.log")

# API keys (used by premium providers when available)
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")
TOMORROW_API_KEY = os.getenv("TOMORROW_API_KEY", "")
VISUAL_CROSSING_KEY = os.getenv("VISUAL_CROSSING_KEY", "")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "")

# Market cities (airport-aligned for Polymarket temperature markets)
CITY_COORDS = {
    "london_city": (51.5053, -0.0553),
    "new_york_laguardia": (40.7769, -73.8740),
    "seoul_incheon": (37.4602, 126.4407),
}

# Aliases to map Polymarket city labels into coord keys.
CITY_ALIASES = {
    "london": "london_city",
    "london city": "london_city",
    "london (city)": "london_city",
    "new york": "new_york_laguardia",
    "new york (laguardia)": "new_york_laguardia",
    "nyc": "new_york_laguardia",
    "laguardia": "new_york_laguardia",
    "seoul": "seoul_incheon",
    "seul": "seoul_incheon",
    "incheon": "seoul_incheon",
    "incheon intl": "seoul_incheon",
}

# Monitor focuses on these city names first (as shown in Polymarket titles).
FOCUS_MARKET_CITIES = (
    "london",
    "new york",
    "seoul",
)
