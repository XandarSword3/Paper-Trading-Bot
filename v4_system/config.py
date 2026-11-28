"""Central configuration for the V4 system."""
import os

V4_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(V4_ROOT, "data")
REPORTS_DIR = os.path.join(V4_ROOT, "reports")
TRIALS_LOG = os.path.join(V4_ROOT, "trials_log.csv")
MANIFEST_PATH = os.path.join(DATA_DIR, "MANIFEST.json")

# Binance USDⓈ-M futures (perp) and spot REST base URLs.
FAPI_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://api.binance.com"

# Default universe — liquid USDⓈ-M perpetuals.
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Timeframes supported by the data engine.
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Binance USDⓈ-M VIP0 fee schedule (fraction, not %).
TAKER_FEE = 0.0005
MAKER_FEE = 0.0002

# HTTP
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
