"""
Secure secrets loader for Binance API credentials
"""
from os import getenv
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

# API credentials
BINANCE_API_KEY = getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = getenv("BINANCE_API_SECRET")
BINANCE_TESTNET = getenv("BINANCE_TESTNET", "true").lower() in ("1", "true", "yes")

def validate_credentials():
    """Check if credentials are set"""
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise ValueError(
            "Binance API credentials not found. "
            "Copy .env.example to .env and add your keys."
        )
    if BINANCE_API_KEY == "your_api_key_here":
        raise ValueError(
            ".env file contains placeholder values. "
            "Replace with your actual Binance API keys."
        )
    return True
