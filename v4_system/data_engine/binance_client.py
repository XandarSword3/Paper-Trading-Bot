"""Rate-limited Binance REST client with retry/backoff.

Public endpoints only — no API key required for market data. All requests
are GETs against fapi.binance.com (USDⓈ-M futures) or api.binance.com
(spot). Timestamps are preserved as UTC; nothing is forward-filled.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

import requests

from v4_system.config import FAPI_BASE, MAX_RETRIES, REQUEST_TIMEOUT, SPOT_BASE


class BinanceAPIError(RuntimeError):
    pass


class BinanceClient:
    """Thin wrapper around Binance public REST with polite rate limiting."""

    def __init__(self, base: str = FAPI_BASE, requests_per_second: float = 2.5):
        self.base = base.rstrip("/")
        self.min_interval = 1.0 / requests_per_second
        self._last_call = 0.0
        self.session = requests.Session()

    # -- low level ---------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base}{path}"
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 429 or resp.status_code >= 500:
                    retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(min(retry_after, 60))
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise BinanceAPIError(f"GET {url} failed after {MAX_RETRIES} attempts: {last_err}")

    def get_signed(self, path: str, params: dict[str, Any] | None = None,
                   api_key: str | None = None, api_secret: str | None = None) -> Any:
        """Signed GET for authenticated READ-ONLY endpoints (e.g.
        /fapi/v1/leverageBracket). Uses HMAC-SHA256 per Binance docs.
        Never use for trading endpoints."""
        import os
        api_key = api_key or os.environ.get("BINANCE_API_KEY")
        api_secret = api_secret or os.environ.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise BinanceAPIError("API key/secret required for signed endpoint")
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        # Binance verifies the signature against the query string AS
        # RECEIVED, so the signed string must match the order requests
        # will send (dict insertion order) — do NOT sort.
        query = "&".join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(api_secret.encode(), query.encode(),
                             hashlib.sha256).hexdigest()
        params["signature"] = signature
        url = f"{self.base}{path}"
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self.session.get(url, params=params,
                                        headers={"X-MBX-APIKEY": api_key},
                                        timeout=REQUEST_TIMEOUT)
                if resp.status_code == 429 or resp.status_code >= 500:
                    retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(min(retry_after, 60))
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise BinanceAPIError(f"GET {url} failed after {MAX_RETRIES} attempts: {last_err}")

    # -- convenience constructors ------------------------------------------
    @classmethod
    def futures(cls, **kw) -> "BinanceClient":
        return cls(base=FAPI_BASE, **kw)

    @classmethod
    def spot(cls, **kw) -> "BinanceClient":
        return cls(base=SPOT_BASE, **kw)
