"""CI entrypoint for refreshing BTC-USDT-SWAP funding-rate history.

Exists as a real file (rather than an inline `python -c "..."` block in the
workflow YAML) because embedding multi-line indented Python inside a YAML
block scalar is fragile — the previous inline version had a leading-indent
IndentationError that made this step fail on every run.

Uses OKX (via okx_funding_rate_fetcher.py), not Binance: Binance's fapi
returns HTTP 451 to GitHub-hosted runner IPs (regulatory geo-block) and
Bybit's public API is blocked at the CloudFront level by country. OKX's
public endpoint was confirmed reachable (200, real data) when probed from
the same runner IP.
"""
import traceback

from okx_funding_rate_fetcher import download_funding_data

if __name__ == "__main__":
    try:
        df = download_funding_data(force_refresh=True)
        print(f"funding: {len(df)} settlements, {df.index[0]} -> {df.index[-1]}")
    except Exception as e:
        print(f"funding fetch FAILED: {e}")
        traceback.print_exc()
        raise
