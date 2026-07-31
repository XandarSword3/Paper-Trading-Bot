"""
CI entry point for fetching ETHUSDT perpetual funding rate history via OKX.

This exists as a standalone script, rather than embedded inline in
fetch-funding-data.yml's `run:` block, because two attempts at embedding
multi-line python -c code inside that YAML step produced steps that
reported "success" in ~2 seconds (too fast for a real paginated network
fetch) while writing no output file — not even the log file the script
itself was supposed to create. The root cause was never confirmed (this
repo's raw CI logs are hosted on Azure blob storage, unreachable from the
sandboxed research environment that authored this fix), but the suspicious
common factor was YAML multi-line string embedding, so removing that
variable entirely by using a real .py file is the safer fix rather than a
third guess at the same embedding pattern.

Run from anywhere — sys.path is set up explicitly below rather than relying
on cwd, which is what made the previous inline version fragile to begin
with (see the DATA_DIR fix in data_fetcher.py for the same root pattern).
"""
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

print(f"[fetch_eth_funding_ci] script location: {__file__}")
print(f"[fetch_eth_funding_ci] cwd: {os.getcwd()}")
print(f"[fetch_eth_funding_ci] sys.path[0]: {sys.path[0]}")

try:
    from okx_funding_rate_fetcher import download_funding_data

    df = download_funding_data(
        inst_id="ETH-USDT-SWAP",
        force_refresh=True,
        filename="ETHUSDT_funding_rate",
    )
    print(f"[fetch_eth_funding_ci] SUCCESS: {len(df)} settlements, "
          f"{df.index[0]} -> {df.index[-1]}")
    sys.exit(0)
except Exception as e:
    print(f"[fetch_eth_funding_ci] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)
