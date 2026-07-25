"""CI entrypoint for refreshing BTCUSDT funding-rate history.

Exists as a real file (rather than an inline `python -c "..."` block in the
workflow YAML) because embedding multi-line indented Python inside a YAML
block scalar is fragile — the previous inline version had a leading-indent
IndentationError that made this step fail on every run.
"""
import traceback

from funding_rate_fetcher import download_funding_data

if __name__ == "__main__":
    try:
        df = download_funding_data(start_date="2019-09-01", force_refresh=True)
        print(f"funding: {len(df)} settlements, {df.index[0]} -> {df.index[-1]}")
    except Exception as e:
        print(f"funding fetch FAILED: {e}")
        traceback.print_exc()
        raise
