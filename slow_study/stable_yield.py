import requests

r = requests.get("https://yields.llama.fi/pools", timeout=30)
d = r.json()["data"]
rows = []
for p in d:
    apy = p.get("apy")
    if not isinstance(apy, dict):
        continue
    if p.get("symbol") in ("USDC", "USDT", "USDS", "DAI"):
        rows.append((apy.get("apy") or 0.0, p.get("project"), p.get("symbol"),
                     p.get("chain"), p.get("tvlUsd") or 0.0))
rows.sort(key=lambda x: -x[0])
with open("stable_yields.txt", "w") as f:
    for a, pr, sym, ch, tvl in rows[:20]:
        f.write(f"{pr:22s} {sym:6s} {str(ch):10s} apy={a:6.2f}% tvl={tvl/1e9:6.1f}B\n")
print("wrote", len(rows), "rows")
