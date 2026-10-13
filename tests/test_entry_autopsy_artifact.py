"""
Traceability test for REPORT.md section 15 (entry autopsy).

Rule enforced here: every number printed in a table of REPORT section 15 must exist
as a value in the committed artifact quant_research/results/entry_autopsy.json, and the
grouped entries CSV must be consistent with the counts in that artifact.
"""
import csv
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "quant_research", "REPORT.md")
RESULTS = os.path.join(ROOT, "quant_research", "results")
ENTRIES = os.path.join(RESULTS, "entry_autopsy_entries.csv")

# A cited number must not be glued to a word/unit (1m, 60m) or be a confidence-level tag (95%).
NUMBER = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.%])")

# Each report section is checked only against the artifacts it is allowed to cite.
SECTION_SOURCES = {
    "## 15. Entry Autopsy": ["entry_autopsy.json"],
    "## 16. Round-2 Closures": ["turtle_sign_fix_comparison.json", "test_3_proper_results.json"],
}


def _section(heading):
    with open(REPORT, "r", encoding="utf-8", newline="") as fh:
        text = fh.read().replace("\r\n", "\n")
    start = text.find(heading)
    assert start != -1, f"REPORT.md no longer contains {heading!r}"
    rest = text[start + len(heading):]
    nxt = rest.find("\n## ")
    return rest if nxt == -1 else rest[:nxt]


def _cited_table_numbers(heading):
    cited = []
    for line in _section(heading).split("\n"):
        if not line.startswith("|"):
            continue
        for cell in line.strip().strip("|").split("|"):
            for m in NUMBER.finditer(cell):
                cited.append(m.group(1))
    return cited


def _artifact(name):
    with open(os.path.join(RESULTS, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _formatted_values(obj, decimals=(0, 1, 2, 3, 4)):
    """Every numeric value in the artifact, rendered the way the report cites it."""
    out = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _formatted_values(v, decimals)
    elif isinstance(obj, list):
        for v in obj:
            out |= _formatted_values(v, decimals)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, int):
        out.add(str(obj))
    elif isinstance(obj, float):
        for d in decimals:
            out.add(("{:." + str(d) + "f}").format(obj))
            out.add(("{:+." + str(d) + "f}").format(obj))
    elif isinstance(obj, str):
        for m in NUMBER.finditer(obj):
            out.add(m.group(1))
        for m in re.finditer(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])", obj):
            out.add(m.group(1))
    return out


def _kind_counts():
    with open(ENTRIES, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    return rows, kinds


@pytest.mark.parametrize("heading", sorted(SECTION_SOURCES))
def test_report_tables_cite_only_artifact_numbers(heading):
    sources = SECTION_SOURCES[heading]
    values = set()
    for name in sources:
        values |= _formatted_values(_artifact(name))
    cited = _cited_table_numbers(heading)
    assert len(cited) > 40, f"expected many cited table numbers in {heading!r}, found {len(cited)}"
    unresolved = sorted({c for c in cited if c not in values})
    assert not unresolved, f"{heading!r} table numbers absent from {sources}: {unresolved}"


def test_artifact_counts_match_the_grouped_entries_csv():
    art = _artifact("entry_autopsy.json")
    counts = art["counts"]
    rows, kinds = _kind_counts()
    assert counts["fills"] == 1786
    assert counts["position_cycles"] == 257
    assert counts["entry_fills_open_plus_add"] == 1003
    assert counts["entry_opens"] == 257
    assert counts["entry_adds"] == 746
    assert len(rows) == counts["entry_fills_open_plus_add"]
    assert kinds["open"] == counts["entry_opens"]
    assert kinds["add"] == counts["entry_adds"]
    assert counts["taker_fills"] + counts["maker_fills"] == counts["fills"]


def test_artifact_shape_and_headline_numbers():
    art = _artifact("entry_autopsy.json")
    assert art["timezone_evidence"]["adopted_offset_hours"] == 3
    assert art["timezone_evidence"]["n_funding_rows"] == 50
    assert art["timezone_evidence"]["settlement_times_in_csv_clock"] == {
        "03:00": 15, "11:00": 20, "19:00": 15}
    assert art["headline"]["n_indicators_tested"] == 13
    assert art["headline"]["n_horizons_tested_per_reference"] == 4

    for label, block_outer in art["results"].items():
        units = block_outer["units"]
        assert set(units) == {"open", "add"}, label
        for unit, block in units.items():
            assert block["indicator_family_size"] == 13
            assert set(block["forward_by_reference"]) == {
                "actual_fill_price", "open_of_fill_bar", "close_of_fill_bar", "next_bar_open"}
            for name in ("open_of_fill_bar", "close_of_fill_bar", "next_bar_open"):
                assert len(block["forward_by_reference"][name]["horizons"]) == 4

    prim = art["results"]["primary_csv_clock_utc_plus_3"]
    assert prim["csv_clock_offset_hours"] == 3
    opens = prim["units"]["open"]
    assert opens["n_entries_used"] == 255
    assert {r["metric"] for r in opens["indicators"] if r["significant_holm_05"]} == {
        "adx14", "daily_trend_ema20", "daily_dist_ema20_bps"}
    fill = opens["forward_by_reference"]["actual_fill_price"]["horizons_bps"]
    assert fill["1m"] == pytest.approx(-1.1374488899407655)
    assert fill["60m"] == pytest.approx(5.8701531508135725)
    assert opens["execution"]["mean_bps"] == pytest.approx(5.142411274603996)
    assert opens["execution"]["ci95"][0] > 0

    sens = art["results"]["sensitivity_csv_clock_utc"]
    assert sens["csv_clock_offset_hours"] == 0
    assert {r["metric"] for r in sens["units"]["open"]["indicators"]
            if r["significant_holm_05"]} == {"daily_trend_ema20", "daily_dist_ema20_bps"}
    for name in ("open_of_fill_bar", "close_of_fill_bar", "next_bar_open"):
        assert not sens["units"]["open"]["forward_by_reference"][name]["any_significant_holm_05"]


def test_sign_fix_comparison_publishes_the_counts_the_report_quotes():
    d = _artifact("turtle_sign_fix_comparison.json")
    assert d["n_configs_net_moved"] == 36
    assert d["n_sign_changes"] == 0
    assert d["n_verdict_changes"] == 0
    assert len(d["configs"]) == 36
    assert len(d["pooled_by_config"]) == 12
    assert sum(1 for r in d["configs"] if r["sign_changed"]) == 0
    flip = [r for r in d["pooled_by_config"]
            if (r["pooled_net_bps_before"] > 0) != (r["pooled_net_bps_after"] > 0)]
    assert len(flip) == 1 and flip[0]["timeframe"] == "1h"
    assert flip[0]["pooled_net_bps_before"] == pytest.approx(-5.38)
    assert flip[0]["verdict_before"] == flip[0]["verdict_after"] == "FAIL"


def test_risk_layer_variants_publish_equity_drawdown_and_worst_trade():
    d = _artifact("test_3_proper_results.json")
    variants = d["replay_variants"]
    assert len(variants) == 6
    for r in variants:
        for key in ("final_equity", "max_drawdown_pct", "worst_trade_usd", "account_ruined"):
            assert key in r, (r["variant"], key)
        assert r["account_ruined"] is False
    # the 5x and 10x cap rows are identical in every performance column; only the cap
    # label differs, so the cap is not binding for this trade sequence.
    perf = ("trades_taken", "stops_triggered", "stops_triggered_pct", "final_equity",
            "total_return_pct", "max_drawdown_pct", "worst_trade_usd", "account_ruined")
    for a, b in ((0, 1), (2, 3), (4, 5)):
        assert {k: variants[a][k] for k in perf} == {k: variants[b][k] for k in perf}
        assert variants[a]["leverage_cap"] != variants[b]["leverage_cap"]
    # no 1% sizing / 3% daily-loss-cap variant exists in any artifact
    assert {r["stop_pct"] for r in variants} == {0.3, 0.5, 1.0}
    assert all("1%" not in r["variant"] and "daily" not in r["variant"] for r in variants)
    assert d["ledger_reconciliation"]["true_max_drawdown_pct"] == pytest.approx(100.0)
