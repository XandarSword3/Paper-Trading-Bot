import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in [
    ROOT,
    ROOT / "research",
    ROOT / "research" / "strategies",
    ROOT / "research" / "validation",
    ROOT / "research" / "data",
]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


def pytest_ignore_collect(collection_path, config):
    p_str = str(collection_path)
    if "test_telegram.py" in p_str or "test_binance_keys.py" in p_str:
        return True
    return False
