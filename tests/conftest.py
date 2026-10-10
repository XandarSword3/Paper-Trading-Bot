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
