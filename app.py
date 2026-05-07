from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT / "src",):
    text = str(path)
    if path.exists() and text not in sys.path:
        sys.path.insert(0, text)

from market_digital_twin.app import main


if __name__ == "__main__":
    main()
