#!/usr/bin/env python3
"""Run Indexatron service.

This is a convenience wrapper. You can also use:
    python -m indexatron.cli
    indexatron  (if installed via pip)
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from indexatron.cli import main

if __name__ == "__main__":
    main()
