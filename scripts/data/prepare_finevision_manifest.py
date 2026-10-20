#!/usr/bin/env python3
"""FineVision single-image first-turn manifest adapter (no model loading)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from litespec.vlm.serving_data import manifest_main

if __name__ == "__main__":
    manifest_main()
