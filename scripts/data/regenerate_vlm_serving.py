#!/usr/bin/env python3
"""Bounded concurrent image-answer generation through SGLang servers."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from litespec.vlm.serving_data import serving_main

if __name__ == "__main__":
    serving_main()
