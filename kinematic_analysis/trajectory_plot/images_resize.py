#!/usr/bin/env python3
"""Resize the stitched trajectory image for display purposes."""
from __future__ import annotations

from PIL import Image

TARGET_WIDTH: int = 360
TARGET_HEIGHT: int = 120
INPUT_FILE: str = "image_stitch.jpg"


def main() -> None:
    Image.open(INPUT_FILE).resize((TARGET_WIDTH, TARGET_HEIGHT)).save(INPUT_FILE)


if __name__ == "__main__":
    main()
