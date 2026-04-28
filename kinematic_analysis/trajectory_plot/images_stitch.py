#!/usr/bin/env python3
"""
Stitch two trajectory images side by side and save the result.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

# Image dimensions (pixels)
IMG_HEIGHT: int = 1200
IMG_WIDTH: int = 1800
_SEP_WIDTH: int = 10  # white separator strip between images

IMAGE_LIST: list[str] = [
    "3D trajectories of left hand of expert and novice surgeon.png",
    "3D trajectories of right hand of expert and novice surgeon.png",
]
OUTPUT_FILE: str = "image_stitch.jpg"


def main() -> None:
    target_width = 2 * IMG_WIDTH + _SEP_WIDTH
    canvas = Image.new("RGB", (target_width, IMG_HEIGHT))

    left = 0
    for count, image_name in enumerate(IMAGE_LIST):
        canvas.paste(Image.open(image_name), (left, 0, left + IMG_WIDTH, IMG_HEIGHT))
        left += IMG_WIDTH
        canvas.paste((255, 255, 255), (left, 0, left + _SEP_WIDTH, IMG_HEIGHT))
        if count >= 1:
            break
        left += _SEP_WIDTH

    canvas.save(OUTPUT_FILE, quality=100)


if __name__ == "__main__":
    main()

