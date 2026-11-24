"""Image composition helpers for figure outputs."""

from __future__ import annotations

from pathlib import Path


def _load_rgb(path: str | Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def stitch_horizontal(
    sources: list[str | Path],
    destination: str | Path,
    *,
    gap: int = 10,
    height: int | None = None,
    background: tuple[int, int, int] = (255, 255, 255),
    quality: int = 95,
) -> Path:
    """Stitch images left-to-right at a common height."""
    from PIL import Image

    if not sources:
        raise ValueError("At least one source image is required")
    images = [_load_rgb(path) for path in sources]
    target_height = height or max(image.height for image in images)

    resized = []
    for image in images:
        if image.height != target_height:
            width = max(1, round(image.width * target_height / image.height))
            image = image.resize((width, target_height), Image.Resampling.LANCZOS)
        resized.append(image)

    width = sum(image.width for image in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (width, target_height), background)
    left = 0
    for index, image in enumerate(resized):
        canvas.paste(image, (left, 0))
        left += image.width
        if index < len(resized) - 1:
            left += gap

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=quality)
    return target


def resize_image(
    source: str | Path,
    destination: str | Path,
    *,
    width: int,
    height: int,
) -> Path:
    """Resize one image with high-quality resampling."""
    from PIL import Image

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.resize((width, height), Image.Resampling.LANCZOS).save(target)
    return target
