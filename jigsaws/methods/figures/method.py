"""Figure utilities: horizontal stitching and resizing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ...images import resize_image, stitch_horizontal
from ..base import Command, Context, Method

LOGGER = logging.getLogger(__name__)


class FiguresMethod(Method):
    name = "figures"
    summary = "Compose and resize publication figures"

    def commands(self) -> tuple[Command, ...]:
        return (
            Command("stitch", "Stitch images side by side", self._stitch, self._configure_stitch),
            Command("resize", "Resize one image", self._resize, self._configure_resize),
        )

    @staticmethod
    def _configure_stitch(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("inputs", nargs="+", type=Path, help="Images, left to right")
        parser.add_argument("--output", type=Path, default=None)
        parser.add_argument("--gap", type=int, default=10, help="White gap in pixels")
        parser.add_argument("--height", type=int, default=None, help="Common height")
        parser.add_argument("--quality", type=int, default=95)

    @staticmethod
    def _configure_resize(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", type=Path)
        parser.add_argument("--output", type=Path, default=None)
        parser.add_argument("--width", type=int, default=360)
        parser.add_argument("--height", type=int, default=120)

    def _stitch(self, ctx: Context, args: argparse.Namespace) -> None:
        target = args.output or (
            ctx.artifact_dir("figures") / "image_stitch.jpg"
        )
        path = stitch_horizontal(
            args.inputs,
            target,
            gap=args.gap,
            height=args.height,
            quality=args.quality,
        )
        LOGGER.info("Wrote stitched image: %s", path)

    def _resize(self, ctx: Context, args: argparse.Namespace) -> None:
        target = args.output or args.input.with_name(
            f"{args.input.stem}_{args.width}x{args.height}{args.input.suffix}"
        )
        if target == args.input:
            target = args.input.with_name(f"{args.input.stem}_resized{args.input.suffix}")
        path = resize_image(args.input, target, width=args.width, height=args.height)
        LOGGER.info("Wrote resized image: %s", path)


METHOD = FiguresMethod()
