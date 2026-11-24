"""Dataset family: inspect, export, annotate and materialise clip videos."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ...dataset import summarize
from ...dataset.clips import (
    build_surgeme_clips,
    capture_random_gesture_frame,
    extract_frames,
)
from ...dataset.annotations import write_annotations, write_split_files
from ...errors import JigsawsError
from ..base import Command, Context, Method

LOGGER = logging.getLogger(__name__)


def _add_split_options(parser: argparse.ArgumentParser, *, default_kind: str) -> None:
    parser.add_argument("--kind", default=default_kind, help="Official split kind")
    parser.add_argument("--scheme", default="SuperTrialOut", help="LOSO scheme directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the default output directory inside the dataset",
    )


class DatasetMethod(Method):
    name = "dataset"
    summary = "Inspect the dataset, export metadata, write splits/annotations, build clips"

    def commands(self) -> tuple[Command, ...]:
        return (
            Command(
                name="info",
                help="Show trial counts, skill balance, labels and available splits",
                run=self._info,
                configure=self._configure_info,
            ),
            Command(
                name="export",
                help="Write a JSON snapshot of scores, grades and surgeme timelines",
                run=self._export,
                configure=self._configure_export,
            ),
            Command(
                name="splits",
                help="Write UCF-style train/test files for the official folds",
                run=self._splits,
                configure=self._configure_splits,
            ),
            Command(
                name="annotations",
                help="Write PyTorchCon3D annotation JSON for the official folds",
                run=self._annotations,
                configure=self._configure_annotations,
            ),
            Command(
                name="clips",
                help="Split trial videos into per-surgeme AVI clips",
                run=self._clips,
                configure=self._configure_clips,
            ),
            Command(
                name="frames",
                help="Extract every frame of the surgeme clips as JPEG",
                run=self._frames,
                configure=self._configure_frames,
            ),
            Command(
                name="capture",
                help="Capture the middle frame of a random clip for one gesture",
                run=self._capture,
                configure=self._configure_capture,
            ),
        )

    # ------------------------------------------------------------------
    # info / export
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_info(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--json", action="store_true", help="Print JSON only")

    @staticmethod
    def _configure_export(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--output", type=Path, default=None)

    def _info(self, ctx: Context, args: argparse.Namespace) -> None:
        info = summarize(ctx.dataset)
        if args.json:
            print(json.dumps(info, indent=2))
            return
        print(f"Task: {info['task']}")
        print(f"Data root: {info['data_root']}")
        print(
            f"Trials: {info['trials_in_metadata']} "
            f"(novice={info['novices']}, expert={info['experts']}, "
            f"kinematics={info['trials_with_kinematics']})"
        )
        print(f"GRS score range: {info['score_range'][0]}..{info['score_range'][1]}")
        print(f"Surgemes ({len(info['surgeme_labels'])}): {', '.join(info['surgeme_labels'])}")
        print(
            "Split kinds: "
            + (", ".join(info["split_kinds"]) if info["split_kinds"] else "none")
        )

    def _export(self, ctx: Context, args: argparse.Namespace) -> None:
        dataset = ctx.dataset
        payload = {
            "task": ctx.task.value,
            "trials": [
                {
                    "name": record.name,
                    "score": record.score,
                    "grade": record.grade,
                    "tier": record.tier,
                    "segments": [
                        {"start": s.start, "end": s.end, "label": s.label}
                        for s in record.segments
                    ],
                }
                for record in dataset.records
            ],
            "labels": list(dataset.surgeme_labels),
            "label_index": dataset.label_index,
        }
        target = args.output or (
            ctx.artifact_dir("dataset") / f"{ctx.task.value}_metadata.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Wrote metadata snapshot: %s", target)

    # ------------------------------------------------------------------
    # splits / annotations
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_splits(parser: argparse.ArgumentParser) -> None:
        _add_split_options(parser, default_kind="GestureClassification")

    @staticmethod
    def _configure_annotations(parser: argparse.ArgumentParser) -> None:
        _add_split_options(parser, default_kind="GestureClassification")

    def _splits(self, ctx: Context, args: argparse.Namespace) -> None:
        paths = write_split_files(
            ctx.dataset, kind=args.kind, scheme=args.scheme, output_dir=args.output_dir
        )
        LOGGER.info("Wrote %d split files", len(paths))
        for path in paths:
            LOGGER.info("  %s", path)

    def _annotations(self, ctx: Context, args: argparse.Namespace) -> None:
        paths = write_annotations(
            ctx.dataset, kind=args.kind, scheme=args.scheme, output_dir=args.output_dir
        )
        LOGGER.info("Wrote %d annotation files", len(paths))
        for path in paths:
            LOGGER.info("  %s", path)

    # ------------------------------------------------------------------
    # clips / frames / capture
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_clips(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--overwrite", action="store_true")

    @staticmethod
    def _configure_frames(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--videos-root", type=Path, default=None)
        parser.add_argument("--images-root", type=Path, default=None)

    @staticmethod
    def _configure_capture(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--gesture", required=True, help="Surgeme label, e.g. G5")
        parser.add_argument("--output", type=Path, default=None)
        parser.add_argument("--seed", type=int, default=None)

    def _clips(self, ctx: Context, args: argparse.Namespace) -> None:
        if not args.overwrite:
            LOGGER.warning("Existing clips are preserved; pass --overwrite to replace them")
        paths = build_surgeme_clips(
            ctx.dataset, output_dir=args.output_dir, overwrite=args.overwrite
        )
        LOGGER.info("Generated %d clips", len(paths))

    def _frames(self, ctx: Context, args: argparse.Namespace) -> None:
        paths = extract_frames(
            ctx.dataset, videos_root=args.videos_root, images_root=args.images_root
        )
        LOGGER.info("Extracted %d frames", len(paths))

    def _capture(self, ctx: Context, args: argparse.Namespace) -> None:
        target = args.output or (
            ctx.artifact_dir("dataset", "captures") / f"{args.gesture}.jpg"
        )
        path = capture_random_gesture_frame(
            ctx.dataset, args.gesture, target, seed=args.seed
        )
        LOGGER.info("Captured frame: %s", path)
        if not path.exists():  # pragma: no cover - defensive
            raise JigsawsError(f"Frame was not written: {path}")


METHOD = DatasetMethod()
