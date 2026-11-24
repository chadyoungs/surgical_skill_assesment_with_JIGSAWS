"""Surgeme video clipping and frame extraction (OpenCV).

Replaces the two helper scripts that relied on module-level ``ROOT``/``TASK``
constants. Everything is driven by the :class:`Dataset` and explicit output
paths, and every video writer is released deterministically.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from ..deps import require
from ..errors import DataError
from .metadata import Dataset, TrialRecord

LOGGER = logging.getLogger(__name__)


def _trial_from_video_stem(stem: str) -> str:
    return "_".join(stem.split("_")[:2])


def _segments_by_frame(record: TrialRecord) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for index, segment in enumerate(record.segments):
        for frame in range(segment.start, segment.end + 1):
            mapping[frame] = index
    return mapping


def build_surgeme_clips(
    dataset: Dataset,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Split full-trial ``*capture1*`` videos into per-surgeme ``.avi`` clips."""
    cv2 = require("cv2", extra="media")

    videos_dir = dataset.settings.videos_dir(dataset.task)
    if not videos_dir.is_dir():
        raise DataError(f"Video directory not found: {videos_dir}")

    target_root = (
        Path(output_dir)
        if output_dir
        else dataset.settings.surgeme_video_dir(dataset.task)
    )
    for label in dataset.surgeme_labels:
        (target_root / label).mkdir(parents=True, exist_ok=True)

    videos = sorted(path for path in videos_dir.glob("*.avi") if "capture1" in path.name)
    if not videos:
        raise DataError(f"No capture1 videos found in {videos_dir}")

    written: list[Path] = []
    for video in videos:
        trial = _trial_from_video_stem(video.stem)
        if trial not in dataset:
            LOGGER.warning("Skipping video with unknown trial: %s", video.name)
            continue
        record = dataset[trial]

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            LOGGER.warning("Cannot open video: %s", video)
            continue
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_to_segment = _segments_by_frame(record)

        active_index: int | None = None
        writer = None
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                segment_index = frame_to_segment.get(frame_index)

                if segment_index != active_index:
                    if writer is not None:
                        writer.release()
                        writer = None
                    active_index = segment_index
                    if segment_index is not None:
                        segment = record.segments[segment_index]
                        clip_path = (
                            target_root
                            / segment.label
                            / f"{trial}_{segment.label}_{segment_index}.avi"
                        )
                        if clip_path.exists() and not overwrite:
                            LOGGER.debug("Clip exists, skipping: %s", clip_path.name)
                            active_index = segment_index
                            continue
                        writer = cv2.VideoWriter(
                            str(clip_path),
                            cv2.VideoWriter_fourcc(*"XVID"),
                            fps,
                            (width, height),
                        )
                        written.append(clip_path)

                if writer is not None and segment_index is not None:
                    writer.write(frame)
                if frame_index >= record.frame_range[1]:
                    break
        finally:
            if writer is not None:
                writer.release()
            capture.release()

    LOGGER.info("Wrote %d surgeme clips under %s", len(written), target_root)
    return tuple(written)


def extract_frames(
    dataset: Dataset,
    *,
    videos_root: str | Path | None = None,
    images_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Dump every frame of every surgeme clip to JPEG files."""
    cv2 = require("cv2", extra="media")

    source = (
        Path(videos_root)
        if videos_root
        else dataset.settings.surgeme_video_dir(dataset.task)
    )
    target = (
        Path(images_root)
        if images_root
        else dataset.settings.surgeme_image_dir(dataset.task)
    )
    if not source.is_dir():
        raise DataError(f"Surgeme video directory not found: {source}")

    written: list[Path] = []
    for video in sorted(source.glob("*/*.avi")):
        folder = target / video.parent.name / video.stem
        folder.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            LOGGER.warning("Cannot open video: %s", video)
            continue
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                path = folder / f"img_{index:05d}.jpg"
                cv2.imwrite(str(path), frame)
                written.append(path)
                index += 1
        finally:
            capture.release()
    LOGGER.info("Extracted %d frames under %s", len(written), target)
    return tuple(written)


def capture_middle_frame(
    video_path: str | Path,
    destination: str | Path,
) -> Path:
    """Save the middle frame of *video_path* as JPEG."""
    cv2 = require("cv2", extra="media")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise DataError(f"Cannot open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    middle = max(total // 2, 1)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            index += 1
            if not ok:
                raise DataError(f"Video ended before middle frame: {video_path}")
            if index >= middle:
                cv2.imwrite(str(target), frame)
                return target
    finally:
        capture.release()


def capture_random_gesture_frame(
    dataset: Dataset,
    gesture: str,
    destination: str | Path,
    *,
    seed: int | None = None,
) -> Path:
    """Pick a random clip of *gesture* and capture its middle frame."""
    directory = dataset.settings.surgeme_video_dir(dataset.task) / gesture
    clips = sorted(directory.glob("*.avi")) if directory.is_dir() else []
    if not clips:
        raise DataError(f"No clips found for gesture '{gesture}' in {directory}")
    rng = random.Random(seed)
    return capture_middle_frame(rng.choice(clips), destination)
