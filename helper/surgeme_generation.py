"""
Surgeme (gesture clip) video generation utilities for the JIGSAWS dataset.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from constant import ROOT, TASK
from metadata_generation import MetaData


def get_metadata() -> dict:
    """Load and return the dataset metadata dictionary."""
    metadata = MetaData()
    metadata.generate_metadata()
    return metadata.metadata_res


def make_dirs(metadata: dict) -> None:
    """Create one output directory per surgeme label under ``surgeme_video``."""
    surgeme_video_root = ROOT / TASK / "surgeme_video"
    for surgeme in metadata["metadata"]["surgeme_list"]:
        (surgeme_video_root / surgeme).mkdir(parents=True, exist_ok=True)


def video_surgeme_generation(metadata: dict) -> None:
    """Split each full-trial video into individual surgeme clip files.

    Args:
        metadata: Dataset metadata dict as returned by :func:`get_metadata`.
    """
    video_root = ROOT / TASK / "video"
    surgeme_video_root = ROOT / TASK / "surgeme_video"

    capture1_video_list = [p for p in video_root.iterdir() if "capture1" in p.name]

    for capture1_video in capture1_video_list:
        trial_name = "_".join(capture1_video.stem.split("_")[:2])

        surgeme_start_end = metadata[trial_name]["surgeme_start_end"]
        start_frames: list[int] = surgeme_start_end["start_frame_idx"]
        end_frames: list[int] = surgeme_start_end["end_frame_idx"]
        surgemes: list[str] = surgeme_start_end["surgeme"]

        cap = cv2.VideoCapture(str(capture1_video))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 1
        surgeme_count = 0
        video_writer: cv2.VideoWriter | None = None
        in_surgeme = False

        if cap.isOpened():
            while True:
                success, frame = cap.read()
                if not success:
                    break

                if not in_surgeme and frame_count == start_frames[surgeme_count]:
                    clip_name = "_".join([trial_name, surgemes[surgeme_count], str(surgeme_count)]) + ".avi"
                    clip_path = surgeme_video_root / surgemes[surgeme_count] / clip_name
                    video_writer = cv2.VideoWriter(
                        str(clip_path),
                        cv2.VideoWriter_fourcc("X", "V", "I", "D"),
                        fps,
                        (frame_width, frame_height),
                    )
                    in_surgeme = True

                if in_surgeme and video_writer is not None:
                    video_writer.write(frame)

                if in_surgeme and frame_count == end_frames[surgeme_count]:
                    in_surgeme = False
                    surgeme_count += 1

                if frame_count == end_frames[-1]:
                    break

                frame_count += 1

        cap.release()


def main() -> None:
    res = get_metadata()
    make_dirs(res)
    video_surgeme_generation(res)


if __name__ == "__main__":
    main()
