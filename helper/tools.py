"""
Utility tools for the JIGSAWS dataset helper.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from constant import ROOT, TASK, IMG_HEIGHT, IMG_WIDTH, IMG_BLANK_WIDTH, STITCH_IMAGE_WIDTH


def frame_capture(gesture: str, capture_file_name: str, task: str = TASK) -> None:
    """Capture the middle frame of a randomly chosen surgeme video clip.

    Args:
        gesture: Gesture label (e.g. ``"G5"``).
        capture_file_name: Destination file path for the captured JPEG frame.
        task: JIGSAWS task name.  Defaults to the global ``TASK`` constant.
    """
    surgeme_video_dir = ROOT / task / "surgeme_video" / gesture
    surgeme_video_list = [p for p in surgeme_video_dir.iterdir() if p.suffix == ".avi"]

    random_idx = np.random.randint(len(surgeme_video_list))
    random_video = surgeme_video_list[random_idx]

    cap = cv2.VideoCapture(str(random_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = total_frames // 2
    current_frame = 1

    if cap.isOpened():
        while True:
            success, frame = cap.read()
            if not success:
                break
            if current_frame == target_frame:
                cv2.imencode(".jpg", frame)[1].tofile(capture_file_name)
                break
            current_frame += 1

    cap.release()


def extract_frame() -> None:
    """Extract every frame from all surgeme videos and save them as JPEG images."""
    video_dataset_root = ROOT / TASK / "surgeme_video"
    for video_dir in video_dataset_root.iterdir():
        for video_path in video_dir.iterdir():
            cap = cv2.VideoCapture(str(video_path))
            count = 0
            if cap.isOpened():
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    img_folder = ROOT / TASK / "surgeme_img" / video_path.parent.name / video_path.name
                    img_folder.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(img_folder / f"img_{count:05d}.jpg"), frame)
                    count += 1
            cap.release()


def image_stitch(
    input_img_list: list[str | Path],
    save_path: str | Path = "image_stitch.jpg",
    quality: int = 95,
) -> None:
    """Stitch up to three images side by side and save the result.

    Args:
        input_img_list: Paths to the images to stitch (up to three).
        save_path: Destination file path for the stitched JPEG image.
        quality: JPEG compression quality (1–95).  Defaults to ``95``.
    """
    canvas = Image.new("RGB", (STITCH_IMAGE_WIDTH, IMG_HEIGHT))

    left = 0
    right = IMG_WIDTH
    for count, image_path in enumerate(input_img_list):
        canvas.paste(Image.open(image_path), (left, 0, right, IMG_HEIGHT))

        # Add a white separator strip to the right of this image
        left += IMG_WIDTH
        right += IMG_WIDTH
        canvas.paste((255, 255, 255), (left, 0, left + IMG_BLANK_WIDTH, IMG_HEIGHT))

        if count == 2:
            break

        left += IMG_BLANK_WIDTH
        right += IMG_BLANK_WIDTH

    canvas.save(str(save_path), quality=quality)


if __name__ == "__main__":
    extract_frame()