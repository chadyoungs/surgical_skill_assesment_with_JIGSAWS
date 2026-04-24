from pathlib import Path

# Dataset root (two levels up from this file)
ROOT: Path = Path(__file__).parent.parent

TASK_LIST = ["Suturing", "Knot_Tying", "Needle_Passing"]
TASK_CHOICE = 0
TASK = TASK_LIST[TASK_CHOICE]

# Image-stitching dimensions
STITCH_IMAGE_COUNT = 3
IMG_HEIGHT = 480
IMG_WIDTH = 640
IMG_BLANK_WIDTH = 10
STITCH_IMAGE_WIDTH = STITCH_IMAGE_COUNT * IMG_WIDTH + (STITCH_IMAGE_COUNT - 1) * IMG_BLANK_WIDTH 