from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_config import TASKS, get_data_root, get_task_index, get_task_name

ROOT: Path = get_data_root()

TASK_LIST = list(TASKS)
TASK_CHOICE = get_task_index()
TASK = get_task_name()

# Image-stitching dimensions
STITCH_IMAGE_COUNT = 3
IMG_HEIGHT = 480
IMG_WIDTH = 640
IMG_BLANK_WIDTH = 10
STITCH_IMAGE_WIDTH = STITCH_IMAGE_COUNT * IMG_WIDTH + (STITCH_IMAGE_COUNT - 1) * IMG_BLANK_WIDTH 