from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_config import get_task_index

# global variable
# 0 for Suturing, 1 for Knot Tying, 2 for Need Passing
TASK_SYMBOL_Global = get_task_index()

# in clustering.py
# for both HOG and HOF
CLUSTERS = 400
SELECTED_FEATURES_No = 100000

# in feature_processing.py
EXPAND_RADIUS = 5
REMOVE_WINDOW_SIZE = 6
SAMPLE_RATE = 2
PADDING_SAMPLES = 20