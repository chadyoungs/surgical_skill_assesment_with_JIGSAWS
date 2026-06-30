from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_config import get_task_index

# 0 for Suturing, 1 for Knot Tying, 2 for Needle Passing
TASK_SYMBOL = get_task_index()

CLUSTERS = 64
N_CLUSTERS = CLUSTERS

# for PCA in [hmm_model_o.py] and [train_observations.py]
PCA_COMPONENTS = 4
