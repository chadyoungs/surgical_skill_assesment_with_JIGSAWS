"""Global configuration constants for the HMM model."""

# 0 for Suturing, 1 for Knot_Tying, 2 for Needle_Passing
TASK_SYMBOL: int = 0

# Number of K-means clusters used for observation quantisation
CLUSTERS: int = 64

# Number of PCA components used in hmm_model_o.py and train_observations.py
PCA_COMPONENTS: int = 4
