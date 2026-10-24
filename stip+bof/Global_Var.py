"""Global configuration constants for the STIP + BoF pipeline."""

# 0 for Suturing, 1 for Knot_Tying, 2 for Needle_Passing
TASK_SYMBOL: int = 0

# Number of K-means clusters used for the BoF vocabulary (HOG and HOF)
CLUSTERS: int = 400

# Maximum number of STIP feature samples used during vocabulary training
SELECTED_FEATURES_NO: int = 100000

# Feature-processing parameters
EXPAND_RADIUS: int = 5
REMOVE_WINDOW_SIZE: int = 6
SAMPLE_RATE: int = 2
PADDING_SAMPLES: int = 20