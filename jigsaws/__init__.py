"""JIGSAWS surgical-skill assessment toolkit.

The package is organised around one shared dataset layer and a pluggable
method layer:

    jigsaws.config      -- path/task resolution (no global mutable state)
    jigsaws.dataset     -- scores, transcripts, LOSO folds, clip catalogue
    jigsaws.evaluation  -- shared metrics and LOSO grid search
    jigsaws.kinematics  -- shared trajectory feature extraction
    jigsaws.methods     -- one package per method, auto-discovered by the CLI
"""

from __future__ import annotations

__version__ = "0.1.0"
