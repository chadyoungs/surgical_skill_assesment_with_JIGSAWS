#!/usr/bin/env python3
"""Repository entry point.

Kept at the top level so the canonical invocation stays:

    python main.py --task Suturing dataset info
    python main.py --task Suturing hmm train
"""

from __future__ import annotations

from jigsaws.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
