#!/usr/bin/env python3
"""Reproducible Q2 evidence pipeline; all outputs go to an existing directory."""

import sys
sys.dont_write_bytecode = True

from q2_pipeline import main


if __name__ == "__main__":
    main()
