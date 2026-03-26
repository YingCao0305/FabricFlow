#!/usr/bin/env python3

from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from reconstruction_workflow import main


if __name__ == "__main__":
    main()
