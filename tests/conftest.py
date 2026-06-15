"""Make stage1_detection importable from tests."""

import sys
from pathlib import Path

STAGE1 = Path(__file__).resolve().parent.parent / "stage1_detection"
if str(STAGE1) not in sys.path:
    sys.path.insert(0, str(STAGE1))
