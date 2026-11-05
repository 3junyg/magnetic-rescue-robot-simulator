import os
from pathlib import Path
import sys


def prepare() -> Path:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.chdir(root)
    return root
