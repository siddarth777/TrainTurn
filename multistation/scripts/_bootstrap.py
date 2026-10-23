"""Allow `python multistation/scripts/name.py` without installation: adds the repo and this folder."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE.parent), str(HERE)]
