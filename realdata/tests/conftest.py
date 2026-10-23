from pathlib import Path
import sys
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE.parent), str(HERE)]
