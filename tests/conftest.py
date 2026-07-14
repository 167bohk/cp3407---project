import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LUPA_DIR = PROJECT_ROOT / "Lupa"

if str(LUPA_DIR) not in sys.path:
    sys.path.insert(0, str(LUPA_DIR))
