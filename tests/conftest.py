# Lets the tests import the project's scripts (they live one folder up).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
