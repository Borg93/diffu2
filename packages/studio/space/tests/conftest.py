"""The Space is no installed package: its folder goes on the path for its tests, as a Space's own folder is."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
