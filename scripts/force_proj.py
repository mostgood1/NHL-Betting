import sys, os
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)
from nhl_betting.cli import props_projections_force
props_projections_force(date='2026-01-27')
print('force projections done')