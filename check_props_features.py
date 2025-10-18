"""Check props NN model feature counts."""
from pathlib import Path
import numpy as np

markets = ['SOG', 'GOALS', 'ASSISTS', 'POINTS', 'SAVES', 'BLOCKS']

for m in markets:
    path = Path(f'data/models/nn_props/{m.lower()}_metadata.npz')
    if path.exists():
        meta = np.load(path, allow_pickle=True)
        features = meta['feature_columns']
        print(f'\n{m}:')
        print(f'  Total features: {len(features)}')
        print(f'  First 5: {list(features[:5])}')
        print(f'  Team features: {sum(1 for f in features if str(f).startswith("team_"))}')
    else:
        print(f'\n{m}: NOT FOUND')
