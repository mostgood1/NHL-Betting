from pathlib import Path
import pandas as pd

p = Path('data/raw/nhl_pbp')
files = sorted(p.glob('pbp_*.parquet'))
print('count', len(files))
print('files', [f.name for f in files[:5]])
if not files:
    raise SystemExit(0)

df = pd.read_parquet(files[0])
print('first file', files[0].name, 'shape', df.shape)
print('columns', list(df.columns))
print(df.head(3).to_string())
