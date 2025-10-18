"""Debug props_recommendations name parsing."""
import pandas as pd
import ast as _ast

def _norm_player(s):
    if s is None:
        return ""
    x = str(s).strip()
    if x.startswith('{') and x.endswith('}'):
        try:
            d = _ast.literal_eval(x)
            if isinstance(d, dict):
                v = d.get('default') or d.get('name') or ''
                if isinstance(v, str):
                    x = v.strip()
        except Exception:
            pass
    return " ".join(x.split())

def _looks_like_player(x: str) -> bool:
    s = (x or '').strip().lower()
    if not s:
        return False
    bad = ['total shots on goal', 'team total', 'first period', 'second period', 'third period']
    return (any(ch.isalpha() for ch in s) and not any(b in s for b in bad))

# Read lines
df = pd.read_parquet('data/props/player_props_lines/date=2025-10-17/oddsapi.parquet')
print(f'Total rows: {len(df)}')

# Test normalization
df['player_display'] = df.apply(lambda r: _norm_player(r.get('player_name') or r.get('player')), axis=1)
print(f'\nFirst 10 player_display:')
print(df['player_display'].head(10).tolist())

# Test filtering
df['is_player'] = df['player_display'].map(_looks_like_player)
print(f'\nFiltered count: {df["is_player"].sum()}')
print(f'\nFirst 10 after filter:')
print(df[df['is_player']]['player_display'].head(10).tolist())

# Check what failed
failed = df[~df['is_player']]
if not failed.empty:
    print(f'\nFailed examples (first 10):')
    print(failed[['player_name', 'player_display']].head(10))
