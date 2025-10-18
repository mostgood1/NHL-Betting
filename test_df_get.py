"""Test DataFrame .get() behavior."""
import pandas as pd

# Create test dataframe
df = pd.DataFrame({
    'player_name': ['Alex Ovechkin', 'Connor McDavid'],
    'market': ['SOG', 'GOALS']
})

# Test .get() on row (Series)
row = df.iloc[0]
print(f"row type: {type(row)}")
print(f"row.get('player_name'): {row.get('player_name')}")
print(f"row['player_name']: {row['player_name']}")

# Test in lambda
def test_get(r):
    return r.get('player_name') or r.get('player')

df['test'] = df.apply(test_get, axis=1)
print(f"\nResult: {df['test'].tolist()}")
