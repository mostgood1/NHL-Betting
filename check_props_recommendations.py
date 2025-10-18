"""Check props recommendations for variability."""
import pandas as pd

df = pd.read_csv('data/processed/props_recommendations_2025-10-17.csv')
print(f'Shape: {df.shape}')
print(f'\nColumns: {df.columns.tolist()}')

if 'p_over' in df.columns and 'proj_lambda' in df.columns:
    print(f'\nSample of predictions:')
    print(df[['player', 'market', 'line', 'proj_lambda', 'p_over', 'ev', 'book']].head(30).to_string())
    
    print(f'\n\n' + '='*80)
    print('VARIABILITY ANALYSIS:')
    print('='*80)
    
    # Group by market
    for market in df['market'].unique():
        market_df = df[df['market'] == market]
        print(f'\n{market}:')
        print(f'  proj_lambda - Mean: {market_df["proj_lambda"].mean():.4f}, Std: {market_df["proj_lambda"].std():.4f}, Range: {market_df["proj_lambda"].max() - market_df["proj_lambda"].min():.4f}')
        print(f'  p_over      - Mean: {market_df["p_over"].mean():.4f}, Std: {market_df["p_over"].std():.4f}, Range: {market_df["p_over"].max() - market_df["p_over"].min():.4f}')
        print(f'  ev          - Mean: {market_df["ev"].mean():.4f}, Std: {market_df["ev"].std():.4f}, Range: {market_df["ev"].max() - market_df["ev"].min():.4f}')
        
        if market_df["p_over"].std() < 0.05:
            print(f'  ❌ WARNING: Very low variability in p_over!')
else:
    print('Missing expected columns!')
    print('Available columns:', df.columns.tolist())
