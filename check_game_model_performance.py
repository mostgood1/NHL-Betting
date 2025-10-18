import pandas as pd
import numpy as np

# Load reconciliations data
df = pd.read_csv('data/processed/reconciliations_log.csv')

print('='*80)
print('GAME MODEL PERFORMANCE ANALYSIS')
print('='*80)

print(f'\nTotal tracked bets: {len(df)}')
if len(df) > 0:
    print(f'Date range: {df["date"].min()} to {df["date"].max()}')
    
    print('\n' + '='*80)
    print('BREAKDOWN BY MARKET')
    print('='*80)
    market_summary = df.groupby('market').agg({
        'market': 'count',
        'result': lambda x: (x == 'win').sum() if 'result' in df.columns else 0,
    }).rename(columns={'market': 'bets', 'result': 'wins'})
    if 'result' in df.columns:
        market_summary['win_rate'] = market_summary['wins'] / market_summary['bets']
    print(market_summary)
    
    if 'result' in df.columns:
        print('\n' + '='*80)
        print('OVERALL RESULTS')
        print('='*80)
        results = df['result'].value_counts()
        print(results)
        
        wins = (df['result'] == 'win').sum()
        losses = (df['result'] == 'loss').sum()
        pushes = (df['result'] == 'push').sum()
        if wins + losses > 0:
            win_rate = wins / (wins + losses)
            print(f'\nWin Rate: {win_rate:.2%} ({wins}W-{losses}L-{pushes}P)')
    
    if 'pnl' in df.columns or 'roi' in df.columns:
        print('\n' + '='*80)
        print('PROFITABILITY')
        print('='*80)
        if 'pnl' in df.columns:
            total_pnl = df['pnl'].sum()
            print(f'Total P&L: ${total_pnl:.2f}')
        if 'roi' in df.columns:
            avg_roi = df['roi'].mean()
            print(f'Average ROI: {avg_roi:.2%}')
    
    print('\n' + '='*80)
    print('RECENT 10 BETS')
    print('='*80)
    cols = ['date', 'home', 'away', 'market']
    if 'odds' in df.columns:
        cols.append('odds')
    if 'result' in df.columns:
        cols.append('result')
    if 'pnl' in df.columns:
        cols.append('pnl')
    
    recent = df.tail(10)[cols]
    print(recent.to_string(index=False))
    
    print('\n' + '='*80)
    print('NEEDS ATTENTION?')
    print('='*80)
    
    if 'result' in df.columns and len(df) >= 20:
        recent_20 = df.tail(20)
        recent_wins = (recent_20['result'] == 'win').sum()
        recent_losses = (recent_20['result'] == 'loss').sum()
        if recent_wins + recent_losses > 0:
            recent_wr = recent_wins / (recent_wins + recent_losses)
            print(f'Last 20 bets win rate: {recent_wr:.2%}')
            if recent_wr < 0.45:
                print('⚠️  WIN RATE BELOW 45% - Consider retraining or recalibration')
            elif recent_wr < 0.52:
                print('⚠️  WIN RATE BELOW 52% - Monitor closely, may need adjustment')
            else:
                print('✓ Win rate healthy')
    else:
        print('Not enough data for trend analysis (need 20+ bets)')

else:
    print('\nNo reconciliation data found.')
    print('Run some predictions and track results to analyze performance.')
