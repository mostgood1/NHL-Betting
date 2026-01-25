import pandas as pd
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    path = root / 'data/processed/props_recommendations_nolines_2026-01-24.csv'
    df = pd.read_csv(path)
    df['market'] = df['market'].astype(str).str.upper()
    def topn(mkt):
        sub = df[df['market']==mkt].sort_values('chosen_prob', ascending=False).head(10)
        cols = ['player','team','opp','market','line','chosen_prob','p_over']
        return sub[cols]
    print('[TOP SAVES]')
    print(topn('SAVES').to_string(index=False))
    print('\n[TOP BLOCKS]')
    print(topn('BLOCKS').to_string(index=False))

if __name__ == '__main__':
    main()
