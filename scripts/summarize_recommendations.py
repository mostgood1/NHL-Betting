import json
import pandas as pd
from pathlib import Path

def main():
    proc = Path('data/processed')
    # Calibration files
    cal = {}
    for name in ['sim_calibration.json','sim_calibration_per_line.json']:
        p = proc / name
        if p.exists():
            try:
                with open(p,'r',encoding='utf-8') as f:
                    cal[name] = json.load(f)
            except Exception as e:
                cal[name] = {'error': str(e)}
    # Recommendations summaries
    summary = {}
    for d in ['2026-01-30','2026-01-31']:
        p = proc / f'props_recommendations_{d}.csv'
        sm = {'exists': p.exists(), 'rows': 0, 'markets': {}, 'ev_min_max': None}
        if p.exists():
            df = pd.read_csv(p)
            sm['rows'] = len(df)
            if 'market' in df.columns:
                sm['markets'] = df['market'].value_counts().to_dict()
            if 'ev' in df.columns and len(df)>0:
                sm['ev_min_max'] = (float(df['ev'].min()), float(df['ev'].max()))
        summary[f'per_market_{d}'] = sm
    for d in ['2026-01-30','2026-01-31']:
        p = proc / f'props_recommendations_sim_{d}.csv'
        sm = {'exists': p.exists(), 'rows': 0, 'markets': {}, 'ev_min_max': None}
        if p.exists():
            df = pd.read_csv(p)
            sm['rows'] = len(df)
            if 'market' in df.columns:
                sm['markets'] = df['market'].value_counts().to_dict()
            if 'ev' in df.columns and len(df)>0:
                sm['ev_min_max'] = (float(df['ev'].min()), float(df['ev'].max()))
        summary[f'sim_backed_{d}'] = sm
    print(json.dumps({'cal': cal, 'summary': summary}, indent=2))

if __name__ == '__main__':
    main()
