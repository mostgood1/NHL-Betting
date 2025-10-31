import sys, os, json, glob, math
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from nhl_betting.utils.io import PROC_DIR


def _pois_to_prob(lam: float) -> float:
    if lam is None or not math.isfinite(lam) or lam < 0:
        return 0.0
    return 1.0 - math.exp(-float(lam))


def _resolve_f10_factor():
    # use the same precedence as core, but keep default 0.55
    try:
        ev = os.getenv('F10_EARLY_FACTOR')
        if ev is not None:
            return float(ev)
    except Exception:
        pass
    try:
        cal = PROC_DIR / 'first10_eval.json'
        if cal.exists():
            obj = json.loads(cal.read_text(encoding='utf-8'))
            s = obj.get('p1_scale')
            if s is not None:
                return float(s)
    except Exception:
        pass
    try:
        cal = PROC_DIR / 'model_calibration.json'
        if cal.exists():
            obj = json.loads(cal.read_text(encoding='utf-8'))
            s = obj.get('f10_early_factor')
            if s is not None:
                return float(s)
    except Exception:
        pass
    return 0.55


def main(argv):
    start = argv[1] if len(argv) >= 2 else '2023-10-01'
    end = argv[2] if len(argv) >= 3 else datetime.today().strftime('%Y-%m-%d')
    # Load team rates
    tr_path = PROC_DIR / 'first10_team_rates.json'
    if not tr_path.exists():
        print('[err] team rates not found, build them first with build_first10_team_rates.py')
        return 2
    TEAM = json.loads(tr_path.read_text(encoding='utf-8'))
    # Load dataset
    paths = sorted(glob.glob(str(PROC_DIR / 'predictions_*.csv')))
    rows = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            ok = df['date'].between(start, end) if 'date' in df.columns else pd.Series([False]*len(df))
            df = df[ok]
            m = df['result_first10'].astype(str).str.lower().isin(['yes','no']) if 'result_first10' in df.columns else []
            df = df[m].copy()
            if df.empty:
                continue
            rows.append(df[['date','home','away','result_first10','period1_home_proj','period1_away_proj']])
        except Exception:
            continue
    if not rows:
        print('[warn] no rows')
        return 0
    data = pd.concat(rows, ignore_index=True)
    data = data.drop_duplicates(subset=['date','home','away'])
    y = data['result_first10'].str.lower().eq('yes').astype(int).values
    # Compute p1-based probs
    f = _resolve_f10_factor()
    h1 = pd.to_numeric(data['period1_home_proj'], errors='coerce').fillna(0.0).values
    a1 = pd.to_numeric(data['period1_away_proj'], errors='coerce').fillna(0.0).values
    p_p1 = 1.0 - np.exp(-f*(h1 + a1))
    # Team average probs
    def team_rate(t):
        obj = TEAM.get(str(t)) or {}
        r = obj.get('yes_rate')
        try:
            return float(r) if r is not None else float('nan')
        except Exception:
            return float('nan')
    th = data['home'].map(team_rate).astype(float).values
    ta = data['away'].map(team_rate).astype(float).values
    p_team = np.maximum(0.0, np.minimum(1.0, 0.5*(np.nan_to_num(th, nan=np.nanmean(th)) + np.nan_to_num(ta, nan=np.nanmean(ta)))))
    # Grid search alpha
    alphas = np.linspace(0.0, 1.0, 41)
    best = None
    def logloss(y_true, p):
        eps = 1e-9
        p = np.clip(p, eps, 1 - eps)
        return float(-(y_true*np.log(p) + (1-y_true)*np.log(1-p)).mean())
    for a in alphas:
        p = a*p_p1 + (1-a)*p_team
        ll = logloss(y, p)
        br = float(((p - y)**2).mean())
        if (best is None) or (ll < best['logloss']):
            best = {'alpha': float(a), 'logloss': ll, 'brier': br}
    # Write to model_calibration.json
    cal_path = PROC_DIR / 'model_calibration.json'
    obj = {}
    if cal_path.exists():
        try:
            obj = json.loads(cal_path.read_text(encoding='utf-8'))
        except Exception:
            obj = {}
    obj['f10_blend_alpha'] = best['alpha'] if best else 0.7
    obj['meta_f10_blend'] = {'start': start, 'end': end, **(best or {})}
    with open(cal_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    print('[ok] calibrated f10_blend_alpha =', obj['f10_blend_alpha'])
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
