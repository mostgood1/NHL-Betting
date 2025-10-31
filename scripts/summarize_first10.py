import sys
import os
from datetime import date as _date
import pandas as pd

# Ensure package import works when run from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nhl_betting.core.recs_shared import recompute_edges_and_recommendations
from nhl_betting.utils.io import PROC_DIR


def main(argv: list[str]) -> int:
    d = None
    if len(argv) >= 2 and argv[1]:
        d = argv[1]
    else:
        d = _date.today().strftime('%Y-%m-%d')
    # Recompute EVs (also writes back p_f10_yes/no if missing)
    _ = recompute_edges_and_recommendations(d)
    pred_path = PROC_DIR / f"predictions_{d}.csv"
    if not pred_path.exists():
        print(f"[err] predictions for {d} not found at {pred_path}")
        return 2
    df = pd.read_csv(pred_path)
    s = df.get('p_f10_yes') if 'p_f10_yes' in df.columns else None
    if s is None:
        print(f"[warn] p_f10_yes not present in predictions for {d}")
        return 0
    s = s.dropna().astype(float)
    if len(s) == 0:
        print(f"[info] no p_f10_yes values for {d}")
        return 0
    print("[date]", d)
    print("[count]", len(s))
    print("[min]", float(s.min()))
    print("[p10]", float(s.quantile(0.10)))
    print("[mean]", float(s.mean()))
    print("[median]", float(s.median()))
    print("[p90]", float(s.quantile(0.90)))
    print("[max]", float(s.max()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
