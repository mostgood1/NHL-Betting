from __future__ import annotations

import sys
import math
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.utils.io import PROC_DIR


def _safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _collect_rows() -> List[Tuple[float, int]]:
    rows: List[Tuple[float, int]] = []
    preds = sorted(PROC_DIR.glob("predictions_*.csv"))
    for p in preds:
        try:
            df = pd.read_csv(p, usecols=[
                "period1_home_proj","period1_away_proj","result_first10"
            ])
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            h1 = _safe_float(r.get("period1_home_proj"))
            a1 = _safe_float(r.get("period1_away_proj"))
            if h1 is None or a1 is None or h1 < 0 or a1 < 0:
                continue
            l = h1 + a1
            if not (l is not None and l >= 0 and math.isfinite(l)):
                continue
            res = str(r.get("result_first10") or "").strip().lower()
            if res not in ("yes","no"):
                continue
            y = 1 if res == "yes" else 0
            rows.append((l, y))
    return rows


def _loglik_for_factor(f: float, obs: List[Tuple[float, int]]) -> float:
    if f <= 0 or not math.isfinite(f):
        return -1e300
    ll = 0.0
    eps = 1e-12
    for l, y in obs:
        lam = f * l
        if lam < 0 or not math.isfinite(lam):
            return -1e300
        if y == 1:
            # log(1 - exp(-lam))
            p1 = max(eps, 1.0 - math.exp(-lam))
            ll += math.log(p1)
        else:
            # log(exp(-lam)) = -lam
            ll += -lam
    return ll


def calibrate(grid_start: float = 0.1, grid_stop: float = 1.0, grid_step: float = 0.02) -> dict:
    obs = _collect_rows()
    n = len(obs)
    if n == 0:
        return {"status": "empty", "n": 0}
    best_f = None
    best_ll = -1e300
    f = grid_start
    while f <= grid_stop + 1e-12:
        ll = _loglik_for_factor(f, obs)
        if ll > best_ll:
            best_ll = ll
            best_f = f
        f += grid_step
    # Compute simple diagnostics at best_f
    def _p_hat(l):
        lam = best_f * l
        return max(0.0, min(1.0, 1.0 - math.exp(-lam)))
    brier = 0.0
    for l, y in obs:
        ph = _p_hat(l)
        brier += (ph - y) ** 2
    brier /= n
    out = {
        "status": "ok",
        "f10_early_factor": best_f,
        "log_likelihood": best_ll,
        "n": n,
        "brier": brier,
        "grid": {"start": grid_start, "stop": grid_stop, "step": grid_step},
    }
    # Merge into model_calibration.json
    path = PROC_DIR / "model_calibration.json"
    try:
        if path.exists():
            base = json.loads(path.read_text(encoding="utf-8"))
        else:
            base = {}
    except Exception:
        base = {}
    base["f10_early_factor"] = best_f
    base["meta_f10"] = {k: v for k, v in out.items() if k != "status"}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2)
        out["written_to"] = str(path)
    except Exception as e:
        out["write_error"] = str(e)
    return out


def main():
    res = calibrate()
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
