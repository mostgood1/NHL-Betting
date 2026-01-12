import json
import itertools
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List
from pathlib import Path
import sys

import pandas as pd

# Ensure repo root is on sys.path for package imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Use the CLI functions directly
from nhl_betting.cli import game_backtest_sim
from nhl_betting.utils.io import PROC_DIR


def _read_backtest_json(start: str, end: str) -> Dict[str, Any]:
    path = PROC_DIR / f"sim_backtest_{start}_to_{end}.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _score(res: Dict[str, Any]) -> Tuple[float, float]:
    """Return (totals_cal_brier, totals_raw_brier) from backtest result; None as large numbers if missing."""
    cal = res.get("totals_cal", {}) or {}
    raw = res.get("totals_raw", {}) or {}
    b_cal = float(cal.get("brier", 1e9)) if cal.get("brier") is not None else 1e9
    b_raw = float(raw.get("brier", 1e9)) if raw.get("brier") is not None else 1e9
    return (b_cal, b_raw)


def run_sweep(
    start: str,
    end: str,
    n_sims: int = 6000,
    prefer_simulations: bool = False,
    grid: Dict[str, List[float]] | None = None,
) -> Dict[str, Any]:
    """Run a parameter sweep and return best configuration by calibrated totals brier.

    grid keys: 'totals_refs_gamma','totals_xg_gamma','totals_penalty_gamma','totals_goalie_form_gamma','totals_rolling_pace_gamma','totals_fatigue_beta'
    """
    if grid is None:
        grid = {
            "totals_refs_gamma": [0.02, 0.03, 0.05],
            "totals_xg_gamma": [0.02, 0.03, 0.05],
            "totals_penalty_gamma": [0.06, 0.08, 0.10],
            "totals_goalie_form_gamma": [0.01, 0.02, 0.03],
            "totals_rolling_pace_gamma": [0.05, 0.10, 0.15],
            "totals_fatigue_beta": [0.05, 0.08, 0.10],
        }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    best = None
    best_score = (1e9, 1e9)
    history: List[Dict[str, Any]] = []
    for idx, vals in enumerate(combos, start=1):
        params = dict(zip(keys, vals))
        try:
            # Execute backtest using CLI function
            game_backtest_sim(
                start=start,
                end=end,
                n_sims=n_sims,
                use_calibrated=True,
                prefer_simulations=prefer_simulations,
                sim_overdispersion_k=0.8,
                sim_shared_k=0.4,
                sim_empty_net_p=0.25,
                sim_empty_net_two_goal_scale=0.5,
                totals_pace_alpha=0.15,
                totals_goalie_beta=0.10,
                totals_fatigue_beta=params.get("totals_fatigue_beta", 0.0),
                totals_rolling_pace_gamma=params.get("totals_rolling_pace_gamma", 0.0),
                totals_pp_gamma=0.0,
                totals_pk_beta=0.0,
                totals_penalty_gamma=params.get("totals_penalty_gamma", 0.0),
                totals_xg_gamma=params.get("totals_xg_gamma", 0.0),
                totals_refs_gamma=params.get("totals_refs_gamma", 0.0),
                totals_goalie_form_gamma=params.get("totals_goalie_form_gamma", 0.0),
            )
            res = _read_backtest_json(start, end)
            s = _score(res)
            row = {"params": params, "score": {"totals_cal_brier": s[0], "totals_raw_brier": s[1]}}
            history.append(row)
            if s < best_score:  # tuple compare prefers cal then raw
                best_score = s
                best = params.copy()
            print(f"[sweep] {idx}/{len(combos)} params={params} score={s}")
        except Exception as e:
            print(f"[sweep] failed for {params}: {e}")
            continue
    out = {
        "start": start,
        "end": end,
        "best": best,
        "best_score": {"totals_cal_brier": best_score[0], "totals_raw_brier": best_score[1]},
        "history": history,
    }
    # Write config file for daily workflow consumption
    cfg_path = PROC_DIR / "totals_multipliers_config.json"
    try:
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[sweep] wrote best config to {cfg_path}")
    except Exception as e:
        print(f"[sweep] failed to write config: {e}")
    return out


if __name__ == "__main__":
    # Default: last 30 days
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    run_sweep(start=start, end=end, n_sims=6000, prefer_simulations=False)
