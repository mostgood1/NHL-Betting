import json
import itertools
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

# Repo paths
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"


def read_summary(start: str, end: str, prefix: str = "sim_daily") -> Dict[str, Any]:
    path = PROC_DIR / f"{prefix}_props_backtest_sim_summary_{start}_to_{end}.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def score(summary: Dict[str, Any]) -> Tuple[float, float]:
    """Return (overall_brier, overall_acc) with fallbacks."""
    overall = summary.get("overall", {}) or {}
    brier = overall.get("brier")
    acc = overall.get("accuracy")
    b = float(brier) if brier is not None else 1e9
    a = float(acc) if acc is not None else 0.0
    return (b, a)


def run_sim_for_range(start: str, end: str, gammas: Dict[str, float], markets: str = "SOG,GOALS,ASSISTS,POINTS,SAVES,BLOCKS") -> None:
    """Re-simulate props for each day in window using supplied gamma params."""
    cur = datetime.strptime(start, "%Y-%m-%d"); end_dt = datetime.strptime(end, "%Y-%m-%d")
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        try:
            args = [
                str(Path(REPO_ROOT / ".venv" / "Scripts" / "python.exe")), "-m", "nhl_betting.cli", "props-simulate",
                "--date", d,
                "--markets", markets,
                "--n-sims", "16000",
                "--sim-shared-k", "1.2",
                "--props-xg-gamma", str(gammas.get("props_xg_gamma", 0.02)),
                "--props-penalty-gamma", str(gammas.get("props_penalty_gamma", 0.06)),
                "--props-goalie-form-gamma", str(gammas.get("props_goalie_form_gamma", 0.02)),
                "--props-strength-gamma", str(gammas.get("props_strength_gamma", 0.04)),
            ]
            subprocess.run(args, check=False)
        except Exception:
            pass
        cur += timedelta(days=1)


def run_backtest(start: str, end: str, prefix: str = "sim_daily", markets: str = "SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS") -> Dict[str, Any]:
    try:
        args = [
            str(Path(REPO_ROOT / ".venv" / "Scripts" / "python.exe")), "-m", "nhl_betting.cli", "props-backtest-from-simulations",
            "--start", start, "--end", end,
            "--stake", "100",
            "--markets", markets,
            "--min-ev", "-1",
            "--out-prefix", prefix,
        ]
        subprocess.run(args, check=False)
    except Exception:
        pass
    return read_summary(start, end, prefix=prefix)


def sweep(start: str, end: str, markets: str = "SOG,GOALS,ASSISTS,POINTS,SAVES,BLOCKS") -> Dict[str, Any]:
    """Grid-search props sim gamma parameters over window; return best and history."""
    grid: Dict[str, List[float]] = {
        "props_xg_gamma": [0.01, 0.02, 0.03],
        "props_penalty_gamma": [0.04, 0.06, 0.08],
        "props_goalie_form_gamma": [0.01, 0.02, 0.03],
        "props_strength_gamma": [0.02, 0.04, 0.06],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    best = None
    best_score = (1e9, 0.0)
    history: List[Dict[str, Any]] = []
    for idx, vals in enumerate(combos, start=1):
        params = dict(zip(keys, vals))
        # Simulate and backtest
        run_sim_for_range(start, end, params, markets=markets)
        summ = run_backtest(start, end, prefix="sim_daily", markets=markets)
        b, a = score(summ)
        row = {"params": params, "score": {"overall_brier": b, "overall_acc": a}, "summary": summ}
        history.append(row)
        if (b < best_score[0]) or (b == best_score[0] and a > best_score[1]):
            best = params.copy()
            best_score = (b, a)
        print(f"[props-sweep] {idx}/{len(combos)} params={params} score=(brier={b:.4f}, acc={a:.4f})")
    out = {
        "start": start,
        "end": end,
        "best": best,
        "best_score": {"overall_brier": best_score[0], "overall_acc": best_score[1]},
        "history": history,
    }
    # Write config file for daily workflow consumption
    cfg_path = PROC_DIR / "props_sim_multipliers_config.json"
    try:
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[props-sweep] wrote best config to {cfg_path}")
    except Exception as e:
        print(f"[props-sweep] failed to write config: {e}")
    return out


if __name__ == "__main__":
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    sweep(start=start, end=end)
