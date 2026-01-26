"""
Run a limited totals multipliers sweep over a fixed 60-day window and
write the best configuration to data/processed/totals_multipliers_config.json.

Window: 2025-11-27 .. 2026-01-25 (ends yesterday)
Grid: 2 levels per parameter (64 combos) for quicker iteration.
"""
from pathlib import Path
from typing import Dict, List

from calibrate_totals_multipliers import run_sweep  # uses nhl_betting.cli under the hood


def main() -> None:
    start = "2025-11-27"
    end = "2026-01-25"
    # 2-level grid for a faster sweep (64 combos)
    grid: Dict[str, List[float]] = {
        "totals_refs_gamma": [0.02, 0.04],
        "totals_xg_gamma": [0.02, 0.04],
        "totals_penalty_gamma": [0.06, 0.10],
        "totals_goalie_form_gamma": [0.01, 0.03],
        "totals_rolling_pace_gamma": [0.05, 0.15],
        "totals_fatigue_beta": [0.05, 0.10],
    }
    # Use fewer sims to speed up the sweep; backtests use more later
    run_sweep(start=start, end=end, n_sims=6000, prefer_simulations=False, grid=grid)


if __name__ == "__main__":
    main()
