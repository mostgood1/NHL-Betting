from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
import sys

# Ensure repo root is on sys.path for package imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nhl_betting.cli import game_simulate


def run_range(start: str, end: str, n_sims: int = 20000,
              sim_overdispersion_k: float = 0.8,
              sim_shared_k: float = 0.4,
              sim_empty_net_p: float = 0.25,
              sim_empty_net_two_goal_scale: float = 0.5,
              totals_pace_alpha: float = 0.15,
              totals_goalie_beta: float = 0.10,
              totals_fatigue_beta: float = 0.08,
              totals_rolling_pace_gamma: float = 0.10,
              totals_pp_gamma: float = 0.0,
              totals_pk_beta: float = 0.0,
              totals_penalty_gamma: float = 0.08,
              totals_xg_gamma: float = 0.03,
              totals_refs_gamma: float = 0.03,
              totals_goalie_form_gamma: float = 0.02):
    """Generate simulations_{date}.csv for each ET date in [start, end]."""
    cur = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    while cur <= end_dt:
        d = cur.strftime('%Y-%m-%d')
        print(f"[simulate_range] simulating {d} …")
        try:
            game_simulate(
                date=d,
                odds_source='oddsapi',
                snapshot=None,
                odds_regions='us',
                odds_markets='h2h,totals,spreads',
                n_sims=n_sims,
                sim_overdispersion_k=sim_overdispersion_k,
                sim_shared_k=sim_shared_k,
                sim_empty_net_p=sim_empty_net_p,
                sim_empty_net_two_goal_scale=sim_empty_net_two_goal_scale,
                totals_pace_alpha=totals_pace_alpha,
                totals_goalie_beta=totals_goalie_beta,
                totals_fatigue_beta=totals_fatigue_beta,
                totals_rolling_pace_gamma=totals_rolling_pace_gamma,
                totals_pp_gamma=totals_pp_gamma,
                totals_pk_beta=totals_pk_beta,
                totals_penalty_gamma=totals_penalty_gamma,
                totals_xg_gamma=totals_xg_gamma,
                totals_refs_gamma=totals_refs_gamma,
                totals_goalie_form_gamma=totals_goalie_form_gamma,
            )
        except Exception as e:
            print(f"[simulate_range] failed for {d}: {e}")
        cur = cur + timedelta(days=1)


if __name__ == '__main__':
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    run_range(start, end)
