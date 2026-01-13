import sys
import subprocess
from datetime import date, datetime, timedelta
from typing import Optional


def run(start: Optional[str] = None, end: Optional[str] = None,
        markets: str = "SAVES,BLOCKS",
        candidate_lines: str = "SAVES=24.5,26.5,28.5,30.5;BLOCKS=1.5,2.5,3.5",
        n_sims: int = 12000,
        sim_shared_k: float = 1.2,
        props_xg_gamma: float = 0.02,
        props_penalty_gamma: float = 0.06,
        props_goalie_form_gamma: float = 0.02) -> None:
    today = date.today()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date() if end else today
    start_dt = datetime.strptime(start, "%Y-%m-%d").date() if start else (end_dt - timedelta(days=30))
    cur = start_dt
    cnt = 0
    while cur <= end_dt:
        d = cur.strftime('%Y-%m-%d')
        print('[nolines] sim', d)
        # Execute via module to avoid import/path issues
        cmd = [sys.executable, '-m', 'nhl_betting.cli', 'props-simulate-unlined', '--date', d,
               '--markets', markets, '--candidate-lines', candidate_lines,
               '--n-sims', str(n_sims), '--sim-shared-k', str(sim_shared_k),
               '--props-xg-gamma', str(props_xg_gamma), '--props-penalty-gamma', str(props_penalty_gamma),
               '--props-goalie-form-gamma', str(props_goalie_form_gamma)]
        try:
            subprocess.run(cmd, check=True)
            cnt += 1
        except subprocess.CalledProcessError as e:
            print('[warn] nolines sim failed for', d, e)
        cur += timedelta(days=1)
    print('[done]', cnt)


if __name__ == '__main__':
    # Accept optional start/end via argv
    start = None
    end = None
    if len(sys.argv) >= 2:
        start = sys.argv[1]
    if len(sys.argv) >= 3:
        end = sys.argv[2]
    run(start=start, end=end)
