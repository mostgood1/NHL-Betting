import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.data.team_stats import fetch_team_special_teams, save_team_special_teams
from nhl_betting.data.penalty_rates import fetch_team_penalty_rates, save_team_penalty_rates


def infer_season_code(date_str: str) -> str:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}{start+1}"


def main():
    ap = argparse.ArgumentParser(description="Fetch and cache team PP/PK and penalty exposure for a season")
    ap.add_argument("--season", help="Season code like 20252026 (default inferred from today)")
    args = ap.parse_args()

    season = args.season or infer_season_code(datetime.utcnow().strftime("%Y-%m-%d"))
    print(f"[build] using season {season}")

    # PP/PK
    try:
        st = fetch_team_special_teams(season)
        save_team_special_teams(st, season=season)
        print(f"[pppk] saved team_special_teams_{season}.json ({len(st)} teams)")
    except Exception as e:
        print("[pppk] failed:", e)

    # Penalty exposure
    try:
        pr = fetch_team_penalty_rates(season)
        save_team_penalty_rates(pr, season=season)
        print(f"[pen] saved team_penalty_rates_{season}.json ({len(pr)} teams)")
    except Exception as e:
        print("[pen] failed:", e)


if __name__ == "__main__":
    main()
