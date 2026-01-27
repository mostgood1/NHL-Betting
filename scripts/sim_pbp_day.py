from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

# Ensure repo root is on path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.sim.pbp_engine import simulate_game, to_dataframe, build_team_from_roster, build_lines
from nhl_betting.web.teams import get_team_assets

PROC = Path('data/processed')


def _read_predictions(date: str) -> pd.DataFrame:
    p = PROC / f'predictions_{date}.csv'
    if not p.exists():
        raise FileNotFoundError(f"Missing predictions_{date}.csv under data/processed")
    df = pd.read_csv(p)
    if {'home','away'}.issubset(df.columns):
        return df[['home','away']].dropna()
    # Fallback: try abbreviations
    cols = [c for c in df.columns if c.lower() in {'home','away','home_abbr','away_abbr'}]
    if not cols:
        raise ValueError("predictions file missing home/away columns")
    return df


def run_day(date: str, seed: int | None = None) -> pd.DataFrame:
    df = _read_predictions(date)
    out_frames = []
    for _, r in df.iterrows():
        home_n = str(r.get('home') or '').strip()
        away_n = str(r.get('away') or '').strip()
        def _abbr(name: str) -> str:
            if not name:
                return ''
            if len(name) <= 3:
                return name.upper()
            a = get_team_assets(name).get('abbr') if name else None
            return str(a).upper() if a else ''
        home = _abbr(home_n) or str(r.get('home_abbr') or '').strip().upper()
        away = _abbr(away_n) or str(r.get('away_abbr') or '').strip().upper()
        if not home or not away:
            continue
        stats = simulate_game(home_abbr=home, away_abbr=away, date=date, seed=seed)
        # Build a mapping from player name -> team abbr via current rosters
        team_map: dict[str, str] = {}
        # Robust: read roster and assemble for both teams
        th = build_team_from_roster(date, home)
        ta = build_team_from_roster(date, away)
        for p in th.roster:
            team_map[p.name] = th.abbr
        if th.goalie:
            team_map[th.goalie.name] = th.abbr
        for p in ta.roster:
            team_map[p.name] = ta.abbr
        if ta.goalie:
            team_map[ta.goalie.name] = ta.abbr
        df_box = to_dataframe(stats, team_map)
        df_box['home'] = home
        df_box['away'] = away
        df_box['date'] = date
        out_frames.append(df_box)
    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Simulate all games for a date and write props boxscores CSV")
    ap.add_argument('--date', required=True, help="Date YYYY-MM-DD")
    ap.add_argument('--seed', type=int, default=42, help="Random seed")
    ap.add_argument('--out', default=None, help="Override output file path")
    args = ap.parse_args()

    df = run_day(args.date, seed=args.seed)
    if df.empty:
        print("[sim] No games or empty output; nothing written")
        return
    out_path = Path(args.out) if args.out else (PROC / f"props_boxscores_sim_{args.date}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[sim] wrote {out_path} rows={len(df)}")


if __name__ == '__main__':
    main()
