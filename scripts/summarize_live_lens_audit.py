"""Summarize /v1/live-lens payload into a compact audit CSV.

Intended for post-game analysis to tune live-lens heuristics and sim calibration.
Fetches play-by-play context even for non-live games (audit mode).

Example:
  python scripts/summarize_live_lens_audit.py --date 2026-02-25

Output:
  data/processed/live_lens_audit_<date>.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


# Ensure repo root is on sys.path for package imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _sum_player_int(arr, key: str):
    try:
        if not isinstance(arr, list) or not arr:
            return None
        total = 0
        seen = False
        for r in arr:
            if not isinstance(r, dict):
                continue
            if r.get(key) is None:
                continue
            try:
                total += int(r.get(key) or 0)
                seen = True
            except Exception:
                continue
        return int(total) if seen else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--regions", default="us")
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--timeout-sec", type=float, default=2.0)
    args = ap.parse_args()

    # Keep network bounded; this script is for local analysis.
    os.environ["LIVE_LENS_ODDS_TIMEOUT_SEC"] = str(args.timeout_sec)
    os.environ["LIVE_LENS_SCHEDULE_TIMEOUT_SEC"] = str(args.timeout_sec)
    os.environ["LIVE_LENS_PBP_TIMEOUT_SEC"] = str(args.timeout_sec)
    os.environ.setdefault("ODDS_API_KEY", "")
    os.environ.setdefault("THE_ODDS_API_KEY", "")

    from fastapi.testclient import TestClient
    from nhl_betting.web.app import app

    client = TestClient(app)
    url = (
        f"/v1/live-lens/{args.date}"
        f"?include_non_live=1"
        f"&include_pbp=1"
        f"&inplay=0"
        f"&regions={args.regions}"
        f"&best={1 if args.best else 0}"
    )

    resp = client.get(url)
    if resp.status_code != 200:
        print("status", resp.status_code)
        print(resp.text[:2000])
        return 2

    obj = resp.json()
    games = obj.get("games") or []

    rows: list[dict] = []
    for g in games:
        away = g.get("away")
        home = g.get("home")
        st = g.get("gameState") or g.get("state")

        score = g.get("score") or {}
        away_goals = score.get("away")
        home_goals = score.get("home")
        try:
            total_goals = int(away_goals) + int(home_goals)
        except Exception:
            total_goals = None
        try:
            margin = int(home_goals) - int(away_goals)
        except Exception:
            margin = None

        lens = g.get("lens") or {}
        totals = lens.get("totals") if isinstance(lens, dict) else None
        away_sog = None
        home_sog = None
        try:
            if isinstance(totals, dict):
                away_sog = (totals.get("away") or {}).get("sog")
                home_sog = (totals.get("home") or {}).get("sog")
        except Exception:
            pass

        # Backfill SOG from players when team totals are missing.
        try:
            if (away_sog is None or home_sog is None) and isinstance(lens, dict):
                players = lens.get("players")
                if isinstance(players, dict):
                    away_sog = away_sog if away_sog is not None else _sum_player_int(players.get("away"), "s")
                    home_sog = home_sog if home_sog is not None else _sum_player_int(players.get("home"), "s")
        except Exception:
            pass

        goalies = lens.get("goalies") if isinstance(lens, dict) else None
        away_sv = None
        home_sv = None
        try:
            if isinstance(goalies, dict):
                if isinstance(goalies.get("away"), list) and goalies.get("away"):
                    away_sv = (goalies.get("away")[0] or {}).get("sv_pct")
                if isinstance(goalies.get("home"), list) and goalies.get("home"):
                    home_sv = (goalies.get("home")[0] or {}).get("sv_pct")
        except Exception:
            pass

        pre = g.get("pregame") or {}
        model_total = _safe_float(pre.get("model_total"))
        model_spread = _safe_float(pre.get("model_spread"))

        guidance = g.get("guidance") or {}
        sog_total = guidance.get("sog_total")
        away_att = guidance.get("away_att")
        home_att = guidance.get("home_att")
        att_pace_60 = guidance.get("att_pace_60")
        away_xg = guidance.get("away_xg_proxy")
        home_xg = guidance.get("home_xg_proxy")

        total_err = None
        if model_total is not None and total_goals is not None:
            total_err = float(total_goals) - float(model_total)

        spread_err = None
        if model_spread is not None and margin is not None:
            spread_err = float(margin) - float(model_spread)

        rows.append(
            {
                "date": args.date,
                "away": away,
                "home": home,
                "gameState": st,
                "away_goals": away_goals,
                "home_goals": home_goals,
                "total_goals": total_goals,
                "margin_home_minus_away": margin,
                "model_total": model_total,
                "model_spread_home_minus_away": model_spread,
                "total_err_actual_minus_model": total_err,
                "spread_err_actual_minus_model": spread_err,
                "away_sog": away_sog,
                "home_sog": home_sog,
                "sog_total": sog_total,
                "away_sv_pct": away_sv,
                "home_sv_pct": home_sv,
                "away_att": away_att,
                "home_att": home_att,
                "att_pace_60": att_pace_60,
                "away_xg_proxy": away_xg,
                "home_xg_proxy": home_xg,
            }
        )

    out_dir = _REPO_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"live_lens_audit_{args.date}.csv"

    fieldnames = list(rows[0].keys()) if rows else ["date"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("ok", bool(obj.get("ok")), "games", len(games))
    print("wrote", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
