"""Inspect /v1/live-lens output for a given date.

This is a dev/debug helper intended to be run locally:
  python scripts/inspect_live_lens.py --date 2026-02-25

It uses FastAPI TestClient to call the app route without running a server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Ensure repo root is on sys.path for package imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--include-non-live", action="store_true", help="Include FINAL games")
    ap.add_argument("--include-pbp", action="store_true", help="Fetch play-by-play even for non-live games (audit mode)")
    ap.add_argument("--inplay", action="store_true", help="Use inplay odds mode")
    ap.add_argument("--regions", default="us")
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--timeout-sec", type=float, default=1.0)
    ap.add_argument(
        "--disk-status",
        action="store_true",
        help="Call /api/live-lens/disk-status after the request and print where snapshots are written",
    )
    ap.add_argument(
        "--disk-status-write-test",
        action="store_true",
        help="When used with --disk-status, attempt a short write/delete test in the snapshot directory",
    )
    ap.add_argument("--dump-game", default="", help="Optional: dump the full game JSON for the first game whose match contains this substring")
    ap.add_argument("--dump-players", action="store_true", help="Print a small sample of player rows for the dumped game")
    args = ap.parse_args()

    # Bound external calls
    os.environ["LIVE_LENS_ODDS_TIMEOUT_SEC"] = str(args.timeout_sec)
    os.environ["LIVE_LENS_SCHEDULE_TIMEOUT_SEC"] = str(args.timeout_sec)
    os.environ["LIVE_LENS_PBP_TIMEOUT_SEC"] = str(args.timeout_sec)
    # If OddsAPI keys aren't configured, keep them blank (client should fail fast)
    os.environ.setdefault("ODDS_API_KEY", "")
    os.environ.setdefault("THE_ODDS_API_KEY", "")

    from fastapi.testclient import TestClient
    from nhl_betting.web.app import app

    client = TestClient(app)
    url = (
        f"/v1/live-lens/{args.date}"
        f"?include_non_live={1 if args.include_non_live else 0}"
        f"&include_pbp={1 if args.include_pbp else 0}"
        f"&inplay={1 if args.inplay else 0}"
        f"&regions={args.regions}"
        f"&best={1 if args.best else 0}"
    )

    resp = client.get(url)
    print("status", resp.status_code)
    obj = resp.json()
    print("ok", obj.get("ok"), "date", obj.get("date"), "games", len(obj.get("games") or []))

    if args.disk_status:
        try:
            ds = client.get(
                f"/api/live-lens/disk-status?write_test={1 if args.disk_status_write_test else 0}"
            )
            print("disk_status", ds.status_code)
            try:
                print(json.dumps(ds.json(), indent=2)[:8000])
            except Exception:
                print(ds.text[:8000])
        except Exception as e:
            print("disk_status_error", str(e))

    games = obj.get("games") or []
    summ = []
    for g in games:
        away = g.get("away")
        home = g.get("home")
        score = g.get("score") or {}
        lens = g.get("lens") or {}
        sig = g.get("signals") or []
        guidance = g.get("guidance") or {}
        has_pbp_ctx = False
        try:
            if isinstance(guidance, dict):
                has_pbp_ctx = any(k in guidance for k in ("home_att", "away_att", "manpower", "home_xg_proxy", "away_xg_proxy"))
        except Exception:
            has_pbp_ctx = False
        summ.append(
            {
                "match": f"{away}@{home}",
                "state": g.get("state") or g.get("game_state"),
                "score": [score.get("away"), score.get("home")],
                "signals_n": len(sig),
                "signals_head": [s.get("label") for s in sig[:5]],
                "has_pbp": bool(has_pbp_ctx),
                "has_totals": bool(lens.get("totals")),
                "has_players": bool((lens.get("players") or {}).get("away") or (lens.get("players") or {}).get("home")),
            }
        )

    print(json.dumps(summ[:12], indent=2))
    print("games_with_signals", sum(1 for x in summ if (x.get("signals_n") or 0) > 0))

    # Print a small histogram of signal labels
    lbl = {}
    for g in games:
        sig = g.get("signals") or []
        for s in sig:
            lab = str(s.get("label") or "").strip() or "(blank)"
            lbl[lab] = lbl.get(lab, 0) + 1
    top = sorted(lbl.items(), key=lambda kv: kv[1], reverse=True)[:25]
    print("top_signal_labels", json.dumps(top, indent=2))

    # Optional deep dump for a single game
    if args.dump_game:
        needle = str(args.dump_game).strip().lower()
        picked = None
        for g in games:
            away = str(g.get("away") or "")
            home = str(g.get("home") or "")
            match = f"{away}@{home}".lower()
            if needle in match:
                picked = g
                break
        if picked is None and games:
            picked = games[0]

        if picked is not None:
            away = picked.get("away")
            home = picked.get("home")
            print("dump_match", f"{away}@{home}")
            lens = picked.get("lens") or {}
            totals = lens.get("totals")
            periods = lens.get("periods")
            players = lens.get("players") or {}
            goalies = lens.get("goalies") or {}
            print(
                "lens_shapes",
                json.dumps(
                    {
                        "totals_keys": sorted(list((totals or {}).keys())) if isinstance(totals, dict) else None,
                        "periods_n": len(periods) if isinstance(periods, list) else None,
                        "players_away_n": len(players.get("away") or []) if isinstance(players, dict) else None,
                        "players_home_n": len(players.get("home") or []) if isinstance(players, dict) else None,
                        "goalies_away_n": len(goalies.get("away") or []) if isinstance(goalies, dict) else None,
                        "goalies_home_n": len(goalies.get("home") or []) if isinstance(goalies, dict) else None,
                        "signals_n": len(lens.get("signals") or []),
                        "has_pbp": bool(lens.get("pbp")),
                    },
                    indent=2,
                ),
            )
            if args.dump_players and isinstance(players, dict):
                a = players.get("away") or []
                h = players.get("home") or []
                print("players_sample_away", json.dumps(a[:8], indent=2)[:6000])
                print("players_sample_home", json.dumps(h[:8], indent=2)[:6000])

            # Full dump last (can be large)
            print("game_json", json.dumps(picked, indent=2)[:12000])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
