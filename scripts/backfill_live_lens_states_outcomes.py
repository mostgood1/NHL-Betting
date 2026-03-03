"""Backfill final outcomes into Live Lens state snapshots.

Reads JSONL rows written by /v1/live-lens persistence:
  data/processed/live_lens/live_lens_states_YYYY-MM-DD.jsonl

Writes labeled JSONL rows:
  data/processed/live_lens/live_lens_states_labeled_YYYY-MM-DD.jsonl

Each output row includes:
  home_goals_final, away_goals_final, home_win (0/1), final_state

This script is best-effort and safe to run in daily_update.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import requests

from nhl_betting.utils.io import PROC_DIR


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _fetch_final_from_statsapi(game_pk: int, timeout: float = 7.0) -> Optional[Dict[str, Any]]:
    """Fetch final state and score from NHL StatsAPI live feed."""
    url = f"https://statsapi.web.nhl.com/api/v1/game/{int(game_pk)}/feed/live"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        obj = r.json()
    except Exception:
        return None

    try:
        status = (((obj or {}).get("gameData") or {}).get("status") or {})
        abstract = str(status.get("abstractGameState") or "").upper()
        detailed = str(status.get("detailedState") or "")

        ls = (((obj or {}).get("liveData") or {}).get("linescore") or {})
        teams = (ls.get("teams") or {})
        h = ((teams.get("home") or {}).get("goals"))
        a = ((teams.get("away") or {}).get("goals"))

        if h is None or a is None:
            return None

        # Consider final-ish states.
        is_final = abstract in {"FINAL", "OFF", "GAMEOVER"} or ("Final" in detailed)
        # StatsAPI sometimes keeps LIVE but with Final state in detailed.
        if not is_final:
            # still return scores/state for logging, but mark not final
            return {"final": False, "home": int(h), "away": int(a), "state": detailed or abstract}

        return {"final": True, "home": int(h), "away": int(a), "state": detailed or abstract}
    except Exception:
        return None


def _daterange(start: str, end: str) -> list[str]:
    s0 = datetime.fromisoformat(start).date()
    s1 = datetime.fromisoformat(end).date()
    out = []
    cur = s0
    while cur <= s1:
        out.append(cur.isoformat())
        cur = cur + timedelta(days=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="Single date YYYY-MM-DD")
    ap.add_argument("--start", default="", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", default="", help="End date YYYY-MM-DD")
    ap.add_argument("--snap-dir", default=str(PROC_DIR / "live_lens"))
    ap.add_argument("--sleep", type=float, default=0.10)
    ap.add_argument("--timeout", type=float, default=7.0)

    args = ap.parse_args()

    if args.date:
        dates = [str(args.date).strip()]
    else:
        end = str(args.end or "").strip()
        start = str(args.start or "").strip()
        if not start or not end:
            # default: last 7 days up to today
            today = datetime.utcnow().date().isoformat()
            end = end or today
            start = start or (datetime.utcnow().date() - timedelta(days=7)).isoformat()
        dates = _daterange(start, end)

    snap_dir = Path(args.snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    # cache final lookups per run
    finals: Dict[int, Optional[Dict[str, Any]]] = {}

    n_in = 0
    n_out = 0
    n_labeled = 0

    for d in dates:
        in_path = snap_dir / f"live_lens_states_{d}.jsonl"
        if (not in_path.exists()) or in_path.stat().st_size <= 0:
            continue

        rows_out: list[dict] = []

        for rec in _iter_jsonl(in_path):
            n_in += 1
            gpk = rec.get("gamePk")
            try:
                gpk_i = int(gpk)
            except Exception:
                rows_out.append(rec)
                continue

            fin = finals.get(gpk_i)
            if fin is None and gpk_i not in finals:
                fin = _fetch_final_from_statsapi(gpk_i, timeout=float(args.timeout))
                finals[gpk_i] = fin
                if args.sleep and args.sleep > 0:
                    time.sleep(float(args.sleep))

            if isinstance(fin, dict) and fin.get("home") is not None and fin.get("away") is not None:
                rec = dict(rec)
                rec["home_goals_final"] = int(fin.get("home"))
                rec["away_goals_final"] = int(fin.get("away"))
                rec["home_win"] = int(int(fin.get("home")) > int(fin.get("away")))
                rec["final_state"] = str(fin.get("state") or "")
                rec["final"] = bool(fin.get("final"))
                n_labeled += 1

            rows_out.append(rec)
            n_out += 1

        out_path = snap_dir / f"live_lens_states_labeled_{d}.jsonl"
        _write_jsonl(out_path, rows_out)
        latest_path = snap_dir / f"live_lens_states_labeled_{d}_latest.json"
        latest_obj = {
            "date": d,
            "n": len(rows_out),
            "labeled": sum(1 for r in rows_out if r.get("home_goals_final") is not None and r.get("away_goals_final") is not None),
        }
        latest_path.write_text(json.dumps(latest_obj, indent=2), encoding="utf-8")

    print(f"[live_lens_outcomes] in={n_in} out={n_out} labeled={n_labeled} dates={len(dates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
