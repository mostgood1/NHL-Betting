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
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure repo root is on sys.path so `import nhl_betting` works
# even when invoked from outside the repository working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.data.nhl_api_web import NHLWebClient
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


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _is_final_state(game_state: Any) -> bool:
    st = str(game_state or "").upper().strip()
    return bool(st) and (st.startswith("FINAL") or st in {"OFF", "GAMEOVER"})


def _load_signal_finals(path: Path) -> dict[int, Dict[str, Any]]:
    out: dict[int, Dict[str, Any]] = {}
    if (not path.exists()) or path.stat().st_size <= 0:
        return out
    for snap in _iter_jsonl(path):
        games = snap.get("games") if isinstance(snap, dict) else None
        if not isinstance(games, list):
            continue
        for g in games:
            if not isinstance(g, dict):
                continue
            if not _is_final_state(g.get("gameState")):
                continue
            game_pk = _safe_int(g.get("gamePk"))
            if game_pk is None:
                continue
            score = g.get("score") if isinstance(g.get("score"), dict) else {}
            home = _safe_int(score.get("home"))
            away = _safe_int(score.get("away"))
            if home is None or away is None:
                continue
            out[int(game_pk)] = {
                "final": True,
                "home": int(home),
                "away": int(away),
                "state": str(g.get("gameState") or "FINAL"),
            }
    return out


def _load_schedule_finals(date: str, client: NHLWebClient) -> dict[int, Dict[str, Any]]:
    out: dict[int, Dict[str, Any]] = {}
    try:
        games = client.schedule_day(date)
    except Exception:
        return out
    for g in games:
        if g.home_goals is None or g.away_goals is None:
            continue
        if not _is_final_state(g.gameState):
            continue
        out[int(g.gamePk)] = {
            "final": True,
            "home": int(g.home_goals),
            "away": int(g.away_goals),
            "state": str(g.gameState or "FINAL"),
        }
    return out


def _extract_final_from_payload(obj: Any) -> Optional[Dict[str, Any]]:
    try:
        if not isinstance(obj, dict):
            return None
        state = str(
            obj.get("gameState")
            or obj.get("gameStatus")
            or obj.get("gameStatusText")
            or obj.get("gameOutcome")
            or ""
        ).strip()

        home = None
        away = None

        home_team = obj.get("homeTeam") if isinstance(obj.get("homeTeam"), dict) else {}
        away_team = obj.get("awayTeam") if isinstance(obj.get("awayTeam"), dict) else {}
        home = _safe_int(home_team.get("score"))
        away = _safe_int(away_team.get("score"))

        if home is None or away is None:
            linescore = obj.get("linescore") if isinstance(obj.get("linescore"), dict) else {}
            teams = linescore.get("teams") if isinstance(linescore.get("teams"), dict) else {}
            if home is None:
                home = _safe_int(((teams.get("home") or {}).get("goals")))
            if away is None:
                away = _safe_int(((teams.get("away") or {}).get("goals")))

        if (home is None or away is None) and isinstance(obj.get("summary"), dict):
            summary = obj.get("summary") or {}
            scoring = summary.get("scoring") if isinstance(summary.get("scoring"), list) else []
            home = 0
            away = 0
            for bucket in scoring:
                if not isinstance(bucket, dict):
                    continue
                for goal in (bucket.get("goals") or []):
                    if not isinstance(goal, dict):
                        continue
                    if goal.get("isHome") is True:
                        home += 1
                    elif goal.get("isHome") is False:
                        away += 1

        if home is None or away is None or not _is_final_state(state):
            return None
        return {"final": True, "home": int(home), "away": int(away), "state": state or "FINAL"}
    except Exception:
        return None


def _fetch_final_from_web(game_pk: int, client: NHLWebClient) -> Optional[Dict[str, Any]]:
    try:
        box = client.boxscore(int(game_pk))
        fin = _extract_final_from_payload(box)
        if isinstance(fin, dict):
            return fin
    except Exception:
        pass
    try:
        landing = client._get(f"/gamecenter/{int(game_pk)}/landing", params=None, retries=2)
        fin = _extract_final_from_payload(landing)
        if isinstance(fin, dict):
            return fin
    except Exception:
        pass
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
    client = NHLWebClient(timeout=float(args.timeout))

    # cache final lookups per run
    finals: Dict[int, Optional[Dict[str, Any]]] = {}

    n_in = 0
    n_out = 0
    n_labeled = 0

    for d in dates:
        in_path = snap_dir / f"live_lens_states_{d}.jsonl"
        if (not in_path.exists()) or in_path.stat().st_size <= 0:
            continue

        signal_path = snap_dir / f"live_lens_signals_{d}.jsonl"
        for game_pk, fin in _load_signal_finals(signal_path).items():
            finals[int(game_pk)] = fin
        for game_pk, fin in _load_schedule_finals(d, client).items():
            finals.setdefault(int(game_pk), fin)

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
                fin = _fetch_final_from_web(gpk_i, client)
                finals[gpk_i] = fin
                if args.sleep and args.sleep > 0:
                    time.sleep(float(args.sleep))

            if isinstance(fin, dict) and bool(fin.get("final")) and fin.get("home") is not None and fin.get("away") is not None:
                rec = dict(rec)
                rec["home_goals_final"] = int(fin.get("home"))
                rec["away_goals_final"] = int(fin.get("away"))
                rec["home_win"] = int(int(fin.get("home")) > int(fin.get("away")))
                rec["final_state"] = str(fin.get("state") or "")
                rec["final"] = True
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
