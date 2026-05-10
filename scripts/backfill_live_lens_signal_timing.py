"""Backfill goal-timing and guidance flow metadata into historical Live Lens signal snapshots.

This enriches saved live_lens_signals_YYYY-MM-DD.jsonl files with timing fields that
can be inferred from the snapshot stream itself:

- driver_meta.last_goal_team
- driver_meta.time_since_last_goal_sec
- driver_meta.score_state_age_sec

It cannot recover full PBP-only context like faceoff or xG proxies, but it gives
historical signal rows enough timing state to support flow-first playoff analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.utils.io import PROC_DIR


def _as_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _daterange(start: str, end: str) -> list[str]:
    start_dt = datetime.fromisoformat(start).date()
    end_dt = datetime.fromisoformat(end).date()
    if end_dt < start_dt:
        raise ValueError("end before start")
    out: list[str] = []
    cur = start_dt
    while cur <= end_dt:
        out.append(cur.isoformat())
        cur = cur + timedelta(days=1)
    return out


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _dedupe_tags(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    arr = values if isinstance(values, list) else [values]
    for value in arr:
        try:
            tag = str(value or "").strip()
        except Exception:
            continue
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def _load_state_guidance_index(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    if (not path.exists()) or path.stat().st_size <= 0:
        return out
    for row in _iter_jsonl(path):
        game_pk = _safe_int(row.get("gamePk"))
        asof_utc = str(row.get("asof_utc") or "").strip()
        guidance = row.get("guidance") if isinstance(row.get("guidance"), dict) else None
        if game_pk is None or not asof_utc or not isinstance(guidance, dict):
            continue
        out[(int(game_pk), asof_utc)] = guidance
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> bool:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        tmp.replace(path)
        return True
    except PermissionError:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def _score_state(home_goals: Any, away_goals: Any) -> Optional[str]:
    hg = _safe_int(home_goals)
    ag = _safe_int(away_goals)
    if hg is None or ag is None:
        return None
    if hg > ag:
        return "home_leading"
    if hg < ag:
        return "away_leading"
    return "tied"


def _snapshot_elapsed_min(game: dict[str, Any]) -> Optional[float]:
    guidance = game.get("guidance") if isinstance(game.get("guidance"), dict) else {}
    elapsed_min = _safe_float(guidance.get("elapsed_min"))
    if elapsed_min is not None:
        return elapsed_min
    for signal in game.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        elapsed_min = _safe_float(signal.get("elapsed_min"))
        if elapsed_min is not None:
            return elapsed_min
    return None


def _apply_timing_meta(
    signal: dict[str, Any],
    game_state: dict[str, Any],
    asof_dt: Optional[datetime],
    *,
    guidance: Optional[dict[str, Any]] = None,
) -> None:
    meta = signal.get("driver_meta") if isinstance(signal.get("driver_meta"), dict) else {}
    guidance = guidance if isinstance(guidance, dict) else {}
    if meta.get("pp_team") in (None, "") and guidance.get("pp_team") not in (None, ""):
        meta["pp_team"] = guidance.get("pp_team")
    if meta.get("pp_sec_remaining_est") is None and guidance.get("pp_sec_remaining_est") is not None:
        meta["pp_sec_remaining_est"] = guidance.get("pp_sec_remaining_est")
    if meta.get("home_empty_net") is None and guidance.get("home_empty_net") is not None:
        meta["home_empty_net"] = guidance.get("home_empty_net")
    if meta.get("away_empty_net") is None and guidance.get("away_empty_net") is not None:
        meta["away_empty_net"] = guidance.get("away_empty_net")
    if meta.get("late_state_mode") in (None, "") and guidance.get("late_state_mode") not in (None, ""):
        meta["late_state_mode"] = guidance.get("late_state_mode")
    if not meta.get("trigger_tags") and guidance.get("projection_driver_tags"):
        meta["trigger_tags"] = _dedupe_tags(guidance.get("projection_driver_tags"))
    pp_team = str(meta.get("pp_team") or "").strip().lower()
    pp_rem = _safe_float(meta.get("pp_sec_remaining_est"))
    if pp_team in {"home", "away"} and pp_rem is not None:
        meta["pp_state_age_sec"] = max(0, min(120, int(round(120.0 - float(pp_rem)))))
    if game_state.get("last_goal_team") is not None:
        meta["last_goal_team"] = game_state.get("last_goal_team")
    if asof_dt is not None:
        last_goal_at = game_state.get("last_goal_at")
        if isinstance(last_goal_at, datetime):
            meta["time_since_last_goal_sec"] = max(0, int((asof_dt - last_goal_at).total_seconds()))
        score_state_started_at = game_state.get("score_state_started_at")
        if isinstance(score_state_started_at, datetime):
            meta["score_state_age_sec"] = max(0, int((asof_dt - score_state_started_at).total_seconds()))
    signal["driver_meta"] = meta


def enrich_signal_snapshots(
    rows: list[dict[str, Any]],
    *,
    state_guidance_by_key: Optional[dict[tuple[int, str], dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    state_by_game: dict[int, dict[str, Any]] = {}
    state_guidance_by_key = state_guidance_by_key if isinstance(state_guidance_by_key, dict) else {}
    for snapshot in rows:
        if not isinstance(snapshot, dict):
            continue
        asof_dt = _as_dt(snapshot.get("asof_utc"))
        asof_utc = str(snapshot.get("asof_utc") or "").strip()
        games = snapshot.get("games") if isinstance(snapshot.get("games"), list) else []
        for game in games:
            if not isinstance(game, dict):
                continue
            game_pk = _safe_int(game.get("gamePk"))
            if game_pk is None:
                continue
            score = game.get("score") if isinstance(game.get("score"), dict) else {}
            home_goals = _safe_int(score.get("home"))
            away_goals = _safe_int(score.get("away"))
            current_score_state = _score_state(home_goals, away_goals)

            game_state = state_by_game.get(game_pk)
            if game_state is None:
                game_state = {
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "score_state": current_score_state,
                    "last_goal_at": None,
                    "last_goal_team": None,
                    "score_state_started_at": None,
                }
                elapsed_min = _snapshot_elapsed_min(game)
                if asof_dt is not None and home_goals == 0 and away_goals == 0 and elapsed_min is not None:
                    game_state["score_state_started_at"] = asof_dt - timedelta(seconds=max(0, int(round(elapsed_min * 60.0))))
                state_by_game[game_pk] = game_state
            else:
                prev_home = _safe_int(game_state.get("home_goals"))
                prev_away = _safe_int(game_state.get("away_goals"))
                delta_home = None if home_goals is None or prev_home is None else int(home_goals) - int(prev_home)
                delta_away = None if away_goals is None or prev_away is None else int(away_goals) - int(prev_away)

                if delta_home is not None and delta_away is not None:
                    if delta_home < 0 or delta_away < 0:
                        game_state["last_goal_at"] = None
                        game_state["last_goal_team"] = None
                        game_state["score_state_started_at"] = None
                    elif delta_home > 0 or delta_away > 0:
                        if asof_dt is not None:
                            game_state["last_goal_at"] = asof_dt
                            game_state["score_state_started_at"] = asof_dt
                        if delta_home > 0 and delta_away == 0:
                            game_state["last_goal_team"] = "home"
                        elif delta_away > 0 and delta_home == 0:
                            game_state["last_goal_team"] = "away"
                        elif delta_home > 0 and delta_away > 0:
                            game_state["last_goal_team"] = "both"
                        else:
                            game_state["last_goal_team"] = None

                game_state["home_goals"] = home_goals
                game_state["away_goals"] = away_goals
                game_state["score_state"] = current_score_state

            for signal in game.get("signals") or []:
                if not isinstance(signal, dict):
                    continue
                guidance = state_guidance_by_key.get((int(game_pk), asof_utc)) if game_pk is not None and asof_utc else None
                _apply_timing_meta(signal, game_state, asof_dt, guidance=guidance)

    return rows


def backfill_signal_file(path: Path) -> bool:
    if (not path.exists()) or path.stat().st_size <= 0:
        return False
    rows = _iter_jsonl(path)
    if not rows:
        return False
    state_path = path.with_name(path.name.replace("signals_", "states_", 1))
    state_guidance_by_key = _load_state_guidance_index(state_path)
    return bool(_write_jsonl(path, enrich_signal_snapshots(rows, state_guidance_by_key=state_guidance_by_key)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill historical Live Lens signal timing metadata")
    parser.add_argument("--date", default="", help="Single date YYYY-MM-DD")
    parser.add_argument("--start", default="", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="", help="End date YYYY-MM-DD")
    parser.add_argument("--all", action="store_true", help="Process all live_lens_signals_*.jsonl files")
    parser.add_argument("--signals-dir", default=str(PROC_DIR / "live_lens"))
    args = parser.parse_args()

    signals_dir = Path(args.signals_dir)
    if args.all:
        paths = sorted(p for p in signals_dir.glob("live_lens_signals_*.jsonl") if p.is_file() and p.stat().st_size > 0)
    elif args.date:
        paths = [signals_dir / f"live_lens_signals_{str(args.date).strip()}.jsonl"]
    else:
        start = str(args.start or "").strip()
        end = str(args.end or "").strip()
        if not start or not end:
            parser.error("Provide --all, or --date, or --start/--end")
        paths = [signals_dir / f"live_lens_signals_{d}.jsonl" for d in _daterange(start, end)]

    touched = 0
    for path in paths:
        if backfill_signal_file(path):
            touched += 1
            print(f"backfilled={path}")
        elif path.exists() and path.stat().st_size > 0:
            print(f"skipped_locked={path}")

    print(f"files_backfilled={touched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())