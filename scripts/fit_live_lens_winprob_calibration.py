"""Fit in-play win-prob calibration for Live Lens.

Primary data source: labeled state snapshots produced by Live Lens and outcome-backfilled:
    data/processed/live_lens/live_lens_states_labeled_YYYY-MM-DD.jsonl

Fallback data source (smaller, edge-triggered only):
    data/processed/live_lens/perf/live_lens_bets_all.jsonl

Fits calibration per segment using a hierarchical time key:
    - prob_source: 'poisson' vs 'logit'
    - full-game 15-second bucket
    - full-game 1-minute bucket
    - REG vs OT phase
    - legacy remaining_min bucket fallback

Calibration types:
    - temp-shift (t,b) on log-odds
    - isotonic regression (PAVA) when enough samples exist

Writes: data/processed/live_lens_winprob_calibration.json
This is safe to run in daily_update.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import numpy as np

# Ensure repo root is on sys.path so `import nhl_betting` works
# even when invoked from outside the repository working directory.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nhl_betting.utils.calibration import (
    BinaryCalibration,
    IsotonicCalibration,
    fit_temp_shift,
    fit_isotonic,
    summarize_binary,
)
from nhl_betting.utils.io import PROC_DIR
from nhl_betting.utils.live_lens_time import live_lens_calibration_segment_candidates


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


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


def _daterange(start: str, end: str) -> list[str]:
    s0 = datetime.fromisoformat(start).date()
    s1 = datetime.fromisoformat(end).date()
    out = []
    cur = s0
    while cur <= s1:
        out.append(cur.isoformat())
        cur = cur + timedelta(days=1)
    return out


def _apply_spec(spec: dict, p: np.ndarray) -> np.ndarray:
    kind = str(spec.get("kind") or "").strip().lower()
    if kind == "isotonic":
        try:
            iso = IsotonicCalibration(x=np.asarray(spec.get("x") or [], dtype=float), y=np.asarray(spec.get("y") or [], dtype=float))
            return iso.apply(p)
        except Exception:
            return p
    # default temp_shift
    try:
        cal = BinaryCalibration(t=float(spec.get("t", 1.0)), b=float(spec.get("b", 0.0)))
        return cal.apply(p)
    except Exception:
        return p


def _fit_spec(prob: Iterable[float], y: Iterable[int], metric: str, min_isotonic: int) -> dict:
    p = np.asarray(list(prob), dtype=float)
    y_arr = np.asarray(list(y), dtype=int)
    if p.size == 0:
        return {"kind": "temp_shift", "t": 1.0, "b": 0.0, "n": 0}

    raw = summarize_binary(y_arr, p)

    # Temp-shift always available
    cal_ts = fit_temp_shift(p, y_arr, metric=str(metric))
    p_ts = cal_ts.apply(p)
    ts = summarize_binary(y_arr, p_ts)

    best = {"kind": "temp_shift", "t": float(cal_ts.t), "b": float(cal_ts.b), "n": int(raw.get("n", int(p.size))), "raw": raw, "post": ts}

    if int(p.size) >= int(min_isotonic):
        try:
            cal_iso = fit_isotonic(p, y_arr)
            p_iso = cal_iso.apply(p)
            iso = summarize_binary(y_arr, p_iso)
            # Choose by metric
            key = "logloss" if str(metric) == "logloss" else "brier"
            if float(iso.get(key, float("inf"))) <= float(ts.get(key, float("inf"))) + 1e-12:
                best = {
                    "kind": "isotonic",
                    "x": [float(x) for x in np.asarray(cal_iso.x, dtype=float).tolist()],
                    "y": [float(v) for v in np.asarray(cal_iso.y, dtype=float).tolist()],
                    "n": int(raw.get("n", int(p.size))),
                    "raw": raw,
                    "post": iso,
                }
        except Exception:
            pass

    return best



def _extract_ml_home_prob(rec: dict) -> Optional[float]:
    """Return P(home wins) implied by this ML signal record."""
    try:
        if str(rec.get("market") or "").upper() != "ML":
            return None
        p = _to_float(rec.get("p_model"))
        if p is None:
            return None
        side = str(rec.get("side") or "").upper().strip()
        if side == "HOME":
            p_home = float(p)
        elif side == "AWAY":
            p_home = float(1.0 - float(p))
        else:
            return None
        return float(max(1e-6, min(1.0 - 1e-6, p_home)))
    except Exception:
        return None


def _extract_home_win_label(rec: dict) -> Optional[int]:
    try:
        if rec.get("final") is False:
            return None
        y = rec.get("home_win")
        if y is not None:
            y_i = int(y)
            if y_i in (0, 1):
                return y_i
        hg = _to_float(rec.get("home_goals_final"))
        ag = _to_float(rec.get("away_goals_final"))
        if hg is None or ag is None:
            return None
        return int(float(hg) > float(ag))
    except Exception:
        return None


def _score_key(rec: dict[str, Any]) -> str:
    try:
        score = rec.get("score")
        if isinstance(score, dict):
            home = score.get("home")
            away = score.get("away")
            return f"{home}-{away}"
        return str(score or "")
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dir", default=str(PROC_DIR / "live_lens"), help="Snapshots directory (default: data/processed/live_lens)")
    ap.add_argument("--start", default="", help="Start date YYYY-MM-DD (default: today-30)")
    ap.add_argument("--end", default="", help="End date YYYY-MM-DD (default: today)")
    ap.add_argument("--fallback-ledger", default=str(PROC_DIR / "live_lens" / "perf" / "live_lens_bets_all.jsonl"), help="Fallback perf ledger JSONL")
    ap.add_argument(
        "--out-json",
        default=str(PROC_DIR / "live_lens_winprob_calibration.json"),
        help="Output calibration JSON (default: data/processed/live_lens_winprob_calibration.json)",
    )
    ap.add_argument("--metric", default="logloss", choices=["logloss", "brier"])
    ap.add_argument("--min-samples", type=int, default=120)
    ap.add_argument("--min-seg-samples", type=int, default=80)
    ap.add_argument("--min-isotonic", type=int, default=800)
    ap.add_argument("--no-dedupe", action="store_true", help="Do not dedupe similar states")

    args = ap.parse_args()
    out_path = Path(args.out_json)

    snap_dir = Path(args.snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    end = str(args.end or "").strip()
    start = str(args.start or "").strip()
    if not end:
        end = datetime.now(timezone.utc).date().isoformat()
    if not start:
        start = (datetime.now(timezone.utc).date() - timedelta(days=30)).isoformat()

    dates = _daterange(start, end)

    dedupe = not bool(args.no_dedupe)

    # Gather dataset from labeled state snapshots
    p_list: list[float] = []
    y_list: list[int] = []
    seg_data: Dict[str, dict] = {}
    seen = set()

    def _add(seg_key: str, p_home: float, y_home: int):
        ent = seg_data.get(seg_key)
        if ent is None:
            ent = {"p": [], "y": []}
            seg_data[seg_key] = ent
        ent["p"].append(float(p_home))
        ent["y"].append(int(y_home))

    used_files = 0
    for d in dates:
        fp = snap_dir / f"live_lens_states_labeled_{d}.jsonl"
        if (not fp.exists()) or fp.stat().st_size <= 0:
            continue
        used_files += 1
        for rec in _iter_jsonl(fp):
            try:
                y_home = _extract_home_win_label(rec)
                if y_home is None:
                    continue
                gpk = rec.get("gamePk")
                guidance = rec.get("guidance") if isinstance(rec.get("guidance"), dict) else {}
                p_home = guidance.get("p_home_win_raw")
                if p_home is None:
                    p_home = guidance.get("p_home_win")
                p_home = _to_float(p_home)
                if p_home is None:
                    continue

                elapsed_min = guidance.get("elapsed_min")
                rm = _to_float(guidance.get("remaining_min"))
                src = str(guidance.get("p_win_prob_source") or "unknown").strip().lower()
                seg_candidates = live_lens_calibration_segment_candidates(
                    src,
                    elapsed_min=elapsed_min,
                    period=rec.get("period"),
                    clock=rec.get("clock"),
                    remaining_min=rm,
                )
                primary_seg_key = seg_candidates[0] if seg_candidates else f"src={src or 'unknown'}"

                if dedupe:
                    key = (gpk, rec.get("period"), _score_key(rec), primary_seg_key)
                    if key in seen:
                        continue
                    seen.add(key)

                p_list.append(float(p_home))
                y_list.append(int(y_home))
                for seg_key in seg_candidates:
                    _add(seg_key, float(p_home), int(y_home))
            except Exception:
                continue

    # Fallback to perf ledger if we have no snapshot data yet.
    fallback_used = False
    if len(p_list) < 20:
        in_path = Path(args.fallback_ledger)
        if in_path.exists() and in_path.stat().st_size > 0:
            fallback_used = True
            for rec in _iter_jsonl(in_path):
                try:
                    if str(rec.get("scope") or "").lower() != "game":
                        continue
                    p_home = _extract_ml_home_prob(rec)
                    y_home = _extract_home_win_label(rec)
                    if p_home is None or y_home is None:
                        continue
                    p_list.append(float(p_home))
                    y_list.append(int(y_home))
                except Exception:
                    continue

    p = np.asarray(p_list, dtype=float)
    y = np.asarray(y_list, dtype=int)

    raw_summary = summarize_binary(y, p)

    fitted = bool(int(len(p_list)) >= int(args.min_samples))
    spec_default: dict = {"kind": "temp_shift", "t": 1.0, "b": 0.0, "n": int(len(p_list)), "raw": raw_summary, "post": raw_summary}
    if fitted:
        spec_default = _fit_spec(p, y, metric=str(args.metric), min_isotonic=int(args.min_isotonic))

    # Fit segments where we have enough data
    seg_specs: Dict[str, dict] = {}
    for k, ent in (seg_data or {}).items():
        try:
            if int(len(ent.get("p") or [])) < int(args.min_seg_samples):
                continue
            seg_specs[k] = _fit_spec(ent.get("p") or [], ent.get("y") or [], metric=str(args.metric), min_isotonic=int(args.min_isotonic))
        except Exception:
            continue

    # For backwards-compatible readers that expect (t,b), expose default temp-shift params when available.
    ml_tb = {"t": 1.0, "b": 0.0}
    if str(spec_default.get("kind")) == "temp_shift":
        ml_tb = {"t": float(spec_default.get("t", 1.0)), "b": float(spec_default.get("b", 0.0))}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj: Dict[str, Any] = {
        "version": 3,
        "moneyline": ml_tb,
        "totals": {"t": 1.0, "b": 0.0},
        "default": spec_default,
        "segments": seg_specs,
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "metric": str(args.metric),
            "dedupe": bool(dedupe),
            "start": start,
            "end": end,
            "snap_dir": str(snap_dir),
            "used_state_files": int(used_files),
            "fallback_used": bool(fallback_used),
            "n": int(len(p_list)),
            "n_segments": int(len(seg_specs)),
            "segment_scheme": [
                "src|phase|t15",
                "src|phase|t1",
                "src|phase",
                "src|rm",
                "src",
            ],
            "raw": raw_summary,
        },
    }
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    print(f"[live_lens_cal] n={len(p_list)} fitted={fitted} default_kind={spec_default.get('kind')}")
    print(f"[live_lens_cal] raw: {raw_summary}")
    try:
        print(f"[live_lens_cal] default_post: {spec_default.get('post')}")
    except Exception:
        pass
    print(f"[live_lens_cal] segments: {len(seg_specs)}")
    print(f"[live_lens_cal] wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
