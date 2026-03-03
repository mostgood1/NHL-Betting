"""Report Live Lens in-play win-prob performance and drift.

Reads labeled Live Lens state snapshots:
  data/processed/live_lens/live_lens_states_labeled_YYYY-MM-DD.jsonl

Computes logloss/Brier for:
  - raw P(home wins) (guidance.p_home_win_raw)
  - calibrated P(home wins) (applied via live_lens_winprob_calibration.json v2)

Writes:
  - data/processed/live_lens/live_lens_winprob_monitor.json
  - data/processed/live_lens/live_lens_winprob_drift_alert.json

Safe to run in daily_update.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# Ensure repo root is on sys.path so `import nhl_betting` works
# even when invoked from outside the repository working directory.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nhl_betting.utils.io import PROC_DIR


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _clamp_prob(p: float) -> float:
    return float(max(1e-6, min(1.0 - 1e-6, float(p))))


def _sigmoid(z: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(z)))
    except Exception:
        if z >= 0:
            return 1.0
        return 0.0


def _logit(p: float) -> float:
    p = _clamp_prob(p)
    return float(math.log(p / (1.0 - p)))


def _rm_bucket(rm: Optional[float]) -> str:
    if rm is None:
        return "unknown"
    try:
        x = float(rm)
    except Exception:
        return "unknown"
    if x < 0:
        x = 0.0
    if x <= 5:
        return "0-5"
    if x <= 10:
        return "5-10"
    if x <= 20:
        return "10-20"
    if x <= 40:
        return "20-40"
    return "40-60"


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
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


def _read_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _apply_temp_shift(spec: dict, p: float) -> float:
    try:
        t = float(spec.get("t", 1.0))
        b = float(spec.get("b", 0.0))
        if not math.isfinite(t) or abs(t) < 1e-9:
            t = 1.0
        if not math.isfinite(b):
            b = 0.0
        z = _logit(float(p))
        return _sigmoid(z / float(t) + float(b))
    except Exception:
        return float(p)


def _apply_isotonic(spec: dict, p: float) -> float:
    try:
        xs = spec.get("x") or []
        ys = spec.get("y") or []
        if not xs or not ys or len(xs) != len(ys):
            return float(p)
        x = float(p)
        # clamp range
        if x <= float(xs[0]):
            return float(ys[0])
        if x >= float(xs[-1]):
            return float(ys[-1])
        # linear interpolate within segment
        for i in range(1, len(xs)):
            x0 = float(xs[i - 1])
            x1 = float(xs[i])
            if x <= x1:
                y0 = float(ys[i - 1])
                y1 = float(ys[i])
                if abs(x1 - x0) < 1e-12:
                    return float(y1)
                t = (x - x0) / (x1 - x0)
                return float(y0 + t * (y1 - y0))
        return float(ys[-1])
    except Exception:
        return float(p)


def _apply_cal_spec(spec: dict, p: float) -> float:
    kind = str(spec.get("kind") or "temp_shift").strip().lower()
    if kind == "isotonic":
        return _apply_isotonic(spec, p)
    return _apply_temp_shift(spec, p)


def _pick_spec(obj: dict, remaining_min: Optional[float], prob_source: Optional[str]) -> dict:
    default = obj.get("default") if isinstance(obj, dict) else None
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(default, dict):
        default = {"kind": "temp_shift", "t": 1.0, "b": 0.0}

    if not isinstance(segments, dict):
        return default

    src = str(prob_source or "unknown").strip().lower() or "unknown"
    rm_key = _rm_bucket(remaining_min)
    seg_key = f"src={src}|rm={rm_key}"
    spec = segments.get(seg_key)
    if isinstance(spec, dict):
        return spec
    return default


def _logloss(y: int, p: float) -> float:
    p = _clamp_prob(p)
    if int(y) == 1:
        return -math.log(p)
    return -math.log(1.0 - p)


def _brier(y: int, p: float) -> float:
    p = float(p)
    return float((float(y) - p) ** 2)


@dataclass
class Accum:
    n: int = 0
    ll: float = 0.0
    br: float = 0.0

    def add(self, y: int, p: float) -> None:
        self.n += 1
        self.ll += float(_logloss(y, p))
        self.br += float(_brier(y, p))

    def as_dict(self) -> dict:
        if self.n <= 0:
            return {"n": 0, "logloss": None, "brier": None}
        return {
            "n": int(self.n),
            "logloss": float(self.ll / float(self.n)),
            "brier": float(self.br / float(self.n)),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dir", default=str(PROC_DIR / "live_lens"))
    ap.add_argument("--cal-json", default=str(PROC_DIR / "live_lens_winprob_calibration.json"))
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--out-json", default=str(PROC_DIR / "live_lens" / "live_lens_winprob_monitor.json"))
    ap.add_argument("--drift-json", default=str(PROC_DIR / "live_lens" / "live_lens_winprob_drift_alert.json"))
    ap.add_argument("--recent-days", type=int, default=7)
    ap.add_argument("--baseline-days", type=int, default=30)
    ap.add_argument("--logloss-delta", type=float, default=0.03)
    ap.add_argument("--brier-delta", type=float, default=0.01)
    args = ap.parse_args()

    snap_dir = Path(args.snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    end = str(args.end or "").strip()
    start = str(args.start or "").strip()
    if not end:
        end = datetime.now(timezone.utc).date().isoformat()
    if not start:
        start = (datetime.now(timezone.utc).date() - timedelta(days=30)).isoformat()

    cal_obj = _read_json(Path(args.cal_json)) or {"default": {"kind": "temp_shift", "t": 1.0, "b": 0.0}, "segments": {}}

    overall_raw = Accum()
    overall_cal = Accum()
    seg_raw: Dict[str, Accum] = {}
    seg_cal: Dict[str, Accum] = {}

    # For drift windows we use file date (YYYY-MM-DD)
    def _acc_map_get(m: Dict[str, Accum], k: str) -> Accum:
        ent = m.get(k)
        if ent is None:
            ent = Accum()
            m[k] = ent
        return ent

    # Compute drift windows (non-overlapping)
    end_date = datetime.fromisoformat(end).date()
    recent_start = (end_date - timedelta(days=int(args.recent_days) - 1)).isoformat()
    baseline_end = (end_date - timedelta(days=int(args.recent_days))).isoformat()
    baseline_start = (end_date - timedelta(days=int(args.recent_days) + int(args.baseline_days) - 1)).isoformat()

    drift_recent_raw = Accum()
    drift_recent_cal = Accum()
    drift_base_raw = Accum()
    drift_base_cal = Accum()

    dates = _daterange(start, end)

    for d in dates:
        fp = snap_dir / f"live_lens_states_labeled_{d}.jsonl"
        if not fp.exists() or fp.stat().st_size <= 0:
            continue

        for rec in _iter_jsonl(fp):
            try:
                if rec.get("final") is False:
                    continue
            except Exception:
                pass

            y = rec.get("home_win")
            if y is None:
                continue
            try:
                y_i = int(y)
            except Exception:
                continue
            if y_i not in (0, 1):
                continue

            guidance = rec.get("guidance") if isinstance(rec.get("guidance"), dict) else {}
            p_raw0 = guidance.get("p_home_win_raw")
            if p_raw0 is None:
                p_raw0 = guidance.get("p_home_win")
            p_raw = _to_float(p_raw0)
            if p_raw is None:
                continue
            p_raw = _clamp_prob(p_raw)

            rm = _to_float(guidance.get("remaining_min"))
            src = str(guidance.get("p_win_prob_source") or "unknown").strip().lower() or "unknown"
            seg_key = f"src={src}|rm={_rm_bucket(rm)}"

            spec = _pick_spec(cal_obj, rm, src)
            p_cal = _clamp_prob(_apply_cal_spec(spec, p_raw))

            overall_raw.add(y_i, p_raw)
            overall_cal.add(y_i, p_cal)
            _acc_map_get(seg_raw, seg_key).add(y_i, p_raw)
            _acc_map_get(seg_cal, seg_key).add(y_i, p_cal)

            # Drift windows by file date
            if recent_start <= d <= end:
                drift_recent_raw.add(y_i, p_raw)
                drift_recent_cal.add(y_i, p_cal)
            if baseline_start <= d <= baseline_end:
                drift_base_raw.add(y_i, p_raw)
                drift_base_cal.add(y_i, p_cal)

    # Build report
    report = {
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "range": {"start": start, "end": end},
        "overall": {"raw": overall_raw.as_dict(), "calibrated": overall_cal.as_dict()},
        "by_segment": {
            k: {"raw": seg_raw[k].as_dict(), "calibrated": seg_cal[k].as_dict()}
            for k in sorted(set(seg_raw.keys()) | set(seg_cal.keys()))
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Drift alert
    base = drift_base_cal.as_dict()
    recent = drift_recent_cal.as_dict()
    alert = False
    reasons: list[str] = []

    try:
        if base.get("n") and recent.get("n") and int(base["n"]) >= 50 and int(recent["n"]) >= 50:
            ll_delta = float(recent.get("logloss") or 0.0) - float(base.get("logloss") or 0.0)
            br_delta = float(recent.get("brier") or 0.0) - float(base.get("brier") or 0.0)
            if ll_delta >= float(args.logloss_delta):
                alert = True
                reasons.append(f"logloss_delta>={args.logloss_delta:.3f}")
            if br_delta >= float(args.brier_delta):
                alert = True
                reasons.append(f"brier_delta>={args.brier_delta:.3f}")
        else:
            ll_delta = None
            br_delta = None
            reasons.append("insufficient_samples")
    except Exception:
        ll_delta = None
        br_delta = None
        reasons.append("drift_calc_failed")

    drift = {
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "recent": {"start": recent_start, "end": end, "calibrated": recent, "raw": drift_recent_raw.as_dict()},
        "baseline": {"start": baseline_start, "end": baseline_end, "calibrated": base, "raw": drift_base_raw.as_dict()},
        "delta": {"logloss": ll_delta, "brier": br_delta},
        "alert": bool(alert),
        "reasons": reasons,
    }

    drift_path = Path(args.drift_json)
    drift_path.parent.mkdir(parents=True, exist_ok=True)
    drift_path.write_text(json.dumps(drift, indent=2), encoding="utf-8")

    print(f"[live_lens_winprob_monitor] n={overall_cal.n} alert={alert} out={out_path} drift={drift_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
