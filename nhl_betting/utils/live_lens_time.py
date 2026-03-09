from __future__ import annotations

import math
import re
from typing import Any, Optional


LIVE_LENS_REGULATION_SECONDS = 60 * 60
LIVE_LENS_OT_SECONDS = 5 * 60
LIVE_LENS_MAX_GAME_SECONDS = LIVE_LENS_REGULATION_SECONDS + LIVE_LENS_OT_SECONDS


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def normalize_live_lens_prob_source(prob_source: Any) -> str:
    try:
        s = str(prob_source or "").strip().lower()
    except Exception:
        s = ""
    return s or "unknown"


def parse_live_lens_mmss_clock(clock: Any) -> Optional[int]:
    try:
        s = str(clock or "").strip()
    except Exception:
        return None
    if not s:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    try:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
    except Exception:
        return None
    if minutes < 0 or seconds < 0 or seconds >= 60:
        return None
    return (minutes * 60) + seconds


def coerce_live_lens_elapsed_seconds(
    *,
    elapsed_min: Any = None,
    period: Any = None,
    clock: Any = None,
) -> Optional[float]:
    sec = None
    em = _safe_float(elapsed_min)
    if em is not None:
        sec = float(em) * 60.0
    else:
        period_i = _safe_int(period)
        clock_sec = parse_live_lens_mmss_clock(clock)
        if period_i is None or clock_sec is None:
            return None
        if 1 <= int(period_i) <= 3:
            per_len = 20 * 60
            sec = float((int(period_i) - 1) * per_len + (per_len - clock_sec))
        elif int(period_i) == 4:
            per_len = LIVE_LENS_OT_SECONDS
            sec = float(LIVE_LENS_REGULATION_SECONDS + (per_len - clock_sec))
        elif int(period_i) >= 5:
            sec = float(LIVE_LENS_MAX_GAME_SECONDS) - 1e-9

    if sec is None or (not math.isfinite(sec)):
        return None
    sec = max(0.0, float(sec))
    if sec >= float(LIVE_LENS_MAX_GAME_SECONDS):
        sec = float(LIVE_LENS_MAX_GAME_SECONDS) - 1e-9
    return sec


def coerce_live_lens_elapsed_minutes(
    *,
    elapsed_min: Any = None,
    period: Any = None,
    clock: Any = None,
) -> Optional[float]:
    sec = coerce_live_lens_elapsed_seconds(elapsed_min=elapsed_min, period=period, clock=clock)
    if sec is None:
        return None
    return float(sec) / 60.0


def live_lens_game_phase(
    *,
    elapsed_min: Any = None,
    period: Any = None,
    clock: Any = None,
) -> Optional[str]:
    sec = coerce_live_lens_elapsed_seconds(elapsed_min=elapsed_min, period=period, clock=clock)
    if sec is None:
        return None
    if float(sec) >= float(LIVE_LENS_REGULATION_SECONDS):
        return "OT"
    return "REG"


def _format_elapsed_mmss(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def live_lens_elapsed_bucket(
    *,
    elapsed_min: Any = None,
    period: Any = None,
    clock: Any = None,
    bin_seconds: int = 15,
    missing_label: str = "(missing)",
) -> str:
    sec = coerce_live_lens_elapsed_seconds(elapsed_min=elapsed_min, period=period, clock=clock)
    if sec is None:
        return str(missing_label)
    bin_seconds = max(1, int(bin_seconds))
    start = int(math.floor(float(sec) / float(bin_seconds))) * bin_seconds
    start = max(0, min(start, LIVE_LENS_MAX_GAME_SECONDS - bin_seconds))
    end = min(LIVE_LENS_MAX_GAME_SECONDS, start + bin_seconds)
    return f"{_format_elapsed_mmss(start)}-{_format_elapsed_mmss(end)}"


def live_lens_remaining_bucket(remaining_min: Any, *, missing_label: str = "unknown") -> str:
    rm = _safe_float(remaining_min)
    if rm is None:
        return str(missing_label)
    x = max(0.0, float(rm))
    if x <= 5.0:
        return "0-5"
    if x <= 10.0:
        return "5-10"
    if x <= 20.0:
        return "10-20"
    if x <= 40.0:
        return "20-40"
    return "40-60"


def live_lens_calibration_segment_candidates(
    prob_source: Any,
    *,
    elapsed_min: Any = None,
    period: Any = None,
    clock: Any = None,
    remaining_min: Any = None,
) -> list[str]:
    src = normalize_live_lens_prob_source(prob_source)
    out: list[str] = []

    phase = live_lens_game_phase(elapsed_min=elapsed_min, period=period, clock=clock)
    if phase:
        bucket_15s = live_lens_elapsed_bucket(
            elapsed_min=elapsed_min,
            period=period,
            clock=clock,
            bin_seconds=15,
            missing_label="",
        )
        bucket_1m = live_lens_elapsed_bucket(
            elapsed_min=elapsed_min,
            period=period,
            clock=clock,
            bin_seconds=60,
            missing_label="",
        )
        if bucket_15s:
            out.append(f"src={src}|phase={phase}|t15={bucket_15s}")
        if bucket_1m:
            out.append(f"src={src}|phase={phase}|t1={bucket_1m}")
        out.append(f"src={src}|phase={phase}")

    legacy_rm = live_lens_remaining_bucket(remaining_min, missing_label="")
    if legacy_rm:
        out.append(f"src={src}|rm={legacy_rm}")

    out.append(f"src={src}")

    deduped: list[str] = []
    seen: set[str] = set()
    for key in out:
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def pick_live_lens_calibration_spec(
    obj: Any,
    prob_source: Any,
    *,
    elapsed_min: Any = None,
    period: Any = None,
    clock: Any = None,
    remaining_min: Any = None,
) -> tuple[dict[str, Any], str]:
    default = obj.get("default") if isinstance(obj, dict) else None
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(default, dict):
        default = {"kind": "temp_shift", "t": 1.0, "b": 0.0}
    if not isinstance(segments, dict):
        return default, "default"

    candidates = live_lens_calibration_segment_candidates(
        prob_source,
        elapsed_min=elapsed_min,
        period=period,
        clock=clock,
        remaining_min=remaining_min,
    )
    for key in candidates:
        spec = segments.get(key)
        if isinstance(spec, dict):
            return spec, str(key)
    return default, "default"