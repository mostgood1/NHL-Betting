from __future__ import annotations

"""NHL Stats API shiftcharts ingestion and co-TOI computation.

Endpoint example:
  https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gameId}

Notes:
- Uses gameId equal to NHL gamePk (e.g., 2024020001).
- Output provides startTime and endTime per player shift; compute pairwise overlaps.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import re

import pandas as pd
import requests

BASE = "https://api.nhle.com/stats/rest/en/shiftcharts"


def _get(game_id: int, retries: int = 3, timeout: float = 25.0) -> Dict:
    last: Optional[Exception] = None
    params = {"cayenneExp": f"gameId={int(game_id)}"}
    for i in range(retries):
        try:
            r = requests.get(BASE, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.5 * (2 ** i))
    if last:
        raise last
    raise RuntimeError("shiftcharts request error")


def _parse_time_str(s: str) -> float:
    """Parse mm:ss (period clock) to seconds."""
    try:
        mm, ss = str(s).split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return 0.0


def shifts_frame(game_id: int) -> pd.DataFrame:
    """Return a tidy shifts DataFrame: columns [team, player_id, period, start_s, end_s]."""
    obj = _get(game_id)
    data = obj.get("data") or obj.get("shiftChart") or []
    rows: List[Dict] = []
    for sh in data:
        try:
            pid = int(sh.get("playerId"))
            tm = sh.get("teamAbbrev") or sh.get("teamAbbrevTricode") or sh.get("teamAbbrevShort")
            per = int(sh.get("period"))
            s = _parse_time_str(sh.get("startTime"))
            e = _parse_time_str(sh.get("endTime"))
            if e < s:
                # Some feeds report countdown clock; swap if needed
                s, e = e, s
            rows.append({
                "team": str(tm or "").upper(),
                "player_id": pid,
                "period": per,
                "start_s": float(s),
                "end_s": float(e),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def co_toi_from_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise EV co-TOI minutes by overlapping shift intervals per team.

    We treat all periods similarly; future refinement can isolate EV vs PP/PK using events.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["team","player_id_a","player_id_b","co_toi_ev"])
    rows: List[Dict] = []
    for team, g in df.groupby("team"):
        # Build list of intervals per player: [(period,start,end), ...]
        intervals: Dict[int, List[Tuple[int,float,float]]] = {}
        for pid, pgrp in g.groupby("player_id"):
            intervals[int(pid)] = [(int(r["period"]), float(r["start_s"]), float(r["end_s"])) for _, r in pgrp.iterrows()]
        pids = sorted(intervals.keys())
        for i in range(len(pids)):
            for j in range(i+1, len(pids)):
                a = pids[i]; b = pids[j]
                tot = 0.0
                # Sum overlap across periods
                for (per_a, sa, ea) in intervals[a]:
                    for (per_b, sb, eb) in intervals[b]:
                        if per_a != per_b:
                            continue
                        # Overlap of [sa,ea] and [sb,eb]
                        lo = max(sa, sb); hi = min(ea, eb)
                        if hi > lo:
                            tot += (hi - lo)
                if tot > 0:
                    rows.append({
                        "team": team,
                        "player_id_a": a,
                        "player_id_b": b,
                        "co_toi_ev": tot / 60.0,  # minutes
                    })
    return pd.DataFrame(rows)


__all__ = ["shifts_frame", "co_toi_from_shifts"]


def avg_toi_from_processed_shift_files(
    processed_dir: Path,
    end_date: str,
    days: int = 45,
) -> pd.DataFrame:
    """Compute rolling average TOI minutes per player from saved shifts_YYYY-MM-DD.csv files.

    Returns columns: team, player_id, avg_toi_minutes
    """
    try:
        end_dt = datetime.strptime(str(end_date), "%Y-%m-%d")
    except Exception:
        return pd.DataFrame(columns=["team", "player_id", "avg_toi_minutes"])
    start_dt = end_dt - timedelta(days=int(days))
    rows: List[Dict] = []
    if processed_dir is None:
        return pd.DataFrame(columns=["team", "player_id", "avg_toi_minutes"])
    try:
        files = list(Path(processed_dir).glob("shifts_*.csv"))
    except Exception:
        files = []
    for fp in files:
        m = re.match(r"shifts_(\d{4}-\d{2}-\d{2})\.csv$", fp.name)
        if not m:
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d")
        except Exception:
            continue
        if not (start_dt <= dt <= end_dt):
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if not {"team", "player_id", "start_s", "end_s"}.issubset(df.columns):
            continue
        df2 = df.copy()
        df2["dur"] = (pd.to_numeric(df2["end_s"], errors="coerce") - pd.to_numeric(df2["start_s"], errors="coerce")).clip(lower=0.0)
        g = df2.groupby(["team", "player_id"], as_index=False)["dur"].sum()
        g["toi_minutes"] = g["dur"].astype(float) / 60.0
        for _, r in g.iterrows():
            try:
                rows.append({
                    "team": str(r.get("team") or "").upper(),
                    "player_id": int(r.get("player_id")),
                    "toi_minutes": float(r.get("toi_minutes") or 0.0),
                })
            except Exception:
                continue
    if not rows:
        return pd.DataFrame(columns=["team", "player_id", "avg_toi_minutes"])
    hist = pd.DataFrame(rows)
    out = hist.groupby(["team", "player_id"], as_index=False)["toi_minutes"].mean()
    out = out.rename(columns={"toi_minutes": "avg_toi_minutes"})
    return out


__all__ = ["shifts_frame", "co_toi_from_shifts", "player_toi_from_shifts", "avg_toi_from_processed_shift_files"]

def player_toi_from_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player total EV TOI minutes from shift intervals.

    Returns columns: team, player_id, toi_ev_minutes
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["team","player_id","toi_ev_minutes"])
    g = df.groupby(["team","player_id"], as_index=False).agg(toi_ev_seconds=("end_s", lambda s: float(s.sum()))).copy()
    # Above aggregation is incorrect: need (end-start) per row. Compute properly.
    df2 = df.copy()
    df2["dur"] = (df2["end_s"].astype(float) - df2["start_s"].astype(float)).clip(lower=0.0)
    gg = df2.groupby(["team","player_id"], as_index=False).agg(toi_ev_seconds=("dur", "sum"))
    gg["toi_ev_minutes"] = gg["toi_ev_seconds"].astype(float) / 60.0
    return gg[["team","player_id","toi_ev_minutes"]]

