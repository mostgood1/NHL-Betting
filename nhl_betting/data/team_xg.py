from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

def _season_code_for_date(d_ymd: Optional[str]) -> str:
    try:
        dt = pd.to_datetime(d_ymd)
    except Exception:
        dt = pd.Timestamp.utcnow()
    y = dt.tz_localize("UTC").tz_convert("America/New_York").year
    return f"{y}-{y+1}"

def load_team_xg(date: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Load team-level expected goals metrics for totals pacing.

    Returns mapping {ABBR: {xgf60: float, xga60: float}} or None if unavailable.
    Looks for season-specific or latest processed CSVs with columns: abbr,xgf60,xga60.
    """
    season = _season_code_for_date(date)
    candidates = [
        PROC_DIR / f"team_xg_{season}.csv",
        PROC_DIR / "team_xg_latest.csv",
    ]
    df = None
    for p in candidates:
        try:
            if p.exists() and getattr(p.stat(), "st_size", 0) > 0:
                df = pd.read_csv(p)
                break
        except Exception:
            continue
    if df is None or df.empty:
        return None
    # Normalize
    cols = {c.lower(): c for c in df.columns}
    ab = cols.get("abbr") or cols.get("team") or cols.get("team_abbr")
    xf = cols.get("xgf60") or cols.get("xgf_per60") or cols.get("xgf60_all")
    xa = cols.get("xga60") or cols.get("xga_per60") or cols.get("xga60_all")
    if not (ab and xf):
        return None
    m: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        try:
            k = str(r[ab]).upper()
            xgf = float(r[xf]) if pd.notna(r[xf]) else None
            xga = float(r[xa]) if (xa and pd.notna(r[xa])) else None
            if k and (xgf is not None):
                m[k] = {"xgf60": xgf}
                if xga is not None:
                    m[k]["xga60"] = xga
        except Exception:
            continue
    return m or None
