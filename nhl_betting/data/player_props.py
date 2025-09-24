from __future__ import annotations
"""Player props collection & normalization (draft).

Responsibilities:
- Collect raw player prop lines from supported books (initial: Bovada).
- Normalize player names to player_id using roster snapshot mapping.
- Combine OVER/UNDER rows into canonical line records.
- Persist Parquet outputs for downstream modeling.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import os

import pandas as pd

from .bovada import BovadaClient


@dataclass
class PropsCollectionConfig:
    output_root: str = "data/props"
    book: str = "bovada"


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def collect_bovada_props(date: str) -> pd.DataFrame:
    client = BovadaClient()
    raw = client.fetch_props_odds(date)
    if raw.empty:
        return raw
    raw["date"] = date
    raw["collected_at"] = _utc_now_iso()
    return raw


def normalize_player_names(raw: pd.DataFrame, roster_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if raw.empty:
        raw["player_id"] = []
        return raw
    df = raw.copy()
    df["player_clean"] = df["player"].str.strip().str.lower()
    if roster_df is not None and not roster_df.empty:
        r = roster_df.copy()
        # Expect roster_df has columns: player_id, full_name
        r["full_name_clean"] = r["full_name"].str.strip().str.lower()
        mapper = dict(zip(r["full_name_clean"], r["player_id"]))
        df["player_id"] = df["player_clean"].map(mapper)
    else:
        df["player_id"] = None
    return df


def combine_over_under(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date","player_id","market","line","over_price","under_price","book","first_seen_at","last_seen_at","is_current"])
    # Filter to known markets
    df = df[df["market"].isin(["SOG","GOALS","SAVES"])]
    # Build key for grouping
    grouped: List[Dict] = []
    now_iso = _utc_now_iso()
    for (date, player_id, market, line, book), g in df.groupby(["date","player_id","market","line","book"], dropna=False):
        over_row = g[g["side"] == "OVER"].sort_values("collected_at").tail(1)
        under_row = g[g["side"] == "UNDER"].sort_values("collected_at").tail(1)
        def parse_price(p):
            if p is None or pd.isna(p):
                return None
            try:
                return int(str(p))
            except Exception:
                return None
        over_price = parse_price(over_row["odds"].iloc[0]) if not over_row.empty else None
        under_price = parse_price(under_row["odds"].iloc[0]) if not under_row.empty else None
        grouped.append({
            "date": date,
            "player_id": player_id,
            "market": market,
            "line": line,
            "over_price": over_price,
            "under_price": under_price,
            "book": book,
            "first_seen_at": over_row["collected_at"].iloc[0] if not over_row.empty else (under_row["collected_at"].iloc[0] if not under_row.empty else now_iso),
            "last_seen_at": now_iso,
            "is_current": True,
        })
    return pd.DataFrame(grouped)


def write_props(df: pd.DataFrame, cfg: PropsCollectionConfig, date: str) -> str:
    if df.empty:
        return ""
    out_dir = os.path.join(cfg.output_root, "player_props_lines", f"date={date}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cfg.book}.parquet")
    df.to_parquet(path, index=False)
    return path


def collect_and_write(date: str, roster_df: Optional[pd.DataFrame] = None, cfg: PropsCollectionConfig | None = None) -> Dict:
    cfg = cfg or PropsCollectionConfig()
    raw = collect_bovada_props(date)
    norm = normalize_player_names(raw, roster_df)
    combined = combine_over_under(norm)
    written_path = write_props(combined, cfg, date)
    return {
        "raw_count": len(raw),
        "combined_count": len(combined),
        "output_path": written_path,
    }


__all__ = [
    "PropsCollectionConfig",
    "collect_bovada_props",
    "normalize_player_names",
    "combine_over_under",
    "write_props",
    "collect_and_write",
]
