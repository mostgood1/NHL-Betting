from __future__ import annotations
"""Player props collection & normalization (draft).

Responsibilities:
- Collect raw player prop lines from supported books (initial: Bovada).
- Normalize player names to player_id using roster snapshot mapping.
- Combine OVER/UNDER rows into canonical line records.
- Persist Parquet outputs for downstream modeling.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

import pandas as pd

from .bovada import BovadaClient
from .odds_api import OddsAPIClient
from . import rosters as _rosters
from ..utils.io import RAW_DIR as _RAW_DIR
from ..web.teams import get_team_assets


@dataclass
class PropsCollectionConfig:
    output_root: str = "data/props"
    # When source=="bovada", book is the fixed file label and book field for rows.
    # When source=="oddsapi", book will come from the bookmaker key per row; file label is "oddsapi".
    book: str = "bovada"
    source: str = "bovada"  # bovada | oddsapi


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


def collect_oddsapi_props(date: str) -> pd.DataFrame:
    """Collect player props via The Odds API historical snapshot.

    Notes:
    - Requires ODDS_API_KEY in environment or passed via OddsAPIClient.
    - Uses a single snapshot at 17:00:00Z (midday US) on the given date to approximate pre-game markets.
    - Markets covered: SOG, GOALS, ASSISTS, POINTS. (SAVES is not available via Odds API v4)
    """
    # Choose a snapshot time on the given date (17:00Z ~ 12:00-13:00 ET depending on DST)
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        snapshot = dt.strftime("%Y-%m-%dT17:00:00Z")
    except Exception:
        snapshot = f"{date}T17:00:00Z"
    client = OddsAPIClient()
    markets = "player_shots_on_goal,player_goals,player_assists,player_points"
    def _empty_df():
        return pd.DataFrame(columns=["market","player","line","odds","side","book","date","collected_at"])
    # Collect event ids for the given date using historical snapshot first
    events: List[Dict] = []
    for sport_key in ("icehockey_nhl", "icehockey_nhl_preseason"):
        try:
            snap, _ = client.historical_list_events(sport_key, snapshot)
            evs = snap.get("data", []) if isinstance(snap, dict) else []
            if evs:
                events = evs
                break
        except Exception:
            continue
    # Filter events to the date window (UTC day)
    def _is_same_day(ev: Dict) -> bool:
        try:
            ct = ev.get("commence_time")
            d = datetime.fromisoformat(str(ct).replace("Z", "+00:00")).strftime("%Y-%m-%d")
            return d == date
        except Exception:
            return False
    events = [e for e in events if _is_same_day(e)]
    rows: List[Dict] = []
    # Map Odds API market keys to our canonical markets
    m_map = {
        "player_shots_on_goal": "SOG",
        "player_goals": "GOALS",
        "player_assists": "ASSISTS",
        "player_points": "POINTS",
    }
    def _parse_event_markets(event_odds_obj: Dict):
        bks = event_odds_obj.get("bookmakers", [])
        for bk in bks:
            book_key = bk.get("key") or "oddsapi"
            for m in bk.get("markets", []):
                mkey = m.get("key")
                market = m_map.get(mkey)
                if not market:
                    continue
                for oc in m.get("outcomes", []):
                    side = (oc.get("name") or "").strip().upper()
                    player = (
                        oc.get("description")
                        or oc.get("participant")
                        or oc.get("player_name")
                        or oc.get("player")
                        or ""
                    )
                    try:
                        line = float(oc.get("point")) if oc.get("point") is not None else None
                    except Exception:
                        line = None
                    odds = oc.get("price")
                    if not player or line is None or odds is None or side not in ("OVER","UNDER"):
                        continue
                    rows.append({
                        "market": market,
                        "player": player,
                        "line": line,
                        "odds": odds,
                        "side": side,
                        "book": book_key,
                        "date": date,
                        "collected_at": _utc_now_iso(),
                    })
    # Query historical event odds for each event
    if events:
        for ev in events:
            ev_id = ev.get("id")
            if not ev_id:
                continue
            for sport_key in ("icehockey_nhl", "icehockey_nhl_preseason"):
                success = False
                for bks in (None, "fanduel,draftkings,betmgm,caesars,pointsbetus,pinnacle", "bovada,betonlineag"):
                    for regs in ("us", "us,us2", "us,eu"):
                        try:
                            eo, _ = client.historical_event_odds(sport_key, ev_id, markets=markets, snapshot_iso=snapshot, bookmakers=bks, regions=regs)
                            if isinstance(eo, dict) and eo.get("bookmakers"):
                                _parse_event_markets(eo)
                                success = True
                                break
                        except Exception:
                            continue
                    if success:
                        break
                if success:
                    break
    # Fallback: use current events/odds for that date window
    if not rows:
        try:
            # current events list (doesn't count) then per-event odds
            from_dt = f"{date}T00:00:00Z"
            to_dt = f"{date}T23:59:59Z"
            for sport_key in ("icehockey_nhl", "icehockey_nhl_preseason"):
                try:
                    evs, _ = client.list_events(sport_key, commence_from_iso=from_dt, commence_to_iso=to_dt)
                    if not isinstance(evs, list) or not evs:
                        continue
                    for ev in evs:
                        ev_id = ev.get("id")
                        if not ev_id:
                            continue
                        success = False
                        for bks in (None, "fanduel,draftkings,betmgm,caesars,pointsbetus,pinnacle", "bovada,betonlineag"):
                            for regs in ("us", "us,us2", "us,eu"):
                                try:
                                    eo, _ = client.event_odds(sport_key, ev_id, markets=markets, bookmakers=bks, regions=regs)
                                    if isinstance(eo, dict) and eo.get("bookmakers"):
                                        _parse_event_markets(eo)
                                        success = True
                                        break
                                except Exception:
                                    continue
                            if success:
                                break
                except Exception:
                    continue
        except Exception:
            pass
    return pd.DataFrame(rows)


def normalize_player_names(raw: pd.DataFrame, roster_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if raw.empty:
        raw["player_id"] = []
        raw["team"] = []
        return raw
    df = raw.copy()
    df["player_clean"] = df["player"].str.strip().str.lower()
    if roster_df is not None and not roster_df.empty:
        r = roster_df.copy()
        # Expect roster_df has columns: player_id, full_name, team (abbr or name)
        r["full_name_clean"] = r["full_name"].astype(str).str.strip().str.lower()
        id_mapper = dict(zip(r["full_name_clean"], r["player_id"]))
        team_mapper = dict(zip(r["full_name_clean"], r.get("team", pd.Series([None]*len(r)))))
        df["player_id"] = df["player_clean"].map(id_mapper)
        # Attach team from roster snapshot where available
        df["team"] = df["player_clean"].map(team_mapper)
        # Normalize team to abbreviation when possible
        def _team_abbr(x):
            try:
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return None
                a = get_team_assets(str(x)) or {}
                ab = a.get("abbr")
                if ab:
                    return str(ab).upper()
                # if 'name' resolves to abbr string
                n = a.get("name")
                if n:
                    b = get_team_assets(str(n)) or {}
                    if b.get("abbr"):
                        return str(b.get("abbr")).upper()
                s = str(x).strip().upper()
                return s if s else None
            except Exception:
                return str(x).strip().upper() if isinstance(x, str) and x.strip() else None
        df["team"] = df["team"].map(_team_abbr)
    else:
        df["player_id"] = None
        df["team"] = None
    return df


def combine_over_under(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date","player_id","player_name","team","market","line","over_price","under_price","book","first_seen_at","last_seen_at","is_current"])
    # Filter to known markets
    df = df[df["market"].isin(["SOG","GOALS","SAVES","ASSISTS","POINTS","BLOCKS"]) ]
    # Build a player grouping key: prefer player_id; fallback to normalized player name
    player_key_col = "player_key_temp"
    def _mk_key(row):
        pid = row.get("player_id")
        if pd.notna(pid):
            return f"id::{pid}"
        name = str(row.get("player") or "").strip().lower()
        return f"name::{name}" if name else None
    df[player_key_col] = df.apply(_mk_key, axis=1)
    # Drop rows without any player identifier
    df = df[df[player_key_col].notna()]
    # Build key for grouping
    grouped: List[Dict] = []
    now_iso = _utc_now_iso()
    for (date, player_key, market, line, book), g in df.groupby(["date", player_key_col, "market", "line", "book"], dropna=False):
        # Extract player_id if present and a representative player_name
        player_id = None
        try:
            # Prefer a non-null player_id within the group
            ids = g["player_id"].dropna()
            if not ids.empty:
                player_id = ids.iloc[0]
        except Exception:
            player_id = None
        # Representative team from group (prefer a non-null recent value)
        team_val = None
        try:
            if "team" in g.columns and g["team"].notna().any():
                team_val = g["team"].dropna().astype(str).iloc[-1]
                # normalize to abbreviation
                try:
                    a = get_team_assets(team_val) or {}
                    ab = a.get("abbr")
                    if ab:
                        team_val = str(ab).upper()
                except Exception:
                    pass
        except Exception:
            team_val = None
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
        # Prefer a non-empty player name from either side
        player_name = None
        try:
            cand_over = over_row["player"].iloc[0] if not over_row.empty else None
            cand_under = under_row["player"].iloc[0] if not under_row.empty else None
            player_name = next((x for x in [cand_over, cand_under] if isinstance(x, str) and x.strip() != ""), None)
        except Exception:
            player_name = None
        grouped.append({
            "date": date,
            "player_id": player_id,
            "player_name": player_name,
            "team": team_val,
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
    # Use file label based on source
    file_label = "bovada" if (cfg.source or "bovada").lower() == "bovada" else "oddsapi"
    path = os.path.join(out_dir, f"{file_label}.parquet")
    df.to_parquet(path, index=False)
    return path


def collect_and_write(date: str, roster_df: Optional[pd.DataFrame] = None, cfg: PropsCollectionConfig | None = None) -> Dict:
    cfg = cfg or PropsCollectionConfig()
    source = (cfg.source or "bovada").lower()
    if source == "bovada":
        raw = collect_bovada_props(date)
    elif source == "oddsapi":
        raw = collect_oddsapi_props(date)
    else:
        raise ValueError(f"Unknown props source: {cfg.source}")
    # If no roster_df provided, attempt to build one for reliable player_id/team mapping
    if roster_df is None:
        roster_df = _build_roster_enrichment()
    norm = normalize_player_names(raw, roster_df)
    combined = combine_over_under(norm)
    written_path = write_props(combined, cfg, date)
    return {
        "raw_count": len(raw),
        "combined_count": len(combined),
        "output_path": written_path,
    }


def _build_roster_enrichment() -> pd.DataFrame:
    """Best-effort build of a roster DataFrame with columns [full_name, player_id, team].

    Strategy:
    1) Try live roster snapshots (team_id -> abbreviation) via rosters.build_all_team_roster_snapshots.
    2) Fallback to historical RAW player_game_stats.csv using last known player_id and team per name.
    """
    # Attempt live roster snapshots
    try:
        snap = _rosters.build_all_team_roster_snapshots()
        if snap is not None and not snap.empty:
            # Map team_id -> abbreviation using list_teams
            try:
                teams = _rosters.list_teams()
                id_to_abbr: Dict[int, str] = {}
                id_to_name: Dict[int, str] = {}
                for t in teams:
                    try:
                        id_to_abbr[int(t.get("id"))] = str(t.get("abbreviation") or "").upper()
                        id_to_name[int(t.get("id"))] = str(t.get("name") or "")
                    except Exception:
                        continue
                snap = snap.copy()
                snap["team"] = snap["team_id"].map(id_to_abbr).fillna(snap["team_id"].map(id_to_name))
            except Exception:
                # If team lookup fails, keep team_id as team string
                snap = snap.copy()
                snap["team"] = snap.get("team_id")
            out = snap.rename(columns={"full_name": "full_name", "player_id": "player_id"})
            return out[["full_name", "player_id", "team"]]
    except Exception:
        pass
    # Fallback to historical stats
    try:
        stats_p = _RAW_DIR / "player_game_stats.csv"
        if stats_p.exists():
            stats = pd.read_csv(stats_p)
            if not stats.empty and {"player","player_id"}.issubset(stats.columns):
                stats = stats.dropna(subset=["player"])  # require a name
                # Order by date to take last known mapping
                try:
                    stats["_date"] = pd.to_datetime(stats["date"], errors="coerce")
                    stats = stats.sort_values("_date")
                except Exception:
                    pass
                last = stats.groupby("player").agg({
                    "player_id": "last",
                    "team": "last",
                }).reset_index().rename(columns={"player": "full_name"})
                # Ensure clean strings
                last["full_name"] = last["full_name"].astype(str)
                return last[["full_name","player_id","team"]]
    except Exception:
        pass
    # As a final fallback, return empty
    return pd.DataFrame(columns=["full_name","player_id","team"])


__all__ = [
    "PropsCollectionConfig",
    "collect_bovada_props",
    "collect_oddsapi_props",
    "normalize_player_names",
    "combine_over_under",
    "write_props",
    "collect_and_write",
]
