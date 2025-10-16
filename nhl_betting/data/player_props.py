from __future__ import annotations
"""Player props collection & normalization (draft).

Responsibilities:
- Collect raw player prop lines from supported books (initial: Bovada).
- Normalize player names to player_id using roster snapshot mapping.
- Combine OVER/UNDER rows into canonical line records.
- Persist Parquet outputs for downstream modeling.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
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
    """Collect player props via The Odds API using a fast, concurrent path.

    Strategy for speed:
    - Prefer current event odds for the UTC day window [00:00Z, 23:59Z] for icehockey_nhl only.
    - Limit to core bookmakers (default: 'fanduel,draftkings,pinnacle') and regions 'us'.
    - Fetch per-event odds concurrently to reduce wall-clock time.
    - Fall back to the slower historical snapshot path only if current path yields no rows.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # Configurable knobs via env
    fast_on = os.environ.get("PROPS_ODDSAPI_FAST", "1").strip().lower() in ("1","true","yes")
    bk_pref = os.environ.get("PROPS_ODDSAPI_BOOKMAKERS", "fanduel,draftkings,pinnacle").strip()
    regions = os.environ.get("PROPS_ODDSAPI_REGIONS", "us").strip()
    max_workers = int(os.environ.get("PROPS_ODDSAPI_WORKERS", "6"))
    markets = "player_shots_on_goal,player_goals,player_assists,player_points"
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
                market = m_map.get(m.get("key"))
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
    # Fast path: current events + concurrent per-event odds
    try:
        client = OddsAPIClient(rate_limit_per_sec=10.0)
        from_dt = f"{date}T00:00:00Z"; to_dt = f"{date}T23:59:59Z"
        evs, _ = client.list_events("icehockey_nhl", commence_from_iso=from_dt, commence_to_iso=to_dt)
        if isinstance(evs, list) and evs:
            def fetch(ev_id: str):
                try:
                    eo, _ = client.event_odds("icehockey_nhl", ev_id, markets=markets, regions=regions, bookmakers=(bk_pref or None))
                    return eo
                except Exception:
                    return None
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(fetch, str(ev.get("id"))): ev for ev in evs if ev.get("id")}
                for fut in as_completed(futs):
                    eo = fut.result()
                    if isinstance(eo, dict) and eo.get("bookmakers"):
                        _parse_event_markets(eo)
        # If we captured rows, return quickly
        if rows:
            return pd.DataFrame(rows)
    except Exception:
        # Continue to fallback
        pass
    # Fallback: slower historical snapshot loop (limited scope to NHL only for speed)
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        snapshot = dt.strftime("%Y-%m-%dT17:00:00Z")
    except Exception:
        snapshot = f"{date}T17:00:00Z"
    client = OddsAPIClient()
    try:
        snap, _ = client.historical_list_events("icehockey_nhl", snapshot)
        data = snap.get("data", []) if isinstance(snap, dict) else []
        def _is_same_day(ev: Dict) -> bool:
            try:
                ct = ev.get("commence_time")
                d = datetime.fromisoformat(str(ct).replace("Z", "+00:00")).strftime("%Y-%m-%d")
                return d == date
            except Exception:
                return False
        events = [e for e in data if _is_same_day(e)]
        for ev in events:
            ev_id = ev.get("id")
            if not ev_id:
                continue
            try:
                eo, _ = client.historical_event_odds("icehockey_nhl", ev_id, markets=markets, snapshot_iso=snapshot, regions=regions, bookmakers=(bk_pref or None))
                if isinstance(eo, dict) and eo.get("bookmakers"):
                    _parse_event_markets(eo)
            except Exception:
                continue
    except Exception:
        pass
    return pd.DataFrame(rows)


def normalize_player_names(raw: pd.DataFrame, roster_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach player_id and team to raw props rows using a robust name matching strategy.

    Strategy tiers:
    1) Exact lowercase full name match
    2) Punctuation/space stripped match (squashed)
    3) Initial+last variants (e.g., "T Bertuzzi", "JG Pageau")

    If PROPS_DEBUG_NAME_MATCH=1, writes a CSV of unmatched names by date.
    """
    if raw.empty:
        raw["player_id"] = []
        raw["team"] = []
        return raw
    df = raw.copy()
    # Some sources (historical stats fallback) may pass player strings that look like dicts
    # (e.g., "{'default': 'C. McDavid'}"). Parse these early so later normalization variants
    # operate on the canonical display form.
    def _unwrap_dict_like_player(val):
        try:
            s = str(val)
            if s.startswith('{') and 'default' in s:
                import ast as _ast
                obj = _ast.literal_eval(s)
                if isinstance(obj, dict):
                    dv = obj.get('default')
                    if isinstance(dv, str) and dv.strip():
                        return dv.strip()
            return val
        except Exception:
            return val
    try:
        if 'player' in df.columns:
            df['player'] = df['player'].map(_unwrap_dict_like_player)
    except Exception:
        pass
    # Basic cleaned form
    def _clean(s: str) -> str:
        import unicodedata, re
        s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s
    def _squash(s: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]", "", str(s or "").lower())
    def _initials_for_first(first: str) -> str:
        # For hyphenated or multi-part first names, take all initials (e.g., Jean-Gabriel -> JG)
        import re
        parts = re.split(r"[-\s]+", first.strip())
        return "".join([p[0] for p in parts if p]) if first else ""
    def _variant_keys(full: str) -> Set[str]:
        full_c = _clean(full)
        if not full_c:
            return set()
        parts = full_c.split(" ")
        last = parts[-1] if len(parts) >= 2 else ""
        first = parts[0] if parts else ""
        keys: Set[str] = {full_c, _squash(full_c)}
        if first and last:
            ini1 = (first[0] if first else "")
            iniall = _initials_for_first(first)
            # Space forms
            if ini1:
                keys.add(f"{ini1} {last}")
                keys.add(f"{ini1}{last}")  # fallback without space
            if iniall and iniall != ini1:
                keys.add(f"{iniall} {last}")
                keys.add(f"{iniall}{last}")
            # Squashed last too
            keys.add(_squash(f"{ini1} {last}"))
            if iniall and iniall != ini1:
                keys.add(_squash(f"{iniall} {last}"))
        return keys

    df["player_clean"] = df["player"].map(_clean)
    df["player_squash"] = df["player_clean"].map(_squash)
    # Build roster indices
    if roster_df is not None and not roster_df.empty:
        r = roster_df.copy()
        r["full_name_clean"] = r["full_name"].astype(str).map(_clean)
        r["full_name_squash"] = r["full_name_clean"].map(_squash)
        # Primary exact map
        map_exact = dict(zip(r["full_name_clean"], r["player_id"]))
        # Squashed map
        map_squash = dict(zip(r["full_name_squash"], r["player_id"]))
        # Variant map: initial(s)+last and squashed variants
        var_map: Dict[str, str] = {}
        for _, row in r.iterrows():
            pid = row["player_id"]
            for k in _variant_keys(row["full_name_clean"]):
                var_map.setdefault(k, pid)
        # Apply in order
        pid_series = (
            df["player_clean"].map(map_exact)
            .fillna(df["player_squash"].map(map_squash))
        )
        # Build props-side variants
        def _player_variants_row(s: str) -> List[str]:
            return list(_variant_keys(s))
        # Try variant lookups for any remaining nulls
        missing_mask = pid_series.isna()
        if missing_mask.any():
            # For performance, we vectorize by expanding to a long form
            sub = df.loc[missing_mask, ["player_clean"]].copy()
            # Map each player_clean to any variant key hit
            def _map_first_hit(name: str) -> Optional[str]:
                for k in _variant_keys(name):
                    v = var_map.get(k)
                    if v is not None:
                        return v
                return None
            pid_fills = sub["player_clean"].map(_map_first_hit)
            pid_series.loc[missing_mask] = pid_series.loc[missing_mask].fillna(pid_fills)
        df["player_id"] = pid_series
        # Team mapping from roster
        team_mapper = dict(zip(r["full_name_clean"], r.get("team", pd.Series([None]*len(r)))))
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

    # Optional debug: write unmatched names
    try:
        if os.environ.get("PROPS_DEBUG_NAME_MATCH", "").strip() in ("1", "true", "yes"):
            miss = df[df["player_id"].isna()].copy()
            if not miss.empty:
                date_val = None
                try:
                    date_val = str(df["date"].dropna().astype(str).unique()[0])
                except Exception:
                    date_val = "unknown"
                out_dir = os.path.join("data", "props")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"name_misses_date={date_val}.csv")
                miss.groupby(["player","market"]).size().reset_index(name="count").to_csv(out_path, index=False)
    except Exception:
        pass

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
        # Require a non-empty player name; drop aggregate rows like 'Total Shots On Goal'
        name_raw = row.get("player")
        name = str(name_raw or "").strip()
        if not name:
            return None
        return f"name::{name.lower()}"
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
    pq_path = os.path.join(out_dir, f"{file_label}.parquet")
    try:
        # Use pyarrow engine explicitly for stability across environments
        df.to_parquet(pq_path, index=False, engine="pyarrow")
        return pq_path
    except Exception:
        # Fallback to CSV if Parquet writer is unavailable/mismatched
        csv_path = os.path.join(out_dir, f"{file_label}.csv")
        df.to_csv(csv_path, index=False)
        return csv_path


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
        # Prefer unified processed roster caches for speed and stability
        try:
            from ..utils.io import PROC_DIR as _PROC
            # Try date-stamped roster first, then master
            p_dated = _PROC / f"roster_{date}.csv"
            p_master = _PROC / "roster_master.csv"
            use = None
            if p_dated.exists() and p_dated.stat().st_size > 64:
                use = p_dated
            elif p_master.exists() and p_master.stat().st_size > 64:
                use = p_master
            if use is not None:
                try:
                    r = pd.read_csv(use)
                    # Normalize to expected columns [full_name, player_id, team]
                    name_col = 'full_name' if 'full_name' in r.columns else ('player' if 'player' in r.columns else None)
                    team_col = None
                    for cand in ('team','team_abbr','teamAbbrev','team_abbrev','team_abbreviation'):
                        if cand in r.columns:
                            team_col = cand; break
                    pid_col = 'player_id' if 'player_id' in r.columns else None
                    if name_col and team_col:
                        roster_df = r.rename(columns={name_col: 'full_name'})
                        if 'team' != team_col:
                            roster_df['team'] = roster_df[team_col]
                        if pid_col and pid_col != 'player_id':
                            roster_df['player_id'] = roster_df[pid_col]
                        roster_df = roster_df[['full_name','player_id','team']]
                    else:
                        roster_df = None
                except Exception:
                    roster_df = None
        except Exception:
            roster_df = None
        # Fallback to dynamic build only if unified roster not present
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
                # Extract a cleaner display name when the raw value is a dict-like string such as
                # "{'default': 'A. Matthews'}" or includes localized variants. This improves
                # downstream variant key matching (initial+last) for player_id enrichment when
                # live roster snapshots are unavailable (e.g., network/DNS issues on cold start).
                def _extract_default(v):
                    try:
                        s = str(v)
                        if s.startswith('{') and 'default' in s:
                            import ast as _ast
                            obj = _ast.literal_eval(s)
                            # Typical structure: {'default': 'N. Schmaltz', 'fr': '...', ...}
                            dval = obj.get('default') if isinstance(obj, dict) else None
                            if isinstance(dval, str) and dval.strip():
                                return dval.strip()
                        return s
                    except Exception:
                        return str(v)
                try:
                    stats['player'] = stats['player'].map(_extract_default)
                except Exception:
                    pass
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
