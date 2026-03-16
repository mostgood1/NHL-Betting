from __future__ import annotations
"""Player props collection & normalization (draft).

Responsibilities:
- Collect raw player prop lines from supported books (OddsAPI-only).
- Normalize player names to player_id using roster snapshot mapping.
- Combine OVER/UNDER rows into canonical line records.
- Persist Parquet outputs for downstream modeling.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import os

import pandas as pd

from .odds_api import OddsAPIClient
from . import rosters as _rosters
from ..utils.io import RAW_DIR as _RAW_DIR
from ..web.teams import get_team_assets


@dataclass
class PropsCollectionConfig:
    output_root: str = "data/props"
    # For OddsAPI, book will come from the bookmaker key per row; file label is "oddsapi".
    book: str = "oddsapi"
    source: str = "oddsapi"  # oddsapi | bovada


def _clean_name_key(value: object) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalize_roster_frame(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if frame is None or frame.empty:
        return None

    name_col = "full_name" if "full_name" in frame.columns else ("player" if "player" in frame.columns else None)
    team_col = None
    for cand in ("team", "team_abbr", "teamAbbrev", "team_abbrev", "team_abbreviation"):
        if cand in frame.columns:
            team_col = cand
            break
    pid_col = "player_id" if "player_id" in frame.columns else None
    if not name_col:
        return None

    out = pd.DataFrame({"full_name": frame[name_col].astype(str).map(lambda s: str(s).strip())})
    out["player_id"] = frame[pid_col] if pid_col else None
    out["team"] = frame[team_col] if team_col else None
    out = out[out["full_name"] != ""].copy()
    if out.empty:
        return None
    out["_name_key"] = out["full_name"].map(_clean_name_key)
    out = out.drop_duplicates(subset=["_name_key"], keep="last")
    return out


def _merge_roster_sources(primary: Optional[pd.DataFrame], fallback: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    primary = _normalize_roster_frame(primary)
    fallback = _normalize_roster_frame(fallback)

    if primary is None and fallback is None:
        return None
    if primary is None:
        return fallback[["full_name", "player_id", "team"]] if fallback is not None else None
    if fallback is None:
        return primary[["full_name", "player_id", "team"]]

    merged = primary.merge(
        fallback[["_name_key", "player_id", "team"]].rename(
            columns={"player_id": "_player_id_fallback", "team": "_team_fallback"}
        ),
        on="_name_key",
        how="left",
    )
    merged["player_id"] = merged["player_id"].where(merged["player_id"].notna(), merged["_player_id_fallback"])
    team_blank = merged["team"].isna() | merged["team"].astype(str).str.strip().isin(["", "nan", "None"])
    merged.loc[team_blank, "team"] = merged.loc[team_blank, "_team_fallback"]

    fallback_only = fallback.loc[~fallback["_name_key"].isin(set(merged["_name_key"]))].copy()
    combined = pd.concat(
        [merged[["full_name", "player_id", "team", "_name_key"]], fallback_only[["full_name", "player_id", "team", "_name_key"]]],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset=["_name_key"], keep="first")
    return combined[["full_name", "player_id", "team"]]


def _load_cached_roster_enrichment(date: str, proc_dir=None) -> Optional[pd.DataFrame]:
    try:
        from ..utils.io import PROC_DIR as _PROC
    except Exception:
        _PROC = None

    proc_root = proc_dir or _PROC
    if proc_root is None:
        return None

    dated_path = proc_root / f"roster_{date}.csv"
    snapshot_path = proc_root / f"roster_snapshot_{date}.csv"
    master_path = proc_root / "roster_master.csv"

    sources: List[pd.DataFrame] = []
    for path in (dated_path, snapshot_path, master_path):
        try:
            if path.exists() and path.stat().st_size > 64:
                norm = _normalize_roster_frame(pd.read_csv(path))
                if norm is not None:
                    sources.append(norm)
        except Exception:
            continue

    try:
        models_dir = _RAW_DIR.parent / "models"
        snapshot_jsons = sorted(models_dir.glob("roster_snapshot_*.json"))
        if snapshot_jsons:
            norm = _normalize_roster_frame(pd.read_json(snapshot_jsons[-1]))
            if norm is not None:
                sources.append(norm)
    except Exception:
        pass

    combined = None
    for source in sources:
        combined = _merge_roster_sources(combined, source)
    return combined


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


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
    # Include core NHL player markets with broad bookmaker support.
    # IMPORTANT: Requesting unsupported markets yields 422; restrict to keys confirmed by provider docs
    # and our live probes: player_points, player_assists, player_goals, player_shots_on_goal.
    markets = ",".join([
        "player_points",
        "player_assists",
        "player_goals",
        "player_shots_on_goal",
    ])
    rows: List[Dict] = []
    # Map Odds API market keys to our canonical markets
    m_map = {
        # Shots on goal variants
        "player_shots_on_goal": "SOG",
        "player_shots_on_goal_alternate": "SOG",
        "shots_on_goal": "SOG",
        "player_shots": "SOG",
        # Goals (including alternates)
        "player_goals": "GOALS",
        "player_goals_alternate": "GOALS",
        # Assists (including alternates)
        "player_assists": "ASSISTS",
        "player_assists_alternate": "ASSISTS",
        # Points (including alternates)
        "player_points": "POINTS",
        "player_points_alternate": "POINTS",
        # Saves
        "player_saves": "SAVES",
        "goalie_saves": "SAVES",
        # Blocks
        "player_blocks": "BLOCKS",
        "player_blocked_shots": "BLOCKS",
    }
    from zoneinfo import ZoneInfo

    def _commence_date_et(commence_time_iso: Optional[str]) -> Optional[str]:
        if not commence_time_iso:
            return None
        try:
            dt = datetime.fromisoformat(str(commence_time_iso).replace("Z", "+00:00"))
            return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        except Exception:
            return None

    def _parse_event_markets(event_odds_obj: Dict):
        ev_id = event_odds_obj.get("id")
        commence_time = event_odds_obj.get("commence_time")
        home_team = event_odds_obj.get("home_team")
        away_team = event_odds_obj.get("away_team")
        commence_date_et = _commence_date_et(commence_time)
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
                        "event_id": ev_id,
                        "commence_time": commence_time,
                        "commence_date_et": commence_date_et,
                        "home_team": home_team,
                        "away_team": away_team,
                        "collected_at": _utc_now_iso(),
                    })
    # Fast path: current events + concurrent per-event odds (no pre-probe to avoid empty keys)
    try:
        client = OddsAPIClient(rate_limit_per_sec=10.0)
        # IMPORTANT: Filter events by ET calendar day.
        # OddsAPI commence_time is in UTC. A 7:00 PM ET game on date D starts at 00:00Z on D+1.
        # We should extend the *end* of the UTC window, not shift the start backward.
        from datetime import timedelta, timezone
        try:
            et_start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=ZoneInfo("America/New_York"))
            utc_start = et_start.astimezone(timezone.utc)
            utc_end = (et_start + timedelta(days=1)).astimezone(timezone.utc)
            from_dt = utc_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            to_dt = utc_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            # Fallback: old logic (will miss late games but better than nothing)
            from_dt = f"{date}T00:00:00Z"; to_dt = f"{date}T23:59:59Z"
        evs, _ = client.list_events("icehockey_nhl", commence_from_iso=from_dt, commence_to_iso=to_dt)
        # Hard filter by ET date to prevent accidentally including previous/next slate games.
        try:
            evs = [
                ev for ev in (evs or [])
                if _commence_date_et(ev.get("commence_time")) == date
            ]
        except Exception:
            pass
        if isinstance(evs, list) and evs:
            # Fallback-aware fetch: probe event-specific available market keys first, then request odds
            def fetch(ev_id: str):
                use_keys = [k for k in markets.split(",")]
                # Request odds for the supported keys; try preferred bookmakers first, then all
                try:
                    eo, _ = client.event_odds("icehockey_nhl", ev_id, markets=",".join(use_keys), regions=regions, bookmakers=(bk_pref or None))
                    if isinstance(eo, dict) and eo.get("bookmakers"):
                        return eo
                except Exception:
                    pass
                # Fallback: no bookmaker filter
                try:
                    eo2, _ = client.event_odds("icehockey_nhl", ev_id, markets=",".join(use_keys), regions=regions, bookmakers=None)
                    if isinstance(eo2, dict) and eo2.get("bookmakers"):
                        return eo2
                except Exception:
                    pass
                # Final fallback: try each market individually to salvage partial coverage
                for single in use_keys:
                    try:
                        eo3, _ = client.event_odds("icehockey_nhl", ev_id, markets=single, regions=regions, bookmakers=None)
                        if isinstance(eo3, dict) and eo3.get("bookmakers"):
                            return eo3
                    except Exception:
                        continue
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
    # Fallback: historical snapshots near each event's commence time (paid plans)
    try:
        # Base snapshot around 17:00Z for the ET day
        dt = datetime.strptime(date, "%Y-%m-%d")
        base_snapshot = dt.strftime("%Y-%m-%dT17:00:00Z")
    except Exception:
        base_snapshot = f"{date}T17:00:00Z"
    client = OddsAPIClient()
    try:
        # List snapshot events; if empty, still attempt with base snapshot only
        snap, _ = client.historical_list_events("icehockey_nhl", base_snapshot)
        data = snap.get("data", []) if isinstance(snap, dict) else []
        def _is_same_day(ev: Dict) -> bool:
            try:
                ct = ev.get("commence_time")
                return _commence_date_et(ct) == date
            except Exception:
                return False
        events = [e for e in data if _is_same_day(e)]
        # If no events resolved via snapshot, still try a generic set of snapshots across the day
        generic_snapshots = [
            base_snapshot,
            f"{date}T19:00:00Z",
            f"{date}T22:00:00Z",
            # early next day window in UTC for late PT games on ET slate
            (datetime.strptime(date, "%Y-%m-%d").replace(hour=2, minute=0).strftime("%Y-%m-%dT%H:%M:00Z") if True else base_snapshot),
        ]
        # Region fallback strategy (broaden coverage)
        region_sets = [regions]
        if regions.strip().lower() != "us,us2,eu":
            region_sets.append("us,us2,eu")
        # For each event, try snapshots near commence time
        for ev in (events or []):
            ev_id = ev.get("id")
            ct = ev.get("commence_time")
            snaps_for_ev: List[str] = []
            try:
                cdt = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
                snaps_for_ev = [
                    (cdt).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    (cdt.replace(minute=0)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    (cdt.replace(minute=0) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    (cdt.replace(minute=0) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ]
            except Exception:
                snaps_for_ev = [base_snapshot]
            for reg in region_sets:
                for snap_iso in snaps_for_ev:
                    if not ev_id:
                        continue
                    try:
                        eo, _ = client.historical_event_odds("icehockey_nhl", ev_id, markets=markets, snapshot_iso=snap_iso, regions=reg, bookmakers=(bk_pref or None))
                        if isinstance(eo, dict) and eo.get("bookmakers"):
                            _parse_event_markets(eo)
                            # continue trying others to maximize coverage across books
                        else:
                            # Fallback: try without bookmaker filter
                            try:
                                eo2, _ = client.historical_event_odds("icehockey_nhl", ev_id, markets=markets, snapshot_iso=snap_iso, regions=reg, bookmakers=None)
                                if isinstance(eo2, dict) and eo2.get("bookmakers"):
                                    _parse_event_markets(eo2)
                            except Exception:
                                pass
                    except Exception:
                        continue
        # If still empty and we had no events, try generic snapshots without event context (unlikely but cheap)
        if not rows:
            for snap_iso in generic_snapshots:
                try:
                    snap2, _ = client.historical_list_events("icehockey_nhl", snap_iso)
                    data2 = snap2.get("data", []) if isinstance(snap2, dict) else []
                    events2 = [e for e in data2 if _is_same_day(e)]
                    for ev2 in events2:
                        ev_id2 = ev2.get("id")
                        if not ev_id2:
                            continue
                        try:
                            eo3, _ = client.historical_event_odds("icehockey_nhl", ev_id2, markets=markets, snapshot_iso=snap_iso, regions=regions, bookmakers=None)
                            if isinstance(eo3, dict) and eo3.get("bookmakers"):
                                _parse_event_markets(eo3)
                        except Exception:
                            continue
                except Exception:
                    continue
    except Exception:
        pass
    return pd.DataFrame(rows)


def collect_bovada_props(date: str) -> pd.DataFrame:
    """Collect player props via Bovada's public JSON services.

    Notes:
    - Uses unauthenticated endpoints intended for public odds display.
    - Parses player Over/Under markets with numeric handicaps only.
    - Returns a long-form DataFrame with columns: market, player, line, odds, side, book, date, collected_at
    """
    import requests, time
    # Bovada serves similar structures on multiple hostnames; try a few.
    hosts = [
        "https://www.bovada.lv",
        "https://www.bovada.com",
        # Regional mirrors often carry identical JSON with different coverage
        "https://www.bodog.eu",
        "https://www.bodog.com",
    ]
    # Candidate endpoint patterns (coupon/events: richer structure with displayGroups)
    paths = [
        # v1 coupon
        "/services/sports/event/coupon/events/A/description/ice-hockey/nhl",
        "/services/sports/event/coupon/events/A/description/hockey/nhl",
        # v2 coupon
        "/services/sports/event/v2/events/A/description/ice-hockey/nhl",
        "/services/sports/event/v2/events/A/description/hockey/nhl",
    ]
    # Try multiple market filters; some mirrors expose player props only when using
    # specific filters like "players" or "props".
    market_filters = ["def", "players", "player", "props", "playerprops"]
    base_params = {
        "preMatchOnly": "true",
        "includeParticipants": "true",
        "lang": "en",
    }
    rows: List[Dict] = []
    # Helper: map market description to canonical
    def _canon_market(desc: str) -> Optional[str]:
        s = (desc or "").strip().lower()
        if not s:
            return None
        # Exclude non-OU player props like anytime goals
        if "anytime" in s or "1st goal" in s or "first goal" in s:
            return None
        if "shots on goal" in s or "shots-on-goal" in s or ("shots" in s and "goal" in s):
            return "SOG"
        if "blocked" in s or "blocks" in s:
            return "BLOCKS"
        if "assists" in s:
            return "ASSISTS"
        if "points" in s:
            return "POINTS"
        if "saves" in s:
            return "SAVES"
        if "goals" in s:
            return "GOALS"
        return None
    def _to_american(price_obj) -> Optional[int]:
        try:
            if isinstance(price_obj, dict):
                a = price_obj.get("american")
                if a is None:
                    return None
                return int(str(a))
        except Exception:
            return None
        return None
    def _float_or_none(v) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None
    def _now() -> str:
        return _utc_now_iso()
    # Iterate endpoints until we successfully parse
    data = None
    last_err = None
    for h in hosts:
        for pth in paths:
            for mf in market_filters:
                url = f"{h}{pth}"
                params = dict(base_params)
                params["marketFilterId"] = mf
                try:
                    r = requests.get(url, params=params, timeout=20)
                    if not r.ok:
                        continue
                    js = r.json()
                    # v1 coupon returns a list of groups; v2 returns dict with events
                    data = js
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(0.3)
            if data is not None:
                break
        if data is not None:
            break
    if data is None:
        return pd.DataFrame(rows)
    # Normalize to a flat list of events with displayGroups/markets
    events: List[Dict] = []
    try:
        if isinstance(data, list):
            # coupon format: list of categories -> each has "events"
            for grp in data:
                evs = grp.get("events") if isinstance(grp, dict) else None
                if isinstance(evs, list):
                    events.extend([e for e in evs if isinstance(e, dict)])
        elif isinstance(data, dict):
            # v2 format: {"events": [...]}
            evs = data.get("events")
            if isinstance(evs, list):
                events.extend([e for e in evs if isinstance(e, dict)])
    except Exception:
        events = []
    # Parse markets/outcomes
    for ev in events:
        dgs = ev.get("displayGroups") or []
        for dg in dgs:
            mkts = dg.get("markets") or []
            for m in mkts:
                mdesc = m.get("description") or m.get("displayKey") or m.get("key") or ""
                canon = _canon_market(mdesc)
                if not canon:
                    continue
                # Only OU with a numeric handicap
                outcs = m.get("outcomes") or []
                for oc in outcs:
                    name = (oc.get("description") or oc.get("participant") or oc.get("name") or "").strip()
                    side_raw = (oc.get("name") or oc.get("type") or "").strip().upper()
                    side = "OVER" if side_raw.startswith("OVER") else ("UNDER" if side_raw.startswith("UNDER") else None)
                    line = _float_or_none(oc.get("handicap") or oc.get("price") and (oc.get("price") or {}).get("handicap"))
                    odds = _to_american(oc.get("price"))
                    if not name or side is None or line is None or odds is None:
                        continue
                    rows.append({
                        "market": canon,
                        "player": name,
                        "line": line,
                        "odds": odds,
                        "side": side,
                        "book": "bovada",
                        "date": date,
                        "collected_at": _now(),
                    })
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
        return _clean_name_key(s)
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
        try:
            team_by_pid = (
                r.loc[r["player_id"].notna(), ["player_id", "team"]]
                .drop_duplicates(subset=["player_id"], keep="last")
                .set_index("player_id")["team"]
            )
            team_missing = df["team"].isna()
            if team_missing.any():
                df.loc[team_missing, "team"] = df.loc[team_missing, "player_id"].map(team_by_pid)
        except Exception:
            pass
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
    # Always write an output artifact, even if empty, to aid downstream verification/monitoring.
    out_dir = os.path.join(cfg.output_root, "player_props_lines", f"date={date}")
    os.makedirs(out_dir, exist_ok=True)
    # Use file label based on source
    file_label = (cfg.source or cfg.book or "oddsapi").strip().lower()
    pq_path = os.path.join(out_dir, f"{file_label}.parquet")
    csv_path = os.path.join(out_dir, f"{file_label}.csv")
    # Load existing data (prefer parquet, then CSV)
    existing = pd.DataFrame()
    try:
        if os.path.exists(pq_path):
            existing = pd.read_parquet(pq_path)
        elif os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
    except Exception:
        existing = pd.DataFrame()
    # If the incoming frame is empty and we already have existing rows, do NOT overwrite
    try:
        if (df is None or df.empty) and (existing is not None and not existing.empty):
            # Ensure CSV sidecar exists
            try:
                if not os.path.exists(csv_path):
                    existing.to_csv(csv_path, index=False)
            except Exception:
                pass
            return pq_path if os.path.exists(pq_path) else csv_path
    except Exception:
        pass
    # Merge: keep latest rows per key, preserve earlier rows for history,
    # but ensure `is_current` reflects membership in the latest fetch.
    try:
        # Normalize required columns presence
        for c in ["date","player_id","player_name","team","market","line","over_price","under_price","book","first_seen_at","last_seen_at","is_current"]:
            if c not in df.columns:
                try:
                    df[c] = None
                except Exception:
                    pass
        for c in ["date","player_id","player_name","team","market","line","over_price","under_price","book","first_seen_at","last_seen_at","is_current"]:
            if c not in existing.columns:
                try:
                    existing[c] = None
                except Exception:
                    pass
        # Build a robust merge key (prefer player_id, else normalized name)
        def _mk_key_row(r):
            try:
                pid = r.get("player_id")
            except Exception:
                pid = None
            if pd.notna(pid):
                return f"id::{pid}"
            try:
                nm = str(r.get("player_name") or "").strip().lower()
            except Exception:
                nm = ""
            return f"name::{nm}" if nm else None

        # Compute merge keys for existing + new separately
        try:
            existing = existing.copy()
            existing["_merge_key"] = existing.apply(_mk_key_row, axis=1)
        except Exception:
            existing["_merge_key"] = None
        try:
            df = df.copy()
            df["_merge_key"] = df.apply(_mk_key_row, axis=1)
        except Exception:
            df["_merge_key"] = None

        # Keys for uniqueness
        subset = ["date", "_merge_key", "market", "line", "book"]

        # Mark currentness: only rows present in the latest fetch are current.
        try:
            existing["is_current"] = False
        except Exception:
            pass
        try:
            df["is_current"] = True
        except Exception:
            pass

        comb = pd.concat([existing, df], ignore_index=True)

        # Compute min first_seen_at per key for historical continuity
        try:
            fst_min = comb.groupby(subset)["first_seen_at"].min().reset_index().rename(columns={"first_seen_at": "_first_seen_min"})
        except Exception:
            fst_min = pd.DataFrame(columns=subset + ["_first_seen_min"])

        # Keep last occurrence per key (so newer prices/timestamps win)
        try:
            comb.sort_values(["date", "last_seen_at"], inplace=True)
        except Exception:
            pass
        try:
            dedup = comb.drop_duplicates(subset=subset, keep="last").copy()
        except Exception:
            dedup = comb.copy()

        # Attach min first_seen_at
        try:
            dedup = dedup.merge(fst_min, on=subset, how="left")
            dedup["first_seen_at"] = dedup["_first_seen_min"].where(dedup["_first_seen_min"].notna(), dedup.get("first_seen_at"))
            if "_first_seen_min" in dedup.columns:
                dedup.drop(columns=["_first_seen_min"], inplace=True)
        except Exception:
            pass

        # Ensure is_current aligns to latest fetched keys (in case of odd sort/dedup)
        try:
            cur_keys = df[subset].drop_duplicates().copy()
            cur_keys["__cur"] = True
            dedup = dedup.merge(cur_keys, on=subset, how="left")
            dedup["is_current"] = dedup["__cur"].fillna(False)
            dedup.drop(columns=["__cur"], inplace=True)
        except Exception:
            pass

        # Use this merged frame for writing
        df = dedup
    except Exception:
        # If merge fails, fall back to incoming df to avoid data loss
        df = df
    try:
        # Use pyarrow engine explicitly for stability across environments
        df.to_parquet(pq_path, index=False, engine="pyarrow")
        # Also write a CSV sidecar to guarantee simple downstream reads
        try:
            df.to_csv(csv_path, index=False)
        except Exception:
            # If CSV write fails, ensure at least an empty schema is present
            cols = [
                "date","player_id","player_name","team","market","line",
                "over_price","under_price","book","first_seen_at","last_seen_at","is_current"
            ]
            import pandas as _pd
            _pd.DataFrame(columns=cols).to_csv(csv_path, index=False)
        return pq_path
    except Exception:
        # Fallback to CSV if Parquet writer is unavailable/mismatched
        try:
            df.to_csv(csv_path, index=False)
        except Exception:
            # As a final fallback, write an empty CSV with standard columns
            cols = ["date","player_id","player_name","team","market","line","over_price","under_price","book","first_seen_at","last_seen_at","is_current"]
            import pandas as _pd
            _pd.DataFrame(columns=cols).to_csv(csv_path, index=False)
        return csv_path


def collect_and_write(date: str, roster_df: Optional[pd.DataFrame] = None, cfg: PropsCollectionConfig | None = None) -> Dict:
    cfg = cfg or PropsCollectionConfig()
    # Choose source
    src = (cfg.source or "oddsapi").strip().lower()
    if src == "bovada":
        raw = collect_bovada_props(date)
    else:
        raw = collect_oddsapi_props(date)
    # If no roster_df provided, attempt to build one for reliable player_id/team mapping
    if roster_df is None:
        try:
            roster_df = _load_cached_roster_enrichment(date)
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
                snap["team"] = snap["team_id"].map(id_to_abbr).fillna(snap["team_id"].map(id_to_name)).fillna(snap.get("team"))
            except Exception:
                # If team lookup fails, keep any existing team abbreviation before falling back to team_id.
                snap = snap.copy()
                snap["team"] = snap.get("team", snap.get("team_id"))
            out = snap.rename(columns={"full_name": "full_name", "player_id": "player_id"})
            return out[["full_name", "player_id", "team"]]
    except Exception:
        pass
    # Fallback to historical stats
    try:
        stats_p = _RAW_DIR / "player_game_stats.csv"
        if stats_p.exists():
            stats = pd.read_csv(stats_p)
            if not stats.empty and {"player"}.issubset(stats.columns):
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
                # Normalize potential team columns
                team_col = None
                for cand in ("team","team_abbr","teamAbbrev","team_abbrev","team_abbreviation"):
                    if cand in stats.columns:
                        team_col = cand; break
                if team_col is None:
                    stats['team'] = None
                    team_col = 'team'
                # Order by date to take last known mapping
                try:
                    stats["_date"] = pd.to_datetime(stats["date"], errors="coerce")
                    stats = stats.sort_values("_date")
                except Exception:
                    pass
                agg_map = {team_col: "last"}
                if "player_id" in stats.columns:
                    agg_map["player_id"] = "last"
                last = stats.groupby("player").agg(agg_map).reset_index().rename(columns={"player": "full_name", team_col: "team"})
                # Ensure clean strings
                last["full_name"] = last["full_name"].astype(str)
                if "player_id" not in last.columns:
                    last["player_id"] = None
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
