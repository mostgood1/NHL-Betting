from __future__ import annotations

import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

BASE_PREFIX = "https://www.bovada.lv/services/sports/event/coupon/events/A/description"
SPORT_PATHS = [
    "hockey/nhl",
    "hockey/nhl-preseason",
]


class BovadaClient:
    def __init__(self, rate_limit_per_sec: float = 3.0, timeout: int = 40):
        self.sleep = 1.0 / rate_limit_per_sec
        self.timeout = timeout

    def _get(self, url: str, params: Dict, referer: Optional[str] = None) -> List[Dict]:
        time.sleep(self.sleep)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://www.bovada.lv",
        }
        if referer:
            headers["Referer"] = referer
        r = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        try:
            return r.json() or []
        except Exception:
            # Bovada sometimes returns HTML (e.g., Cloudflare); treat as no data
            return []

    def fetch_events(self, pre_match_only: bool = True, market_filter: str = "def") -> List[Dict]:
        params = {
            "preMatchOnly": "true" if pre_match_only else "false",
            "lang": "en",
            "marketFilterId": market_filter,
            "eventsLimit": 200,
        }
        all_groups: List[Dict] = []
        for path in SPORT_PATHS:
            url = f"{BASE_PREFIX}/{path}"
            try:
                referer = f"https://www.bovada.lv/sports/{path}"
                groups = self._get(url, params, referer=referer)
                if groups:
                    all_groups.extend(groups)
            except Exception:
                # Ignore path-specific failures
                continue
        return all_groups

    @staticmethod
    def _ms_to_iso(ms: int) -> str:
        try:
            return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            return None

    @staticmethod
    def _event_teams(ev: Dict) -> Tuple[Optional[str], Optional[str]]:
        # Bovada events often have a 'competitors' list with home/away marking
        comps = ev.get("competitors") or []
        home = away = None
        for c in comps:
            side = (c.get("homeAway") or c.get("position") or "").upper()
            name = c.get("name") or c.get("description")
            if side in ("HOME", "H"):
                home = name
            elif side in ("AWAY", "A"):
                away = name
        # Fallback: try to parse from description like "Team A @ Team B"
        if not home or not away:
            desc = ev.get("description") or ""
            if " @ " in desc:
                parts = desc.split(" @ ")
                away = away or parts[0].strip()
                home = home or parts[1].strip()
        return home, away

    @staticmethod
    def _collect_markets(ev: Dict) -> Dict[str, any]:
        out: Dict[str, any] = {}
        def _norm_american(v):
            if v is None:
                return None
            s = str(v).strip().upper()
            if s in ("EVEN", "EV", "E"):
                return "+100"
            return s
        dgs = ev.get("displayGroups") or []
        for dg in dgs:
            markets = dg.get("markets") or []
            for m in markets:
                mdesc = (m.get("description") or "").lower()
                outcomes = m.get("outcomes") or []
                # Moneyline
                if "moneyline" in mdesc or m.get("key", "").lower() == "moneyline":
                    for oc in outcomes:
                        name = (oc.get("description") or oc.get("name") or "").strip()
                        price = oc.get("price") or {}
                        american = _norm_american(price.get("american"))
                        if not american:
                            continue
                        out[f"ml::{name}"] = american
                # Totals
                if "total" in mdesc:
                    for oc in outcomes:
                        ocdesc = (oc.get("description") or oc.get("name") or "").strip()
                        price = oc.get("price") or {}
                        american = _norm_american(price.get("american"))
                        # Bovada sometimes places the handicap on the price object
                        point = oc.get("handicap") if oc.get("handicap") is not None else price.get("handicap")
                        if ocdesc.lower() in ("over", "under"):
                            out[f"tot::{ocdesc}"] = american
                            # Preserve the first discovered total point
                            if point is not None and out.get("tot::point") is None:
                                try:
                                    out["tot::point"] = float(point)
                                except Exception:
                                    out["tot::point"] = point
                # Puckline / Spread
                if any(k in mdesc for k in ["puck line", "puckline", "spread"]):
                    for oc in outcomes:
                        ocname = (oc.get("description") or oc.get("name") or "").strip()
                        price = oc.get("price") or {}
                        american = _norm_american(price.get("american"))
                        try:
                            # Handicap may be on the price object
                            hcap = oc.get("handicap") if oc.get("handicap") is not None else price.get("handicap")
                            pt = float(hcap)
                        except Exception:
                            pt = None
                        if american and pt is not None and abs(pt) == 1.5:
                            out[f"pl::{ocname}::{pt}"] = american
        return out

    def fetch_game_odds(self, date: str) -> pd.DataFrame:
        """Fetch Bovada odds for a given YYYY-MM-DD.

        Strategy:
        - Try pre-match only (default markets), then broaden to 'all'.
        - If no matches for the requested date, retry with pre_match_only=False (includes in-play), for both filters.
        - Filter returned rows to the requested date to avoid cross-date noise.
        """
        # Try a few combinations
        attempts = [
            (True, "def"),
            (True, "all"),
            (False, "def"),
            (False, "all"),
        ]
        events: List[Dict] = []
        for prematch, mf in attempts:
            try:
                groups = self.fetch_events(pre_match_only=prematch, market_filter=mf)
                if groups:
                    events = groups
                    break
            except Exception:
                continue
        rows: List[Dict] = []
        for group in events:
            evs = group.get("events") or []
            for ev in evs:
                start_ms = ev.get("startTime")
                iso = self._ms_to_iso(start_ms) if start_ms else None
                date_key = None
                try:
                    if iso:
                        # Map Bovada start time to US/Eastern calendar day to match slate grouping
                        dt_utc = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                        dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
                        date_key = dt_et.strftime("%Y-%m-%d")
                except Exception:
                    pass
                # Filter strictly to requested date if we have a date_key
                if date_key and date_key != date:
                    continue
                home, away = self._event_teams(ev)
                if not home or not away:
                    continue
                prices = self._collect_markets(ev)
                row = {
                    "date": date_key,
                    "home": home,
                    "away": away,
                    "home_ml": prices.get(f"ml::{home}"),
                    "away_ml": prices.get(f"ml::{away}"),
                    "over": prices.get("tot::Over") or prices.get("tot::over"),
                    "under": prices.get("tot::Under") or prices.get("tot::under"),
                    "total_line": prices.get("tot::point"),
                    "home_pl_-1.5": prices.get(f"pl::{home}::-1.5"),
                    "away_pl_+1.5": prices.get(f"pl::{away}::1.5"),
                    "home_ml_book": "bovada",
                    "away_ml_book": "bovada",
                    "over_book": "bovada",
                    "under_book": "bovada",
                    "home_pl_-1.5_book": "bovada",
                    "away_pl_+1.5_book": "bovada",
                }
                rows.append(row)
        # If we couldn't determine a date_key, as a last resort include undated rows for this fetch
        df = pd.DataFrame(rows)
        return df

    def fetch_props_odds(self, date: str) -> pd.DataFrame:
        # Attempt to fetch more markets; Bovada's props are grouped differently
        events = self.fetch_events(pre_match_only=True, market_filter="all")
        rows: List[Dict] = []
        def _et_date(iso: Optional[str]) -> Optional[str]:
            try:
                if not iso:
                    return None
                dt_utc = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
                return dt_et.strftime("%Y-%m-%d")
            except Exception:
                return None
        def _market_to_code(mdesc: str) -> Optional[str]:
            md = (mdesc or "").lower()
            if "shots on goal" in md or "sog" in md:
                return "SOG"
            if ("player" in md and "goals" in md) or md.strip() == "goals":
                return "GOALS"
            if "saves" in md:
                return "SAVES"
            if ("player" in md and "assists" in md) or md.strip() == "assists":
                return "ASSISTS"
            if ("player" in md and "points" in md) or md.strip() == "points":
                return "POINTS"
            return None
        def _extract_player_and_side(oc: Dict, market_desc: str) -> Tuple[Optional[str], Optional[str]]:
            # Try explicit Over/Under in outcome description/name
            side = None
            desc = (oc.get("description") or "").strip()
            name = (oc.get("name") or "").strip()
            txt = f"{desc} {name}".strip()
            lower = txt.lower()
            if "over" in lower and not "under" in lower:
                side = "OVER"
            elif "under" in lower and not "over" in lower:
                side = "UNDER"
            # Player candidates
            player = None
            if side == "OVER":
                player = txt.replace("Over", "").replace("over", "").strip(" -") or None
            elif side == "UNDER":
                player = txt.replace("Under", "").replace("under", "").strip(" -") or None
            if not player:
                player = oc.get("participant") or oc.get("competitor") or oc.get("competitorName")
            if not player:
                if " - " in (market_desc or ""):
                    player = market_desc.split(" - ")[0].strip()
            if player:
                player = str(player).strip()
                if player.lower() in ("over", "under"):
                    player = None
            return player, side
        for group in events:
            evs = group.get("events") or []
            for ev in evs:
                start_ms = ev.get("startTime")
                iso = self._ms_to_iso(start_ms) if start_ms else None
                date_key = _et_date(iso)
                if date_key != date:
                    continue
                dgs = ev.get("displayGroups") or []
                for dg in dgs:
                    markets = dg.get("markets") or []
                    for m in markets:
                        mdesc = (m.get("description") or "")
                        market_code = _market_to_code(mdesc)
                        if not market_code:
                            continue
                        outcomes = m.get("outcomes") or []
                        for oc in outcomes:
                            price = oc.get("price") or {}
                            american = price.get("american")
                            hcap = oc.get("handicap") if oc.get("handicap") is not None else price.get("handicap")
                            try:
                                line = float(hcap) if hcap is not None else None
                            except Exception:
                                line = None
                            player, side = _extract_player_and_side(oc, mdesc)
                            if not side:
                                od = (oc.get("description") or "").strip().lower()
                                if od in ("over", "under"):
                                    side = od.upper()
                            if not player:
                                continue
                            if american is None or line is None or side not in ("OVER","UNDER"):
                                continue
                            rows.append({
                                "market": market_code,
                                "player": player,
                                "line": line,
                                "odds": american,
                                "side": side,
                                "book": "bovada",
                            })
        return pd.DataFrame(rows)
