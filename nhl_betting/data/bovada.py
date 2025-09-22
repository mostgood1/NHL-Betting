from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

BASE_PREFIX = "https://www.bovada.lv/services/sports/event/coupon/events/A/description"
SPORT_PATHS = [
    "ice-hockey/nhl",
    "ice-hockey/nhl-preseason",
]


class BovadaClient:
    def __init__(self, rate_limit_per_sec: float = 3.0, timeout: int = 40):
        self.sleep = 1.0 / rate_limit_per_sec
        self.timeout = timeout

    def _get(self, url: str, params: Dict) -> List[Dict]:
        time.sleep(self.sleep)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
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
        }
        all_groups: List[Dict] = []
        for path in SPORT_PATHS:
            url = f"{BASE_PREFIX}/{path}"
            try:
                groups = self._get(url, params)
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
                        american = price.get("american")
                        if not american:
                            continue
                        out[f"ml::{name}"] = american
                # Totals
                if "total" in mdesc:
                    for oc in outcomes:
                        ocdesc = (oc.get("description") or oc.get("name") or "").strip()
                        price = oc.get("price") or {}
                        american = price.get("american")
                        point = oc.get("handicap")
                        if ocdesc.lower() in ("over", "under"):
                            out[f"tot::{ocdesc}"] = american
                            out["tot::point"] = point
                # Puckline / Spread
                if any(k in mdesc for k in ["puck line", "puckline", "spread"]):
                    for oc in outcomes:
                        ocname = (oc.get("description") or oc.get("name") or "").strip()
                        price = oc.get("price") or {}
                        american = price.get("american")
                        try:
                            pt = float(oc.get("handicap"))
                        except Exception:
                            pt = None
                        if american and pt is not None and abs(pt) == 1.5:
                            out[f"pl::{ocname}::{pt}"] = american
        return out

    def fetch_game_odds(self, date: str) -> pd.DataFrame:
        # Fetch default markets and filter by date
        # Try default markets; if empty, fall back to 'all' to broaden
        events = self.fetch_events(pre_match_only=True, market_filter="def")
        if not events:
            events = self.fetch_events(pre_match_only=True, market_filter="all")
        rows: List[Dict] = []
        y, m, d = date.split("-")
        for group in events:
            evs = group.get("events") or []
            for ev in evs:
                start_ms = ev.get("startTime")
                iso = self._ms_to_iso(start_ms) if start_ms else None
                date_key = None
                try:
                    if iso:
                        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                        date_key = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
                if date_key != date:
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
        return pd.DataFrame(rows)

    def fetch_props_odds(self, date: str) -> pd.DataFrame:
        # Attempt to fetch more markets; Bovada's props are grouped differently
        events = self.fetch_events(pre_match_only=True, market_filter="all")
        rows: List[Dict] = []
        for group in events:
            evs = group.get("events") or []
            for ev in evs:
                start_ms = ev.get("startTime")
                iso = self._ms_to_iso(start_ms) if start_ms else None
                date_key = None
                try:
                    if iso:
                        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                        date_key = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
                if date_key != date:
                    continue
                dgs = ev.get("displayGroups") or []
                for dg in dgs:
                    markets = dg.get("markets") or []
                    for m in markets:
                        mdesc = (m.get("description") or "").lower()
                        # Map common player prop markets
                        market_code = None
                        if "shots on goal" in mdesc:
                            market_code = "SOG"
                        elif "player goals" in mdesc or ("goals" in mdesc and "player" in mdesc):
                            market_code = "GOALS"
                        elif "saves" in mdesc:
                            market_code = "SAVES"
                        if not market_code:
                            continue
                        # Player name often appears in market description like "Player Name - Shots on Goal"
                        # Try extracting text before a hyphen
                        raw_desc = m.get("description") or ""
                        player = None
                        if " - " in raw_desc:
                            player = raw_desc.split(" - ")[0].strip()
                        outcomes = m.get("outcomes") or []
                        for oc in outcomes:
                            ocdesc = (oc.get("description") or oc.get("name") or "").strip().lower()
                            if ocdesc not in ("over", "under"):
                                continue
                            price = oc.get("price") or {}
                            american = price.get("american")
                            try:
                                line = float(oc.get("handicap")) if oc.get("handicap") is not None else None
                            except Exception:
                                line = None
                            if american is None or line is None:
                                continue
                            rows.append({
                                "market": market_code,
                                "player": player or "",
                                "line": line,
                                "odds": american,
                                "side": ocdesc.upper(),
                                "book": "bovada",
                            })
        return pd.DataFrame(rows)
