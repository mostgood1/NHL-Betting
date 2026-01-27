from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import time

import pandas as pd

from .player_props import PropsCollectionConfig, write_props, normalize_player_names, combine_over_under, _utc_now_iso  # type: ignore


def _canon_market(desc: str) -> Optional[str]:
    s = (desc or "").strip().lower()
    if not s:
        return None
    if "shots on goal" in s or "shots-on-goal" in s or ("shots" in s and "goal" in s):
        return "SOG"
    if "blocked" in s or "blocks" in s:
        return "BLOCKS"
    if "assists" in s:
        return "ASSISTS"
    if "points" in s:
        return "POINTS"
    if "saves" in s or "goalie saves" in s:
        return "SAVES"
    if "goals" in s:
        return "GOALS"
    return None


def _extract_rows_from_json_obj(obj: dict, date: str) -> List[Dict]:
    rows: List[Dict] = []
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
    # Bovada JSON typically has events -> displayGroups -> markets -> outcomes
    events = []
    if isinstance(obj, dict):
        if isinstance(obj.get("events"), list):
            events = [e for e in obj["events"] if isinstance(e, dict)]
        elif isinstance(obj.get("event"), dict):
            events = [obj["event"]]
    elif isinstance(obj, list):
        for g in obj:
            if isinstance(g, dict) and isinstance(g.get("events"), list):
                events.extend([e for e in g["events"] if isinstance(e, dict)])
    for ev in events:
        dgs = ev.get("displayGroups") or []
        for dg in dgs:
            mkts = dg.get("markets") or []
            for m in mkts:
                mdesc = m.get("description") or m.get("displayKey") or m.get("key") or ""
                canon = _canon_market(mdesc)
                if not canon:
                    continue
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
                        "collected_at": _utc_now_iso(),
                    })
    return rows


def collect_from_event_page(url: str, date: str, wait_sec: float = 6.0, headless: bool = True) -> pd.DataFrame:
    """Render a Bovada event page with Playwright, capture network JSON, and parse markets.

    Targets Goalie Saves / Blocked Shots and any OU numeric markets.
    """
    from playwright.sync_api import sync_playwright
    rows: List[Dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"),
            viewport={"width": 1366, "height": 900},
            java_script_enabled=True,
        )
        page = ctx.new_page()
        # Capture JSON responses
        def on_response(resp):
            try:
                ct = resp.headers.get("content-type", "")
            except Exception:
                ct = ""
            url_l = resp.url.lower()
            if "services/sports/event" in url_l and "json" in ct:
                try:
                    txt = resp.text()
                    obj = json.loads(txt)
                    rows.extend(_extract_rows_from_json_obj(obj, date))
                except Exception:
                    return
        page.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded")
        # Give some time for lazy XHRs
        time.sleep(max(wait_sec, 8.0))
        # As a fallback, evaluate window.__INITIAL_STATE__ if present
        try:
            data = page.evaluate("() => window.__INITIAL_STATE__ || null")
            if data:
                rows.extend(_extract_rows_from_json_obj(data, date))
        except Exception:
            pass
        ctx.close(); browser.close()
    # Dedup
    if not rows:
        return pd.DataFrame(columns=["market","player","line","odds","side","book","date","collected_at"])
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["date","player","market","line","side","book"], keep="last")
    return df


def collect_and_write_from_page(url: str, date: str) -> Dict:
    raw = collect_from_event_page(url, date)
    # Build roster enrichment best-effort
    roster_df = None
    try:
        from ..utils.io import PROC_DIR as _PROC
        p_master = _PROC / "roster_master.csv"
        if p_master.exists() and getattr(p_master.stat(), "st_size", 0) > 64:
            r = pd.read_csv(p_master)
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
    except Exception:
        roster_df = None
    norm = normalize_player_names(raw, roster_df)
    comb = combine_over_under(norm)
    cfg = PropsCollectionConfig(output_root="data/props", book="bovada", source="bovada")
    out_path = write_props(comb, cfg, date)
    return {"raw_count": len(raw), "combined_count": len(comb), "output_path": out_path}


__all__ = [
    "collect_from_event_page",
    "collect_and_write_from_page",
]


def collect_from_category_page(url: str, date: str, visit_events: bool = True, max_events: int = 20, wait_sec: float = 8.0, headless: bool = True) -> pd.DataFrame:
    """Scrape the NHL category page and optionally visit each event to capture props.

    Strategy:
    - Open the category page and capture any coupon/event JSON responses.
    - Extract event links from DOM that look like NHL game pages.
    - Visit up to max_events event pages and reuse the same response-capture logic.
    """
    from playwright.sync_api import sync_playwright
    import re
    rows: List[Dict] = []
    def _cap_rows(obj):
        try:
            rows.extend(_extract_rows_from_json_obj(obj, date))
        except Exception:
            pass
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"),
            viewport={"width": 1366, "height": 900},
            java_script_enabled=True,
        )
        page = ctx.new_page()
        def on_response(resp):
            try:
                ct = resp.headers.get("content-type", "")
            except Exception:
                ct = ""
            url_l = resp.url.lower()
            if "services/sports/event" in url_l and ("json" in ct or url_l.endswith("json")):
                try:
                    txt = resp.text()
                    obj = json.loads(txt)
                    _cap_rows(obj)
                except Exception:
                    return
        page.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded")
        time.sleep(wait_sec)
        # Capture from window state if present
        try:
            data = page.evaluate("() => window.__INITIAL_STATE__ || null")
            if data:
                _cap_rows(data)
        except Exception:
            pass
        # Collect event links
        event_links: List[str] = []
        try:
            hrefs = page.evaluate("() => Array.from(document.querySelectorAll('a')).map(a=>a.href)")
            if isinstance(hrefs, list):
                for h in hrefs:
                    try:
                        s = str(h)
                        if "/sports/hockey/nhl/" in s and re.search(r"\d{12}$", s):
                            event_links.append(s)
                    except Exception:
                        continue
        except Exception:
            pass
        # Visit each event page (limited)
        if visit_events and event_links:
            seen = set()
            for ev_url in event_links[:max_events]:
                if ev_url in seen: continue
                seen.add(ev_url)
                try:
                    page.goto(ev_url, wait_until="domcontentloaded")
                    time.sleep(wait_sec)
                    try:
                        data = page.evaluate("() => window.__INITIAL_STATE__ || null")
                        if data:
                            _cap_rows(data)
                    except Exception:
                        pass
                except Exception:
                    continue
        ctx.close(); browser.close()
    if not rows:
        return pd.DataFrame(columns=["market","player","line","odds","side","book","date","collected_at"])
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["date","player","market","line","side","book"], keep="last")
    return df


def collect_and_write_from_category(url: str, date: str) -> Dict:
    raw = collect_from_category_page(url, date)
    # Roster enrichment
    roster_df = None
    try:
        from ..utils.io import PROC_DIR as _PROC
        p_master = _PROC / "roster_master.csv"
        if p_master.exists() and getattr(p_master.stat(), "st_size", 0) > 64:
            r = pd.read_csv(p_master)
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
    except Exception:
        roster_df = None
    norm = normalize_player_names(raw, roster_df)
    comb = combine_over_under(norm)
    cfg = PropsCollectionConfig(output_root="data/props", book="bovada", source="bovada")
    out_path = write_props(comb, cfg, date)
    return {"raw_count": len(raw), "combined_count": len(comb), "output_path": out_path}

