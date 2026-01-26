from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

TEAM_SLUGS: Dict[str, str] = {
    # Eastern Conference
    "BOS": "boston-bruins",
    "BUF": "buffalo-sabres",
    "DET": "detroit-red-wings",
    "FLA": "florida-panthers",
    "MTL": "montreal-canadiens",
    "OTT": "ottawa-senators",
    "TBL": "tampa-bay-lightning",
    "TOR": "toronto-maple-leafs",
    "CAR": "carolina-hurricanes",
    "CBJ": "columbus-blue-jackets",
    "NJD": "new-jersey-devils",
    "NYI": "new-york-islanders",
    "NYR": "new-york-rangers",
    "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins",
    "WSH": "washington-capitals",
    # Western Conference
    "ANA": "anaheim-ducks",
    "ARI": "arizona-coyotes",
    "CGY": "calgary-flames",
    "CHI": "chicago-blackhawks",
    "COL": "colorado-avalanche",
    "DAL": "dallas-stars",
    "EDM": "edmonton-oilers",
    "LAK": "los-angeles-kings",
    "MIN": "minnesota-wild",
    "NSH": "nashville-predators",
    "SEA": "seattle-kraken",
    "SJS": "san-jose-sharks",
    "STL": "st-louis-blues",
    "VAN": "vancouver-canucks",
    "VGK": "vegas-golden-knights",
    "WPG": "winnipeg-jets",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; arm64) NHL-Betting Bot",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

class LineupItem(Dict):
    """Dict-style lineup item with fields:
    player_name, team, position, line_slot, pp_unit, pk_unit, confidence
    """


def _extract_next_data(soup: BeautifulSoup) -> Optional[dict]:
    # Try Next.js __NEXT_DATA__ payload first
    for script in soup.find_all("script"):
        t = script.get_text(strip=True)
        if not t:
            continue
        if "__NEXT_DATA__" in t or t.startswith("{"):
            # Heuristic: find the largest JSON-looking blob
            try:
                data = json.loads(t)
                return data
            except Exception:
                continue
    return None


def _simple_text_parse(soup: BeautifulSoup) -> List[LineupItem]:
    items: List[LineupItem] = []
    # Fallback: parse anchor tags under obvious sections
    # Group by visible headings like "Forwards" / "Defense" / "Power Play" / "Penalty Kill"
    sections = {}
    for h in soup.find_all(["h2", "h3", "h4"]):
        txt = (h.get_text(" ", strip=True) or "").upper()
        if any(k in txt for k in ("FORWARDS", "DEFENSE", "POWER PLAY", "PENALTY KILL")):
            sections[txt] = h
    def collect_players(start_tag) -> List[str]:
        names = []
        parent = start_tag.find_parent()
        if not parent:
            parent = start_tag.next_sibling or start_tag
        for a in parent.find_all("a"):
            href = a.get("href") or ""
            name = a.get_text(" ", strip=True)
            if name and "/players/" in href:
                names.append(name)
        return names
    # Forwards
    fw = []
    for k, h in sections.items():
        if "FORWARDS" in k:
            fw = collect_players(h)
    # Defense
    df = []
    for k, h in sections.items():
        if "DEFENSE" in k:
            df = collect_players(h)
    # PP / PK units
    pp = []
    pk = []
    for k, h in sections.items():
        if "POWER PLAY" in k:
            pp = collect_players(h)
        if "PENALTY KILL" in k:
            pk = collect_players(h)
    # Assign naive slots by order
    def _mk_items(names: List[str], pos_hint: str, slot_prefix: str) -> List[LineupItem]:
        res = []
        line_num = 1
        count_in_line = 0
        for nm in names:
            res.append(LineupItem(player_name=nm, position=pos_hint, line_slot=f"{slot_prefix}{line_num}", pp_unit=None, pk_unit=None, confidence=0.4))
            count_in_line += 1
            if slot_prefix == "L" and count_in_line >= 3:  # LW-C-RW grouped
                line_num += 1
                count_in_line = 0
            elif slot_prefix == "D" and count_in_line >= 2:
                line_num += 1
                count_in_line = 0
        return res
    items.extend(_mk_items(fw, "F", "L"))
    items.extend(_mk_items(df, "D", "D"))
    # Tag PP/PK membership
    def _tag_unit(names: List[str], col: str, unit_name: str):
        for nm in names:
            for it in items:
                if it.get("player_name") == nm:
                    it[col] = unit_name
                    it["confidence"] = max(it.get("confidence", 0.4), 0.5)
    if pp:
        _tag_unit(pp[:5], "pp_unit", "PP1")
        _tag_unit(pp[5:], "pp_unit", "PP2")
    if pk:
        _tag_unit(pk[:4], "pk_unit", "PK1")
        _tag_unit(pk[4:], "pk_unit", "PK2")
    return items


def fetch_dailyfaceoff_team_lineups(team_abbr: str, date: str) -> List[LineupItem]:
    slug = TEAM_SLUGS.get(team_abbr.upper())
    if not slug:
        return []
    url = f"https://www.dailyfaceoff.com/teams/{slug}/line-combinations/?date={date}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Try to parse Next.js data first; fallback to simple text parse
        data = _extract_next_data(soup)
        items: List[LineupItem] = []
        if isinstance(data, dict):
            # Heuristic: search for line combination arrays in JSON
            blob = json.dumps(data)
            # Very defensive: pull player names by known key words
            names = []
            for key in ("playerName", "fullName", "name"):
                # crude scan to avoid tight coupling with schema
                # Note: not ideal; improves over time if schema reverse-engineered
                pass
            # Fallback to text parse when schema unknown
            items = _simple_text_parse(soup)
        else:
            items = _simple_text_parse(soup)
        # Attach team
        for it in items:
            it["team"] = team_abbr.upper()
        # Polite throttle
        time.sleep(0.5)
        return items
    except Exception:
        return []


def fetch_dailyfaceoff_lineups(date: str, team_abbrs: Optional[List[str]] = None) -> List[LineupItem]:
    abbrs = team_abbrs or list(TEAM_SLUGS.keys())
    out: List[LineupItem] = []
    for ab in abbrs:
        out.extend(fetch_dailyfaceoff_team_lineups(ab, date))
    return out

__all__ = [
    "fetch_dailyfaceoff_lineups",
    "fetch_dailyfaceoff_team_lineups",
    "TEAM_SLUGS",
]


def fetch_dailyfaceoff_starting_goalies(date: str) -> List[Dict]:
    """Fetch a best-effort list of starting goalies for the given date.

    Returns rows with: team, goalie, status, confidence, source.
    """
    url = "https://www.dailyfaceoff.com/starting-goalies"
    rows: List[Dict] = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Best-effort parsing: find team blocks and extract goalie names
        # This is intentionally loose to avoid brittle coupling.
        for card in soup.find_all(["article", "div"], class_=lambda c: c and ("starting-goalies" in c or "Card" in c or "goalies" in c)):
            txt = card.get_text(" ", strip=True)
            # Extract a goalie name by players link
            goalies = []
            for a in card.find_all("a"):
                href = a.get("href") or ""
                nm = a.get_text(" ", strip=True)
                if nm and "/players/" in href:
                    goalies.append(nm)
            team = None
            # Try to infer team by presence of team slug in links/images nearby
            for a in card.find_all("a"):
                href = a.get("href") or ""
                for ab, slug in TEAM_SLUGS.items():
                    if slug in href:
                        team = ab
                        break
                if team:
                    break
            status = "Probable"
            conf = 0.5
            if goalies:
                rows.append({"team": team, "goalie": goalies[0], "status": status, "confidence": conf, "source": "dailyfaceoff"})
        # Polite throttle
        time.sleep(0.2)
    except Exception:
        pass
    return rows

