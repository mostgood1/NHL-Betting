from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
from ..utils.odds import american_to_decimal
import pandas as pd


ODDS_API_BASE = "https://api.the-odds-api.com/v4"


BOOKMAKER_PRIORITY = [
    "pinnacle",
    "fanduel",
    "draftkings",
    "betmgm",
    "caesars",
    "pointsbetus",
    "betonlineag",
    "bovada",
    "unibet",
    "betrivers",
    "sugarhouse",
    "barstool",
]


class OddsAPIClient:
    def __init__(self, api_key: Optional[str] = None, rate_limit_per_sec: float = 3.0):
        # Attempt to load .env at repo root so users can store secrets locally
        if load_dotenv is not None:
            try:
                root = Path(__file__).resolve().parents[2]
                dotenv_path = root / ".env"
                if dotenv_path.exists():
                    load_dotenv(dotenv_path)
            except Exception:
                pass
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set ODDS_API_KEY env var or pass api_key to OddsAPIClient.")
        self.sleep = 1.0 / rate_limit_per_sec

    def _get(self, path: str, params: Dict) -> Tuple[Dict, Dict[str, str]]:
        time.sleep(self.sleep)
        url = f"{ODDS_API_BASE}{path}"
        r = requests.get(url, params=params, timeout=40)
        hdrs = {k.lower(): v for k, v in r.headers.items()}
        r.raise_for_status()
        return r.json(), hdrs

    def historical_odds_snapshot(
        self,
        sport: str,
        snapshot_iso: str,
        regions: str = "us",
        markets: str = "h2h,totals,spreads",
        odds_format: str = "american",
    ) -> Tuple[Dict, Dict[str, str]]:
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "date": snapshot_iso,
            "oddsFormat": odds_format,
        }
        return self._get(f"/historical/sports/{sport}/odds", params)


def _pick_bookmaker(bookmakers: List[Dict], preferred: Optional[str]) -> Optional[Dict]:
    if not bookmakers:
        return None
    if preferred and any(b.get("key") == preferred for b in bookmakers):
        return next(b for b in bookmakers if b.get("key") == preferred)
    for bk in BOOKMAKER_PRIORITY:
        for b in bookmakers:
            if b.get("key") == bk:
                return b
    return bookmakers[0]


def _extract_prices_from_markets(markets: List[Dict]) -> Dict[str, any]:
    out: Dict[str, any] = {}
    # Moneyline
    m_h2h = next((m for m in markets if m.get("key") == "h2h"), None)
    if m_h2h:
        for oc in m_h2h.get("outcomes", []):
            name = str(oc.get("name"))
            price = oc.get("price")
            # store under dynamic keys; caller will map by team name
            out[f"ml::{name}"] = price
    # Totals (single main line assumed)
    m_tot = next((m for m in markets if m.get("key") == "totals"), None)
    if m_tot:
        # Often two outcomes: Over/Under
        pts = None
        for oc in m_tot.get("outcomes", []):
            if oc.get("name") in ("Over", "Under"):
                if pts is None:
                    pts = oc.get("point")
                out[f"tot::{oc.get('name')}"] = oc.get("price")
                out["tot::point"] = pts
    # Spreads (puck line) - look for +/-1.5
    m_spr = next((m for m in markets if m.get("key") == "spreads"), None)
    if m_spr:
        for oc in m_spr.get("outcomes", []):
            try:
                pt = float(oc.get("point"))
            except Exception:
                continue
            if abs(pt) == 1.5:
                out[f"pl::{oc.get('name')}::{pt}"] = oc.get("price")
    return out


def normalize_snapshot_to_rows(
    snapshot,
    bookmaker: Optional[str] = None,
    best_of_all: bool = False,
) -> pd.DataFrame:
    """
    Convert Odds API historical odds snapshot into a flat DataFrame with columns:
    [date, home, away, home_ml, away_ml, over, under, total_line, home_pl_-1.5, away_pl_+1.5]
    """
    # Support both historical (dict with 'data') and current (list) responses
    if isinstance(snapshot, list):
        data = snapshot
    else:
        data = snapshot.get("data", [])
    rows: List[Dict] = []
    for ev in data:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        dstr = None
        try:
            dstr = datetime.fromisoformat(commence.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except Exception:
            dstr = None
        if not best_of_all:
            book = _pick_bookmaker(ev.get("bookmakers", []), bookmaker)
            if not book:
                continue
            prices = _extract_prices_from_markets(book.get("markets", []))

            # Moneyline
            home_ml = prices.get(f"ml::{home}")
            away_ml = prices.get(f"ml::{away}")
            # Totals
            over = prices.get("tot::Over")
            under = prices.get("tot::Under")
            total_line = prices.get("tot::point")
            # Puck line
            home_pl = prices.get(f"pl::{home}::-1.5")
            away_pl = prices.get(f"pl::{away}::1.5")

            rows.append(
                {
                    "date": dstr,
                    "home": home,
                    "away": away,
                    "home_ml": home_ml,
                    "away_ml": away_ml,
                    "over": over,
                    "under": under,
                    "total_line": total_line,
                    "home_pl_-1.5": home_pl,
                    "away_pl_+1.5": away_pl,
                    "home_ml_book": book.get("key"),
                    "away_ml_book": book.get("key"),
                    "over_book": book.get("key"),
                    "under_book": book.get("key"),
                    "home_pl_-1.5_book": book.get("key"),
                    "away_pl_+1.5_book": book.get("key"),
                }
            )
        else:
            # Aggregate best odds across all bookmakers for key markets
            bks: List[Dict] = ev.get("bookmakers", [])
            best = {
                "home_ml": (None, None),  # (american, book)
                "away_ml": (None, None),
                "over": (None, None),
                "under": (None, None),
                "home_pl_-1.5": (None, None),
                "away_pl_+1.5": (None, None),
            }
            # Gather totals points to pick the modal total line
            totals_points: List[float] = []
            book_prices: List[Tuple[str, Dict[str, any]]] = []
            for b in bks:
                pr = _extract_prices_from_markets(b.get("markets", []))
                book_prices.append((b.get("key"), pr))
                if "tot::point" in pr and pr["tot::point"] is not None:
                    try:
                        totals_points.append(float(pr["tot::point"]))
                    except Exception:
                        pass
            totals_point = None
            if totals_points:
                # mode; if tie, pick the first
                totals_point = max(set(totals_points), key=totals_points.count)
            # Helper to update best (maximize decimal odds)
            def upd(key: str, american: Optional[float], book_key: str):
                if american is None:
                    return
                try:
                    dec = american_to_decimal(float(american))
                except Exception:
                    return
                cur = best[key][0]
                cur_dec = american_to_decimal(float(cur)) if cur is not None else -1.0
                if dec > cur_dec:
                    best[key] = (american, book_key)

            for bkey, pr in book_prices:
                # Moneyline
                upd("home_ml", pr.get(f"ml::{home}"), bkey)
                upd("away_ml", pr.get(f"ml::{away}"), bkey)
                # Totals at chosen point
                if totals_point is not None and pr.get("tot::point") is not None:
                    try:
                        if float(pr.get("tot::point")) == float(totals_point):
                            upd("over", pr.get("tot::Over"), bkey)
                            upd("under", pr.get("tot::Under"), bkey)
                    except Exception:
                        pass
                # Puckline at +/-1.5
                upd("home_pl_-1.5", pr.get(f"pl::{home}::-1.5"), bkey)
                upd("away_pl_+1.5", pr.get(f"pl::{away}::1.5"), bkey)

            rows.append(
                {
                    "date": dstr,
                    "home": home,
                    "away": away,
                    "home_ml": best["home_ml"][0],
                    "away_ml": best["away_ml"][0],
                    "over": best["over"][0],
                    "under": best["under"][0],
                    "total_line": totals_point,
                    "home_pl_-1.5": best["home_pl_-1.5"][0],
                    "away_pl_+1.5": best["away_pl_+1.5"][0],
                    "home_ml_book": best["home_ml"][1],
                    "away_ml_book": best["away_ml"][1],
                    "over_book": best["over"][1],
                    "under_book": best["under"][1],
                    "home_pl_-1.5_book": best["home_pl_-1.5"][1],
                    "away_pl_+1.5_book": best["away_pl_+1.5"][1],
                }
            )
    df = pd.DataFrame(rows)
    return df
