from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
import re
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
        def _load_env_file_fallback(path: Path) -> dict[str, str]:
            out: dict[str, str] = {}
            try:
                for raw in path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("\"").strip("'")
                    if k:
                        out[k] = v
            except Exception:
                return {}
            return out

        # Attempt to load .env at repo root so users can store secrets locally
        try:
            root = Path(__file__).resolve().parents[2]
            dotenv_path = root / ".env"
            if dotenv_path.exists():
                if load_dotenv is not None:
                    # override=True so local .env wins over a stale system ODDS_API_KEY
                    load_dotenv(dotenv_path, override=True)
                else:
                    # Minimal fallback for environments without python-dotenv.
                    env_map = _load_env_file_fallback(dotenv_path)
                    if "ODDS_API_KEY" in env_map:
                        os.environ["ODDS_API_KEY"] = env_map["ODDS_API_KEY"]
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
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raw_url = str(getattr(r, "url", ""))
            safe_url = re.sub(r"(apiKey=)[^&]+", r"\1REDACTED", raw_url)
            raise requests.HTTPError(f"{r.status_code} Client Error for url: {safe_url}") from None
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

    def list_events(
        self,
        sport: str,
        commence_from_iso: Optional[str] = None,
        commence_to_iso: Optional[str] = None,
        date_format: str = "iso",
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """List current events for a sport. Does not count against quota."""
        params = {
            "apiKey": self.api_key,
            "dateFormat": date_format,
        }
        if commence_from_iso:
            params["commenceTimeFrom"] = commence_from_iso
        if commence_to_iso:
            params["commenceTimeTo"] = commence_to_iso
        return self._get(f"/sports/{sport}/events", params)

    def historical_list_events(
        self,
        sport: str,
        snapshot_iso: str,
        date_format: str = "iso",
    ) -> Tuple[Dict, Dict[str, str]]:
        """List historical events snapshot for a sport at a timestamp (costs 1 if data returned)."""
        params = {
            "apiKey": self.api_key,
            "date": snapshot_iso,
            "dateFormat": date_format,
        }
        return self._get(f"/historical/sports/{sport}/events", params)

    def event_odds(
        self,
        sport: str,
        event_id: str,
        markets: str,
        regions: str = "us",
        bookmakers: Optional[str] = None,
        odds_format: str = "american",
        date_format: str = "iso",
    ) -> Tuple[Dict, Dict[str, str]]:
        """Get current event odds for specified markets (supports player_* markets)."""
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return self._get(f"/sports/{sport}/events/{event_id}/odds", params)

    def historical_event_odds(
        self,
        sport: str,
        event_id: str,
        markets: str,
        snapshot_iso: str,
        regions: str = "us",
        bookmakers: Optional[str] = None,
        odds_format: str = "american",
        date_format: str = "iso",
    ) -> Tuple[Dict, Dict[str, str]]:
        """Get historical event odds snapshot for specified markets (supports player_* markets)."""
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
            "date": snapshot_iso,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return self._get(f"/historical/sports/{sport}/events/{event_id}/odds", params)

    def event_markets(
        self,
        sport: str,
        event_id: str,
        regions: str = "us",
        bookmakers: Optional[str] = None,
        date_format: str = "iso",
    ) -> Tuple[Dict, Dict[str, str]]:
        """Get available market keys for a single event (returns keys only)."""
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "dateFormat": date_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return self._get(f"/sports/{sport}/events/{event_id}/markets", params)

    def flat_snapshot(
        self,
        iso_date: str,
        regions: str = "us",
        markets: str = "h2h,totals,spreads",
        snapshot_iso: Optional[str] = None,
        odds_format: str = "american",
        bookmaker: Optional[str] = None,
        best: bool = False,
        inplay: bool = False,
    ) -> pd.DataFrame:
        """
        Convenience: fetch odds (historical if snapshot_iso provided, else current),
        and return a normalized DataFrame via normalize_snapshot_to_rows.

        - Tries regular season sport key first, then preseason fallback
        - Supports best-of-all-bookmakers aggregation when best=True
        """

        def _utc_window_for_et_date(d_ymd: str) -> Tuple[Optional[str], Optional[str]]:
            try:
                tz_et = ZoneInfo("America/New_York")
                d0 = datetime.strptime(str(d_ymd), "%Y-%m-%d").replace(tzinfo=tz_et)
                d1 = d0 + timedelta(days=1)
                start = d0.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                end = d1.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                return start, end
            except Exception:
                return None, None

        sport_keys = ["icehockey_nhl", "icehockey_nhl_preseason"]
        df = pd.DataFrame([])
        # Historical snapshot path
        if snapshot_iso:
            for sk in sport_keys:
                try:
                    snap, _ = self.historical_odds_snapshot(
                        sport=sk,
                        snapshot_iso=snapshot_iso,
                        regions=regions,
                        markets=markets,
                        odds_format=odds_format,
                    )
                    tmp = normalize_snapshot_to_rows(snap, bookmaker=bookmaker, best_of_all=best)
                    if tmp is not None and not tmp.empty:
                        df = tmp
                        break
                except Exception:
                    continue
        # Current odds fallback
        if df is None or df.empty:
            # Prefer per-event odds when caller requests in-play lines.
            if inplay and not snapshot_iso:
                start_iso, end_iso = _utc_window_for_et_date(iso_date)
                for sk in sport_keys:
                    try:
                        events, _ = self.list_events(sk, commence_from_iso=start_iso, commence_to_iso=end_iso)
                        if not events:
                            continue
                        event_rows: List[Dict] = []
                        # If best-of-all is requested, omit bookmakers to allow best aggregation across books.
                        bookmakers_param = None if best else bookmaker
                        for ev in events:
                            try:
                                eid = str((ev or {}).get("id") or "").strip()
                                if not eid:
                                    continue
                                data, _ = self.event_odds(
                                    sport=sk,
                                    event_id=eid,
                                    markets=markets,
                                    regions=regions,
                                    bookmakers=bookmakers_param,
                                    odds_format=odds_format,
                                    date_format="iso",
                                )
                                if isinstance(data, dict) and data:
                                    event_rows.append(data)
                            except Exception:
                                continue

                        if event_rows:
                            tmp = normalize_snapshot_to_rows(event_rows, bookmaker=bookmaker, best_of_all=best)
                            if tmp is not None and not tmp.empty:
                                df = tmp
                                break
                    except Exception:
                        continue

        if df is None or df.empty:
            import requests as _rq
            base = ODDS_API_BASE
            params = {
                "apiKey": self.api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
                "dateFormat": "iso",
            }
            for sk in sport_keys:
                try:
                    url = f"{base}/sports/{sk}/odds"
                    r = _rq.get(url, params=params, timeout=40)
                    if r.ok:
                        tmp = normalize_snapshot_to_rows(r.json(), bookmaker=bookmaker, best_of_all=best)
                        if tmp is not None and not tmp.empty:
                            df = tmp
                            break
                except Exception:
                    continue
        if df is not None and not df.empty:
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        return df


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

    def _extract_totals_market(market_key: str, out_prefix: str) -> None:
        m_tot = next((m for m in markets if m.get("key") == market_key), None)
        if not m_tot:
            return
        pts = None
        for oc in m_tot.get("outcomes", []):
            if oc.get("name") in ("Over", "Under"):
                if pts is None:
                    pts = oc.get("point")
                out[f"{out_prefix}::{oc.get('name')}"] = oc.get("price")
                out[f"{out_prefix}::point"] = pts

    def _extract_h2h_market(market_key: str, out_prefix: str) -> None:
        m = next((m for m in markets if m.get("key") == market_key), None)
        if not m:
            return
        for oc in m.get("outcomes", []):
            name = str(oc.get("name"))
            out[f"{out_prefix}::{name}"] = oc.get("price")

    def _extract_h2h_3way_market(market_key: str, out_prefix: str) -> None:
        # Outcomes are typically: home team, away team, and draw/tie.
        m = next((m for m in markets if m.get("key") == market_key), None)
        if not m:
            return
        for oc in m.get("outcomes", []):
            name = str(oc.get("name"))
            out[f"{out_prefix}::{name}"] = oc.get("price")

    def _extract_spreads_market(market_key: str, out_prefix: str, abs_points: Optional[set[float]] = None) -> None:
        m = next((m for m in markets if m.get("key") == market_key), None)
        if not m:
            return
        for oc in m.get("outcomes", []):
            try:
                pt = float(oc.get("point"))
            except Exception:
                continue
            if abs_points is not None:
                try:
                    if float(abs(pt)) not in abs_points:
                        continue
                except Exception:
                    continue
            name = str(oc.get("name"))
            out[f"{out_prefix}::{name}"] = oc.get("price")
            out[f"{out_prefix}::{name}::point"] = pt

    # Moneyline
    m_h2h = next((m for m in markets if m.get("key") == "h2h"), None)
    if m_h2h:
        for oc in m_h2h.get("outcomes", []):
            name = str(oc.get("name"))
            price = oc.get("price")
            # store under dynamic keys; caller will map by team name
            out[f"ml::{name}"] = price

    # Totals (single main line assumed)
    _extract_totals_market("totals", "tot")

    # Period totals (if requested by markets=...)
    _extract_totals_market("totals_1st_period", "p1tot")
    _extract_totals_market("totals_2nd_period", "p2tot")
    _extract_totals_market("totals_3rd_period", "p3tot")
    # Alternate key scheme used by OddsAPI for some sports/books
    _extract_totals_market("totals_p1", "p1tot")
    _extract_totals_market("totals_p2", "p2tot")
    _extract_totals_market("totals_p3", "p3tot")

    # Period moneyline (if requested)
    _extract_h2h_market("h2h_1st_period", "p1ml")
    _extract_h2h_market("h2h_2nd_period", "p2ml")
    _extract_h2h_market("h2h_3rd_period", "p3ml")
    _extract_h2h_market("h2h_p1", "p1ml")
    _extract_h2h_market("h2h_p2", "p2ml")
    _extract_h2h_market("h2h_p3", "p3ml")

    # Regulation + period 3-way (if requested)
    _extract_h2h_3way_market("h2h_3_way", "reg3")
    _extract_h2h_3way_market("h2h_3_way_1st_period", "p13w")
    _extract_h2h_3way_market("h2h_3_way_2nd_period", "p23w")
    _extract_h2h_3way_market("h2h_3_way_3rd_period", "p33w")
    _extract_h2h_3way_market("h2h_3_way_p1", "p13w")
    _extract_h2h_3way_market("h2h_3_way_p2", "p23w")
    _extract_h2h_3way_market("h2h_3_way_p3", "p33w")

    # Period spreads (typically +/-0.5)
    _extract_spreads_market("spreads_1st_period", "p1spr", abs_points={0.5})
    _extract_spreads_market("spreads_2nd_period", "p2spr", abs_points={0.5})
    _extract_spreads_market("spreads_3rd_period", "p3spr", abs_points={0.5})
    _extract_spreads_market("spreads_p1", "p1spr", abs_points={0.5})
    _extract_spreads_market("spreads_p2", "p2spr", abs_points={0.5})
    _extract_spreads_market("spreads_p3", "p3spr", abs_points={0.5})

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
    Convert Odds API odds snapshot into a flat DataFrame with columns including core game markets
    and (when requested) period totals.

    Core columns:
    [date, home, away, home_ml, away_ml, over, under, total_line, home_pl_-1.5, away_pl_+1.5]

    Optional period totals columns:
    [p1_over, p1_under, p1_total_line, p2_over, p2_under, p2_total_line, p3_over, p3_under, p3_total_line]

    Optional regulation/period markets columns:
    - Regulation 3-way: [reg3_home, reg3_draw, reg3_away]
    - Period ML: [p1_ml_home, p1_ml_away, ...]
    - Period spreads (abs 0.5 only): [p1_spr_home, p1_spr_home_point, p1_spr_away, p1_spr_away_point, ...]
    - Period 3-way: [p1_3w_home, p1_3w_draw, p1_3w_away, ...]
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
            # Bucket events by US/Eastern slate date (not UTC) so late games
            # don't roll into the next day when commence_time is after 00:00Z.
            dt_utc = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
            dstr = dt_et.strftime("%Y-%m-%d")
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

            # Period totals
            p1_over = prices.get("p1tot::Over")
            p1_under = prices.get("p1tot::Under")
            p1_total_line = prices.get("p1tot::point")
            p2_over = prices.get("p2tot::Over")
            p2_under = prices.get("p2tot::Under")
            p2_total_line = prices.get("p2tot::point")
            p3_over = prices.get("p3tot::Over")
            p3_under = prices.get("p3tot::Under")
            p3_total_line = prices.get("p3tot::point")

            # Regulation 3-way (home/draw/away)
            reg3_home = prices.get(f"reg3::{home}")
            reg3_away = prices.get(f"reg3::{away}")
            reg3_draw = None
            try:
                # pick the non-team outcome as draw/tie
                for k, v in prices.items():
                    if not str(k).startswith("reg3::"):
                        continue
                    if str(k) in {f"reg3::{home}", f"reg3::{away}"}:
                        continue
                    reg3_draw = v
                    break
            except Exception:
                reg3_draw = None

            # Period moneyline
            p1_ml_home = prices.get(f"p1ml::{home}")
            p1_ml_away = prices.get(f"p1ml::{away}")
            p2_ml_home = prices.get(f"p2ml::{home}")
            p2_ml_away = prices.get(f"p2ml::{away}")
            p3_ml_home = prices.get(f"p3ml::{home}")
            p3_ml_away = prices.get(f"p3ml::{away}")

            # Period spreads (abs 0.5 only)
            p1_spr_home = prices.get(f"p1spr::{home}")
            p1_spr_home_point = prices.get(f"p1spr::{home}::point")
            p1_spr_away = prices.get(f"p1spr::{away}")
            p1_spr_away_point = prices.get(f"p1spr::{away}::point")
            p2_spr_home = prices.get(f"p2spr::{home}")
            p2_spr_home_point = prices.get(f"p2spr::{home}::point")
            p2_spr_away = prices.get(f"p2spr::{away}")
            p2_spr_away_point = prices.get(f"p2spr::{away}::point")
            p3_spr_home = prices.get(f"p3spr::{home}")
            p3_spr_home_point = prices.get(f"p3spr::{home}::point")
            p3_spr_away = prices.get(f"p3spr::{away}")
            p3_spr_away_point = prices.get(f"p3spr::{away}::point")

            # Period 3-way
            p1_3w_home = prices.get(f"p13w::{home}")
            p1_3w_away = prices.get(f"p13w::{away}")
            p1_3w_draw = None
            p2_3w_home = prices.get(f"p23w::{home}")
            p2_3w_away = prices.get(f"p23w::{away}")
            p2_3w_draw = None
            p3_3w_home = prices.get(f"p33w::{home}")
            p3_3w_away = prices.get(f"p33w::{away}")
            p3_3w_draw = None
            try:
                for k, v in prices.items():
                    if str(k).startswith("p13w::") and str(k) not in {f"p13w::{home}", f"p13w::{away}"}:
                        p1_3w_draw = v
                        break
                for k, v in prices.items():
                    if str(k).startswith("p23w::") and str(k) not in {f"p23w::{home}", f"p23w::{away}"}:
                        p2_3w_draw = v
                        break
                for k, v in prices.items():
                    if str(k).startswith("p33w::") and str(k) not in {f"p33w::{home}", f"p33w::{away}"}:
                        p3_3w_draw = v
                        break
            except Exception:
                pass

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
                    "p1_over": p1_over,
                    "p1_under": p1_under,
                    "p1_total_line": p1_total_line,
                    "p2_over": p2_over,
                    "p2_under": p2_under,
                    "p2_total_line": p2_total_line,
                    "p3_over": p3_over,
                    "p3_under": p3_under,
                    "p3_total_line": p3_total_line,
                    "reg3_home": reg3_home,
                    "reg3_draw": reg3_draw,
                    "reg3_away": reg3_away,
                    "p1_ml_home": p1_ml_home,
                    "p1_ml_away": p1_ml_away,
                    "p2_ml_home": p2_ml_home,
                    "p2_ml_away": p2_ml_away,
                    "p3_ml_home": p3_ml_home,
                    "p3_ml_away": p3_ml_away,
                    "p1_spr_home": p1_spr_home,
                    "p1_spr_home_point": p1_spr_home_point,
                    "p1_spr_away": p1_spr_away,
                    "p1_spr_away_point": p1_spr_away_point,
                    "p2_spr_home": p2_spr_home,
                    "p2_spr_home_point": p2_spr_home_point,
                    "p2_spr_away": p2_spr_away,
                    "p2_spr_away_point": p2_spr_away_point,
                    "p3_spr_home": p3_spr_home,
                    "p3_spr_home_point": p3_spr_home_point,
                    "p3_spr_away": p3_spr_away,
                    "p3_spr_away_point": p3_spr_away_point,
                    "p1_3w_home": p1_3w_home,
                    "p1_3w_draw": p1_3w_draw,
                    "p1_3w_away": p1_3w_away,
                    "p2_3w_home": p2_3w_home,
                    "p2_3w_draw": p2_3w_draw,
                    "p2_3w_away": p2_3w_away,
                    "p3_3w_home": p3_3w_home,
                    "p3_3w_draw": p3_3w_draw,
                    "p3_3w_away": p3_3w_away,
                    "home_pl_-1.5": home_pl,
                    "away_pl_+1.5": away_pl,
                    "home_ml_book": book.get("key"),
                    "away_ml_book": book.get("key"),
                    "over_book": book.get("key"),
                    "under_book": book.get("key"),
                    "p1_over_book": book.get("key") if p1_over is not None else None,
                    "p1_under_book": book.get("key") if p1_under is not None else None,
                    "p2_over_book": book.get("key") if p2_over is not None else None,
                    "p2_under_book": book.get("key") if p2_under is not None else None,
                    "p3_over_book": book.get("key") if p3_over is not None else None,
                    "p3_under_book": book.get("key") if p3_under is not None else None,
                    "reg3_home_book": book.get("key") if reg3_home is not None else None,
                    "reg3_draw_book": book.get("key") if reg3_draw is not None else None,
                    "reg3_away_book": book.get("key") if reg3_away is not None else None,
                    "p1_ml_home_book": book.get("key") if p1_ml_home is not None else None,
                    "p1_ml_away_book": book.get("key") if p1_ml_away is not None else None,
                    "p2_ml_home_book": book.get("key") if p2_ml_home is not None else None,
                    "p2_ml_away_book": book.get("key") if p2_ml_away is not None else None,
                    "p3_ml_home_book": book.get("key") if p3_ml_home is not None else None,
                    "p3_ml_away_book": book.get("key") if p3_ml_away is not None else None,
                    "p1_spr_home_book": book.get("key") if p1_spr_home is not None else None,
                    "p1_spr_away_book": book.get("key") if p1_spr_away is not None else None,
                    "p2_spr_home_book": book.get("key") if p2_spr_home is not None else None,
                    "p2_spr_away_book": book.get("key") if p2_spr_away is not None else None,
                    "p3_spr_home_book": book.get("key") if p3_spr_home is not None else None,
                    "p3_spr_away_book": book.get("key") if p3_spr_away is not None else None,
                    "p1_3w_home_book": book.get("key") if p1_3w_home is not None else None,
                    "p1_3w_draw_book": book.get("key") if p1_3w_draw is not None else None,
                    "p1_3w_away_book": book.get("key") if p1_3w_away is not None else None,
                    "p2_3w_home_book": book.get("key") if p2_3w_home is not None else None,
                    "p2_3w_draw_book": book.get("key") if p2_3w_draw is not None else None,
                    "p2_3w_away_book": book.get("key") if p2_3w_away is not None else None,
                    "p3_3w_home_book": book.get("key") if p3_3w_home is not None else None,
                    "p3_3w_draw_book": book.get("key") if p3_3w_draw is not None else None,
                    "p3_3w_away_book": book.get("key") if p3_3w_away is not None else None,
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
                "p1_over": (None, None),
                "p1_under": (None, None),
                "p2_over": (None, None),
                "p2_under": (None, None),
                "p3_over": (None, None),
                "p3_under": (None, None),
                "reg3_home": (None, None),
                "reg3_draw": (None, None),
                "reg3_away": (None, None),
                "p1_ml_home": (None, None),
                "p1_ml_away": (None, None),
                "p2_ml_home": (None, None),
                "p2_ml_away": (None, None),
                "p3_ml_home": (None, None),
                "p3_ml_away": (None, None),
                "p1_spr_home": (None, None),
                "p1_spr_away": (None, None),
                "p2_spr_home": (None, None),
                "p2_spr_away": (None, None),
                "p3_spr_home": (None, None),
                "p3_spr_away": (None, None),
                "p1_3w_home": (None, None),
                "p1_3w_draw": (None, None),
                "p1_3w_away": (None, None),
                "p2_3w_home": (None, None),
                "p2_3w_draw": (None, None),
                "p2_3w_away": (None, None),
                "p3_3w_home": (None, None),
                "p3_3w_draw": (None, None),
                "p3_3w_away": (None, None),
                "home_pl_-1.5": (None, None),
                "away_pl_+1.5": (None, None),
            }
            # Gather totals points to pick the modal total line
            totals_points: List[float] = []
            p1_points: List[float] = []
            p2_points: List[float] = []
            p3_points: List[float] = []
            book_prices: List[Tuple[str, Dict[str, any]]] = []
            for b in bks:
                pr = _extract_prices_from_markets(b.get("markets", []))
                book_prices.append((b.get("key"), pr))
                if "tot::point" in pr and pr["tot::point"] is not None:
                    try:
                        totals_points.append(float(pr["tot::point"]))
                    except Exception:
                        pass
                if "p1tot::point" in pr and pr["p1tot::point"] is not None:
                    try:
                        p1_points.append(float(pr["p1tot::point"]))
                    except Exception:
                        pass
                if "p2tot::point" in pr and pr["p2tot::point"] is not None:
                    try:
                        p2_points.append(float(pr["p2tot::point"]))
                    except Exception:
                        pass
                if "p3tot::point" in pr and pr["p3tot::point"] is not None:
                    try:
                        p3_points.append(float(pr["p3tot::point"]))
                    except Exception:
                        pass
            totals_point = None
            if totals_points:
                # mode; if tie, pick the first
                totals_point = max(set(totals_points), key=totals_points.count)

            p1_point = None
            if p1_points:
                p1_point = max(set(p1_points), key=p1_points.count)
            p2_point = None
            if p2_points:
                p2_point = max(set(p2_points), key=p2_points.count)
            p3_point = None
            if p3_points:
                p3_point = max(set(p3_points), key=p3_points.count)

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

                # Regulation 3-way
                upd("reg3_home", pr.get(f"reg3::{home}"), bkey)
                upd("reg3_away", pr.get(f"reg3::{away}"), bkey)
                # Draw/tie: pick non-team outcome
                try:
                    for k, v in pr.items():
                        if not str(k).startswith("reg3::"):
                            continue
                        if str(k) in {f"reg3::{home}", f"reg3::{away}"}:
                            continue
                        upd("reg3_draw", v, bkey)
                        break
                except Exception:
                    pass

                # Period ML
                upd("p1_ml_home", pr.get(f"p1ml::{home}"), bkey)
                upd("p1_ml_away", pr.get(f"p1ml::{away}"), bkey)
                upd("p2_ml_home", pr.get(f"p2ml::{home}"), bkey)
                upd("p2_ml_away", pr.get(f"p2ml::{away}"), bkey)
                upd("p3_ml_home", pr.get(f"p3ml::{home}"), bkey)
                upd("p3_ml_away", pr.get(f"p3ml::{away}"), bkey)

                # Period spreads (abs 0.5 only)
                upd("p1_spr_home", pr.get(f"p1spr::{home}"), bkey)
                upd("p1_spr_away", pr.get(f"p1spr::{away}"), bkey)
                upd("p2_spr_home", pr.get(f"p2spr::{home}"), bkey)
                upd("p2_spr_away", pr.get(f"p2spr::{away}"), bkey)
                upd("p3_spr_home", pr.get(f"p3spr::{home}"), bkey)
                upd("p3_spr_away", pr.get(f"p3spr::{away}"), bkey)

                # Period 3-way
                upd("p1_3w_home", pr.get(f"p13w::{home}"), bkey)
                upd("p1_3w_away", pr.get(f"p13w::{away}"), bkey)
                try:
                    for k, v in pr.items():
                        if str(k).startswith("p13w::") and str(k) not in {f"p13w::{home}", f"p13w::{away}"}:
                            upd("p1_3w_draw", v, bkey)
                            break
                except Exception:
                    pass
                upd("p2_3w_home", pr.get(f"p23w::{home}"), bkey)
                upd("p2_3w_away", pr.get(f"p23w::{away}"), bkey)
                try:
                    for k, v in pr.items():
                        if str(k).startswith("p23w::") and str(k) not in {f"p23w::{home}", f"p23w::{away}"}:
                            upd("p2_3w_draw", v, bkey)
                            break
                except Exception:
                    pass
                upd("p3_3w_home", pr.get(f"p33w::{home}"), bkey)
                upd("p3_3w_away", pr.get(f"p33w::{away}"), bkey)
                try:
                    for k, v in pr.items():
                        if str(k).startswith("p33w::") and str(k) not in {f"p33w::{home}", f"p33w::{away}"}:
                            upd("p3_3w_draw", v, bkey)
                            break
                except Exception:
                    pass
                # Totals at chosen point
                if totals_point is not None and pr.get("tot::point") is not None:
                    try:
                        if float(pr.get("tot::point")) == float(totals_point):
                            upd("over", pr.get("tot::Over"), bkey)
                            upd("under", pr.get("tot::Under"), bkey)
                    except Exception:
                        pass

                # Period totals at chosen points
                if p1_point is not None and pr.get("p1tot::point") is not None:
                    try:
                        if float(pr.get("p1tot::point")) == float(p1_point):
                            upd("p1_over", pr.get("p1tot::Over"), bkey)
                            upd("p1_under", pr.get("p1tot::Under"), bkey)
                    except Exception:
                        pass
                if p2_point is not None and pr.get("p2tot::point") is not None:
                    try:
                        if float(pr.get("p2tot::point")) == float(p2_point):
                            upd("p2_over", pr.get("p2tot::Over"), bkey)
                            upd("p2_under", pr.get("p2tot::Under"), bkey)
                    except Exception:
                        pass
                if p3_point is not None and pr.get("p3tot::point") is not None:
                    try:
                        if float(pr.get("p3tot::point")) == float(p3_point):
                            upd("p3_over", pr.get("p3tot::Over"), bkey)
                            upd("p3_under", pr.get("p3tot::Under"), bkey)
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
                    "p1_over": best["p1_over"][0],
                    "p1_under": best["p1_under"][0],
                    "p1_total_line": p1_point,
                    "p2_over": best["p2_over"][0],
                    "p2_under": best["p2_under"][0],
                    "p2_total_line": p2_point,
                    "p3_over": best["p3_over"][0],
                    "p3_under": best["p3_under"][0],
                    "p3_total_line": p3_point,
                    "reg3_home": best["reg3_home"][0],
                    "reg3_draw": best["reg3_draw"][0],
                    "reg3_away": best["reg3_away"][0],
                    "p1_ml_home": best["p1_ml_home"][0],
                    "p1_ml_away": best["p1_ml_away"][0],
                    "p2_ml_home": best["p2_ml_home"][0],
                    "p2_ml_away": best["p2_ml_away"][0],
                    "p3_ml_home": best["p3_ml_home"][0],
                    "p3_ml_away": best["p3_ml_away"][0],
                    "p1_spr_home": best["p1_spr_home"][0],
                    "p1_spr_away": best["p1_spr_away"][0],
                    "p2_spr_home": best["p2_spr_home"][0],
                    "p2_spr_away": best["p2_spr_away"][0],
                    "p3_spr_home": best["p3_spr_home"][0],
                    "p3_spr_away": best["p3_spr_away"][0],
                    "p1_3w_home": best["p1_3w_home"][0],
                    "p1_3w_draw": best["p1_3w_draw"][0],
                    "p1_3w_away": best["p1_3w_away"][0],
                    "p2_3w_home": best["p2_3w_home"][0],
                    "p2_3w_draw": best["p2_3w_draw"][0],
                    "p2_3w_away": best["p2_3w_away"][0],
                    "p3_3w_home": best["p3_3w_home"][0],
                    "p3_3w_draw": best["p3_3w_draw"][0],
                    "p3_3w_away": best["p3_3w_away"][0],
                    "home_pl_-1.5": best["home_pl_-1.5"][0],
                    "away_pl_+1.5": best["away_pl_+1.5"][0],
                    "home_ml_book": best["home_ml"][1],
                    "away_ml_book": best["away_ml"][1],
                    "over_book": best["over"][1],
                    "under_book": best["under"][1],
                    "p1_over_book": best["p1_over"][1],
                    "p1_under_book": best["p1_under"][1],
                    "p2_over_book": best["p2_over"][1],
                    "p2_under_book": best["p2_under"][1],
                    "p3_over_book": best["p3_over"][1],
                    "p3_under_book": best["p3_under"][1],
                    "reg3_home_book": best["reg3_home"][1],
                    "reg3_draw_book": best["reg3_draw"][1],
                    "reg3_away_book": best["reg3_away"][1],
                    "p1_ml_home_book": best["p1_ml_home"][1],
                    "p1_ml_away_book": best["p1_ml_away"][1],
                    "p2_ml_home_book": best["p2_ml_home"][1],
                    "p2_ml_away_book": best["p2_ml_away"][1],
                    "p3_ml_home_book": best["p3_ml_home"][1],
                    "p3_ml_away_book": best["p3_ml_away"][1],
                    "p1_spr_home_book": best["p1_spr_home"][1],
                    "p1_spr_away_book": best["p1_spr_away"][1],
                    "p2_spr_home_book": best["p2_spr_home"][1],
                    "p2_spr_away_book": best["p2_spr_away"][1],
                    "p3_spr_home_book": best["p3_spr_home"][1],
                    "p3_spr_away_book": best["p3_spr_away"][1],
                    "p1_3w_home_book": best["p1_3w_home"][1],
                    "p1_3w_draw_book": best["p1_3w_draw"][1],
                    "p1_3w_away_book": best["p1_3w_away"][1],
                    "p2_3w_home_book": best["p2_3w_home"][1],
                    "p2_3w_draw_book": best["p2_3w_draw"][1],
                    "p2_3w_away_book": best["p2_3w_away"][1],
                    "p3_3w_home_book": best["p3_3w_home"][1],
                    "p3_3w_draw_book": best["p3_3w_draw"][1],
                    "p3_3w_away_book": best["p3_3w_away"][1],
                    "home_pl_-1.5_book": best["home_pl_-1.5"][1],
                    "away_pl_+1.5_book": best["away_pl_+1.5"][1],
                }
            )
    df = pd.DataFrame(rows)
    return df
