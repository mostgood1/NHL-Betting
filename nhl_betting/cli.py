import json

# CRITICAL: Import onnxruntime and torch BEFORE numpy/pandas to avoid DLL conflicts
# NumPy's MKL DLLs interfere with ONNX Runtime's pybind11 state initialization
# However, make these optional to avoid breaking web server which imports cli
try:
    import onnxruntime as _ort
except (ImportError, OSError):
    _ort = None  # Optional dependency, may fail if numpy already loaded
try:
    import torch as _torch
except (ImportError, OSError):
    _torch = None  # Optional dependency, may fail if numpy already loaded

import numpy as np
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional, List

import pandas as pd
import typer
from rich import print

from .utils.io import RAW_DIR, PROC_DIR, MODEL_DIR, save_df, load_df
from .utils.dates import parse_date, ymd, today_utc
from .data.nhl_api import NHLClient
from .data.nhl_api_web import NHLWebClient
from .features.engineering import make_team_game_features
from .models.elo import Elo
from .models.elo import Elo, EloConfig

def _load_elo_config() -> EloConfig:
    """Load EloConfig from MODEL_DIR/config.json if present, else defaults.

    Keys:
      - elo_k (float)
      - elo_home_adv (float, Elo points)
    """
    cfg_path = MODEL_DIR / "config.json"
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            k = float(obj.get("elo_k", 20.0))
            ha = float(obj.get("elo_home_adv", 50.0))
            return EloConfig(k=k, home_adv=ha)
        except Exception:
            return EloConfig()
    return EloConfig()

from .models.trends import TrendAdjustments, team_keys, get_adjustment
from .utils.odds import american_to_decimal, decimal_to_implied_prob, remove_vig_two_way, ev_unit, kelly_stake
from .data.collect import collect_player_game_stats
from .models.props import SkaterShotsModel, GoalieSavesModel, SkaterGoalsModel, SkaterAssistsModel, SkaterPointsModel, SkaterBlocksModel
from .props.utils import compute_props_lam_scale_mean
from .data.odds_api import OddsAPIClient, normalize_snapshot_to_rows
from .data import player_props as props_data
from .data.rosters import build_all_team_roster_snapshots, build_roster_snapshot, infer_lines, project_toi, TEAM_ABBRS
from .data.lineups import build_lineup_snapshot, build_lineup_snapshot_from_source
from .data.lineups_sources import fetch_dailyfaceoff_starting_goalies
from .data.lineups_sources import fetch_dailyfaceoff_starting_goalies
from .data.injuries import build_injury_snapshot
from .data.co_toi import build_co_toi_from_lineups
from .data.shifts_api import shifts_frame, co_toi_from_shifts, player_toi_from_shifts
from .sim.engine import GameSimulator, SimConfig
from .sim.models import RateModels
from .web.teams import get_team_assets

app = typer.Typer(help="NHL Betting predictive engine CLI")


@app.command()
def fetch(
    season: Optional[int] = typer.Option(None, help="Season start year, e.g., 2023"),
    start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD"),
    source: str = typer.Option("web", help="Data source: 'web' (api-web.nhle.com), 'stats' (api.nhle.com/stats/rest), or 'nhlpy' (nhl-api-py)"),
):
    """Fetch schedule and results into data/raw/games.csv"""
    # Choose client based on source
    source = (source or "web").lower()
    if source == "web":
        client = NHLWebClient()
    elif source == "stats":
        client = NHLClient()
    elif source == "nhlpy":
        try:
            from .data.nhl_api_nhlpy import NHLNhlPyClient  # lazy import
        except Exception as e:
            print("nhl-api-py adapter not available:", e)
            raise typer.Exit(code=1)
        client = NHLNhlPyClient()
    else:
        print(f"Unknown source '{source}'. Use one of: web, stats, nhlpy")
        raise typer.Exit(code=1)
    if season and (not start and not end):
        # NHL season spans two years; we'll fetch from Oct 1 to July 15
        start = f"{season}-09-01"
        end = f"{season+1}-08-01"
    if not start:
        start = ymd(today_utc())
    if not end:
        end = ymd(today_utc())

    # Support both client interfaces: schedule_range(start,end) or schedule(start,end)
    if hasattr(client, "schedule_range"):
        games = client.schedule_range(start, end)
    else:
        games = client.schedule(start, end)
    rows = []
    # Helper for ET calendar day
    def _to_et_date(iso_utc: str) -> str:
        try:
            s = str(iso_utc).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        except Exception:
            try:
                dt = datetime.fromisoformat(str(iso_utc)[:19]).replace(tzinfo=timezone.utc)
                return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            except Exception:
                return None
    for g in games:
        rows.append({
            "gamePk": g.gamePk,
            "date": g.gameDate,  # ISO UTC
            "date_et": _to_et_date(g.gameDate),  # US/Eastern calendar day
            "season": g.season,
            "type": g.gameType,
            "home": g.home,
            "away": g.away,
            "home_goals": g.home_goals,
            "away_goals": g.away_goals,
        })
    df = pd.DataFrame(rows)
    path = RAW_DIR / "games.csv"
    save_df(df, path)
    print(f"Saved {len(df)} games to {path}")


@app.command()
def featurize():
    path = RAW_DIR / "games.csv"
    df = load_df(path)
    out = make_team_game_features(df)
    out_path = PROC_DIR / "team_games.csv"
    save_df(out, out_path)
    print(f"Saved features to {out_path}")


@app.command()
def train():
    # Train Elo from historical completed games
    path = RAW_DIR / "games.csv"
    df = load_df(path)
    df = df.dropna(subset=["home_goals", "away_goals"])  # completed
    # Keep only NHL vs NHL games (exclude overseas, all-star, non-NHL opponents)
    try:
        from .web.teams import get_team_assets
        df = df[df["home"].apply(lambda x: bool(get_team_assets(str(x)).get("abbr"))) & df["away"].apply(lambda x: bool(get_team_assets(str(x)).get("abbr")))]
    except Exception:
        # If mapping not available, proceed without filtering
        pass
    df = df.sort_values("date")
    # Use configured Elo parameters for training
    elo = Elo(cfg=_load_elo_config())
    for _, g in df.iterrows():
        elo.update_game(g["home"], g["away"], int(g["home_goals"]), int(g["away_goals"]))
    # Save ratings
    ratings_path = MODEL_DIR / "elo_ratings.json"
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ratings_path, "w", encoding="utf-8") as f:
        json.dump(elo.ratings, f, indent=2)
    # Compute league-average per-team goals (reg season + playoffs treated same)
    total_goals = (df["home_goals"].astype(int) + df["away_goals"].astype(int)).sum()
    games_count = len(df)
    # per-team per-game mean goals lambda (for PoissonGoals base)
    base_mu = float(total_goals / (2 * games_count)) if games_count > 0 else 3.0
    # Update config.json: preserve existing keys (e.g., elo_k, elo_home_adv) and only update base_mu
    cfg_path = MODEL_DIR / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_obj = {}
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg_obj = json.load(f) or {}
        except Exception:
            cfg_obj = {}
    cfg_obj["base_mu"] = float(base_mu)
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg_obj, f, indent=2)
    print(f"Saved Elo ratings to {ratings_path} and config to {cfg_path} (base_mu={base_mu:.3f})")


def _season_code_for_date(d_ymd: Optional[str]) -> str:
    """Return NHL season code like '20252026' for a given YYYY-MM-DD (uses ET boundary in July)."""
    try:
        if d_ymd and isinstance(d_ymd, str):
            dt = datetime.fromisoformat(d_ymd)
        else:
            dt = datetime.now(ZoneInfo("America/New_York"))
        start_year = dt.year if dt.month >= 7 else (dt.year - 1)
        return f"{start_year}{start_year+1}"
    except Exception:
        y = datetime.now(ZoneInfo("America/New_York")).year
        return f"{y}{y+1}"


def _gh_raw_read_csv(rel_path: str, timeout_sec: float = 4.0) -> pd.DataFrame:
    """Read a CSV from GitHub raw given a repo-relative path."""
    import os, requests
    try:
        repo = os.getenv("GITHUB_REPO", "mostgood1/NHL-Betting").strip() or "mostgood1/NHL-Betting"
        branch = os.getenv("GITHUB_BRANCH", "master").strip() or "master"
        rel = rel_path.lstrip("/")
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{rel}"
        r = requests.get(url, timeout=timeout_sec)
        if r.ok and r.text:
            from io import StringIO
            return pd.read_csv(StringIO(r.text))
    except Exception:
        pass
    return pd.DataFrame()


def _gh_raw_read_parquet(rel_path: str, timeout_sec: float = 6.0) -> pd.DataFrame:
    """Read a Parquet from GitHub raw given a repo-relative path."""
    import os, requests, io
    try:
        repo = os.getenv("GITHUB_REPO", "mostgood1/NHL-Betting").strip() or "mostgood1/NHL-Betting"
        branch = os.getenv("GITHUB_BRANCH", "master").strip() or "master"
        rel = rel_path.lstrip("/")
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{rel}"
        r = requests.get(url, timeout=timeout_sec)
        if r.ok and r.content:
            return pd.read_parquet(io.BytesIO(r.content))
    except Exception:
        pass
    return pd.DataFrame()


@app.command()
def roster_master(date: Optional[str] = typer.Option(None, help="ET date YYYY-MM-DD to stamp output and choose season code")):
    """Build a master roster CSV of current NHL players with player_id, name, position, team, and image URL.

    Writes data/processed/roster_{date}.csv and data/processed/roster_master.csv
    """
    from .web.teams import get_team_assets as _assets
    d = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    season = _season_code_for_date(d)
    rows = []
    built_via_web_api = False
    # Primary path: NHL api-web for teams and rosters
    try:
        import requests, time
        BASE = "https://api-web.nhle.com/v1"
        def _get(path: str, params=None, retries=3, timeout=20):
            last = None
            for i in range(retries):
                try:
                    r = requests.get(f"{BASE}{path}", params=params, timeout=timeout)
                    if r.ok:
                        return r.json()
                except Exception as e:
                    last = e
                    time.sleep(0.4 * (2 ** i))
            if last:
                raise last
            raise RuntimeError("web api error")
        # teams via standings/now
        st = _get("/standings/now")
        # standings/now structure: {'standings': [...]} with entries containing teamAbbrev, teamName, teamCommonName, teamId
        teams_info = []
        def _txt(v):
            if isinstance(v, dict):
                # handle both 'default' and 'DEFAULT' keys
                return v.get('default') or v.get('DEFAULT') or v.get('en') or v.get('name') or ''
            return str(v or '')
        if isinstance(st, dict):
            lst = st.get('standings') or st.get('standingsByDivision') or st.get('records') or []
            # try flatten common cases
            if isinstance(lst, dict):
                # sometimes wrapped by divisions
                tmp = []
                for v in lst.values():
                    if isinstance(v, list): tmp.extend(v)
                lst = tmp
            if isinstance(lst, list):
                for t in lst:
                    try:
                        ab = _txt(t.get('teamAbbrev') or t.get('teamAbbrevTricode') or t.get('teamAbbrevShort') or '').upper()
                        tid = t.get('teamId') or t.get('id')
                        nm = _txt(t.get('teamName') or t.get('teamCommonName') or t.get('teamFullName'))
                        if ab:
                            teams_info.append({'abbr': ab, 'id': int(tid) if tid else None, 'name': nm})
                    except Exception:
                        continue
        # Unique by abbr
        seen_abbr = set(); uniq = []
        for t in teams_info:
            if t['abbr'] in seen_abbr: continue
            seen_abbr.add(t['abbr']); uniq.append(t)
        # Fetch roster per team
        for t in uniq:
            ab = str(t['abbr']).upper(); tid = t.get('id'); tname = t.get('name')
            try:
                rjson = _get(f"/roster/{ab}/current")
            except Exception as e:
                # skip if roster endpoint fails for one team
                print(f"[roster] roster fetch failed for {ab}: {e}")
                continue
            # Expect keys like 'forwards','defensemen','goalies' or similar; also some APIs use 'forwards','defense','goalies'
            buckets = []
            if isinstance(rjson, dict):
                for k in ('forwards','defense','defensemen','goalies','skaters','roster'):  # cover variants
                    v = rjson.get(k)
                    if isinstance(v, list):
                        buckets.extend(v)
            def _norm_pos(ps):
                s = str(ps or '').strip().upper()
                if not s:
                    return None
                # Goalies
                if s in ('G', 'GOALIE', 'GOALTENDER'):
                    return 'G'
                if s.startswith('G'):
                    return 'G'
                # Defense
                if s in ('D', 'DEF', 'DEFENSE', 'DEFENCE', 'DEFENCEMAN', 'DEFENSEMAN'):
                    return 'D'
                if s.startswith('D'):
                    return 'D'
                # Forwards (centers/wings)
                if s in ('F', 'FORWARD', 'C', 'LW', 'RW', 'L', 'R', 'W'):
                    return 'F'
                if s.startswith('F') or s.startswith('W') or s.startswith('L') or s.startswith('R') or s.startswith('C'):
                    return 'F'
                return None
            for p in buckets:
                try:
                    # Extract id, name, position
                    pid = p.get('id') or (p.get('person') or {}).get('id')
                    if pid is None:
                        continue
                    pid = int(pid)
                    def _txt(v):
                        if isinstance(v, dict):
                            return v.get('default') or v.get('full') or v.get('name') or ''
                        return str(v or '')
                    first = _txt(p.get('firstName') or (p.get('person') or {}).get('firstName'))
                    last = _txt(p.get('lastName') or (p.get('person') or {}).get('lastName'))
                    full = (first + ' ' + last).strip() or _txt(p.get('fullName'))
                    pos_raw = p.get('positionCode') or p.get('position') or (p.get('positionObj') or {}).get('code')
                    pos = _norm_pos(pos_raw)
                    img = f"https://assets.nhle.com/mugs/nhl/{season}/{ab}/{pid}.png"
                    rows.append({
                        'player_id': pid,
                        'player': full,
                        'full_name': full,
                        'position': pos,
                        'team_id': tid,
                        'team_abbr': ab,
                        'team_name': tname,
                        'image_url': img,
                    })
                except Exception:
                    continue
        built_via_web_api = len(rows) > 0
    except Exception as e:
        print("[roster] api-web unavailable, will try canonical lines + stats fallback:", e)
    # Fallback path: canonical lines + stats
    if not built_via_web_api:
        try:
            # Pick a lines date: prefer provided date; else try the most recent available partition
            from .utils.io import PROC_DIR as _PROC
            base_root = _PROC.parent / "props" / "player_props_lines"
            date_dir = None
            cand = base_root / f"date={d}"
            if cand.exists():
                date_dir = cand
            else:
                # find latest date=YYYY-MM-DD dir
                dirs = [p for p in base_root.glob("date=*") if p.is_dir()]
                if dirs:
                    date_dir = sorted(dirs, key=lambda p: p.name)[-1]
            import pandas as _pd
            parts = []
            if date_dir is not None:
                for name in ("oddsapi.parquet", "oddsapi.csv"):
                    p = date_dir / name
                    if p.exists():
                        try:
                            parts.append(_pd.read_parquet(p) if p.suffix == ".parquet" else _pd.read_csv(p))
                        except Exception:
                            continue
            # If still empty, try GitHub raw for the provided date and yesterday
            if not parts:
                from datetime import datetime as _dt, timedelta as _td
                dates_try = [d]
                try:
                    d0 = _dt.fromisoformat(d)
                    dates_try.append((d0 - _td(days=1)).strftime('%Y-%m-%d'))
                except Exception:
                    pass
                for dtry in dates_try:
                    for name in ("oddsapi.parquet",):
                        rel = f"data/props/player_props_lines/date={dtry}/{name}"
                        gdf = _gh_raw_read_parquet(rel)
                        if gdf is not None and not gdf.empty:
                            parts.append(gdf)
                    for name in ("oddsapi.csv",):
                        rel = f"data/props/player_props_lines/date={dtry}/{name}"
                        gdf = _gh_raw_read_csv(rel)
                        if gdf is not None and not gdf.empty:
                            parts.append(gdf)
            lines = _pd.concat(parts, ignore_index=True) if parts else _pd.DataFrame()
            if lines.empty:
                print("[roster] No canonical lines found; cannot build roster fallback.")
            else:
                # Unique by player_id where available, else by player_name
                # Standardize columns
                name_col = "player_name" if "player_name" in lines.columns else ("player" if "player" in lines.columns else None)
                team_col = "team" if "team" in lines.columns else None
                pid_col = "player_id" if "player_id" in lines.columns else None
                if not name_col or not team_col:
                    print("[roster] lines missing required columns; aborting fallback.")
                else:
                    # Last known position from stats by player_id if possible (fallback by name)
                    from .utils.io import RAW_DIR as _RAW
                    stats = _pd.read_csv(_RAW / "player_game_stats.csv") if (_RAW / "player_game_stats.csv").exists() else _pd.DataFrame()
                    pos_by_pid = {}
                    pos_by_name = {}
                    if not stats.empty:
                        try:
                            stats["_d"] = _pd.to_datetime(stats.get("date"), errors="coerce")
                            stats = stats.sort_values("_d")
                        except Exception:
                            pass
                        if {"player_id", "primary_position"}.issubset(stats.columns):
                            last_pos = stats.dropna(subset=["primary_position"]).groupby("player_id")["primary_position"].last()
                            pos_by_pid = {int(k): str(v).upper() for k, v in last_pos.to_dict().items() if _pd.notna(v)}
                        if {"player", "primary_position"}.issubset(stats.columns):
                            last_pos2 = stats.dropna(subset=["primary_position"]).groupby("player")["primary_position"].last()
                            pos_by_name = {str(k): str(v).upper() for k, v in last_pos2.to_dict().items() if _pd.notna(v)}
                    # Build set of players
                    lines = lines.dropna(subset=[name_col, team_col])
                    seen = set()
                    for _, r in lines.iterrows():
                        nm = str(r.get(name_col) or "").strip()
                        tm = str(r.get(team_col) or "").strip()
                        ab = (_assets(tm).get("abbr") or "").upper() if _assets else ""
                        pid_val = r.get(pid_col) if pid_col else None
                        pid = None
                        try:
                            if pid_val is not None and str(pid_val).strip() != "":
                                pid = int(float(pid_val))
                        except Exception:
                            pid = None
                        key = pid if pid is not None else (nm, ab)
                        if key in seen:
                            continue
                        seen.add(key)
                        pos_raw = pos_by_pid.get(pid) if pid is not None else pos_by_name.get(nm)
                        pos = None
                        if pos_raw:
                            if pos_raw in ("C","LW","RW"): pos = "F"
                            elif pos_raw in ("F","D","G"): pos = pos_raw
                        img = f"https://assets.nhle.com/mugs/nhl/{season}/{ab}/{pid}.png" if (pid and ab) else (f"https://cms.nhl.bamgrid.com/images/headshots/current/168x168/{pid}.jpg" if pid else None)
                        rows.append({
                            "player_id": pid,
                            "player": nm,
                            "full_name": nm,
                            "position": pos,
                            "team_id": None,
                            "team_abbr": ab,
                            "team_name": tm,
                            "image_url": img,
                        })
        except Exception as e:
            print("[roster] fallback failed:", e)
    df = pd.DataFrame(rows)
    out_dated = PROC_DIR / f"roster_{d}.csv"
    out_master = PROC_DIR / "roster_master.csv"
    save_df(df, out_dated)
    save_df(df, out_master)
    print(f"Wrote {len(df)} players to {out_dated.name} and roster_master.csv")


def predict_core(
    date: str,
    total_line: float = 6.0,
    odds_csv: Optional[str] = None,
    source: str = "web",
    odds_source: str = "csv",  # csv | oddsapi
    snapshot: Optional[str] = None,
    odds_regions: str = "us",
    odds_markets: str = "h2h,totals,spreads",
    odds_bookmaker: Optional[str] = None,
    odds_best: bool = False,
    bankroll: float = 0.0,
    kelly_fraction_part: float = 0.5,
) -> Path:
    # Helper: convert ISO UTC (e.g., 2025-09-22T23:00:00Z) to ET calendar day YYYY-MM-DD
    def iso_to_et_date(iso_utc: str) -> str:
        if not iso_utc:
            return ""
        try:
            s = str(iso_utc).replace("Z", "+00:00")
            dt_utc = datetime.fromisoformat(s)
            et = ZoneInfo("America/New_York")
            dt_et = dt_utc.astimezone(et)
            return dt_et.strftime("%Y-%m-%d")
        except Exception:
            try:
                dt_utc = datetime.fromisoformat(str(iso_utc)[:19]).replace(tzinfo=timezone.utc)
                et = ZoneInfo("America/New_York")
                dt_et = dt_utc.astimezone(et)
                return dt_et.strftime("%Y-%m-%d")
            except Exception:
                return ""
    # Helper: normalize team names for robust matching
    import re, unicodedata
    def norm_team(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = s.lower()
        # Strip common Bovada/ preseason qualifiers that break exact matching
        # e.g., "New Jersey Devils Split Squad" -> "New Jersey Devils"
        s = re.sub(r"\bsplit\s*squad\b", " ", s)
        s = re.sub(r"\bprospects?\b", " ", s)
        s = re.sub(r"\brookie\b", " ", s)
        s = re.sub(r"\bpreseason\b", " ", s)
        s = re.sub(r"\bexhibition\b", " ", s)
        s = re.sub(r"\bteam\s*[ab]\b", " ", s)  # Team A / Team B if ever present
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s
    # Load schedule for date and ratings (non-CLI core)
    source = (source or "web").lower()
    if source == "web":
        client = NHLWebClient()
        games = client.schedule_range(date, date)
    elif source == "stats":
        client = NHLClient()
        games = client.schedule(date, date)
    elif source == "nhlpy":
        try:
            from .data.nhl_api_nhlpy import NHLNhlPyClient  # lazy import
        except Exception as e:
            print("nhl-api-py adapter not available:", e)
            raise typer.Exit(code=1)
        client = NHLNhlPyClient()
        games = client.schedule_range(date, date)
    else:
        print(f"Unknown source '{source}'. Use one of: web, stats, nhlpy")
        raise typer.Exit(code=1)
    # Load ratings
    ratings_path = MODEL_DIR / "elo_ratings.json"
    if not ratings_path.exists():
        print("No ratings found. Run 'train' first.")
        raise typer.Exit(code=1)
    with open(ratings_path, "r", encoding="utf-8") as f:
        ratings = json.load(f)
    elo = Elo()
    elo = Elo(cfg=_load_elo_config()); elo.ratings = ratings

    # Load model configuration (sigma for normal approximations, etc.)
    base_mu = None
    sigma_total = 1.9  # default std dev for total goals distribution
    sigma_diff = 1.6   # default std dev for goal differential distribution
    cfg_path = MODEL_DIR / "config.json"
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                _cfg = json.load(f) or {}
            base_mu = float(_cfg.get("base_mu", None)) if _cfg.get("base_mu") is not None else None
            if _cfg.get("sigma_total") is not None:
                sigma_total = float(_cfg.get("sigma_total"))
            if _cfg.get("sigma_diff") is not None:
                sigma_diff = float(_cfg.get("sigma_diff"))
        except Exception:
            pass
    # Load subgroup adjustments
    trends = TrendAdjustments.load()

    rows = []
    odds_df = None
    # Load odds based on selected source
    odds_source = (odds_source or "csv").lower()
    if odds_source == "csv":
        if odds_csv and Path(odds_csv).exists():
            odds_df = pd.read_csv(odds_csv)
            odds_df["date"] = pd.to_datetime(odds_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            # Normalize team names and add abbreviations for robust matching
            odds_df["home_norm"] = odds_df["home"].apply(norm_team)
            odds_df["away_norm"] = odds_df["away"].apply(norm_team)
            try:
                from .web.teams import get_team_assets as _assets
                def to_abbr(x):
                    try:
                        return (_assets(str(x)).get("abbr") or "").upper()
                    except Exception:
                        return ""
                odds_df["home_abbr"] = odds_df["home"].apply(to_abbr)
                odds_df["away_abbr"] = odds_df["away"].apply(to_abbr)
            except Exception:
                odds_df["home_abbr"] = ""
                odds_df["away_abbr"] = ""
    elif odds_source == "oddsapi":
        if not snapshot:
            print("snapshot is required when odds_source='oddsapi' (e.g., 2024-03-01T12:00:00Z)")
        else:
            try:
                client_oa = OddsAPIClient()
                snap, _ = client_oa.historical_odds_snapshot(
                    sport="icehockey_nhl",
                    snapshot_iso=snapshot,
                    regions=odds_regions,
                    markets=odds_markets,
                    odds_format="american",
                )
                df = normalize_snapshot_to_rows(snap, bookmaker=odds_bookmaker, best_of_all=odds_best)
                # Fallback: preseason market key if no rows
                if df is None or df.empty:
                    snap2, _ = client_oa.historical_odds_snapshot(
                        sport="icehockey_nhl_preseason",
                        snapshot_iso=snapshot,
                        regions=odds_regions,
                        markets=odds_markets,
                        odds_format="american",
                    )
                    df = normalize_snapshot_to_rows(snap2, bookmaker=odds_bookmaker, best_of_all=odds_best)
                # Fallback: try current odds endpoint when historical has gaps
                if df is None or df.empty:
                    # current odds: GET /sports/{sport}/odds
                    import requests
                    base = "https://api.the-odds-api.com/v4"
                    params = {
                        "apiKey": client_oa.api_key,
                        "regions": odds_regions,
                        "markets": odds_markets,
                        "oddsFormat": "american",
                        "dateFormat": "iso",
                    }
                    url = f"{base}/sports/icehockey_nhl/odds"
                    r = requests.get(url, params=params, timeout=40)
                    if r.ok:
                        df_cur = normalize_snapshot_to_rows(r.json(), bookmaker=odds_bookmaker, best_of_all=odds_best)
                        if df_cur is not None and not df_cur.empty:
                            df = df_cur
                    if df is None or df.empty:
                        url2 = f"{base}/sports/icehockey_nhl_preseason/odds"
                        r2 = requests.get(url2, params=params, timeout=40)
                        if r2.ok:
                            df_cur2 = normalize_snapshot_to_rows(r2.json(), bookmaker=odds_bookmaker, best_of_all=odds_best)
                            if df_cur2 is not None and not df_cur2.empty:
                                df = df_cur2
                # Keep all rows; we'll match by date+teams first then fallback to teams if needed (handles UTC offsets)
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                df["home_norm"] = df["home"].apply(norm_team)
                df["away_norm"] = df["away"].apply(norm_team)
                # Add NHL team abbreviations for robust matching (handles LA vs Los Angeles, Utah rebrand, etc.)
                try:
                    from .web.teams import get_team_assets as _assets
                    def to_abbr(x):
                        try:
                            return (_assets(str(x)).get("abbr") or "").upper()
                        except Exception:
                            return ""
                    df["home_abbr"] = df["home"].apply(to_abbr)
                    df["away_abbr"] = df["away"].apply(to_abbr)
                except Exception:
                    df["home_abbr"] = ""
                    df["away_abbr"] = ""
                odds_df = df.copy()
            except Exception as e:
                print("Failed to fetch odds from The Odds API:", e)
                odds_df = None
    else:
        print("Unknown odds_source. Use 'csv' or 'oddsapi'.")
    
    # Load NN models for predictions (lazy import to avoid DLL issues on web server)
    period_model = None
    first_10min_model = None
    # NEW: Load game outcome models (TOTAL_GOALS, MONEYLINE, GOAL_DIFF)
    total_goals_nn = None
    moneyline_nn = None
    goal_diff_nn = None
    try:
        from .models.nn_games import NNGameModel  # Lazy import
        period_model = NNGameModel(model_type="PERIOD_GOALS", model_dir=MODEL_DIR / "nn_games")
        # Check if EITHER PyTorch model OR ONNX session loaded
        if period_model.model is None and period_model.onnx_session is None:
            period_model = None
        first_10min_model = NNGameModel(model_type="FIRST_10MIN", model_dir=MODEL_DIR / "nn_games")
        # Check if EITHER PyTorch model OR ONNX session loaded
        if first_10min_model.model is None and first_10min_model.onnx_session is None:
            first_10min_model = None
        
        # NEW: Load game outcome models
        total_goals_nn = NNGameModel(model_type="TOTAL_GOALS", model_dir=MODEL_DIR / "nn_games")
        if total_goals_nn.model is None and total_goals_nn.onnx_session is None:
            total_goals_nn = None
        
        moneyline_nn = NNGameModel(model_type="MONEYLINE", model_dir=MODEL_DIR / "nn_games")
        if moneyline_nn.model is None and moneyline_nn.onnx_session is None:
            moneyline_nn = None
        
        goal_diff_nn = NNGameModel(model_type="GOAL_DIFF", model_dir=MODEL_DIR / "nn_games")
        if goal_diff_nn.model is None and goal_diff_nn.onnx_session is None:
            goal_diff_nn = None
            
        # Log which models are available
        models_loaded = []
        if period_model: models_loaded.append("PERIOD_GOALS")
        if first_10min_model: models_loaded.append("FIRST_10MIN")
        if total_goals_nn: models_loaded.append("TOTAL_GOALS")
        if moneyline_nn: models_loaded.append("MONEYLINE")
        if goal_diff_nn: models_loaded.append("GOAL_DIFF")
        if models_loaded:
            print(f"[NN] Loaded models: {', '.join(models_loaded)}")
    except Exception as e:
        print(f"[warn] NN models not available: {e}")
        period_model = None
        first_10min_model = None
        total_goals_nn = None
        moneyline_nn = None
        goal_diff_nn = None
    
    # Load historical games for computing recent form features
    historical_games_df = None
    team_last_game = {}  # For rest days calculation
    team_games_played = {}  # For season progress calculation
    if period_model is not None or first_10min_model is not None:
        try:
            hist_path = RAW_DIR / "games_with_features.csv"
            if hist_path.exists():
                historical_games_df = pd.read_csv(hist_path, parse_dates=["date"])
                # Build team state: last game date and games played through historical data
                for _, game in historical_games_df.iterrows():
                    game_date = pd.to_datetime(game["date"])
                    home_team = game["home"]
                    away_team = game["away"]
                    team_last_game[home_team] = game_date
                    team_last_game[away_team] = game_date
                    team_games_played[home_team] = team_games_played.get(home_team, 0) + 1
                    team_games_played[away_team] = team_games_played.get(away_team, 0) + 1
        except Exception:
            pass
    
    def compute_recent_form_features(team, date, historical_df, window=10):
        """Compute recent form stats for a team before a given date."""
        if historical_df is None:
            return {"goals_last10": 0, "goals_against_last10": 0, "wins_last10": 0}
        
        # Get team's recent games before this date
        team_games = historical_df[
            ((historical_df["home"] == team) | (historical_df["away"] == team)) &
            (historical_df["date"] < date)
        ].tail(window)
        
        goals_for = []
        goals_against = []
        wins = 0
        
        for _, g in team_games.iterrows():
            if g["home"] == team:
                goals_for.append(g["home_goals"])
                goals_against.append(g["away_goals"])
                if g["home_goals"] > g["away_goals"]:
                    wins += 1
            else:
                goals_for.append(g["away_goals"])
                goals_against.append(g["home_goals"])
                if g["away_goals"] > g["home_goals"]:
                    wins += 1
        
        return {
            "goals_last10": float(np.mean(goals_for)) if goals_for else 0.0,
            "goals_against_last10": float(np.mean(goals_against)) if goals_against else 0.0,
            "wins_last10": float(wins)
        }
    
    for g in games:
        # Skip non-NHL matchups if any slipped through
        try:
            from .web.teams import get_team_assets
            if not get_team_assets(g.home).get("abbr") or not get_team_assets(g.away).get("abbr"):
                continue
        except Exception:
            pass
        # Get initial predictions from Elo (used as fallback if NN unavailable)
        p_home_elo, p_away_elo = elo.predict_moneyline_prob(g.home, g.away)
        
        # Start with Elo predictions (will be overridden by NN if available)
        p_home = p_home_elo
        p_away = p_away_elo
        # Per-game total line: use odds total_line if available, else provided total_line
        per_game_total = total_line
        match_info = None  # store the matched odds row for later EV/edge calc
        match_reversed = False  # whether matched odds row has home/away reversed vs schedule
        if odds_df is not None:
            # Use ET calendar day for matching (Bovada odds are keyed to ET date)
            key_date = iso_to_et_date(getattr(g, "gameDate")) or pd.to_datetime(g.gameDate).strftime("%Y-%m-%d")
            g_home_n = norm_team(g.home)
            g_away_n = norm_team(g.away)
            # Also derive team abbreviations for robust matching
            try:
                from .web.teams import get_team_assets as _assets
                g_home_abbr = (_assets(g.home).get("abbr") or "").upper()
                g_away_abbr = (_assets(g.away).get("abbr") or "").upper()
            except Exception:
                g_home_abbr = ""
                g_away_abbr = ""
            # Try exact date+team match (abbr preferred)
            m = pd.DataFrame()
            if {"home_abbr","away_abbr"}.issubset(set(odds_df.columns)) and g_home_abbr and g_away_abbr:
                m = odds_df[(odds_df["date"] == key_date) & (odds_df["home_abbr"] == g_home_abbr) & (odds_df["away_abbr"] == g_away_abbr)]
            if m.empty:
                m = odds_df[(odds_df["date"] == key_date) & (odds_df["home_norm"] == g_home_n) & (odds_df["away_norm"] == g_away_n)]
            if m.empty:
                # Fallback: team-only
                if {"home_abbr","away_abbr"}.issubset(set(odds_df.columns)) and g_home_abbr and g_away_abbr:
                    m = odds_df[(odds_df["home_abbr"] == g_home_abbr) & (odds_df["away_abbr"] == g_away_abbr)]
                if m.empty:
                    m = odds_df[(odds_df["home_norm"] == g_home_n) & (odds_df["away_norm"] == g_away_n)]
            if m.empty:
                # Fallback: reversed sides
                if {"home_abbr","away_abbr"}.issubset(set(odds_df.columns)) and g_home_abbr and g_away_abbr:
                    m = odds_df[(odds_df["home_abbr"] == g_away_abbr) & (odds_df["away_abbr"] == g_home_abbr)]
                if m.empty:
                    m = odds_df[(odds_df["home_norm"] == g_away_n) & (odds_df["away_norm"] == g_home_n)]
            if not m.empty:
                rec = m.iloc[0].to_dict()
                # Detect if matched row sides align with schedule; if not, mark reversed
                try:
                    mh_abbr = str(rec.get("home_abbr") or "").upper()
                    ma_abbr = str(rec.get("away_abbr") or "").upper()
                    if mh_abbr and ma_abbr and g_home_abbr and g_away_abbr:
                        match_reversed = not (mh_abbr == g_home_abbr and ma_abbr == g_away_abbr)
                    else:
                        # Fallback: compare normalized names
                        mh_n = str(rec.get("home_norm") or "").strip()
                        ma_n = str(rec.get("away_norm") or "").strip()
                        match_reversed = not (mh_n == g_home_n and ma_n == g_away_n)
                except Exception:
                    match_reversed = False
                rec["_reversed_vs_schedule"] = match_reversed
                match_info = rec
                if "total_line" in match_info:
                    val = match_info.get("total_line")
                    try:
                        if pd.notna(val):
                            per_game_total = float(val)
                    except Exception:
                        per_game_total = total_line

        # Initialize projections from odds total and neutral spread (will be overridden by NN below)
        model_total = float(per_game_total if per_game_total is not None else (2.0 * (base_mu or 3.05)))
        model_spread = 0.0
        # Split per-team projections equally for now; NN will refine
        proj_home_goals = model_total / 2.0
        proj_away_goals = model_total / 2.0

        # Prepare placeholders for calibrated probabilities
        p_home_cal = None
        p_away_cal = None
        p_over_cal = None
        p_under_cal = None
        
        # Period-by-period predictions if model available
        p1_home, p1_away, p2_home, p2_away, p3_home, p3_away = None, None, None, None, None, None
        first_10min_proj = None  # Expected goals in first 10 minutes (lambda)
        # Configurable knobs for first-10 projection
        import os as _os
        # Prefer deriving FIRST_10MIN from Period 1 by default (more stable, PBP-calibrated); allow override via env
        _F10_FROM_P1 = str(_os.getenv("FIRST10_FROM_P1", "1")).lower() not in ("0", "false", "no")
        # Calibrated defaults from PBP backtest (2023-24): use 0.55 and 0.15; still override via env if set
        _F10_SCALE = float(_os.getenv("FIRST10_SCALE", "0.55"))  # fraction of P1 goals happening in first 10 min
        # Fallback when P1 projections unavailable: share of total assigned to P1 (~35%) times 10/20 (=0.5)
        _F10_TOTAL_SCALE = float(_os.getenv("FIRST10_TOTAL_SCALE", "0.15"))
        if period_model is not None or first_10min_model is not None:
            try:
                # Build complete feature dict for neural network models (95 features expected)
                game_date = pd.to_datetime(getattr(g, "gameDate"))
                
                # Convert team names to abbreviations for team encoding (model trained with abbrs)
                try:
                    from .web.teams import get_team_assets
                    home_abbr = (get_team_assets(g.home).get("abbr") or "").upper()
                    away_abbr = (get_team_assets(g.away).get("abbr") or "").upper()
                except Exception:
                    home_abbr = g.home
                    away_abbr = g.away
                
                # Base ELO features
                home_elo = elo.get(g.home)
                away_elo = elo.get(g.away)
                game_features = {
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "elo_diff": home_elo - away_elo,
                }
                
                # Recent form features (last 10 games) - use full names for lookup
                home_form = compute_recent_form_features(g.home, game_date, historical_games_df)
                away_form = compute_recent_form_features(g.away, game_date, historical_games_df)
                game_features.update({
                    "home_goals_last10": home_form["goals_last10"],
                    "home_goals_against_last10": home_form["goals_against_last10"],
                    "home_wins_last10": home_form["wins_last10"],
                    "away_goals_last10": away_form["goals_last10"],
                    "away_goals_against_last10": away_form["goals_against_last10"],
                    "away_wins_last10": away_form["wins_last10"],
                })
                
                # Rest days (days since last game)
                home_rest = 1  # Default to 1 day rest
                away_rest = 1
                if g.home in team_last_game:
                    home_rest = max(0, (game_date - team_last_game[g.home]).days)
                if g.away in team_last_game:
                    away_rest = max(0, (game_date - team_last_game[g.away]).days)
                game_features["home_rest_days"] = float(home_rest)
                game_features["away_rest_days"] = float(away_rest)
                
                # Season progress (games played / 82)
                home_gp = team_games_played.get(g.home, 0)
                away_gp = team_games_played.get(g.away, 0)
                season_progress = (home_gp + away_gp) / (2.0 * 82.0)
                game_features["season_progress"] = float(season_progress)
                
                # Home indicator
                game_features["is_home"] = 1.0
                
                # Team one-hot encodings (use abbrs to match model training)
                game_features[f"home_team_{home_abbr}"] = 1.0
                game_features[f"away_team_{away_abbr}"] = 1.0
                
                # Update team state for next game (if processing chronologically)
                team_last_game[g.home] = game_date
                team_last_game[g.away] = game_date
                team_games_played[g.home] = home_gp + 1
                team_games_played[g.away] = away_gp + 1
                
                # NEW: Use NN models to override Elo predictions if available
                if moneyline_nn is not None:
                    try:
                        # Get NN moneyline prediction (home win probability)
                        p_home_nn = moneyline_nn.predict(home_abbr, away_abbr, game_features)
                        if p_home_nn is not None:
                            # Override Elo prediction
                            p_home = float(p_home_nn)
                            p_away = 1.0 - p_home
                    except Exception:
                        pass  # Fall back to Elo if NN fails
                
                if total_goals_nn is not None:
                    try:
                        # Get NN total goals prediction
                        total_nn = total_goals_nn.predict(home_abbr, away_abbr, game_features)
                        if total_nn is not None:
                            model_total = float(total_nn)
                    except Exception:
                        pass  # Fall back to odds total if NN fails
                
                if goal_diff_nn is not None:
                    try:
                        # Get NN goal differential prediction
                        diff_nn = goal_diff_nn.predict(home_abbr, away_abbr, game_features)
                        if diff_nn is not None:
                            model_spread = float(diff_nn)
                            # Derive per-team projections from total and differential
                            proj_home_goals = (model_total / 2.0) + (model_spread / 2.0)
                            proj_away_goals = (model_total / 2.0) - (model_spread / 2.0)
                            # Projections now fully from NN
                    except Exception:
                        pass  # Fall back to neutral spread if NN fails
                
                # Apply ML subgroup adjustment AFTER NN prediction (home side bias)
                h_keys = team_keys(g.home)
                ml_delta = get_adjustment(trends.ml_home, h_keys)
                if ml_delta:
                    p_home = min(max(p_home + ml_delta, 0.01), 0.99)
                    p_away = 1.0 - p_home
                
            except Exception:
                # If feature computation fails, skip predictions
                game_features = None

        if period_model is not None and game_features is not None:
            try:
                # Predict returns array [p1_home, p1_away, p2_home, p2_away, p3_home, p3_away]
                period_preds = period_model.predict(g.home, g.away, game_features)
                if period_preds is not None and len(period_preds) == 6:
                    p1_home, p1_away, p2_home, p2_away, p3_home, p3_away = period_preds
            except Exception:
                pass
        
        # First 10 minutes projection
        # Preferred: derive from P1 projections (more stable and better calibrated), then fall back to NN model, then total-based heuristic
        _first10_from_p1_val = None
        _first10_source = None
        if (p1_home is not None) and (p1_away is not None):
            try:
                # Expected goals in first 10 = (expected goals in P1) * (10/20) ~= 0.5, tunable via FIRST10_SCALE
                _first10_from_p1_val = max(0.0, (float(p1_home) + float(p1_away)) * _F10_SCALE)
            except Exception:
                _first10_from_p1_val = None

        _first10_nn_val = None
        if first_10min_model is not None and game_features is not None:
            try:
                # Predict returns single value (expected goals in first 10 min)
                _first10_nn_val = float(first_10min_model.predict(g.home, g.away, game_features))
            except Exception:
                _first10_nn_val = None

        # Choose value per preference
        if _F10_FROM_P1 and (_first10_from_p1_val is not None):
            first_10min_proj = _first10_from_p1_val
            _first10_source = "P1"
        elif _first10_nn_val is not None:
            first_10min_proj = _first10_nn_val
            _first10_source = "NN"
        else:
            # Heuristic fallback from total goals if nothing else available
            try:
                first_10min_proj = max(0.0, float(model_total) * _F10_TOTAL_SCALE)
                _first10_source = "TOTAL_HEURISTIC"
            except Exception:
                first_10min_proj = None

        # Derive first-10 YES probability from lambda if available
        _first10_prob = None
        try:
            if first_10min_proj is not None:
                import math as _math
                _first10_prob = 1.0 - _math.exp(-float(first_10min_proj))
        except Exception:
            _first10_prob = None

        # Compute betting probabilities; prefer Monte Carlo simulation from model outputs
        import math as _math
        try:
            from .models.simulator import simulate_from_period_lambdas, simulate_from_totals_diff, SimConfig
            _sim_available = True
        except Exception:
            _sim_available = False
        # Moneyline: base on NN/Elo probability then calibrate if available
        try:
            from .utils.calibration import load_calibration as _load_cal
            _cal_path = PROC_DIR / "model_calibration.json"
            _ml_cal, _tot_cal = _load_cal(_cal_path)
        except Exception:
            _ml_cal, _tot_cal = None, None

        # Calibrate ML prob if possible
        try:
            p_home_cal = float(_ml_cal.apply(np.array([p_home]))[0]) if _ml_cal is not None else float(p_home)
            p_away_cal = 1.0 - p_home_cal
        except Exception:
            p_home_cal = float(p_home)
            p_away_cal = 1.0 - p_home_cal

        # Totals & puckline probabilities via simulation (fallback to normal approx if simulation unavailable)
        if _sim_available:
            try:
                sim_cfg = SimConfig(n_sims=int(_os.getenv("SIM_N_SIMS", "20000")))  # uses default seed for reproducibility
                if (p1_home is not None) and (p1_away is not None) and (p2_home is not None) and (p2_away is not None) and (p3_home is not None) and (p3_away is not None):
                    sim = simulate_from_period_lambdas(
                        home_periods=[float(p1_home), float(p2_home), float(p3_home)],
                        away_periods=[float(p1_away), float(p2_away), float(p3_away)],
                        total_line=per_game_total,
                        puck_line=-1.5,
                        cfg=sim_cfg,
                    )
                else:
                    sim = simulate_from_totals_diff(
                        total_mean=float(model_total),
                        diff_mean=float(model_spread),
                        total_line=per_game_total,
                        puck_line=-1.5,
                        cfg=sim_cfg,
                    )
                # Calibrate totals prob if calibrator available
                if sim.get("over") is not None and not np.isnan(sim["over"]):
                    p_over_raw = float(sim["over"])  # Monte Carlo estimate
                    p_over_cal = float(_tot_cal.apply(np.array([p_over_raw]))[0]) if _tot_cal is not None else p_over_raw
                    p_under_cal = 1.0 - p_over_cal
                else:
                    p_over_cal, p_under_cal = None, None
                # Puck line from simulation
                p_home_pl = float(sim.get("home_puckline_-1.5")) if sim.get("home_puckline_-1.5") is not None else None
                p_away_pl = float(sim.get("away_puckline_+1.5")) if sim.get("away_puckline_+1.5") is not None else None
            except Exception:
                # Fallback to normal approximations if simulation fails
                try:
                    thr = float(per_game_total)
                    if abs(thr - round(thr)) < 1e-6:
                        thr = thr + 0.5
                    z = (thr - model_total) / max(1e-6, sigma_total)
                    p_over_raw = 1.0 - (0.5 * (1.0 + _math.erf(z / (_math.sqrt(2.0)))))
                    p_over_cal = float(_tot_cal.apply(np.array([p_over_raw]))[0]) if _tot_cal is not None else float(p_over_raw)
                    p_under_cal = 1.0 - p_over_cal
                except Exception:
                    p_over_cal = None
                    p_under_cal = None
                try:
                    z_pl = (1.5 - model_spread) / max(1e-6, sigma_diff)
                    p_home_pl = 1.0 - (0.5 * (1.0 + _math.erf(z_pl / (_math.sqrt(2.0)))))
                    p_away_pl = 1.0 - p_home_pl
                except Exception:
                    p_home_pl = None
                    p_away_pl = None
        else:
            # No simulator available: keep normal approximations
            try:
                thr = float(per_game_total)
                if abs(thr - round(thr)) < 1e-6:
                    thr = thr + 0.5
                z = (thr - model_total) / max(1e-6, sigma_total)
                p_over_raw = 1.0 - (0.5 * (1.0 + _math.erf(z / (_math.sqrt(2.0)))))
                p_over_cal = float(_tot_cal.apply(np.array([p_over_raw]))[0]) if _tot_cal is not None else float(p_over_raw)
                p_under_cal = 1.0 - p_over_cal
            except Exception:
                p_over_cal = None
                p_under_cal = None
            try:
                z_pl = (1.5 - model_spread) / max(1e-6, sigma_diff)
                p_home_pl = 1.0 - (0.5 * (1.0 + _math.erf(z_pl / (_math.sqrt(2.0)))))
                p_away_pl = 1.0 - p_home_pl
            except Exception:
                p_home_pl = None
                p_away_pl = None

        # Build base row with model probabilities
        row = {
            # Keep UTC ISO in 'date' for precision; add ET calendar day for grouping/UI
            "date": getattr(g, "gameDate"),
            "date_et": iso_to_et_date(getattr(g, "gameDate")),
            "home": getattr(g, "home"),
            "away": getattr(g, "away"),
            "total_line_used": float(per_game_total),
            "proj_home_goals": round(proj_home_goals, 2),
            "proj_away_goals": round(proj_away_goals, 2),
            "model_total": round(model_total, 2),
            "model_spread": round(model_spread, 2),
            # Period projections
            "period1_home_proj": round(float(p1_home), 2) if p1_home is not None else None,
            "period1_away_proj": round(float(p1_away), 2) if p1_away is not None else None,
            "period2_home_proj": round(float(p2_home), 2) if p2_home is not None else None,
            "period2_away_proj": round(float(p2_away), 2) if p2_away is not None else None,
            "period3_home_proj": round(float(p3_home), 2) if p3_home is not None else None,
            "period3_away_proj": round(float(p3_away), 2) if p3_away is not None else None,
            # First 10 minutes projection (lambda) and probability
            "first_10min_proj": round(float(first_10min_proj), 3) if first_10min_proj is not None else None,
            "first_10min_prob": round(float(_first10_prob), 4) if _first10_prob is not None else None,
            "first_10min_source": _first10_source,
            "p_home_ml": p_home_cal,
            "p_away_ml": p_away_cal,
            "p_over": p_over_cal,
            "p_under": p_under_cal,
            "p_home_pl_-1.5": float(p_home_pl) if p_home_pl is not None else None,
            "p_away_pl_+1.5": float(p_away_pl) if p_away_pl is not None else None,
        }

        # Helper to coerce odds to float
        def _f(v):
            try:
                return float(v)
            except Exception:
                try:
                    s = str(v).strip().replace(",", "")
                    return float(s)
                except Exception:
                    return None

        # If we matched odds for this game, compute EV/edge and record odds
        if match_info is not None:
            # Moneyline
            if "home_ml" in match_info or "away_ml" in match_info:
                # If matched odds row is reversed vs schedule, swap home/away mapping
                _rev = bool(match_info.get("_reversed_vs_schedule"))
                _home_ml_src = "away_ml" if _rev else "home_ml"
                _away_ml_src = "home_ml" if _rev else "away_ml"
                dec_home = american_to_decimal(_f(match_info.get(_home_ml_src))) if match_info.get(_home_ml_src) is not None else None
                dec_away = american_to_decimal(_f(match_info.get(_away_ml_src))) if match_info.get(_away_ml_src) is not None else None
                if dec_home is not None and dec_away is not None:
                    imp_h = decimal_to_implied_prob(dec_home)
                    imp_a = decimal_to_implied_prob(dec_away)
                    nv_h, nv_a = remove_vig_two_way(imp_h, imp_a)
                    row["ev_home_ml"] = round(ev_unit(row["p_home_ml"], dec_home), 4)
                    row["ev_away_ml"] = round(ev_unit(row["p_away_ml"], dec_away), 4)
                    row["edge_home_ml"] = round(row["p_home_ml"] - nv_h, 4)
                    row["edge_away_ml"] = round(row["p_away_ml"] - nv_a, 4)
                # Assign odds and books respecting reversal
                row["home_ml_odds"] = _f(match_info.get(_home_ml_src)) if match_info.get(_home_ml_src) is not None else None
                row["away_ml_odds"] = _f(match_info.get(_away_ml_src)) if match_info.get(_away_ml_src) is not None else None
                _home_ml_book_src = ("away_ml_book" if _rev else "home_ml_book")
                _away_ml_book_src = ("home_ml_book" if _rev else "away_ml_book")
                if match_info.get(_home_ml_book_src) is not None:
                    row["home_ml_book"] = match_info.get(_home_ml_book_src)
                if match_info.get(_away_ml_book_src) is not None:
                    row["away_ml_book"] = match_info.get(_away_ml_book_src)
                if bankroll > 0 and dec_home is not None and dec_away is not None:
                    row["stake_home_ml"] = round(kelly_stake(row["p_home_ml"], dec_home, bankroll, kelly_fraction_part), 2)
                    row["stake_away_ml"] = round(kelly_stake(row["p_away_ml"], dec_away, bankroll, kelly_fraction_part), 2)

            # Totals
            if (match_info.get("over") is not None) and (match_info.get("under") is not None):
                dec_over = american_to_decimal(_f(match_info.get("over")))
                dec_under = american_to_decimal(_f(match_info.get("under")))
                if dec_over is not None and dec_under is not None:
                    imp_o = decimal_to_implied_prob(dec_over)
                    imp_u = decimal_to_implied_prob(dec_under)
                    nv_o, nv_u = remove_vig_two_way(imp_o, imp_u)
                    row["ev_over"] = round(ev_unit(row["p_over"], dec_over), 4)
                    row["ev_under"] = round(ev_unit(row["p_under"], dec_under), 4)
                    row["edge_over"] = round(row["p_over"] - nv_o, 4)
                    row["edge_under"] = round(row["p_under"] - nv_u, 4)
                row["over_odds"] = _f(match_info.get("over"))
                row["under_odds"] = _f(match_info.get("under"))
                if match_info.get("over_book") is not None:
                    row["over_book"] = match_info.get("over_book")
                if match_info.get("under_book") is not None:
                    row["under_book"] = match_info.get("under_book")
                if bankroll > 0 and dec_over is not None and dec_under is not None:
                    row["stake_over"] = round(kelly_stake(row["p_over"], dec_over, bankroll, kelly_fraction_part), 2)
                    row["stake_under"] = round(kelly_stake(row["p_under"], dec_under, bankroll, kelly_fraction_part), 2)

            # Puck line (home -1.5 / away +1.5)
            if (match_info.get("home_pl_-1.5") is not None) and (match_info.get("away_pl_+1.5") is not None):
                _rev = bool(match_info.get("_reversed_vs_schedule"))
                # If reversed vs schedule, we cannot reliably map -1.5 for schedule home from the flattened row.
                # To avoid misleading prices, only compute/display PL EV/prices when sides align.
                if not _rev:
                    dec_hpl = american_to_decimal(_f(match_info.get("home_pl_-1.5")))
                    dec_apl = american_to_decimal(_f(match_info.get("away_pl_+1.5")))
                    if dec_hpl is not None and dec_apl is not None:
                        imp_hpl = decimal_to_implied_prob(dec_hpl)
                        imp_apl = decimal_to_implied_prob(dec_apl)
                        nv_hpl, nv_apl = remove_vig_two_way(imp_hpl, imp_apl)
                        row["ev_home_pl_-1.5"] = round(ev_unit(row["p_home_pl_-1.5"], dec_hpl), 4)
                        row["ev_away_pl_+1.5"] = round(ev_unit(row["p_away_pl_+1.5"], dec_apl), 4)
                        row["edge_home_pl_-1.5"] = round(row["p_home_pl_-1.5"] - nv_hpl, 4)
                        row["edge_away_pl_+1.5"] = round(row["p_away_pl_+1.5"] - nv_apl, 4)
                    row["home_pl_-1.5_odds"] = _f(match_info.get("home_pl_-1.5"))
                    row["away_pl_+1.5_odds"] = _f(match_info.get("away_pl_+1.5"))
                    if match_info.get("home_pl_-1.5_book") is not None:
                        row["home_pl_-1.5_book"] = match_info.get("home_pl_-1.5_book")
                    if match_info.get("away_pl_+1.5_book") is not None:
                        row["away_pl_+1.5_book"] = match_info.get("away_pl_+1.5_book")
                    if bankroll > 0 and dec_hpl is not None and dec_apl is not None:
                        row["stake_home_pl_-1.5"] = round(kelly_stake(row["p_home_pl_-1.5"], dec_hpl, bankroll, kelly_fraction_part), 2)
                        row["stake_away_pl_+1.5"] = round(kelly_stake(row["p_away_pl_+1.5"], dec_apl, bankroll, kelly_fraction_part), 2)

        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        print("No eligible NHL games found for date.")
        out_path = PROC_DIR / f"predictions_{date}.csv"
        save_df(out, out_path)
        return out_path
    out_path = PROC_DIR / f"predictions_{date}.csv"
    # If we didn't attach any odds this run, but an older file exists with odds, merge them and recompute EVs
    try:
        has_new_odds = any(col in out.columns for col in [
            "home_ml_odds","away_ml_odds","over_odds","under_odds","home_pl_-1.5_odds","away_pl_+1.5_odds"
        ]) and any(out.get(c).notna().any() for c in out.columns if c.endswith("_odds"))
    except Exception:
        has_new_odds = False
    if not has_new_odds and out_path.exists():
        try:
            prev = pd.read_csv(out_path)
            # Identify odds/book columns to carry over
            carry_cols = [
                "home_ml_odds","away_ml_odds","over_odds","under_odds","home_pl_-1.5_odds","away_pl_+1.5_odds",
                "home_ml_book","away_ml_book","over_book","under_book","home_pl_-1.5_book","away_pl_+1.5_book",
                "total_line_used",
                # Preserve captured openers and closings to avoid losing them on odds-miss runs
                "open_home_ml_odds","open_away_ml_odds","open_over_odds","open_under_odds",
                "open_home_pl_-1.5_odds","open_away_pl_+1.5_odds","open_total_line_used",
                "open_home_ml_book","open_away_ml_book","open_over_book","open_under_book",
                "open_home_pl_-1.5_book","open_away_pl_+1.5_book","open_snapshot",
                "close_home_ml_odds","close_away_ml_odds","close_over_odds","close_under_odds",
                "close_home_pl_-1.5_odds","close_away_pl_+1.5_odds","close_total_line_used",
                "close_home_ml_book","close_away_ml_book","close_over_book","close_under_book",
                "close_home_pl_-1.5_book","close_away_pl_+1.5_book","close_snapshot",
            ]
            keys = ["date","home","away"]
            if set(keys).issubset(set(prev.columns)):
                # Merge carry-over columns
                prev_carry = prev[keys + [c for c in carry_cols if c in prev.columns]].copy()
                out = out.merge(prev_carry, on=keys, how="left", suffixes=("", "_prev"))
                # Prefer current total_line_used but backfill from previous if missing
                if "total_line_used_prev" in out.columns:
                    out["total_line_used"] = out["total_line_used"].fillna(out["total_line_used_prev"]) 
                    out = out.drop(columns=[c for c in out.columns if c.endswith("_prev")])
                # Recompute EV/Edge where both prices exist and probs are present
                def _recalc_row(r):
                    # Moneyline
                    try:
                        if pd.notna(r.get("home_ml_odds")) and pd.notna(r.get("away_ml_odds")):
                            dec_h = american_to_decimal(float(r.get("home_ml_odds")))
                            dec_a = american_to_decimal(float(r.get("away_ml_odds")))
                            imp_h = decimal_to_implied_prob(dec_h)
                            imp_a = decimal_to_implied_prob(dec_a)
                            nv_h, nv_a = remove_vig_two_way(imp_h, imp_a)
                            r["ev_home_ml"] = round(ev_unit(float(r.get("p_home_ml")), dec_h), 4)
                            r["ev_away_ml"] = round(ev_unit(float(r.get("p_away_ml")), dec_a), 4)
                            r["edge_home_ml"] = round(float(r.get("p_home_ml")) - nv_h, 4)
                            r["edge_away_ml"] = round(float(r.get("p_away_ml")) - nv_a, 4)
                        elif pd.notna(r.get("open_home_ml_odds")) and pd.notna(r.get("open_away_ml_odds")):
                            dec_h = american_to_decimal(float(r.get("open_home_ml_odds")))
                            dec_a = american_to_decimal(float(r.get("open_away_ml_odds")))
                            imp_h = decimal_to_implied_prob(dec_h)
                            imp_a = decimal_to_implied_prob(dec_a)
                            nv_h, nv_a = remove_vig_two_way(imp_h, imp_a)
                            r["ev_home_ml"] = round(ev_unit(float(r.get("p_home_ml")), dec_h), 4)
                            r["ev_away_ml"] = round(ev_unit(float(r.get("p_away_ml")), dec_a), 4)
                            r["edge_home_ml"] = round(float(r.get("p_home_ml")) - nv_h, 4)
                            r["edge_away_ml"] = round(float(r.get("p_away_ml")) - nv_a, 4)
                    except Exception:
                        pass
                    # Totals
                    try:
                        if pd.notna(r.get("over_odds")) and pd.notna(r.get("under_odds")):
                            dec_o = american_to_decimal(float(r.get("over_odds")))
                            dec_u = american_to_decimal(float(r.get("under_odds")))
                            imp_o = decimal_to_implied_prob(dec_o)
                            imp_u = decimal_to_implied_prob(dec_u)
                            nv_o, nv_u = remove_vig_two_way(imp_o, imp_u)
                            r["ev_over"] = round(ev_unit(float(r.get("p_over")), dec_o), 4)
                            r["ev_under"] = round(ev_unit(float(r.get("p_under")), dec_u), 4)
                            r["edge_over"] = round(float(r.get("p_over")) - nv_o, 4)
                            r["edge_under"] = round(float(r.get("p_under")) - nv_u, 4)
                        elif pd.notna(r.get("open_over_odds")) and pd.notna(r.get("open_under_odds")):
                            dec_o = american_to_decimal(float(r.get("open_over_odds")))
                            dec_u = american_to_decimal(float(r.get("open_under_odds")))
                            imp_o = decimal_to_implied_prob(dec_o)
                            imp_u = decimal_to_implied_prob(dec_u)
                            nv_o, nv_u = remove_vig_two_way(imp_o, imp_u)
                            r["ev_over"] = round(ev_unit(float(r.get("p_over")), dec_o), 4)
                            r["ev_under"] = round(ev_unit(float(r.get("p_under")), dec_u), 4)
                            r["edge_over"] = round(float(r.get("p_over")) - nv_o, 4)
                            r["edge_under"] = round(float(r.get("p_under")) - nv_u, 4)
                    except Exception:
                        pass
                    # Puckline
                    try:
                        if pd.notna(r.get("home_pl_-1.5_odds")) and pd.notna(r.get("away_pl_+1.5_odds")):
                            dec_hpl = american_to_decimal(float(r.get("home_pl_-1.5_odds")))
                            dec_apl = american_to_decimal(float(r.get("away_pl_+1.5_odds")))
                            imp_hpl = decimal_to_implied_prob(dec_hpl)
                            imp_apl = decimal_to_implied_prob(dec_apl)
                            nv_hpl, nv_apl = remove_vig_two_way(imp_hpl, imp_apl)
                            r["ev_home_pl_-1.5"] = round(ev_unit(float(r.get("p_home_pl_-1.5")), dec_hpl), 4)
                            r["ev_away_pl_+1.5"] = round(ev_unit(float(r.get("p_away_pl_+1.5")), dec_apl), 4)
                            r["edge_home_pl_-1.5"] = round(float(r.get("p_home_pl_-1.5")) - nv_hpl, 4)
                            r["edge_away_pl_+1.5"] = round(float(r.get("p_away_pl_+1.5")) - nv_apl, 4)
                        elif pd.notna(r.get("open_home_pl_-1.5_odds")) and pd.notna(r.get("open_away_pl_+1.5_odds")):
                            dec_hpl = american_to_decimal(float(r.get("open_home_pl_-1.5_odds")))
                            dec_apl = american_to_decimal(float(r.get("open_away_pl_+1.5_odds")))
                            imp_hpl = decimal_to_implied_prob(dec_hpl)
                            imp_apl = decimal_to_implied_prob(dec_apl)
                            nv_hpl, nv_apl = remove_vig_two_way(imp_hpl, imp_apl)
                            r["ev_home_pl_-1.5"] = round(ev_unit(float(r.get("p_home_pl_-1.5")), dec_hpl), 4)
                            r["ev_away_pl_+1.5"] = round(ev_unit(float(r.get("p_away_pl_+1.5")), dec_apl), 4)
                            r["edge_home_pl_-1.5"] = round(float(r.get("p_home_pl_-1.5")) - nv_hpl, 4)
                            r["edge_away_pl_+1.5"] = round(float(r.get("p_away_pl_+1.5")) - nv_apl, 4)
                    except Exception:
                        pass
                    return r
                out = out.apply(_recalc_row, axis=1)
        except Exception:
            # best-effort merge; ignore errors
            pass
    # If no games (empty frame), avoid writing empty files
    if out is None or (hasattr(out, "empty") and bool(out.empty)):
        print("No eligible NHL games found for date.")
        return out_path
    save_df(out, out_path)
    print(out)
    print(f"Saved predictions to {out_path}")
    # If EV columns exist, also save a long-form edges report sorted by EV desc
    ev_cols = [c for c in out.columns if c.startswith("ev_")]
    if ev_cols:
        edges_long = out.melt(id_vars=["date", "home", "away"], value_vars=ev_cols, var_name="market", value_name="ev").dropna()
        edges_long = edges_long.sort_values("ev", ascending=False)
        edges_path = PROC_DIR / f"edges_{date}.csv"
        save_df(edges_long, edges_path)
        print("Top edges:")
        print(edges_long.head(10))
        print(f"Saved edges to {edges_path}")
    return out_path


@app.command()
def predict(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD"),
    total_line: float = 6.0,
    odds_csv: Optional[str] = typer.Option(None, help="Path to odds CSV with columns: date,home,away,home_ml,away_ml,over,under,home_pl_-1.5,away_pl_+1.5 (American odds)"),
    source: str = typer.Option("web", help="Data source: 'web' (api-web.nhle.com), 'stats' (api.nhle.com/stats/rest), or 'nhlpy' (nhl-api-py)"),
    odds_source: str = typer.Option("oddsapi", help="Odds source: 'csv' (provide --odds-csv) or 'oddsapi' (provide --snapshot)"),
    snapshot: Optional[str] = typer.Option(None, help="When odds_source=oddsapi, ISO snapshot like 2024-03-01T12:00:00Z"),
    odds_regions: str = typer.Option("us", help="Odds API regions, e.g., us or us,us2"),
    odds_markets: str = typer.Option("h2h,totals,spreads", help="Odds API markets"),
    odds_bookmaker: Optional[str] = typer.Option(None, help="Preferred bookmaker key (e.g., pinnacle)"),
    odds_best: bool = typer.Option(False, help="Use best available odds across all bookmakers in snapshot"),
    bankroll: float = typer.Option(0.0, help="Bankroll amount for Kelly sizing (0 to disable)"),
    kelly_fraction_part: float = typer.Option(0.5, help="Fraction of Kelly to bet (e.g., 0.5 = half-Kelly)"),
):
    # Ensure we're running on ARM64 Python when available (helps ONNX QNN provider)
    try:
        import platform, sys as _sys
        _arch = (platform.machine() or "").upper()
        if "ARM" not in _arch:
            print(f"[warn] Non-ARM64 Python detected: {_sys.executable} arch={_arch}. For best performance on this machine, use the repo's ARM64 .venv.")
    except Exception:
        pass
    predict_core(
        date=date,
        total_line=total_line,
        odds_csv=odds_csv,
        source=source,
        odds_source=odds_source,
        snapshot=snapshot,
        odds_regions=odds_regions,
        odds_markets=odds_markets,
        odds_bookmaker=odds_bookmaker,
        odds_best=odds_best,
        bankroll=bankroll,
        kelly_fraction_part=kelly_fraction_part,
    )


@app.command()
def smoke():
    print("Running smoke test...")
    # Fake tiny dataset
    data = pd.DataFrame([
        {"date": "2024-01-01", "home": "Team A", "away": "Team B", "home_goals": 3, "away_goals": 2},
        {"date": "2024-01-03", "home": "Team B", "away": "Team A", "home_goals": 1, "away_goals": 4},
    ])
    save_df(data, RAW_DIR / "games.csv")
    featurize()
    train()
    # Use Elo ratings to predict a dummy game
    with open(MODEL_DIR / "elo_ratings.json", "r", encoding="utf-8") as f:
        ratings = json.load(f)
    elo = Elo()
    elo.ratings = ratings
    p_home, p_away = elo.predict_moneyline_prob("Team A", "Team B")
    print({"p_home_ml": p_home})
    print("Smoke test complete.")


@app.command()
def collect_props(start: str = typer.Option(...), end: str = typer.Option(...), source: str = typer.Option("stats", help="Source for schedule/boxscores: web | stats | nhlpy")):
    df = collect_player_game_stats(start, end, source=source)
    print(f"Collected {len(df)} player-game rows into data/raw/player_game_stats.csv")


@app.command(name="game-simulate")
def game_simulate(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    odds_source: str = typer.Option("oddsapi", help="Odds source: 'csv' or 'oddsapi'"),
    odds_csv: Optional[str] = typer.Option(None, help="When odds_source=csv, path to odds CSV"),
    snapshot: Optional[str] = typer.Option(None, help="When odds_source=oddsapi, ISO snapshot like 2025-12-01T12:00:00Z"),
    odds_regions: str = typer.Option("us", help="Odds API regions"),
    odds_markets: str = typer.Option("h2h,totals,spreads", help="Odds markets to fetch"),
    n_sims: int = typer.Option(20000, help="Monte Carlo samples"),
    sim_overdispersion_k: float = typer.Option(0.0, help="Gamma-Poisson overdispersion k (0=off)"),
    sim_shared_k: float = typer.Option(0.0, help="Shared pace Gamma k (correlation; 0=off)"),
    sim_empty_net_p: float = typer.Option(0.0, help="Empty-net extra goal probability when leading by 1 (0=off)"),
    sim_empty_net_two_goal_scale: float = typer.Option(0.0, help="Scale factor for empty-net probability when leading by 2 (0=off)"),
    totals_pace_alpha: float = typer.Option(0.0, help="Strength of SOG-based pace adjustment for totals (0=off)"),
    totals_goalie_beta: float = typer.Option(0.0, help="Strength of goalie SAVES-based defensive adjustment (0=off)"),
    totals_fatigue_beta: float = typer.Option(0.0, help="Strength of fatigue (B2B) offensive reduction (0=off)"),
    totals_rolling_pace_gamma: float = typer.Option(0.0, help="Strength of rolling team goals pace adjustment (last 10; 0=off)"),
    totals_pp_gamma: float = typer.Option(0.0, help="Strength of PP offense adjustment from team PP% (0=off)"),
    totals_pk_beta: float = typer.Option(0.0, help="Strength of PK defensive adjustment from team PK% (applied to opponent; 0=off)"),
    totals_penalty_gamma: float = typer.Option(0.0, help="Strength of penalty exposure adjustment from team committed/drawn rates (0=off)"),
    totals_xg_gamma: float = typer.Option(0.0, help="Strength of expected-goals (xGF/60) pace adjustment (0=off)"),
    totals_refs_gamma: float = typer.Option(0.0, help="Strength of referee penalty-rate adjustment (0=off)"),
    totals_goalie_form_gamma: float = typer.Option(0.0, help="Strength of goalie recent form adjustment (0=off)"),
):
    """Run Monte Carlo simulations for the given slate, producing probabilities from NN outputs.

    Writes data/processed/simulations_{date}.csv
    """
    # Load schedule for date and ratings
    source = "web"
    client = NHLWebClient() if source == "web" else NHLClient()
    games = client.schedule_range(date, date)
    # Load ratings
    ratings_path = MODEL_DIR / "elo_ratings.json"
    if not ratings_path.exists():
        print("No ratings found. Run 'train' first.")
        raise typer.Exit(code=1)
    with open(ratings_path, "r", encoding="utf-8") as f:
        ratings = json.load(f)
    elo = Elo(cfg=_load_elo_config()); elo.ratings = ratings
    # Config
    cfg_path = MODEL_DIR / "config.json"
    base_mu = None
    try:
        if cfg_path.exists():
            _cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            base_mu = float(_cfg.get("base_mu", None)) if _cfg.get("base_mu") is not None else None
    except Exception:
        base_mu = None

    # Odds (for total_line)
    odds_df = None
    def norm_team(s: str) -> str:
        import re, unicodedata
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s
    if (odds_source or "oddsapi").lower() == "csv":
        if odds_csv and Path(odds_csv).exists():
            odds_df = pd.read_csv(odds_csv)
            odds_df["date"] = pd.to_datetime(odds_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            odds_df["home_norm"] = odds_df["home"].apply(norm_team)
            odds_df["away_norm"] = odds_df["away"].apply(norm_team)
    else:
        try:
            from .data.odds_api import OddsAPIClient
            _client = OddsAPIClient()
            iso_date = pd.to_datetime(date).strftime("%Y-%m-%d")
            snap = snapshot
            df = _client.flat_snapshot(iso_date, regions=odds_regions, markets=odds_markets, snapshot_iso=snap, best=False)
            df["home_norm"] = df["home"].apply(norm_team)
            df["away_norm"] = df["away"].apply(norm_team)
            odds_df = df
        except Exception as e:
            print("Failed to fetch odds from The Odds API:", e)
            odds_df = None

    # NN models
    try:
        from .models.nn_games import NNGameModel
        period_model = NNGameModel(model_type="PERIOD_GOALS", model_dir=MODEL_DIR / "nn_games")
        if period_model.model is None and period_model.onnx_session is None:
            period_model = None
        total_goals_nn = NNGameModel(model_type="TOTAL_GOALS", model_dir=MODEL_DIR / "nn_games")
        if total_goals_nn.model is None and total_goals_nn.onnx_session is None:
            total_goals_nn = None
        goal_diff_nn = NNGameModel(model_type="GOAL_DIFF", model_dir=MODEL_DIR / "nn_games")
        if goal_diff_nn.model is None and goal_diff_nn.onnx_session is None:
            goal_diff_nn = None
    except Exception as e:
        print(f"[warn] NN models not available: {e}")
        period_model = None; total_goals_nn = None; goal_diff_nn = None

    # Calibration for totals
    try:
        from .utils.calibration import load_calibration as _load_cal
        _ml_cal, _tot_cal = _load_cal(PROC_DIR / "model_calibration.json")
    except Exception:
        _ml_cal, _tot_cal = None, None
    # Optional simulation-specific calibration (moneyline/totals/puckline)
    sim_cal_path = PROC_DIR / "sim_calibration.json"
    sim_ml_cal = sim_tot_cal = sim_pl_cal = None
    try:
        if sim_cal_path.exists() and sim_cal_path.stat().st_size > 0:
            obj = json.loads(sim_cal_path.read_text(encoding="utf-8"))
            def _mk(o):
                from .utils.calibration import BinaryCalibration
                return BinaryCalibration(float(o.get("t", 1.0)), float(o.get("b", 0.0)))
            if obj.get("moneyline"): sim_ml_cal = _mk(obj["moneyline"])
            if obj.get("totals"): sim_tot_cal = _mk(obj["totals"])
            if obj.get("puckline"): sim_pl_cal = _mk(obj["puckline"])
    except Exception:
        pass
    # Optional per-total-line calibrations
    sim_cal_line_path = PROC_DIR / "sim_calibration_per_line.json"
    tot_line_cals: Dict[float, Any] = {}
    try:
        if sim_cal_line_path.exists() and getattr(sim_cal_line_path.stat(), "st_size", 0) > 0:
            obj = json.loads(sim_cal_line_path.read_text(encoding="utf-8"))
            from .utils.calibration import BinaryCalibration
            for k, v in (obj.get("totals", {}) or {}).items():
                try:
                    line = float(k)
                    tot_line_cals[line] = BinaryCalibration(float(v.get("t", 1.0)), float(v.get("b", 0.0)))
                except Exception:
                    continue
    except Exception:
        tot_line_cals = {}

    from .models.simulator import simulate_from_period_lambdas, simulate_from_totals_diff, SimConfig
    sim_cfg = SimConfig(
        n_sims=int(n_sims),
        overdispersion_k=(sim_overdispersion_k if sim_overdispersion_k and sim_overdispersion_k > 0 else None),
        shared_k=(sim_shared_k if sim_shared_k and sim_shared_k > 0 else None),
        empty_net_p=(sim_empty_net_p if sim_empty_net_p and sim_empty_net_p > 0 else None),
        empty_net_two_goal_scale=(sim_empty_net_two_goal_scale if sim_empty_net_two_goal_scale and sim_empty_net_two_goal_scale > 0 else None),
    )

    # Load props projections for feature-driven totals adjustments (optional)
    props_df = None
    try:
        props_path = PROC_DIR / f"props_projections_all_{date}.csv"
        if props_path.exists() and getattr(props_path.stat(), "st_size", 0) > 0:
            props_df = pd.read_csv(props_path)
    except Exception:
        props_df = None

    rows = []
    # Load games for rest-day calculation
    games_raw = None
    try:
        games_raw = load_df(RAW_DIR / "games.csv")
        games_raw["date_et"] = pd.to_datetime(games_raw["date_et"], errors="coerce")
    except Exception:
        games_raw = None

    # Optional team special teams
    team_st: Optional[Dict[str, Dict[str, float]]] = None
    try:
        from .data.team_stats import load_team_special_teams
        team_st = load_team_special_teams(date)
    except Exception:
        team_st = None
    # Optional team penalty rates
    team_pr: Optional[Dict[str, Dict[str, float]]] = None
    try:
        if totals_penalty_gamma > 0.0:
            from .data.penalty_rates import load_team_penalty_rates
            team_pr = load_team_penalty_rates(date)
    except Exception:
        team_pr = None
    # Optional team expected goals rates
    team_xg: Optional[Dict[str, Dict[str, float]]] = None
    try:
        if totals_xg_gamma > 0.0:
            from .data.team_xg import load_team_xg
            team_xg = load_team_xg(date)
    except Exception:
        team_xg = None
    # Optional referee rates (per-game) and goalie recent form
    ref_map_base = None
    goalie_form_map: Optional[Dict[str, float]] = None
    try:
        if totals_refs_gamma > 0.0:
            from .data.referee_rates import load_referee_rates
            rr = load_referee_rates(date)
            if rr is not None:
                ref_map_base = rr  # (map, base)
    except Exception:
        ref_map_base = None
    try:
        if totals_goalie_form_gamma > 0.0:
            from .data.goalie_form import load_goalie_form
            goalie_form_map = load_goalie_form(date)
    except Exception:
        goalie_form_map = None

    for g in games:
        # Get total_line if available
        per_game_total = 2.0 * (base_mu or 3.05)
        match_info = None
        if odds_df is not None:
            try:
                key_date = pd.to_datetime(g.gameDate).strftime("%Y-%m-%d")
                m = odds_df[(odds_df["date"] == key_date) & (odds_df["home_norm"] == norm_team(g.home)) & (odds_df["away_norm"] == norm_team(g.away))]
                if m.empty:
                    m = odds_df[(odds_df["home_norm"] == norm_team(g.home)) & (odds_df["away_norm"] == norm_team(g.away))]
                if not m.empty:
                    rec = m.iloc[0].to_dict(); match_info = rec
                    if "total_line" in rec and pd.notna(rec.get("total_line")):
                        per_game_total = float(rec.get("total_line"))
            except Exception:
                pass

        # Build features and NN predictions
        p1_home = p1_away = p2_home = p2_away = p3_home = p3_away = None
        model_total = float(per_game_total)
        model_spread = 0.0
        try:
            game_date = pd.to_datetime(g.gameDate)
            try:
                from .web.teams import get_team_assets
                home_abbr = (get_team_assets(g.home).get("abbr") or "").upper()
                away_abbr = (get_team_assets(g.away).get("abbr") or "").upper()
            except Exception:
                home_abbr = g.home; away_abbr = g.away
            home_elo = elo.get(g.home); away_elo = elo.get(g.away)
            game_features = {
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": home_elo - away_elo,
                "home_rest_days": 1.0,
                "away_rest_days": 1.0,
                "season_progress": 0.5,
                "is_home": 1.0,
            }
            game_features[f"home_team_{home_abbr}"] = 1.0
            game_features[f"away_team_{away_abbr}"] = 1.0
            if period_model is not None:
                try:
                    arr = period_model.predict(g.home, g.away, game_features)
                    if arr is not None and len(arr) == 6:
                        p1_home, p1_away, p2_home, p2_away, p3_home, p3_away = [float(x) for x in arr]
                except Exception:
                    pass
            if total_goals_nn is not None:
                try:
                    model_total = float(total_goals_nn.predict(home_abbr, away_abbr, game_features))
                except Exception:
                    pass
            if goal_diff_nn is not None:
                try:
                    model_spread = float(goal_diff_nn.predict(home_abbr, away_abbr, game_features))
                except Exception:
                    pass
        except Exception:
            game_features = None

        # Derive pace, goalie defensive, fatigue, and rolling pace multipliers (optional)
        pace_mult_home = pace_mult_away = 1.0
        goalie_def_home = goalie_def_away = 1.0
        fatigue_mult_home = fatigue_mult_away = 1.0
        roll_mult_home = roll_mult_away = 1.0
        pp_mult_home = pp_mult_away = 1.0
        pk_mult_home_def = pk_mult_away_def = 1.0
        pen_mult_home = pen_mult_away = 1.0
        xg_mult_home = xg_mult_away = 1.0
        refs_mult = 1.0
        goalie_form_def_home = goalie_form_def_away = 1.0
        # Special teams: PP/PK adjustments
        try:
            if team_st is not None and (totals_pp_gamma > 0.0 or totals_pk_beta > 0.0):
                # Abbreviations
                try:
                    from .web.teams import get_team_assets
                    h_abbr = (get_team_assets(g.home).get("abbr") or "").upper()
                    a_abbr = (get_team_assets(g.away).get("abbr") or "").upper()
                except Exception:
                    h_abbr = str(g.home).upper(); a_abbr = str(g.away).upper()
                # Baselines
                vals = list((team_st or {}).values())
                pp_base = float(np.mean([v.get("pp_pct", 0.2) for v in vals])) if vals else 0.2
                pk_base = float(np.mean([v.get("pk_pct", 0.8) for v in vals])) if vals else 0.8
                th = (team_st or {}).get(h_abbr) or {}
                ta = (team_st or {}).get(a_abbr) or {}
                h_pp = float(th.get("pp_pct", pp_base)); a_pp = float(ta.get("pp_pct", pp_base))
                h_pk = float(th.get("pk_pct", pk_base)); a_pk = float(ta.get("pk_pct", pk_base))
                if totals_pp_gamma > 0.0 and pp_base > 0.0:
                    pp_mult_home = float(np.clip(1.0 + totals_pp_gamma * ((h_pp - pp_base) / pp_base), 0.85, 1.20))
                    pp_mult_away = float(np.clip(1.0 + totals_pp_gamma * ((a_pp - pp_base) / pp_base), 0.85, 1.20))
                if totals_pk_beta > 0.0 and pk_base > 0.0:
                    pk_mult_home_def = float(np.clip(1.0 - totals_pk_beta * ((h_pk - pk_base) / pk_base), 0.80, 1.15))
                    pk_mult_away_def = float(np.clip(1.0 - totals_pk_beta * ((a_pk - pk_base) / pk_base), 0.80, 1.15))
        except Exception:
            pass
        # Penalty exposure adjustment (committed/drawn per 60)
        try:
            if team_pr is not None and totals_penalty_gamma > 0.0:
                from .web.teams import get_team_assets
                try:
                    h_abbr = (get_team_assets(g.home).get("abbr") or "").upper()
                    a_abbr = (get_team_assets(g.away).get("abbr") or "").upper()
                except Exception:
                    h_abbr = str(g.home).upper(); a_abbr = str(g.away).upper()
                vals = list(team_pr.values())
                # Baseline exposure ~ avg committed + avg drawn
                c_base = float(np.nanmean([v.get("committed_per60") for v in vals])) if vals else None
                d_base = float(np.nanmean([v.get("drawn_per60") for v in vals])) if vals else None
                base_exp = (c_base or 0.0) + (d_base or 0.0)
                th = team_pr.get(h_abbr) or {}
                ta = team_pr.get(a_abbr) or {}
                exp_h = float((th.get("drawn_per60") or 0.0) + (ta.get("committed_per60") or 0.0))
                exp_a = float((ta.get("drawn_per60") or 0.0) + (th.get("committed_per60") or 0.0))
                if base_exp > 0.0:
                    pen_mult_home = float(np.clip(1.0 + totals_penalty_gamma * ((exp_h - base_exp) / base_exp), 0.85, 1.20))
                    pen_mult_away = float(np.clip(1.0 + totals_penalty_gamma * ((exp_a - base_exp) / base_exp), 0.85, 1.20))
        except Exception:
            pass
        # Expected goals pace adjustment (xGF/60)
        try:
            if team_xg is not None and totals_xg_gamma > 0.0:
                from .web.teams import get_team_assets
                try:
                    h_abbr = (get_team_assets(g.home).get("abbr") or "").upper()
                    a_abbr = (get_team_assets(g.away).get("abbr") or "").upper()
                except Exception:
                    h_abbr = str(g.home).upper(); a_abbr = str(g.away).upper()
                vals = [v.get("xgf60") for v in team_xg.values() if v.get("xgf60")]
                base_xg = float(np.nanmean(vals)) if vals else None
                th = team_xg.get(h_abbr) or {}
                ta = team_xg.get(a_abbr) or {}
                xg_h = float(th.get("xgf60")) if th.get("xgf60") else None
                xg_a = float(ta.get("xgf60")) if ta.get("xgf60") else None
                if base_xg and base_xg > 0.0:
                    if xg_h:
                        xg_mult_home = float(np.clip(1.0 + totals_xg_gamma * ((xg_h - base_xg) / base_xg), 0.85, 1.20))
                    if xg_a:
                        xg_mult_away = float(np.clip(1.0 + totals_xg_gamma * ((xg_a - base_xg) / base_xg), 0.85, 1.20))
        except Exception:
            pass
        # Referee rate multiplier (applied symmetrically to both teams)
        try:
            if ref_map_base is not None and totals_refs_gamma > 0.0:
                from .web.teams import get_team_assets
                try:
                    h_abbr = (get_team_assets(g.home).get("abbr") or "").upper()
                    a_abbr = (get_team_assets(g.away).get("abbr") or "").upper()
                except Exception:
                    h_abbr = str(g.home).upper(); a_abbr = str(g.away).upper()
                per_game_map, base_rate = ref_map_base
                key = f"{h_abbr}|{a_abbr}"
                rate = per_game_map.get(key)
                if rate and base_rate and base_rate > 0.0:
                    refs_mult = float(np.clip(1.0 + totals_refs_gamma * ((float(rate) - float(base_rate)) / float(base_rate)), 0.9, 1.2))
        except Exception:
            pass
        # Goalie recent form: defensive scaling of opponent scoring
        try:
            if goalie_form_map is not None and totals_goalie_form_gamma > 0.0:
                from .web.teams import get_team_assets
                try:
                    h_abbr = (get_team_assets(g.home).get("abbr") or "").upper()
                    a_abbr = (get_team_assets(g.away).get("abbr") or "").upper()
                except Exception:
                    h_abbr = str(g.home).upper(); a_abbr = str(g.away).upper()
                vals = [v for v in goalie_form_map.values() if v is not None]
                base_form = float(np.nanmean(vals)) if vals else None
                gh = goalie_form_map.get(h_abbr); ga = goalie_form_map.get(a_abbr)
                if base_form and base_form != 0.0:
                    if gh is not None:
                        goalie_form_def_home = float(np.clip(1.0 - totals_goalie_form_gamma * ((float(gh) - base_form) / abs(base_form)), 0.7, 1.3))
                    if ga is not None:
                        goalie_form_def_away = float(np.clip(1.0 - totals_goalie_form_gamma * ((float(ga) - base_form) / abs(base_form)), 0.7, 1.3))
        except Exception:
            pass
        # Fatigue: reduce offense on back-to-backs
        try:
            if games_raw is not None and totals_fatigue_beta > 0.0:
                d_et = pd.to_datetime(g.gameDate, utc=True).tz_convert("America/New_York").normalize()
                sub_home = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == g.home) | (games_raw["away"] == g.home))]
                sub_away = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == g.away) | (games_raw["away"] == g.away))]
                prev_home = pd.to_datetime(sub_home["date_et"].max()) if not sub_home.empty else None
                prev_away = pd.to_datetime(sub_away["date_et"].max()) if not sub_away.empty else None
                if prev_home is not None:
                    rd = int((d_et - prev_home.normalize()).days)
                    if rd == 1:
                        fatigue_mult_home = float(np.clip(1.0 - totals_fatigue_beta, 0.8, 1.0))
                if prev_away is not None:
                    rd = int((d_et - prev_away.normalize()).days)
                    if rd == 1:
                        fatigue_mult_away = float(np.clip(1.0 - totals_fatigue_beta, 0.8, 1.0))
        except Exception:
            pass
        # Rolling pace: last 10 games goals per game vs baseline
        try:
            if games_raw is not None and totals_rolling_pace_gamma > 0.0:
                d_et = pd.to_datetime(g.gameDate, utc=True).tz_convert("America/New_York").normalize()
                base_per_team = float(base_mu) if base_mu is not None else 3.05
                # Home team
                sub_h = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == g.home) | (games_raw["away"] == g.home))].copy()
                sub_h.sort_values("date_et", inplace=True)
                if not sub_h.empty:
                    # compute team goals per game regardless of home/away
                    sub_h["team_goals"] = np.where(sub_h["home"] == g.home, sub_h["home_goals"], sub_h["away_goals"])
                    last_h = sub_h.tail(10)
                    gpg_h = float(last_h["team_goals"].mean()) if len(last_h) > 0 else None
                    if gpg_h and gpg_h > 0.0 and base_per_team > 0.0:
                        roll_mult_home = float(np.clip(1.0 + totals_rolling_pace_gamma * ((gpg_h - base_per_team) / base_per_team), 0.8, 1.2))
                # Away team
                sub_a = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == g.away) | (games_raw["away"] == g.away))].copy()
                sub_a.sort_values("date_et", inplace=True)
                if not sub_a.empty:
                    sub_a["team_goals"] = np.where(sub_a["home"] == g.away, sub_a["home_goals"], sub_a["away_goals"])
                    last_a = sub_a.tail(10)
                    gpg_a = float(last_a["team_goals"].mean()) if len(last_a) > 0 else None
                    if gpg_a and gpg_a > 0.0 and base_per_team > 0.0:
                        roll_mult_away = float(np.clip(1.0 + totals_rolling_pace_gamma * ((gpg_a - base_per_team) / base_per_team), 0.8, 1.2))
        except Exception:
            pass
        try:
            if props_df is not None and (totals_pace_alpha > 0.0 or totals_goalie_beta > 0.0):
                # Normalize team keys
                try:
                    from .web.teams import get_team_assets
                    home_abbr = (get_team_assets(g.home).get("abbr") or "").upper()
                    away_abbr = (get_team_assets(g.away).get("abbr") or "").upper()
                except Exception:
                    home_abbr = str(g.home).upper(); away_abbr = str(g.away).upper()
                df = props_df.copy()
                # Robust column names
                mcol = "market"; tcol = "team"; lcol = next((c for c in ["proj_lambda", "lambda"] if c in df.columns), None)
                # Compute baselines across slate for SOG and SAVES
                sog_baseline = None; saves_baseline = None
                try:
                    sog_baseline = float(df[df[mcol] == "SOG"][lcol].mean()) if lcol and (lcol in df.columns) else None
                except Exception:
                    sog_baseline = None
                try:
                    saves_baseline = float(df[df[mcol] == "SAVES"][lcol].mean()) if lcol and (lcol in df.columns) else None
                except Exception:
                    saves_baseline = None
                # Team aggregates
                home_sog = away_sog = None
                home_goalie_saves = away_goalie_saves = None
                try:
                    if lcol and (lcol in df.columns) and tcol in df.columns and mcol in df.columns:
                        home_sog = float(df[(df[mcol] == "SOG") & (df[tcol].str.upper() == home_abbr)][lcol].sum())
                        away_sog = float(df[(df[mcol] == "SOG") & (df[tcol].str.upper() == away_abbr)][lcol].sum())
                        # For goalie saves, use max per team to proxy starter strength
                        h_g = df[(df[mcol] == "SAVES") & (df[tcol].str.upper() == home_abbr)][lcol]
                        a_g = df[(df[mcol] == "SAVES") & (df[tcol].str.upper() == away_abbr)][lcol]
                        home_goalie_saves = float(h_g.max()) if len(h_g) else None
                        away_goalie_saves = float(a_g.max()) if len(a_g) else None
                        # Starter gating: infer certainty from distribution of team SAVES props
                        def _starter_cert(series: pd.Series) -> float:
                            try:
                                vals = sorted([float(x) for x in series.dropna().tolist()], reverse=True)
                                if len(vals) == 0:
                                    return 0.25
                                if len(vals) == 1:
                                    return 1.0
                                top, second = vals[0], vals[1]
                                return (1.0 if (top - second) >= 2.0 else 0.5)
                            except Exception:
                                return 0.25
                        starter_cert_home = _starter_cert(h_g)
                        starter_cert_away = _starter_cert(a_g)
                    else:
                        starter_cert_home = 0.25; starter_cert_away = 0.25
                except Exception:
                    starter_cert_home = 0.25; starter_cert_away = 0.25
                # Pace multipliers from SOG totals
                try:
                    if totals_pace_alpha > 0.0 and sog_baseline and sog_baseline > 0.0 and home_sog and away_sog:
                        pace_mult_home = float(np.clip(1.0 + totals_pace_alpha * ((home_sog - sog_baseline) / sog_baseline), 0.7, 1.3))
                        pace_mult_away = float(np.clip(1.0 + totals_pace_alpha * ((away_sog - sog_baseline) / sog_baseline), 0.7, 1.3))
                except Exception:
                    pass
                # Goalie defensive multipliers (reductions applied to opponent scoring)
                try:
                    if totals_goalie_beta > 0.0 and saves_baseline and saves_baseline > 0.0:
                        # Apply starter gating by scaling effect with certainty
                        b_home = float(totals_goalie_beta * (starter_cert_home if 'starter_cert_home' in locals() else 0.25))
                        b_away = float(totals_goalie_beta * (starter_cert_away if 'starter_cert_away' in locals() else 0.25))
                        if home_goalie_saves:
                            goalie_def_home = float(np.clip(1.0 - b_home * ((home_goalie_saves - saves_baseline) / saves_baseline), 0.7, 1.2))
                        if away_goalie_saves:
                            goalie_def_away = float(np.clip(1.0 - b_away * ((away_goalie_saves - saves_baseline) / saves_baseline), 0.7, 1.2))
                        # Recent form adjustment: use last-5 days props 'SAVES' top goalie per team
                        try:
                            base_recent = None
                            h_recent = None; a_recent = None
                            prev_dates = []
                            try:
                                d0 = pd.to_datetime(date)
                                for k in range(1, 6):
                                    prev_dates.append((d0 - pd.Timedelta(days=k)).strftime('%Y-%m-%d'))
                            except Exception:
                                prev_dates = []
                            if prev_dates:
                                vals_all = []
                                vals_h = []; vals_a = []
                                for dd in prev_dates:
                                    p_path = PROC_DIR / f"props_projections_all_{dd}.csv"
                                    if p_path.exists() and getattr(p_path.stat(), 'st_size', 0) > 0:
                                        try:
                                            p_df = pd.read_csv(p_path)
                                            if mcol in p_df.columns and tcol in p_df.columns:
                                                # all teams baseline
                                                ss = p_df[p_df[mcol] == 'SAVES']
                                                if lcol and (lcol in ss.columns) and not ss.empty:
                                                    vals_all.extend([float(x) for x in ss[lcol].dropna().tolist()])
                                                # per-team top goalie proxy
                                                h_ser = p_df[(p_df[mcol] == 'SAVES') & (p_df[tcol].str.upper() == home_abbr)][lcol] if lcol in p_df.columns else None
                                                a_ser = p_df[(p_df[mcol] == 'SAVES') & (p_df[tcol].str.upper() == away_abbr)][lcol] if lcol in p_df.columns else None
                                                if h_ser is not None and len(h_ser):
                                                    vals_h.append(float(h_ser.max()))
                                                if a_ser is not None and len(a_ser):
                                                    vals_a.append(float(a_ser.max()))
                                        except Exception:
                                            pass
                                if vals_all:
                                    base_recent = float(np.mean(vals_all))
                                if vals_h:
                                    h_recent = float(np.mean(vals_h))
                                if vals_a:
                                    a_recent = float(np.mean(vals_a))
                            # Apply recent form as additional scaling
                            beta_r = float(totals_goalie_beta * 0.5)
                            if beta_r > 0.0 and base_recent and base_recent > 0.0:
                                if h_recent:
                                    rec_mult_home = float(np.clip(1.0 - beta_r * ((h_recent - base_recent) / base_recent), 0.75, 1.20))
                                    goalie_def_home = float(np.clip(goalie_def_home * rec_mult_home, 0.65, 1.25))
                                if a_recent:
                                    rec_mult_away = float(np.clip(1.0 - beta_r * ((a_recent - base_recent) / base_recent), 0.75, 1.20))
                                    goalie_def_away = float(np.clip(goalie_def_away * rec_mult_away, 0.65, 1.25))
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

        # Run simulation
        sim = None
        if (p1_home is not None) and (p1_away is not None) and (p2_home is not None) and (p2_away is not None) and (p3_home is not None) and (p3_away is not None):
            # Apply pace + goalie defensive multipliers per team if enabled
            h_periods = [max(0.0, float(x)) * pace_mult_home * xg_mult_home * refs_mult * goalie_def_away * goalie_form_def_away * fatigue_mult_home * roll_mult_home * pp_mult_home * pk_mult_away_def * pen_mult_home for x in [p1_home, p2_home, p3_home]]
            a_periods = [max(0.0, float(x)) * pace_mult_away * xg_mult_away * refs_mult * goalie_def_home * goalie_form_def_home * fatigue_mult_away * roll_mult_away * pp_mult_away * pk_mult_home_def * pen_mult_away for x in [p1_away, p2_away, p3_away]]
            sim = simulate_from_period_lambdas(
                home_periods=h_periods,
                away_periods=a_periods,
                total_line=per_game_total,
                puck_line=-1.5,
                cfg=sim_cfg,
            )
        else:
            # Derive team lambdas and adjust by pace + goalie defensive factors
            try:
                from .models.simulator import derive_team_lambdas
                lh, la = derive_team_lambdas(float(model_total), float(model_spread))
                lh_adj = float(np.clip(lh * pace_mult_home * xg_mult_home * refs_mult * goalie_def_away * goalie_form_def_away * fatigue_mult_home * roll_mult_home * pp_mult_home * pk_mult_away_def * pen_mult_home, 0.05, 8.0))
                la_adj = float(np.clip(la * pace_mult_away * xg_mult_away * refs_mult * goalie_def_home * goalie_form_def_home * fatigue_mult_away * roll_mult_away * pp_mult_away * pk_mult_home_def * pen_mult_away, 0.05, 8.0))
                adj_total = lh_adj + la_adj
                adj_diff = lh_adj - la_adj
            except Exception:
                adj_total = float(model_total); adj_diff = float(model_spread)
            sim = simulate_from_totals_diff(
                total_mean=adj_total,
                diff_mean=adj_diff,
                total_line=per_game_total,
                puck_line=-1.5,
                cfg=sim_cfg,
            )

        # Calibrate totals and puckline/moneyline if available
        p_over_cal = None; p_under_cal = None
        try:
            p_over_raw = float(sim.get("over"))
            # prefer simulation calibration if present
            if tot_line_cals and (per_game_total in tot_line_cals):
                p_over_cal = float(tot_line_cals[per_game_total].apply(np.array([p_over_raw]))[0])
            elif sim_tot_cal is not None:
                p_over_cal = float(sim_tot_cal.apply(np.array([p_over_raw]))[0])
            elif _tot_cal is not None:
                p_over_cal = float(_tot_cal.apply(np.array([p_over_raw]))[0])
            else:
                p_over_cal = p_over_raw
            p_under_cal = 1.0 - p_over_cal
        except Exception:
            pass

        p_ml_cal = None
        try:
            p_ml_raw = float(sim.get("home_ml"))
            p_ml_cal = float(sim_ml_cal.apply(np.array([p_ml_raw]))[0]) if sim_ml_cal is not None else p_ml_raw
        except Exception:
            pass

        p_pl_cal = None
        try:
            p_pl_raw = float(sim.get("home_puckline_-1.5"))
            p_pl_cal = float(sim_pl_cal.apply(np.array([p_pl_raw]))[0]) if sim_pl_cal is not None else p_pl_raw
        except Exception:
            pass

        rows.append({
            "date": g.gameDate,
            "date_et": pd.to_datetime(g.gameDate).tz_convert("America/New_York").strftime("%Y-%m-%d") if hasattr(pd.Timestamp(g.gameDate), 'tz_convert') else pd.to_datetime(g.gameDate).strftime("%Y-%m-%d"),
            "home": g.home,
            "away": g.away,
            "total_line_used": float(per_game_total),
            "model_total": float(model_total),
            "model_spread": float(model_spread),
            "pace_mult_home": float(pace_mult_home),
            "pace_mult_away": float(pace_mult_away),
            "goalie_def_home": float(goalie_def_home),
            "goalie_def_away": float(goalie_def_away),
            "p_home_ml_sim": float(sim.get("home_ml")),
            "p_away_ml_sim": float(sim.get("away_ml")),
            "p_over_sim": float(sim.get("over")) if sim.get("over") is not None else None,
            "p_under_sim": float(sim.get("under")) if sim.get("under") is not None else None,
            "p_over_sim_cal": float(p_over_cal) if p_over_cal is not None else None,
            "p_under_sim_cal": float(p_under_cal) if p_under_cal is not None else None,
            "p_home_pl_-1.5_sim": float(sim.get("home_puckline_-1.5")) if sim.get("home_puckline_-1.5") is not None else None,
            "p_away_pl_+1.5_sim": float(sim.get("away_puckline_+1.5")) if sim.get("away_puckline_+1.5") is not None else None,
            "p_home_ml_sim_cal": float(p_ml_cal) if p_ml_cal is not None else None,
            "p_home_pl_-1.5_sim_cal": float(p_pl_cal) if p_pl_cal is not None else None,
            "n_sims": int(n_sims),
        })

    out = pd.DataFrame(rows)
    out_path = PROC_DIR / f"simulations_{date}.csv"
    save_df(out, out_path)
    print(out.head())
    print(f"Saved simulations to {out_path}")


@app.command(name="game-backtest-sim")
def game_backtest_sim(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    n_sims: int = typer.Option(20000, help="Monte Carlo samples per game"),
    use_calibrated: bool = typer.Option(True, help="Use calibrated probabilities if available"),
    prefer_simulations: bool = typer.Option(False, help="Prefer precomputed simulations_{date}.csv over inline predictions"),
    sim_overdispersion_k: float = typer.Option(0.0, help="Gamma-Poisson overdispersion k (0=off)"),
    sim_shared_k: float = typer.Option(0.0, help="Shared pace Gamma k (correlation; 0=off)"),
    sim_empty_net_p: float = typer.Option(0.0, help="Empty-net extra goal probability when leading by 1 (0=off)"),
    sim_empty_net_two_goal_scale: float = typer.Option(0.0, help="Scale factor for empty-net probability when leading by 2 (0=off)"),
    totals_pace_alpha: float = typer.Option(0.0, help="Strength of SOG-based pace adjustment for totals (0=off)"),
    totals_goalie_beta: float = typer.Option(0.0, help="Strength of goalie SAVES-based defensive adjustment (0=off)"),
    totals_fatigue_beta: float = typer.Option(0.0, help="Strength of fatigue (B2B) offensive reduction (0=off)"),
    totals_rolling_pace_gamma: float = typer.Option(0.0, help="Strength of rolling team goals pace adjustment (last 10; 0=off)"),
    totals_pp_gamma: float = typer.Option(0.0, help="Strength of PP offense adjustment from team PP% (0=off)"),
    totals_pk_beta: float = typer.Option(0.0, help="Strength of PK defensive adjustment from team PK% (applied to opponent; 0=off)"),
    totals_penalty_gamma: float = typer.Option(0.0, help="Strength of penalty exposure adjustment from team committed/drawn rates (0=off)"),
    totals_xg_gamma: float = typer.Option(0.0, help="Strength of expected goals (xGF/60) pace adjustment (0=off)"),
    totals_refs_gamma: float = typer.Option(0.0, help="Strength of referee penalty-rate adjustment (0=off)"),
    totals_goalie_form_gamma: float = typer.Option(0.0, help="Strength of goalie recent form adjustment (0=off)"),
):
    """Backtest simulation probabilities vs outcomes over a date range using predictions files.

    Reads data/processed/predictions_{date}.csv for each date and uses period projections or totals/diff to simulate ML/PL/Totals.
    Joins actual results from data/raw/games.csv.
    """
    from datetime import datetime as _dt, timedelta as _td
    # Load actuals
    games_raw = load_df(RAW_DIR / "games.csv")
    games_raw["date_et"] = pd.to_datetime(games_raw["date"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    # Metrics accumulators
    ml_y: List[int] = []; ml_p: List[float] = []; ml_p_cal: List[float] = []
    tot_y: List[int] = []; tot_p: List[float] = []; tot_p_cal: List[float] = []
    pl_y: List[int] = []; pl_p: List[float] = []; pl_p_cal: List[float] = []
    from .models.simulator import simulate_from_period_lambdas, simulate_from_totals_diff, SimConfig
    sim_cfg = SimConfig(
        n_sims=int(n_sims),
        random_state=123,
        overdispersion_k=(sim_overdispersion_k if sim_overdispersion_k and sim_overdispersion_k > 0 else None),
        shared_k=(sim_shared_k if sim_shared_k and sim_shared_k > 0 else None),
        empty_net_p=(sim_empty_net_p if sim_empty_net_p and sim_empty_net_p > 0 else None),
        empty_net_two_goal_scale=(sim_empty_net_two_goal_scale if sim_empty_net_two_goal_scale and sim_empty_net_two_goal_scale > 0 else None),
    )
    # Load simulation calibration if present
    sim_cal_path = PROC_DIR / "sim_calibration.json"
    sim_ml_cal = sim_tot_cal = sim_pl_cal = None
    try:
        if use_calibrated and sim_cal_path.exists() and getattr(sim_cal_path.stat(), "st_size", 0) > 0:
            obj = json.loads(sim_cal_path.read_text(encoding="utf-8"))
            from .utils.calibration import BinaryCalibration as _BC
            if obj.get("moneyline"): sim_ml_cal = _BC(float(obj["moneyline"].get("t", 1.0)), float(obj["moneyline"].get("b", 0.0)))
            if obj.get("totals"): sim_tot_cal = _BC(float(obj["totals"].get("t", 1.0)), float(obj["totals"].get("b", 0.0)))
            if obj.get("puckline"): sim_pl_cal = _BC(float(obj["puckline"].get("t", 1.0)), float(obj["puckline"].get("b", 0.0)))
    except Exception:
        pass
    cur = _dt.strptime(start, "%Y-%m-%d")
    end_dt = _dt.strptime(end, "%Y-%m-%d")
    # Preload props projections cache by date for totals feature adjustments
    props_cache: Dict[str, Optional[pd.DataFrame]] = {}
    # Precompute last game date per team for rest-day fatigue
    last_game_map: Dict[str, pd.Timestamp] = {}
    try:
        if games_raw is not None:
            for team in pd.unique(pd.concat([games_raw["home"], games_raw["away"]], ignore_index=True)):
                sub = games_raw[(games_raw["home"] == team) | (games_raw["away"] == team)]
                if not sub.empty:
                    last_game_map[str(team)] = pd.to_datetime(sub["date_et"], utc=True).dt.tz_convert("America/New_York").max()
    except Exception:
        last_game_map = {}
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        pred_path = PROC_DIR / f"predictions_{d}.csv"
        use_df = None
        use_mode = None  # 'pred' or 'sim'
        if (not prefer_simulations) and pred_path.exists() and getattr(pred_path.stat(), "st_size", 0) > 0:
            try:
                use_df = pd.read_csv(pred_path)
                use_mode = 'pred'
            except Exception:
                use_df = None
        if use_df is None:
            sim_path = PROC_DIR / f"simulations_{d}.csv"
            if sim_path.exists() and getattr(sim_path.stat(), "st_size", 0) > 0:
                try:
                    use_df = pd.read_csv(sim_path)
                    use_mode = 'sim'
                except Exception:
                    use_df = None
        if use_df is None:
            cur += _td(days=1)
            continue

        # Optional props projections for this date
        props_df = None
        try:
            if totals_pace_alpha > 0.0 or totals_goalie_beta > 0.0:
                if d not in props_cache:
                    ppath = PROC_DIR / f"props_projections_all_{d}.csv"
                    if ppath.exists() and getattr(ppath.stat(), "st_size", 0) > 0:
                        try:
                            props_cache[d] = pd.read_csv(ppath)
                        except Exception:
                            props_cache[d] = None
                    else:
                        props_cache[d] = None
                props_df = props_cache.get(d)
        except Exception:
            props_df = None

        # Optional team special teams for this date
        team_st: Optional[Dict[str, Dict[str, float]]] = None
        try:
            if totals_pp_gamma > 0.0 or totals_pk_beta > 0.0:
                from .data.team_stats import load_team_special_teams
                team_st = load_team_special_teams(d)
        except Exception:
            team_st = None
        # Optional team penalty rates for this date
        team_pr: Optional[Dict[str, Dict[str, float]]] = None
        try:
            if totals_penalty_gamma > 0.0:
                from .data.penalty_rates import load_team_penalty_rates
                team_pr = load_team_penalty_rates(d)
        except Exception:
            team_pr = None

        # Optional team expected goals rates for this date
        team_xg: Optional[Dict[str, Dict[str, float]]] = None
        try:
            if totals_xg_gamma > 0.0:
                from .data.team_xg import load_team_xg
                team_xg = load_team_xg(d)
        except Exception:
            team_xg = None
        # Optional referee per-game rates and goalie form
        ref_map_base = None
        goalie_form_map: Optional[Dict[str, float]] = None
        try:
            if totals_refs_gamma > 0.0:
                from .data.referee_rates import load_referee_rates
                rr = load_referee_rates(d)
                if rr is not None:
                    ref_map_base = rr
        except Exception:
            ref_map_base = None
        try:
            if totals_goalie_form_gamma > 0.0:
                from .data.goalie_form import load_goalie_form
                goalie_form_map = load_goalie_form(d)
        except Exception:
            goalie_form_map = None
            # Optional referee per-game rates + goalie form maps
            ref_map_base = None
            goalie_form_map: Optional[Dict[str, float]] = None
            try:
                if totals_refs_gamma > 0.0:
                    from .data.referee_rates import load_referee_rates
                    rr = load_referee_rates(d)
                    if rr is not None:
                        ref_map_base = rr
            except Exception:
                ref_map_base = None
            try:
                if totals_goalie_form_gamma > 0.0:
                    from .data.goalie_form import load_goalie_form
                    goalie_form_map = load_goalie_form(d)
            except Exception:
                goalie_form_map = None

        for _, r in use_df.iterrows():
            # actuals
            try:
                sub = games_raw[(games_raw["date_et"] == str(r.get("date_et"))) & (games_raw["home"] == r.get("home")) & (games_raw["away"] == r.get("away"))]
                if sub.empty:
                    continue
                ah = int(sub.iloc[0]["home_goals"]); aa = int(sub.iloc[0]["away_goals"]); atot = ah + aa; adiff = ah - aa
            except Exception:
                continue

            if use_mode == 'pred':
                # simulate from predictions row
                try:
                    if pd.notna(r.get("period1_home_proj")) and pd.notna(r.get("period1_away_proj")) and pd.notna(r.get("period2_home_proj")) and pd.notna(r.get("period2_away_proj")) and pd.notna(r.get("period3_home_proj")) and pd.notna(r.get("period3_away_proj")):
                        # Optionally adjust period lambdas by props-derived pace and goalie defensive multipliers
                        h_periods = [float(r.get("period1_home_proj")), float(r.get("period2_home_proj")), float(r.get("period3_home_proj"))]
                        a_periods = [float(r.get("period1_away_proj")), float(r.get("period2_away_proj")), float(r.get("period3_away_proj"))]
                        # Fatigue multipliers from rest-day detection (B2B)
                        fat_h = fat_a = 1.0
                        try:
                            if totals_fatigue_beta > 0.0:
                                d_et = pd.to_datetime(r.get("date_et"), utc=True).tz_convert("America/New_York").normalize() if pd.notna(r.get("date_et")) else None
                                home = str(r.get("home")); away = str(r.get("away"))
                                if d_et is not None:
                                    # last game strictly before current date
                                    prev_h = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == home) | (games_raw["away"] == home))]
                                    prev_a = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == away) | (games_raw["away"] == away))]
                                    prev_h_dt = pd.to_datetime(prev_h["date_et"].max()) if not prev_h.empty else None
                                    prev_a_dt = pd.to_datetime(prev_a["date_et"].max()) if not prev_a.empty else None
                                    if prev_h_dt is not None and int((d_et - prev_h_dt.normalize()).days) == 1:
                                        fat_h = float(np.clip(1.0 - totals_fatigue_beta, 0.8, 1.0))
                                    if prev_a_dt is not None and int((d_et - prev_a_dt.normalize()).days) == 1:
                                        fat_a = float(np.clip(1.0 - totals_fatigue_beta, 0.8, 1.0))
                        except Exception:
                            pass
                        # Rolling pace multipliers from last 10 games (team goals per game vs baseline)
                        roll_h = roll_a = 1.0
                        try:
                            if totals_rolling_pace_gamma > 0.0:
                                d_et = pd.to_datetime(r.get("date_et"), utc=True).tz_convert("America/New_York").normalize() if pd.notna(r.get("date_et")) else None
                                base_per_team = 3.05
                                if d_et is not None:
                                    home = str(r.get("home")); away = str(r.get("away"))
                                    sub_h = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == home) | (games_raw["away"] == home))].copy()
                                    sub_h.sort_values("date_et", inplace=True)
                                    if not sub_h.empty:
                                        sub_h["team_goals"] = np.where(sub_h["home"] == home, sub_h["home_goals"], sub_h["away_goals"]) 
                                        last_h = sub_h.tail(10)
                                        gpg_h = float(last_h["team_goals"].mean()) if len(last_h) > 0 else None
                                        if gpg_h and gpg_h > 0.0:
                                            roll_h = float(np.clip(1.0 + totals_rolling_pace_gamma * ((gpg_h - base_per_team) / base_per_team), 0.8, 1.2))
                                    sub_a = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == away) | (games_raw["away"] == away))].copy()
                                    sub_a.sort_values("date_et", inplace=True)
                                    if not sub_a.empty:
                                        sub_a["team_goals"] = np.where(sub_a["home"] == away, sub_a["home_goals"], sub_a["away_goals"]) 
                                        last_a = sub_a.tail(10)
                                        gpg_a = float(last_a["team_goals"].mean()) if len(last_a) > 0 else None
                                        if gpg_a and gpg_a > 0.0:
                                            roll_a = float(np.clip(1.0 + totals_rolling_pace_gamma * ((gpg_a - base_per_team) / base_per_team), 0.8, 1.2))
                        except Exception:
                            pass
                        # xG pacing multipliers
                        xg_mult_home = xg_mult_away = 1.0
                        try:
                            if team_xg is not None and totals_xg_gamma > 0.0:
                                vals = list(team_xg.values())
                                base_xgf = float(np.nanmean([v.get("xgf60") for v in vals])) if vals else None
                                h_abbr = str(r.get("home")).upper(); a_abbr = str(r.get("away")).upper()
                                h_xgf = float((team_xg.get(h_abbr) or {}).get("xgf60") or np.nan)
                                a_xgf = float((team_xg.get(a_abbr) or {}).get("xgf60") or np.nan)
                                if base_xgf and base_xgf > 0.0 and pd.notna(h_xgf) and pd.notna(a_xgf):
                                    xg_mult_home = float(np.clip(1.0 + totals_xg_gamma * ((h_xgf - base_xgf) / base_xgf), 0.85, 1.20))
                                    xg_mult_away = float(np.clip(1.0 + totals_xg_gamma * ((a_xgf - base_xgf) / base_xgf), 0.85, 1.20))
                        except Exception:
                            pass
                        try:
                            if props_df is not None and (totals_pace_alpha > 0.0 or totals_goalie_beta > 0.0):
                                mcol = "market"; tcol = "team"; lcol = next((c for c in ["proj_lambda", "lambda"] if c in props_df.columns), None)
                                home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                sog_baseline = float(props_df[props_df[mcol] == "SOG"][lcol].mean()) if lcol in props_df.columns else None
                                saves_baseline = float(props_df[props_df[mcol] == "SAVES"][lcol].mean()) if lcol in props_df.columns else None
                                home_sog = float(props_df[(props_df[mcol] == "SOG") & (props_df[tcol].str.upper() == home_abbr)][lcol].sum()) if lcol in props_df.columns else None
                                away_sog = float(props_df[(props_df[mcol] == "SOG") & (props_df[tcol].str.upper() == away_abbr)][lcol].sum()) if lcol in props_df.columns else None
                                h_g = props_df[(props_df[mcol] == "SAVES") & (props_df[tcol].str.upper() == home_abbr)][lcol] if lcol in props_df.columns else None
                                a_g = props_df[(props_df[mcol] == "SAVES") & (props_df[tcol].str.upper() == away_abbr)][lcol] if lcol in props_df.columns else None
                                home_goalie_saves = float(h_g.max()) if (isinstance(h_g, pd.Series) and len(h_g)) else None
                                away_goalie_saves = float(a_g.max()) if (isinstance(a_g, pd.Series) and len(a_g)) else None
                                pace_mult_home = pace_mult_away = 1.0
                                goalie_def_home = goalie_def_away = 1.0
                                if totals_pace_alpha > 0.0 and sog_baseline and sog_baseline > 0.0 and home_sog and away_sog:
                                    pace_mult_home = float(np.clip(1.0 + totals_pace_alpha * ((home_sog - sog_baseline) / sog_baseline), 0.7, 1.3))
                                    pace_mult_away = float(np.clip(1.0 + totals_pace_alpha * ((away_sog - sog_baseline) / sog_baseline), 0.7, 1.3))
                                if totals_goalie_beta > 0.0 and saves_baseline and saves_baseline > 0.0:
                                    if home_goalie_saves:
                                        goalie_def_home = float(np.clip(1.0 - totals_goalie_beta * ((home_goalie_saves - saves_baseline) / saves_baseline), 0.7, 1.2))
                                    if away_goalie_saves:
                                        goalie_def_away = float(np.clip(1.0 - totals_goalie_beta * ((away_goalie_saves - saves_baseline) / saves_baseline), 0.7, 1.2))
                                # Special teams multipliers
                                pp_mult_home = pp_mult_away = 1.0
                                pk_mult_home_def = pk_mult_away_def = 1.0
                                pen_mult_home = pen_mult_away = 1.0
                                refs_mult = 1.0
                                goalie_form_def_home = goalie_form_def_away = 1.0
                                try:
                                    if team_st is not None and (totals_pp_gamma > 0.0 or totals_pk_beta > 0.0):
                                        # Team abbreviations
                                        h_abbr = home_abbr; a_abbr = away_abbr
                                        vals = list(team_st.values())
                                        pp_base = float(np.mean([v.get("pp_pct", 0.2) for v in vals])) if vals else 0.2
                                        pk_base = float(np.mean([v.get("pk_pct", 0.8) for v in vals])) if vals else 0.8
                                        th = team_st.get(h_abbr) or {}
                                        ta = team_st.get(a_abbr) or {}
                                        h_pp = float(th.get("pp_pct", pp_base)); a_pp = float(ta.get("pp_pct", pp_base))
                                        h_pk = float(th.get("pk_pct", pk_base)); a_pk = float(ta.get("pk_pct", pk_base))
                                        if totals_pp_gamma > 0.0 and pp_base > 0.0:
                                            pp_mult_home = float(np.clip(1.0 + totals_pp_gamma * ((h_pp - pp_base) / pp_base), 0.85, 1.20))
                                            pp_mult_away = float(np.clip(1.0 + totals_pp_gamma * ((a_pp - pp_base) / pp_base), 0.85, 1.20))
                                        if totals_pk_beta > 0.0 and pk_base > 0.0:
                                            pk_mult_home_def = float(np.clip(1.0 - totals_pk_beta * ((h_pk - pk_base) / pk_base), 0.80, 1.15))
                                            pk_mult_away_def = float(np.clip(1.0 - totals_pk_beta * ((a_pk - pk_base) / pk_base), 0.80, 1.15))
                                except Exception:
                                    pass
                                # Penalty exposure multipliers
                                try:
                                    if team_pr is not None and totals_penalty_gamma > 0.0:
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        vals = list(team_pr.values())
                                        c_base = float(np.nanmean([v.get("committed_per60") for v in vals])) if vals else None
                                        d_base = float(np.nanmean([v.get("drawn_per60") for v in vals])) if vals else None
                                        base_exp = (c_base or 0.0) + (d_base or 0.0)
                                        th = team_pr.get(home_abbr) or {}
                                        ta = team_pr.get(away_abbr) or {}
                                        exp_h = float((th.get("drawn_per60") or 0.0) + (ta.get("committed_per60") or 0.0))
                                        exp_a = float((ta.get("drawn_per60") or 0.0) + (th.get("committed_per60") or 0.0))
                                        if base_exp > 0.0:
                                            pen_mult_home = float(np.clip(1.0 + totals_penalty_gamma * ((exp_h - base_exp) / base_exp), 0.85, 1.20))
                                            pen_mult_away = float(np.clip(1.0 + totals_penalty_gamma * ((exp_a - base_exp) / base_exp), 0.85, 1.20))
                                except Exception:
                                    pass
                                # Referee multiplier
                                try:
                                    if ref_map_base is not None and totals_refs_gamma > 0.0:
                                        vals = ref_map_base; (per_game_map, base_rate) = vals
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        key = f"{home_abbr}|{away_abbr}"
                                        rate = per_game_map.get(key)
                                        if rate and base_rate and base_rate > 0.0:
                                            refs_mult = float(np.clip(1.0 + totals_refs_gamma * ((float(rate) - float(base_rate)) / float(base_rate)), 0.9, 1.2))
                                except Exception:
                                    pass
                                # Goalie recent form defensive scaling
                                try:
                                    if goalie_form_map is not None and totals_goalie_form_gamma > 0.0:
                                        vals = [v for v in goalie_form_map.values() if v is not None]
                                        base_form = float(np.nanmean(vals)) if vals else None
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        gh = goalie_form_map.get(home_abbr); ga = goalie_form_map.get(away_abbr)
                                        if base_form and base_form != 0.0:
                                            if gh is not None:
                                                goalie_form_def_home = float(np.clip(1.0 - totals_goalie_form_gamma * ((float(gh) - base_form) / abs(base_form)), 0.7, 1.3))
                                            if ga is not None:
                                                goalie_form_def_away = float(np.clip(1.0 - totals_goalie_form_gamma * ((float(ga) - base_form) / abs(base_form)), 0.7, 1.3))
                                except Exception:
                                    pass
                                h_periods = [max(0.0, float(x)) * pace_mult_home * refs_mult * goalie_def_away * goalie_form_def_away * fat_h * roll_h * pp_mult_home * pk_mult_away_def * pen_mult_home * xg_mult_home for x in h_periods]
                                a_periods = [max(0.0, float(x)) * pace_mult_away * refs_mult * goalie_def_home * goalie_form_def_home * fat_a * roll_a * pp_mult_away * pk_mult_home_def * pen_mult_away * xg_mult_away for x in a_periods]
                        except Exception:
                            pass
                        sim = simulate_from_period_lambdas(
                            home_periods=h_periods,
                            away_periods=a_periods,
                            total_line=float(r.get("total_line_used")),
                            puck_line=-1.5,
                            cfg=sim_cfg,
                        )
                    else:
                        # Apply optional totals feature adjustments when simulating inline
                        adj_total = float(r.get("model_total")); adj_diff = float(r.get("model_spread"))
                        try:
                            if props_df is not None and (totals_pace_alpha > 0.0 or totals_goalie_beta > 0.0):
                                mcol = "market"; tcol = "team"; lcol = next((c for c in ["proj_lambda", "lambda"] if c in props_df.columns), None)
                                # Normalize team keys
                                home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                # Baselines
                                sog_baseline = float(props_df[props_df[mcol] == "SOG"][lcol].mean()) if lcol and (lcol in props_df.columns) else None
                                saves_baseline = float(props_df[props_df[mcol] == "SAVES"][lcol].mean()) if lcol and (lcol in props_df.columns) else None
                                # Team aggregates
                                home_sog = float(props_df[(props_df[mcol] == "SOG") & (props_df[tcol].str.upper() == home_abbr)][lcol].sum()) if lcol and (lcol in props_df.columns) else None
                                away_sog = float(props_df[(props_df[mcol] == "SOG") & (props_df[tcol].str.upper() == away_abbr)][lcol].sum()) if lcol and (lcol in props_df.columns) else None
                                h_g = props_df[(props_df[mcol] == "SAVES") & (props_df[tcol].str.upper() == home_abbr)][lcol] if lcol and (lcol in props_df.columns) else None
                                a_g = props_df[(props_df[mcol] == "SAVES") & (props_df[tcol].str.upper() == away_abbr)][lcol] if lcol and (lcol in props_df.columns) else None
                                home_goalie_saves = float(h_g.max()) if (isinstance(h_g, pd.Series) and len(h_g)) else None
                                away_goalie_saves = float(a_g.max()) if (isinstance(a_g, pd.Series) and len(a_g)) else None
                                # Multipliers
                                pace_mult_home = pace_mult_away = 1.0
                                goalie_def_home = goalie_def_away = 1.0
                                if totals_pace_alpha > 0.0 and sog_baseline and sog_baseline > 0.0 and home_sog and away_sog:
                                    pace_mult_home = float(np.clip(1.0 + totals_pace_alpha * ((home_sog - sog_baseline) / sog_baseline), 0.7, 1.3))
                                    pace_mult_away = float(np.clip(1.0 + totals_pace_alpha * ((away_sog - sog_baseline) / sog_baseline), 0.7, 1.3))
                                if totals_goalie_beta > 0.0 and saves_baseline and saves_baseline > 0.0:
                                    if home_goalie_saves:
                                        goalie_def_home = float(np.clip(1.0 - totals_goalie_beta * ((home_goalie_saves - saves_baseline) / saves_baseline), 0.7, 1.2))
                                    if away_goalie_saves:
                                        goalie_def_away = float(np.clip(1.0 - totals_goalie_beta * ((away_goalie_saves - saves_baseline) / saves_baseline), 0.7, 1.2))
                                from .models.simulator import derive_team_lambdas
                                lh, la = derive_team_lambdas(adj_total, adj_diff)
                                # Apply fatigue multipliers (offensive reduction on B2B)
                                fat_h = fat_a = 1.0
                                roll_h = roll_a = 1.0
                                # Special teams multipliers
                                pp_mult_home = pp_mult_away = 1.0
                                pk_mult_home_def = pk_mult_away_def = 1.0
                                pen_mult_home = pen_mult_away = 1.0
                                refs_mult = 1.0
                                goalie_form_def_home = goalie_form_def_away = 1.0
                                try:
                                    if team_st is not None and (totals_pp_gamma > 0.0 or totals_pk_beta > 0.0):
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        vals = list(team_st.values())
                                        pp_base = float(np.mean([v.get("pp_pct", 0.2) for v in vals])) if vals else 0.2
                                        pk_base = float(np.mean([v.get("pk_pct", 0.8) for v in vals])) if vals else 0.8
                                        th = team_st.get(home_abbr) or {}
                                        ta = team_st.get(away_abbr) or {}
                                        h_pp = float(th.get("pp_pct", pp_base)); a_pp = float(ta.get("pp_pct", pp_base))
                                        h_pk = float(th.get("pk_pct", pk_base)); a_pk = float(ta.get("pk_pct", pk_base))
                                        if totals_pp_gamma > 0.0 and pp_base > 0.0:
                                            pp_mult_home = float(np.clip(1.0 + totals_pp_gamma * ((h_pp - pp_base) / pp_base), 0.85, 1.20))
                                            pp_mult_away = float(np.clip(1.0 + totals_pp_gamma * ((a_pp - pp_base) / pp_base), 0.85, 1.20))
                                        if totals_pk_beta > 0.0 and pk_base > 0.0:
                                            pk_mult_home_def = float(np.clip(1.0 - totals_pk_beta * ((h_pk - pk_base) / pk_base), 0.80, 1.15))
                                            pk_mult_away_def = float(np.clip(1.0 - totals_pk_beta * ((a_pk - pk_base) / pk_base), 0.80, 1.15))
                                except Exception:
                                    pass
                                # Referee multiplier
                                try:
                                    if ref_map_base is not None and totals_refs_gamma > 0.0:
                                        per_game_map, base_rate = ref_map_base
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        key = f"{home_abbr}|{away_abbr}"
                                        rate = per_game_map.get(key)
                                        if rate and base_rate and base_rate > 0.0:
                                            refs_mult = float(np.clip(1.0 + totals_refs_gamma * ((float(rate) - float(base_rate)) / float(base_rate)), 0.9, 1.2))
                                except Exception:
                                    pass
                                # Goalie recent form defensive scaling
                                try:
                                    if goalie_form_map is not None and totals_goalie_form_gamma > 0.0:
                                        vals = [v for v in goalie_form_map.values() if v is not None]
                                        base_form = float(np.nanmean(vals)) if vals else None
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        gh = goalie_form_map.get(home_abbr); ga = goalie_form_map.get(away_abbr)
                                        if base_form and base_form != 0.0:
                                            if gh is not None:
                                                goalie_form_def_home = float(np.clip(1.0 - totals_goalie_form_gamma * ((float(gh) - base_form) / abs(base_form)), 0.7, 1.3))
                                            if ga is not None:
                                                goalie_form_def_away = float(np.clip(1.0 - totals_goalie_form_gamma * ((float(ga) - base_form) / abs(base_form)), 0.7, 1.3))
                                except Exception:
                                    pass
                                # Penalty exposure multipliers
                                try:
                                    if team_pr is not None and totals_penalty_gamma > 0.0:
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        vals = list(team_pr.values())
                                        c_base = float(np.nanmean([v.get("committed_per60") for v in vals])) if vals else None
                                        d_base = float(np.nanmean([v.get("drawn_per60") for v in vals])) if vals else None
                                        base_exp = (c_base or 0.0) + (d_base or 0.0)
                                        th = team_pr.get(home_abbr) or {}
                                        ta = team_pr.get(away_abbr) or {}
                                        exp_h = float((th.get("drawn_per60") or 0.0) + (ta.get("committed_per60") or 0.0))
                                        exp_a = float((ta.get("drawn_per60") or 0.0) + (th.get("committed_per60") or 0.0))
                                        if base_exp > 0.0:
                                            pen_mult_home = float(np.clip(1.0 + totals_penalty_gamma * ((exp_h - base_exp) / base_exp), 0.85, 1.20))
                                            pen_mult_away = float(np.clip(1.0 + totals_penalty_gamma * ((exp_a - base_exp) / base_exp), 0.85, 1.20))
                                except Exception:
                                    pass
                                try:
                                    if totals_fatigue_beta > 0.0:
                                        d_et = pd.to_datetime(r.get("date_et"), utc=True).tz_convert("America/New_York").normalize() if pd.notna(r.get("date_et")) else None
                                        home = str(r.get("home")); away = str(r.get("away"))
                                        if d_et is not None:
                                            prev_h = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == home) | (games_raw["away"] == home))]
                                            prev_a = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == away) | (games_raw["away"] == away))]
                                            prev_h_dt = pd.to_datetime(prev_h["date_et"].max()) if not prev_h.empty else None
                                            prev_a_dt = pd.to_datetime(prev_a["date_et"].max()) if not prev_a.empty else None
                                            if prev_h_dt is not None and int((d_et - prev_h_dt.normalize()).days) == 1:
                                                fat_h = float(np.clip(1.0 - totals_fatigue_beta, 0.8, 1.0))
                                            if prev_a_dt is not None and int((d_et - prev_a_dt.normalize()).days) == 1:
                                                fat_a = float(np.clip(1.0 - totals_fatigue_beta, 0.8, 1.0))
                                except Exception:
                                    pass
                                # Rolling pace multipliers
                                try:
                                    if totals_rolling_pace_gamma > 0.0:
                                        d_et = pd.to_datetime(r.get("date_et"), utc=True).tz_convert("America/New_York").normalize() if pd.notna(r.get("date_et")) else None
                                        base_per_team = 3.05
                                        if d_et is not None:
                                            home = str(r.get("home")); away = str(r.get("away"))
                                            sub_h = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == home) | (games_raw["away"] == home))].copy()
                                            sub_h.sort_values("date_et", inplace=True)
                                            if not sub_h.empty:
                                                sub_h["team_goals"] = np.where(sub_h["home"] == home, sub_h["home_goals"], sub_h["away_goals"]) 
                                                last_h = sub_h.tail(10)
                                                gpg_h = float(last_h["team_goals"].mean()) if len(last_h) > 0 else None
                                                if gpg_h and gpg_h > 0.0:
                                                    roll_h = float(np.clip(1.0 + totals_rolling_pace_gamma * ((gpg_h - base_per_team) / base_per_team), 0.8, 1.2))
                                            sub_a = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == away) | (games_raw["away"] == away))].copy()
                                            sub_a.sort_values("date_et", inplace=True)
                                            if not sub_a.empty:
                                                sub_a["team_goals"] = np.where(sub_a["home"] == away, sub_a["home_goals"], sub_a["away_goals"]) 
                                                last_a = sub_a.tail(10)
                                                gpg_a = float(last_a["team_goals"].mean()) if len(last_a) > 0 else None
                                                if gpg_a and gpg_a > 0.0:
                                                    roll_a = float(np.clip(1.0 + totals_rolling_pace_gamma * ((gpg_a - base_per_team) / base_per_team), 0.8, 1.2))
                                except Exception:
                                    pass
                                lh_adj = float(np.clip(lh * pace_mult_home * refs_mult * goalie_def_away * goalie_form_def_away * fat_h * roll_h * pp_mult_home * pk_mult_away_def * pen_mult_home, 0.05, 8.0))
                                la_adj = float(np.clip(la * pace_mult_away * refs_mult * goalie_def_home * goalie_form_def_home * fat_a * roll_a * pp_mult_away * pk_mult_home_def * pen_mult_away, 0.05, 8.0))
                                # xG pacing multipliers for totals/diff path
                                xg_mult_home = xg_mult_away = 1.0
                                try:
                                    if team_xg is not None and totals_xg_gamma > 0.0:
                                        vals = list(team_xg.values())
                                        base_xgf = float(np.nanmean([v.get("xgf60") for v in vals])) if vals else None
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        h_xgf = float((team_xg.get(home_abbr) or {}).get("xgf60") or np.nan)
                                        a_xgf = float((team_xg.get(away_abbr) or {}).get("xgf60") or np.nan)
                                        if base_xgf and base_xgf > 0.0 and pd.notna(h_xgf) and pd.notna(a_xgf):
                                            xg_mult_home = float(np.clip(1.0 + totals_xg_gamma * ((h_xgf - base_xgf) / base_xgf), 0.85, 1.20))
                                            xg_mult_away = float(np.clip(1.0 + totals_xg_gamma * ((a_xgf - base_xgf) / base_xgf), 0.85, 1.20))
                                except Exception:
                                    pass
                                lh_adj = float(np.clip(lh_adj * xg_mult_home, 0.05, 8.0))
                                la_adj = float(np.clip(la_adj * xg_mult_away, 0.05, 8.0))
                                adj_total = lh_adj + la_adj
                                adj_diff = lh_adj - la_adj
                        except Exception:
                            pass
                        sim = simulate_from_totals_diff(
                            total_mean=adj_total,
                            diff_mean=adj_diff,
                            total_line=float(r.get("total_line_used")),
                            puck_line=-1.5,
                            cfg=sim_cfg,
                        )
                except Exception:
                    continue
                p_ml = float(sim.get("home_ml", np.nan))
                p_over = float(sim.get("over", np.nan))
                p_pl = float(sim.get("home_puckline_-1.5", np.nan))
                # apply calibration if available
                p_ml_cal_row = p_ml
                p_over_cal_row = p_over
                p_pl_cal_row = p_pl
                try:
                    if use_calibrated and sim_ml_cal is not None:
                        p_ml_cal_row = float(sim_ml_cal.apply(np.array([p_ml]))[0])
                    # Prefer per-line totals calibration if available
                    if use_calibrated:
                        # determine line for this row
                        line_val = r.get("close_total_line_used")
                        if line_val is None or (isinstance(line_val, float) and np.isnan(line_val)):
                            line_val = r.get("total_line_used")
                        line = float(line_val) if (line_val is not None and not (isinstance(line_val, float) and np.isnan(line_val))) else None
                        tot_line_map = {}
                        try:
                            obj = json.loads((PROC_DIR / "sim_calibration_per_line.json").read_text(encoding="utf-8"))
                            tot_line_map = obj.get("totals", {}) or {}
                        except Exception:
                            tot_line_map = {}
                        if (line is not None) and (str(line) in tot_line_map):
                            from .utils.calibration import BinaryCalibration
                            c = tot_line_map[str(line)]
                            cal = BinaryCalibration(float(c.get("t", 1.0)), float(c.get("b", 0.0)))
                            p_over_cal_row = float(cal.apply(np.array([p_over]))[0])
                        elif sim_tot_cal is not None:
                            p_over_cal_row = float(sim_tot_cal.apply(np.array([p_over]))[0])
                    if use_calibrated and sim_pl_cal is not None:
                        p_pl_cal_row = float(sim_pl_cal.apply(np.array([p_pl]))[0])
                except Exception:
                    pass
            else:
                # use precomputed simulations probabilities
                p_ml = float(r.get("p_home_ml_sim", np.nan))
                p_over = float(r.get("p_over_sim", np.nan))
                p_pl = float(r.get("p_home_pl_-1.5_sim", np.nan))
                p_ml_cal_row = float(r.get("p_home_ml_sim_cal", p_ml))
                p_over_cal_row = float(r.get("p_over_sim_cal", p_over))
                p_pl_cal_row = float(r.get("p_home_pl_-1.5_sim_cal", p_pl))

            # record ML
            ml_y.append(1 if ah > aa else 0)
            ml_p.append(p_ml)
            if use_calibrated: ml_p_cal.append(p_ml_cal_row)
            # record totals (exclude pushes)
            try:
                # prefer closing total line if present
                line_val = r.get("close_total_line_used")
                if line_val is None or (isinstance(line_val, float) and np.isnan(line_val)):
                    line_val = r.get("total_line_used")
                line = float(line_val)
                if atot != line:
                    tot_y.append(1 if atot > line else 0)
                    tot_p.append(p_over)
                    if use_calibrated: tot_p_cal.append(p_over_cal_row)
            except Exception:
                pass
            # record puck line
            pl_y.append(1 if adiff > 1.5 else 0)
            pl_p.append(p_pl)
            if use_calibrated: pl_p_cal.append(p_pl_cal_row)
        cur += _td(days=1)
    # Summaries
    def _brier(y, p):
        if not y or not p: return None
        arr_y = np.array(y, dtype=np.float32); arr_p = np.array(p, dtype=np.float32)
        return float(np.mean((arr_p - arr_y) ** 2))
    def _accuracy(y, p):
        if not y or not p: return None
        return float(np.mean(((np.array(p) >= 0.5).astype(int) == np.array(y).astype(int))))
    res = {
        "range": {"start": start, "end": end},
        "moneyline_raw": {"n": len(ml_y), "brier": _brier(ml_y, ml_p), "accuracy": _accuracy(ml_y, ml_p)},
        "totals_raw": {"n": len(tot_y), "brier": _brier(tot_y, tot_p), "accuracy": _accuracy(tot_y, tot_p)},
        "puckline_raw": {"n": len(pl_y), "brier": _brier(pl_y, pl_p), "accuracy": _accuracy(pl_y, pl_p)},
    }
    if use_calibrated and ml_p_cal and tot_p_cal and pl_p_cal:
        res.update({
            "moneyline_cal": {"n": len(ml_y), "brier": _brier(ml_y, ml_p_cal), "accuracy": _accuracy(ml_y, ml_p_cal)},
            "totals_cal": {"n": len(tot_y), "brier": _brier(tot_y, tot_p_cal), "accuracy": _accuracy(tot_y, tot_p_cal)},
            "puckline_cal": {"n": len(pl_y), "brier": _brier(pl_y, pl_p_cal), "accuracy": _accuracy(pl_y, pl_p_cal)},
        })
    out_path = PROC_DIR / f"sim_backtest_{start}_to_{end}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(res)
    print(f"Saved backtest to {out_path}")


@app.command(name="game-calibrate-sim")
def game_calibrate_sim(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    metric: str = typer.Option("brier", help="Calibration metric: 'brier' or 'logloss'"),
):
    """Fit simple temperature+bias calibrations for moneyline, totals, and puckline from simulated probabilities.

    Saves calibrations to data/processed/sim_calibration.json
    """
    from .utils.calibration import fit_temp_shift
    # Collect probabilities and outcomes by market
    ml_p: List[float] = []; ml_y: List[int] = []
    tot_p: List[float] = []; tot_y: List[int] = []
    pl_p: List[float] = []; pl_y: List[int] = []
    # Load actuals
    games_raw = load_df(RAW_DIR / "games.csv")
    games_raw["date_et"] = pd.to_datetime(games_raw["date"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    from datetime import datetime as _dt, timedelta as _td
    cur = _dt.strptime(start, "%Y-%m-%d"); end_dt = _dt.strptime(end, "%Y-%m-%d")
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        sim_path = PROC_DIR / f"simulations_{d}.csv"
        if not (sim_path.exists() and getattr(sim_path.stat(), "st_size", 0) > 0):
            cur += _td(days=1)
            continue
        try:
            df = pd.read_csv(sim_path)
        except Exception:
            cur += _td(days=1)
            continue
        for _, r in df.iterrows():
            try:
                sub = games_raw[(games_raw["date_et"] == str(r.get("date_et"))) & (games_raw["home"] == r.get("home")) & (games_raw["away"] == r.get("away"))]
                if sub.empty:
                    continue
                ah = int(sub.iloc[0]["home_goals"]); aa = int(sub.iloc[0]["away_goals"]); atot = ah + aa; adiff = ah - aa
                # ML
                ml_p.append(float(r.get("p_home_ml_sim", np.nan)))
                ml_y.append(1 if ah > aa else 0)
                # Totals (exclude pushes)
                line = float(r.get("total_line_used"))
                if atot != line:
                    tot_p.append(float(r.get("p_over_sim", np.nan)))
                    tot_y.append(1 if atot > line else 0)
                # Puckline
                pl_p.append(float(r.get("p_home_pl_-1.5_sim", np.nan)))
                pl_y.append(1 if adiff > 1.5 else 0)
            except Exception:
                continue
        cur += _td(days=1)
    # Fit calibrations
    ml_cal = fit_temp_shift(ml_p, ml_y, metric=("logloss" if metric=="logloss" else "brier"))
    tot_cal = fit_temp_shift(tot_p, tot_y, metric=("logloss" if metric=="logloss" else "brier"))
    pl_cal = fit_temp_shift(pl_p, pl_y, metric=("logloss" if metric=="logloss" else "brier"))
    out = {
        "moneyline": {"t": ml_cal.t, "b": ml_cal.b},
        "totals": {"t": tot_cal.t, "b": tot_cal.b},
        "puckline": {"t": pl_cal.t, "b": pl_cal.b},
        "meta": {"start": start, "end": end, "metric": metric}
    }
    out_path = PROC_DIR / "sim_calibration.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(out)
    print(f"Saved simulation calibration to {out_path}")


@app.command(name="game-calibrate-sim-per-line")
def game_calibrate_sim_per_line(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
):
    """Fit per-total-line calibrations for simulated Over/Under probabilities and save mapping.

    Output: data/processed/sim_calibration_per_line.json with {totals: {line: {t,b}}}
    """
    # Load actuals
    games_raw = load_df(RAW_DIR / "games.csv")
    games_raw["date_et"] = pd.to_datetime(games_raw["date"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    from datetime import datetime as _dt, timedelta as _td
    cur = _dt.strptime(start, "%Y-%m-%d"); end_dt = _dt.strptime(end, "%Y-%m-%d")
    # Collect by line
    buckets: Dict[float, Dict[str, list]] = {}
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        sim_path = PROC_DIR / f"simulations_{d}.csv"
        if not (sim_path.exists() and getattr(sim_path.stat(), "st_size", 0) > 0):
            cur += _td(days=1); continue
        try:
            df = pd.read_csv(sim_path)
        except Exception:
            cur += _td(days=1); continue
        for _, r in df.iterrows():
            try:
                sub = games_raw[(games_raw["date_et"] == str(r.get("date_et"))) & (games_raw["home"] == r.get("home")) & (games_raw["away"] == r.get("away"))]
                if sub.empty: continue
                ah = int(sub.iloc[0]["home_goals"]); aa = int(sub.iloc[0]["away_goals"]); atot = ah + aa
                # prefer closing line if present
                line_val = r.get("close_total_line_used")
                if line_val is None or (isinstance(line_val, float) and np.isnan(line_val)):
                    line_val = r.get("total_line_used")
                line = float(line_val)
                if atot == line: continue
                p_over = float(r.get("p_over_sim", np.nan))
                if np.isnan(p_over): continue
                b = buckets.setdefault(line, {"p": [], "y": []})
                b["p"].append(p_over)
                b["y"].append(1 if atot > line else 0)
            except Exception:
                continue
        cur += _td(days=1)
    # Fit per-line
    from .utils.calibration import fit_temp_shift
    out_map: Dict[str, Dict[str, float]] = {}
    for line, data in buckets.items():
        try:
            cal = fit_temp_shift(data["p"], data["y"], metric="brier")
            out_map[str(line)] = {"t": cal.t, "b": cal.b}
        except Exception:
            continue
    out = {"totals": out_map, "meta": {"start": start, "end": end}}
    out_path = PROC_DIR / "sim_calibration_per_line.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(out)
    print(f"Saved per-line simulation calibration to {out_path}")


@app.command(name="game-backtest-sim-thresholds")
def game_backtest_sim_thresholds(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    thresholds: str = typer.Option("0.50,0.55,0.60,0.62,0.65,0.70", help="Comma-separated thresholds to evaluate"),
    prefer_simulations: bool = typer.Option(False, help="Prefer precomputed simulations_{date}.csv over inline predictions"),
    sim_overdispersion_k: float = typer.Option(0.0, help="Gamma-Poisson overdispersion k (0=off)"),
    sim_shared_k: float = typer.Option(0.0, help="Shared pace Gamma k (correlation; 0=off)"),
    sim_empty_net_p: float = typer.Option(0.0, help="Empty-net extra goal probability when leading by 1 (0=off)"),
    sim_empty_net_two_goal_scale: float = typer.Option(0.0, help="Scale factor for empty-net probability when leading by 2 (0=off)"),
    totals_pace_alpha: float = typer.Option(0.0, help="Strength of SOG-based pace adjustment for totals (0=off)"),
    totals_goalie_beta: float = typer.Option(0.0, help="Strength of goalie SAVES-based defensive adjustment (0=off)"),
    totals_fatigue_beta: float = typer.Option(0.0, help="Strength of fatigue (B2B) offensive reduction (0=off)"),
    totals_rolling_pace_gamma: float = typer.Option(0.0, help="Strength of rolling team goals pace adjustment (last 10; 0=off)"),
    totals_pp_gamma: float = typer.Option(0.0, help="Strength of PP offense adjustment from team PP% (0=off)"),
    totals_pk_beta: float = typer.Option(0.0, help="Strength of PK defensive adjustment from team PK% (applied to opponent; 0=off)"),
    totals_penalty_gamma: float = typer.Option(0.0, help="Strength of penalty exposure adjustment from team committed/drawn rates (0=off)"),
    totals_xg_gamma: float = typer.Option(0.0, help="Strength of expected goals (xGF/60) pace adjustment (0=off)"),
    totals_refs_gamma: float = typer.Option(0.0, help="Strength of referee penalty-rate adjustment (0=off)"),
    totals_goalie_form_gamma: float = typer.Option(0.0, help="Strength of goalie recent form adjustment (0=off)"),
):
    """Evaluate accuracy vs decision thresholds for ML, Totals, and Puckline using calibrated simulation probabilities.

    If predictions_{date}.csv exist, simulates calibrated probabilities inline; otherwise reads simulations_{date}.csv calibrated fields.
    Outputs JSON summary with accuracy and coverage per threshold.
    """
    from datetime import datetime as _dt, timedelta as _td
    thrs = [float(x) for x in thresholds.split(",") if x.strip()]
    # Load actuals
    games_raw = load_df(RAW_DIR / "games.csv")
    games_raw["date_et"] = pd.to_datetime(games_raw["date"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    # Load simulation calibration
    sim_cal_path = PROC_DIR / "sim_calibration.json"
    sim_ml_cal = sim_tot_cal = sim_pl_cal = None
    try:
        if sim_cal_path.exists() and getattr(sim_cal_path.stat(), "st_size", 0) > 0:
            obj = json.loads(sim_cal_path.read_text(encoding="utf-8"))
            from .utils.calibration import BinaryCalibration as _BC
            if obj.get("moneyline"): sim_ml_cal = _BC(float(obj["moneyline"].get("t", 1.0)), float(obj["moneyline"].get("b", 0.0)))
            if obj.get("totals"): sim_tot_cal = _BC(float(obj["totals"].get("t", 1.0)), float(obj["totals"].get("b", 0.0)))
            if obj.get("puckline"): sim_pl_cal = _BC(float(obj["puckline"].get("t", 1.0)), float(obj["puckline"].get("b", 0.0)))
    except Exception:
        pass

    from .models.simulator import simulate_from_period_lambdas, simulate_from_totals_diff, SimConfig
    sim_cfg = SimConfig(
        n_sims=8000,
        random_state=123,
        overdispersion_k=(sim_overdispersion_k if sim_overdispersion_k and sim_overdispersion_k > 0 else None),
        shared_k=(sim_shared_k if sim_shared_k and sim_shared_k > 0 else None),
        empty_net_p=(sim_empty_net_p if sim_empty_net_p and sim_empty_net_p > 0 else None),
        empty_net_two_goal_scale=(sim_empty_net_two_goal_scale if sim_empty_net_two_goal_scale and sim_empty_net_two_goal_scale > 0 else None),
    )  # moderate samples for speed

    # Accumulators per threshold
    acc_ml = {t: {"correct": 0, "total": 0} for t in thrs}
    acc_tot = {t: {"correct": 0, "total": 0} for t in thrs}
    acc_pl = {t: {"correct": 0, "total": 0} for t in thrs}

    cur = _dt.strptime(start, "%Y-%m-%d"); end_dt = _dt.strptime(end, "%Y-%m-%d")
    # Cache props projections by date for totals feature adjustments
    props_cache: Dict[str, Optional[pd.DataFrame]] = {}
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        # choose source
        use_df = None; use_mode = None
        pred_path = PROC_DIR / f"predictions_{d}.csv"
        if (not prefer_simulations) and pred_path.exists() and getattr(pred_path.stat(), "st_size", 0) > 0:
            try:
                use_df = pd.read_csv(pred_path); use_mode = 'pred'
            except Exception:
                use_df = None
        if use_df is None:
            sim_path = PROC_DIR / f"simulations_{d}.csv"
            if sim_path.exists() and getattr(sim_path.stat(), "st_size", 0) > 0:
                try:
                    use_df = pd.read_csv(sim_path); use_mode = 'sim'
                except Exception:
                    use_df = None
        if use_df is None:
            cur += _td(days=1); continue

        # Optional team special teams for this date
        team_st: Optional[Dict[str, Dict[str, float]]] = None
        try:
            if totals_pp_gamma > 0.0 or totals_pk_beta > 0.0:
                from .data.team_stats import load_team_special_teams
                team_st = load_team_special_teams(d)
        except Exception:
            team_st = None
        # Optional team penalty rates for this date
        team_pr: Optional[Dict[str, Dict[str, float]]] = None
        try:
            if totals_penalty_gamma > 0.0:
                from .data.penalty_rates import load_team_penalty_rates
                team_pr = load_team_penalty_rates(d)
        except Exception:
            team_pr = None
        # Optional team expected goals rates for this date
        team_xg: Optional[Dict[str, Dict[str, float]]] = None
        try:
            if totals_xg_gamma > 0.0:
                from .data.team_xg import load_team_xg
                team_xg = load_team_xg(d)
        except Exception:
            team_xg = None

        for _, r in use_df.iterrows():
            # actuals
            try:
                sub = games_raw[(games_raw["date_et"] == str(r.get("date_et"))) & (games_raw["home"] == r.get("home")) & (games_raw["away"] == r.get("away"))]
                if sub.empty: continue
                ah = int(sub.iloc[0]["home_goals"]); aa = int(sub.iloc[0]["away_goals"]); atot = ah + aa; adiff = ah - aa
                line_val = r.get("close_total_line_used")
                if line_val is None or (isinstance(line_val, float) and np.isnan(line_val)):
                    line_val = r.get("total_line_used")
                line = float(line_val)
            except Exception:
                continue

            # calibrated probabilities
            if use_mode == 'pred':
                try:
                    if pd.notna(r.get("period1_home_proj")) and pd.notna(r.get("period1_away_proj")) and pd.notna(r.get("period2_home_proj")) and pd.notna(r.get("period2_away_proj")) and pd.notna(r.get("period3_home_proj")) and pd.notna(r.get("period3_away_proj")):
                        # Optionally adjust period lambdas by props-derived + fatigue multipliers
                        h_periods = [float(r.get("period1_home_proj")), float(r.get("period2_home_proj")), float(r.get("period3_home_proj"))]
                        a_periods = [float(r.get("period1_away_proj")), float(r.get("period2_away_proj")), float(r.get("period3_away_proj"))]
                        fat_h = fat_a = 1.0
                        roll_h = roll_a = 1.0
                        try:
                            if totals_fatigue_beta > 0.0:
                                d_et = pd.to_datetime(r.get("date_et"), utc=True).tz_convert("America/New_York").normalize() if pd.notna(r.get("date_et")) else None
                                home = str(r.get("home")); away = str(r.get("away"))
                                if d_et is not None:
                                    prev_h = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == home) | (games_raw["away"] == home))]
                                    prev_a = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == away) | (games_raw["away"] == away))]
                                    prev_h_dt = pd.to_datetime(prev_h["date_et"].max()) if not prev_h.empty else None
                                    prev_a_dt = pd.to_datetime(prev_a["date_et"].max()) if not prev_a.empty else None
                                    if prev_h_dt is not None and int((d_et - prev_h_dt.normalize()).days) == 1:
                                        fat_h = float(np.clip(1.0 - totals_fatigue_beta, 0.8, 1.0))
                                    if prev_a_dt is not None and int((d_et - prev_a_dt.normalize()).days) == 1:
                                        fat_a = float(np.clip(1.0 - totals_fatigue_beta, 0.8, 1.0))
                        except Exception:
                            pass
                        # Rolling pace multipliers
                        try:
                            if totals_rolling_pace_gamma > 0.0:
                                d_et = pd.to_datetime(r.get("date_et"), utc=True).tz_convert("America/New_York").normalize() if pd.notna(r.get("date_et")) else None
                                base_per_team = 3.05
                                if d_et is not None:
                                    home = str(r.get("home")); away = str(r.get("away"))
                                    sub_h = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == home) | (games_raw["away"] == home))].copy()
                                    sub_h.sort_values("date_et", inplace=True)
                                    if not sub_h.empty:
                                        sub_h["team_goals"] = np.where(sub_h["home"] == home, sub_h["home_goals"], sub_h["away_goals"]) 
                                        last_h = sub_h.tail(10)
                                        gpg_h = float(last_h["team_goals"].mean()) if len(last_h) > 0 else None
                                        if gpg_h and gpg_h > 0.0:
                                            roll_h = float(np.clip(1.0 + totals_rolling_pace_gamma * ((gpg_h - base_per_team) / base_per_team), 0.8, 1.2))
                                    sub_a = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == away) | (games_raw["away"] == away))].copy()
                                    sub_a.sort_values("date_et", inplace=True)
                                    if not sub_a.empty:
                                        sub_a["team_goals"] = np.where(sub_a["home"] == away, sub_a["home_goals"], sub_a["away_goals"]) 
                                        last_a = sub_a.tail(10)
                                        gpg_a = float(last_a["team_goals"].mean()) if len(last_a) > 0 else None
                                        if gpg_a and gpg_a > 0.0:
                                            roll_a = float(np.clip(1.0 + totals_rolling_pace_gamma * ((gpg_a - base_per_team) / base_per_team), 0.8, 1.2))
                        except Exception:
                            pass
                        try:
                            # Load props projections for date if needed
                            props_df = None
                            if totals_pace_alpha > 0.0 or totals_goalie_beta > 0.0:
                                if d not in props_cache:
                                    ppath = PROC_DIR / f"props_projections_all_{d}.csv"
                                    if ppath.exists() and getattr(ppath.stat(), "st_size", 0) > 0:
                                        try:
                                            props_cache[d] = pd.read_csv(ppath)
                                        except Exception:
                                            props_cache[d] = None
                                    else:
                                        props_cache[d] = None
                                props_df = props_cache.get(d)
                            mcol = "market"; tcol = "team"; lcol = "lambda"
                            home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                            # Pace and goalie defensive multipliers (only if props available)
                            sog_baseline = float(props_df[props_df[mcol] == "SOG"][lcol].mean()) if (props_df is not None and (lcol in props_df.columns)) else None
                            saves_baseline = float(props_df[props_df[mcol] == "SAVES"][lcol].mean()) if (props_df is not None and (lcol in props_df.columns)) else None
                            home_sog = float(props_df[(props_df[mcol] == "SOG") & (props_df[tcol].str.upper() == home_abbr)][lcol].sum()) if (props_df is not None and (lcol in props_df.columns)) else None
                            away_sog = float(props_df[(props_df[mcol] == "SOG") & (props_df[tcol].str.upper() == away_abbr)][lcol].sum()) if (props_df is not None and (lcol in props_df.columns)) else None
                            h_g = (props_df[(props_df[mcol] == "SAVES") & (props_df[tcol].str.upper() == home_abbr)][lcol] if (props_df is not None and (lcol in props_df.columns)) else None)
                            a_g = (props_df[(props_df[mcol] == "SAVES") & (props_df[tcol].str.upper() == away_abbr)][lcol] if (props_df is not None and (lcol in props_df.columns)) else None)
                            home_goalie_saves = float(h_g.max()) if (isinstance(h_g, pd.Series) and len(h_g)) else None
                            away_goalie_saves = float(a_g.max()) if (isinstance(a_g, pd.Series) and len(a_g)) else None
                            pace_mult_home = pace_mult_away = 1.0
                            goalie_def_home = goalie_def_away = 1.0
                            if (props_df is not None) and totals_pace_alpha > 0.0 and sog_baseline and sog_baseline > 0.0 and home_sog and away_sog:
                                pace_mult_home = float(np.clip(1.0 + totals_pace_alpha * ((home_sog - sog_baseline) / sog_baseline), 0.7, 1.3))
                                pace_mult_away = float(np.clip(1.0 + totals_pace_alpha * ((away_sog - sog_baseline) / sog_baseline), 0.7, 1.3))
                            if (props_df is not None) and totals_goalie_beta > 0.0 and saves_baseline and saves_baseline > 0.0:
                                if home_goalie_saves:
                                    goalie_def_home = float(np.clip(1.0 - totals_goalie_beta * ((home_goalie_saves - saves_baseline) / saves_baseline), 0.7, 1.2))
                                if away_goalie_saves:
                                    goalie_def_away = float(np.clip(1.0 - totals_goalie_beta * ((away_goalie_saves - saves_baseline) / saves_baseline), 0.7, 1.2))
                            # Special teams multipliers
                            pp_mult_home = pp_mult_away = 1.0
                            pk_mult_home_def = pk_mult_away_def = 1.0
                            try:
                                if team_st is not None and (totals_pp_gamma > 0.0 or totals_pk_beta > 0.0):
                                    h_abbr = home_abbr; a_abbr = away_abbr
                                    vals = list(team_st.values())
                                    pp_base = float(np.mean([v.get("pp_pct", 0.2) for v in vals])) if vals else 0.2
                                    pk_base = float(np.mean([v.get("pk_pct", 0.8) for v in vals])) if vals else 0.8
                                    th = team_st.get(h_abbr) or {}
                                    ta = team_st.get(a_abbr) or {}
                                    h_pp = float(th.get("pp_pct", pp_base)); a_pp = float(ta.get("pp_pct", pp_base))
                                    h_pk = float(th.get("pk_pct", pk_base)); a_pk = float(ta.get("pk_pct", pk_base))
                                    if totals_pp_gamma > 0.0 and pp_base > 0.0:
                                        pp_mult_home = float(np.clip(1.0 + totals_pp_gamma * ((h_pp - pp_base) / pp_base), 0.85, 1.20))
                                        pp_mult_away = float(np.clip(1.0 + totals_pp_gamma * ((a_pp - pp_base) / pp_base), 0.85, 1.20))
                                    if totals_pk_beta > 0.0 and pk_base > 0.0:
                                        pk_mult_home_def = float(np.clip(1.0 - totals_pk_beta * ((h_pk - pk_base) / pk_base), 0.80, 1.15))
                                        pk_mult_away_def = float(np.clip(1.0 - totals_pk_beta * ((a_pk - pk_base) / pk_base), 0.80, 1.15))
                            except Exception:
                                pass
                            # Penalty exposure multipliers
                            pen_mult_home = pen_mult_away = 1.0
                            try:
                                if team_pr is not None and totals_penalty_gamma > 0.0:
                                    vals = list(team_pr.values())
                                    c_base = float(np.nanmean([v.get("committed_per60") for v in vals])) if vals else None
                                    d_base = float(np.nanmean([v.get("drawn_per60") for v in vals])) if vals else None
                                    base_exp = (c_base or 0.0) + (d_base or 0.0)
                                    th = team_pr.get(home_abbr) or {}
                                    ta = team_pr.get(away_abbr) or {}
                                    exp_h = float((th.get("drawn_per60") or 0.0) + (ta.get("committed_per60") or 0.0))
                                    exp_a = float((ta.get("drawn_per60") or 0.0) + (th.get("committed_per60") or 0.0))
                                    if base_exp > 0.0:
                                        pen_mult_home = float(np.clip(1.0 + totals_penalty_gamma * ((exp_h - base_exp) / base_exp), 0.85, 1.20))
                                        pen_mult_away = float(np.clip(1.0 + totals_penalty_gamma * ((exp_a - base_exp) / base_exp), 0.85, 1.20))
                            except Exception:
                                pass
                            # Referee multiplier
                            refs_mult = 1.0
                            try:
                                if ref_map_base is not None and totals_refs_gamma > 0.0:
                                    per_game_map, base_rate = ref_map_base
                                    key = f"{home_abbr}|{away_abbr}"
                                    rate = per_game_map.get(key)
                                    if rate and base_rate and base_rate > 0.0:
                                        refs_mult = float(np.clip(1.0 + totals_refs_gamma * ((float(rate) - float(base_rate)) / float(base_rate)), 0.9, 1.2))
                            except Exception:
                                pass
                            # Goalie recent form defensive scaling
                            goalie_form_def_home = goalie_form_def_away = 1.0
                            try:
                                if goalie_form_map is not None and totals_goalie_form_gamma > 0.0:
                                    vals = [v for v in goalie_form_map.values() if v is not None]
                                    base_form = float(np.nanmean(vals)) if vals else None
                                    gh = goalie_form_map.get(home_abbr); ga = goalie_form_map.get(away_abbr)
                                    if base_form and base_form != 0.0:
                                        if gh is not None:
                                            goalie_form_def_home = float(np.clip(1.0 - totals_goalie_form_gamma * ((float(gh) - base_form) / abs(base_form)), 0.7, 1.3))
                                        if ga is not None:
                                            goalie_form_def_away = float(np.clip(1.0 - totals_goalie_form_gamma * ((float(ga) - base_form) / abs(base_form)), 0.7, 1.3))
                            except Exception:
                                pass
                            # xG pacing multipliers
                            xg_mult_home = xg_mult_away = 1.0
                            try:
                                if team_xg is not None and totals_xg_gamma > 0.0:
                                    vals = list(team_xg.values())
                                    base_xgf = float(np.nanmean([v.get("xgf60") for v in vals])) if vals else None
                                    h_xgf = float((team_xg.get(home_abbr) or {}).get("xgf60") or np.nan)
                                    a_xgf = float((team_xg.get(away_abbr) or {}).get("xgf60") or np.nan)
                                    if base_xgf and base_xgf > 0.0 and pd.notna(h_xgf) and pd.notna(a_xgf):
                                        xg_mult_home = float(np.clip(1.0 + totals_xg_gamma * ((h_xgf - base_xgf) / base_xgf), 0.85, 1.20))
                                        xg_mult_away = float(np.clip(1.0 + totals_xg_gamma * ((a_xgf - base_xgf) / base_xgf), 0.85, 1.20))
                            except Exception:
                                pass
                            h_periods = [max(0.0, float(x)) * pace_mult_home * refs_mult * goalie_def_away * goalie_form_def_away * fat_h * roll_h * pp_mult_home * pk_mult_away_def * pen_mult_home * xg_mult_home for x in h_periods]
                            a_periods = [max(0.0, float(x)) * pace_mult_away * refs_mult * goalie_def_home * goalie_form_def_home * fat_a * roll_a * pp_mult_away * pk_mult_home_def * pen_mult_away * xg_mult_away for x in a_periods]
                        except Exception:
                            pass
                        sim = simulate_from_period_lambdas(
                            home_periods=h_periods,
                            away_periods=a_periods,
                            total_line=line,
                            puck_line=-1.5,
                            cfg=sim_cfg,
                        )
                    else:
                        # Apply optional totals feature adjustments
                        adj_total = float(r.get("model_total")); adj_diff = float(r.get("model_spread"))
                        try:
                            props_df = None
                            if totals_pace_alpha > 0.0 or totals_goalie_beta > 0.0:
                                if d not in props_cache:
                                    ppath = PROC_DIR / f"props_projections_all_{d}.csv"
                                    if ppath.exists() and getattr(ppath.stat(), "st_size", 0) > 0:
                                        try:
                                            props_cache[d] = pd.read_csv(ppath)
                                        except Exception:
                                            props_cache[d] = None
                                    else:
                                        props_cache[d] = None
                                props_df = props_cache.get(d)
                            if props_df is not None:
                                mcol = "market"; tcol = "team"; lcol = "lambda"
                                home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                sog_baseline = float(props_df[props_df[mcol] == "SOG"][lcol].mean()) if lcol in props_df.columns else None
                                saves_baseline = float(props_df[props_df[mcol] == "SAVES"][lcol].mean()) if lcol in props_df.columns else None
                                home_sog = float(props_df[(props_df[mcol] == "SOG") & (props_df[tcol].str.upper() == home_abbr)][lcol].sum()) if lcol in props_df.columns else None
                                away_sog = float(props_df[(props_df[mcol] == "SOG") & (props_df[tcol].str.upper() == away_abbr)][lcol].sum()) if lcol in props_df.columns else None
                                h_g = props_df[(props_df[mcol] == "SAVES") & (props_df[tcol].str.upper() == home_abbr)][lcol] if lcol in props_df.columns else None
                                a_g = props_df[(props_df[mcol] == "SAVES") & (props_df[tcol].str.upper() == away_abbr)][lcol] if lcol in props_df.columns else None
                                home_goalie_saves = float(h_g.max()) if (isinstance(h_g, pd.Series) and len(h_g)) else None
                                away_goalie_saves = float(a_g.max()) if (isinstance(a_g, pd.Series) and len(a_g)) else None
                                pace_mult_home = pace_mult_away = 1.0
                                goalie_def_home = goalie_def_away = 1.0
                                if totals_pace_alpha > 0.0 and sog_baseline and sog_baseline > 0.0 and home_sog and away_sog:
                                    pace_mult_home = float(np.clip(1.0 + totals_pace_alpha * ((home_sog - sog_baseline) / sog_baseline), 0.7, 1.3))
                                    pace_mult_away = float(np.clip(1.0 + totals_pace_alpha * ((away_sog - sog_baseline) / sog_baseline), 0.7, 1.3))
                                if totals_goalie_beta > 0.0 and saves_baseline and saves_baseline > 0.0:
                                    if home_goalie_saves:
                                        goalie_def_home = float(np.clip(1.0 - totals_goalie_beta * ((home_goalie_saves - saves_baseline) / saves_baseline), 0.7, 1.2))
                                    if away_goalie_saves:
                                        goalie_def_away = float(np.clip(1.0 - totals_goalie_beta * ((away_goalie_saves - saves_baseline) / saves_baseline), 0.7, 1.2))
                                from .models.simulator import derive_team_lambdas
                                lh, la = derive_team_lambdas(adj_total, adj_diff)
                                # Fatigue multipliers
                                fat_h = fat_a = 1.0
                                roll_h = roll_a = 1.0
                                # Special teams multipliers
                                pp_mult_home = pp_mult_away = 1.0
                                pk_mult_home_def = pk_mult_away_def = 1.0
                                try:
                                    if team_st is not None and (totals_pp_gamma > 0.0 or totals_pk_beta > 0.0):
                                        vals = list(team_st.values())
                                        pp_base = float(np.mean([v.get("pp_pct", 0.2) for v in vals])) if vals else 0.2
                                        pk_base = float(np.mean([v.get("pk_pct", 0.8) for v in vals])) if vals else 0.8
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        th = team_st.get(home_abbr) or {}
                                        ta = team_st.get(away_abbr) or {}
                                        h_pp = float(th.get("pp_pct", pp_base)); a_pp = float(ta.get("pp_pct", pp_base))
                                        h_pk = float(th.get("pk_pct", pk_base)); a_pk = float(ta.get("pk_pct", pk_base))
                                        if totals_pp_gamma > 0.0 and pp_base > 0.0:
                                            pp_mult_home = float(np.clip(1.0 + totals_pp_gamma * ((h_pp - pp_base) / pp_base), 0.85, 1.20))
                                            pp_mult_away = float(np.clip(1.0 + totals_pp_gamma * ((a_pp - pp_base) / pp_base), 0.85, 1.20))
                                        if totals_pk_beta > 0.0 and pk_base > 0.0:
                                            pk_mult_home_def = float(np.clip(1.0 - totals_pk_beta * ((h_pk - pk_base) / pk_base), 0.80, 1.15))
                                            pk_mult_away_def = float(np.clip(1.0 - totals_pk_beta * ((a_pk - pk_base) / pk_base), 0.80, 1.15))
                                except Exception:
                                    pass
                                try:
                                    if totals_fatigue_beta > 0.0:
                                        d_et = pd.to_datetime(r.get("date_et"), utc=True).tz_convert("America/New_York").normalize() if pd.notna(r.get("date_et")) else None
                                        home = str(r.get("home")); away = str(r.get("away"))
                                        if d_et is not None:
                                            prev_h = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == home) | (games_raw["away"] == home))]
                                            prev_a = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == away) | (games_raw["away"] == away))]
                                            prev_h_dt = pd.to_datetime(prev_h["date_et"].max()) if not prev_h.empty else None
                                            prev_a_dt = pd.to_datetime(prev_a["date_et"].max()) if not prev_a.empty else None
                                            if prev_h_dt is not None and int((d_et - prev_h_dt.normalize()).days) == 1:
                                                fat_h = float(np.clip(1.0 - totals_fatigue_beta, 0.8, 1.0))
                                            if prev_a_dt is not None and int((d_et - prev_a_dt.normalize()).days) == 1:
                                                fat_a = float(np.clip(1.0 - totals_fatigue_beta, 0.8, 1.0))
                                except Exception:
                                    pass
                                # Rolling pace multipliers
                                try:
                                    if totals_rolling_pace_gamma > 0.0:
                                        d_et = pd.to_datetime(r.get("date_et"), utc=True).tz_convert("America/New_York").normalize() if pd.notna(r.get("date_et")) else None
                                        base_per_team = 3.05
                                        if d_et is not None:
                                            home = str(r.get("home")); away = str(r.get("away"))
                                            sub_h = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == home) | (games_raw["away"] == home))].copy()
                                            sub_h.sort_values("date_et", inplace=True)
                                            if not sub_h.empty:
                                                sub_h["team_goals"] = np.where(sub_h["home"] == home, sub_h["home_goals"], sub_h["away_goals"]) 
                                                last_h = sub_h.tail(10)
                                                gpg_h = float(last_h["team_goals"].mean()) if len(last_h) > 0 else None
                                                if gpg_h and gpg_h > 0.0:
                                                    roll_h = float(np.clip(1.0 + totals_rolling_pace_gamma * ((gpg_h - base_per_team) / base_per_team), 0.8, 1.2))
                                            sub_a = games_raw[(games_raw["date_et"] < d_et) & ((games_raw["home"] == away) | (games_raw["away"] == away))].copy()
                                            sub_a.sort_values("date_et", inplace=True)
                                            if not sub_a.empty:
                                                sub_a["team_goals"] = np.where(sub_a["home"] == away, sub_a["home_goals"], sub_a["away_goals"]) 
                                                last_a = sub_a.tail(10)
                                                gpg_a = float(last_a["team_goals"].mean()) if len(last_a) > 0 else None
                                                if gpg_a and gpg_a > 0.0:
                                                    roll_a = float(np.clip(1.0 + totals_rolling_pace_gamma * ((gpg_a - base_per_team) / base_per_team), 0.8, 1.2))
                                except Exception:
                                    pass
                                lh_adj = float(np.clip(lh * pace_mult_home * goalie_def_away * fat_h * roll_h * pp_mult_home * pk_mult_away_def * (1.0), 0.05, 8.0))
                                la_adj = float(np.clip(la * pace_mult_away * goalie_def_home * fat_a * roll_a * pp_mult_away * pk_mult_home_def * (1.0), 0.05, 8.0))
                                # Referee multiplier for totals/diff path
                                try:
                                    if ref_map_base is not None and totals_refs_gamma > 0.0:
                                        per_game_map, base_rate = ref_map_base
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        key = f"{home_abbr}|{away_abbr}"
                                        rate = per_game_map.get(key)
                                        if rate and base_rate and base_rate > 0.0:
                                            refs_mult = float(np.clip(1.0 + totals_refs_gamma * ((float(rate) - float(base_rate)) / float(base_rate)), 0.9, 1.2))
                                            lh_adj = float(np.clip(lh_adj * refs_mult, 0.05, 8.0))
                                            la_adj = float(np.clip(la_adj * refs_mult, 0.05, 8.0))
                                except Exception:
                                    pass
                                # Goalie recent form defensive scaling for totals/diff
                                try:
                                    if goalie_form_map is not None and totals_goalie_form_gamma > 0.0:
                                        vals = [v for v in goalie_form_map.values() if v is not None]
                                        base_form = float(np.nanmean(vals)) if vals else None
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        gh = goalie_form_map.get(home_abbr); ga = goalie_form_map.get(away_abbr)
                                        if base_form and base_form != 0.0:
                                            if gh is not None:
                                                # Away team shoots vs home goalie (home goalie form reduces away scoring)
                                                la_adj = float(np.clip(la_adj * (1.0 - totals_goalie_form_gamma * ((float(gh) - base_form) / abs(base_form))), 0.05, 8.0))
                                            if ga is not None:
                                                # Home team shoots vs away goalie (away goalie form reduces home scoring)
                                                lh_adj = float(np.clip(lh_adj * (1.0 - totals_goalie_form_gamma * ((float(ga) - base_form) / abs(base_form))), 0.05, 8.0))
                                except Exception:
                                    pass
                                # Penalty exposure multipliers for totals/diff path
                                try:
                                    if team_pr is not None and totals_penalty_gamma > 0.0:
                                        vals = list(team_pr.values())
                                        c_base = float(np.nanmean([v.get("committed_per60") for v in vals])) if vals else None
                                        d_base = float(np.nanmean([v.get("drawn_per60") for v in vals])) if vals else None
                                        base_exp = (c_base or 0.0) + (d_base or 0.0)
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        th = team_pr.get(home_abbr) or {}
                                        ta = team_pr.get(away_abbr) or {}
                                        exp_h = float((th.get("drawn_per60") or 0.0) + (ta.get("committed_per60") or 0.0))
                                        exp_a = float((ta.get("drawn_per60") or 0.0) + (th.get("committed_per60") or 0.0))
                                        if base_exp > 0.0:
                                            pen_mult_home = float(np.clip(1.0 + totals_penalty_gamma * ((exp_h - base_exp) / base_exp), 0.85, 1.20))
                                            pen_mult_away = float(np.clip(1.0 + totals_penalty_gamma * ((exp_a - base_exp) / base_exp), 0.85, 1.20))
                                            lh_adj = float(np.clip(lh_adj * pen_mult_home, 0.05, 8.0))
                                            la_adj = float(np.clip(la_adj * pen_mult_away, 0.05, 8.0))
                                except Exception:
                                    pass
                                # xG pacing multipliers for totals/diff path
                                try:
                                    if team_xg is not None and totals_xg_gamma > 0.0:
                                        vals = list(team_xg.values())
                                        base_xgf = float(np.nanmean([v.get("xgf60") for v in vals])) if vals else None
                                        home_abbr = str(r.get("home")).upper(); away_abbr = str(r.get("away")).upper()
                                        h_xgf = float((team_xg.get(home_abbr) or {}).get("xgf60") or np.nan)
                                        a_xgf = float((team_xg.get(away_abbr) or {}).get("xgf60") or np.nan)
                                        if base_xgf and base_xgf > 0.0 and pd.notna(h_xgf) and pd.notna(a_xgf):
                                            xg_mult_home = float(np.clip(1.0 + totals_xg_gamma * ((h_xgf - base_xgf) / base_xgf), 0.85, 1.20))
                                            xg_mult_away = float(np.clip(1.0 + totals_xg_gamma * ((a_xgf - base_xgf) / base_xgf), 0.85, 1.20))
                                            lh_adj = float(np.clip(lh_adj * xg_mult_home, 0.05, 8.0))
                                            la_adj = float(np.clip(la_adj * xg_mult_away, 0.05, 8.0))
                                except Exception:
                                    pass
                                adj_total = lh_adj + la_adj
                                adj_diff = lh_adj - la_adj
                        except Exception:
                            pass
                        sim = simulate_from_totals_diff(
                            total_mean=adj_total,
                            diff_mean=adj_diff,
                            total_line=line,
                            puck_line=-1.5,
                            cfg=sim_cfg,
                        )
                except Exception:
                    continue
                p_ml = float(sim.get("home_ml", np.nan))
                p_over = float(sim.get("over", np.nan))
                p_pl = float(sim.get("home_puckline_-1.5", np.nan))
            else:
                p_ml = float(r.get("p_home_ml_sim", np.nan))
                p_over = float(r.get("p_over_sim", np.nan))
                p_pl = float(r.get("p_home_pl_-1.5_sim", np.nan))

            # apply calibration
            try:
                if sim_ml_cal is not None: p_ml = float(sim_ml_cal.apply(np.array([p_ml]))[0])
                # apply per-line totals calibration if available
                # load per-line cal map locally
                tot_line_map = {}
                try:
                    obj = json.loads((PROC_DIR / "sim_calibration_per_line.json").read_text(encoding="utf-8"))
                    tot_line_map = obj.get("totals", {}) or {}
                except Exception:
                    tot_line_map = {}
                if str(line) in tot_line_map:
                    from .utils.calibration import BinaryCalibration
                    c = tot_line_map[str(line)]
                    cal = BinaryCalibration(float(c.get("t", 1.0)), float(c.get("b", 0.0)))
                    p_over = float(cal.apply(np.array([p_over]))[0])
                elif sim_tot_cal is not None:
                    p_over = float(sim_tot_cal.apply(np.array([p_over]))[0])
                if sim_pl_cal is not None: p_pl = float(sim_pl_cal.apply(np.array([p_pl]))[0])
            except Exception:
                pass

            # evaluate thresholds
            for t in thrs:
                # ML: pick side only if confident
                if not np.isnan(p_ml):
                    if p_ml >= t or (1 - p_ml) >= t:
                        acc_ml[t]["total"] += 1
                        pick_home = p_ml >= t
                        correct = (ah > aa) if pick_home else (aa > ah)
                        acc_ml[t]["correct"] += 1 if correct else 0
                # Totals: choose side if confident (avoid pushes)
                if not np.isnan(p_over) and atot != line:
                    p_under = 1.0 - p_over
                    if p_over >= t or p_under >= t:
                        acc_tot[t]["total"] += 1
                        pick_over = p_over >= t
                        correct = (atot > line) if pick_over else (atot < line)
                        acc_tot[t]["correct"] += 1 if correct else 0
                # Puckline: home -1.5 or away +1.5 if confident
                if not np.isnan(p_pl):
                    p_away_pl = 1.0 - p_pl
                    if p_pl >= t or p_away_pl >= t:
                        acc_pl[t]["total"] += 1
                        pick_home_pl = p_pl >= t
                        correct = (adiff > 1.5) if pick_home_pl else (adiff < 1.5)
                        acc_pl[t]["correct"] += 1 if correct else 0
        cur += _td(days=1)

    # summarize
    def _summ(d):
        return {"n": d["total"], "acc": (float(d["correct"]) / max(1, d["total"]))}
    res = {
        "range": {"start": start, "end": end},
        "thresholds": thrs,
        "moneyline": {str(t): _summ(acc_ml[t]) for t in thrs},
        "totals": {str(t): _summ(acc_tot[t]) for t in thrs},
        "puckline": {str(t): _summ(acc_pl[t]) for t in thrs},
    }
    out_path = PROC_DIR / f"sim_thresholds_{start}_to_{end}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(res)
    print(f"Saved thresholds summary to {out_path}")


@app.command(name="game-recommendations-sim")
def game_recommendations_sim(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    include_ml: bool = typer.Option(True, help="Include moneyline picks"),
    include_totals: bool = typer.Option(False, help="Include totals picks (over/under)"),
    include_puckline: bool = typer.Option(True, help="Include puckline picks"),
    ml_thr: float = typer.Option(0.65, help="Threshold for ML confidence"),
    tot_thr: float = typer.Option(0.55, help="Threshold for totals confidence"),
    pl_thr: float = typer.Option(0.60, help="Threshold for puckline confidence"),
):
    """Emit threshold-gated recommendations from simulations_{date}.csv using calibrated probabilities if present.

    Writes data/processed/sim_picks_{date}.csv
    """
    path = PROC_DIR / f"simulations_{date}.csv"
    if not (path.exists() and getattr(path.stat(), "st_size", 0) > 0):
        print(f"Missing {path}; run game-simulate first.")
        raise typer.Exit(code=1)
    df = pd.read_csv(path)
    out_rows = []
    def _safe_float(v, dflt=np.nan):
        try:
            return float(v)
        except Exception:
            return dflt
    for _, r in df.iterrows():
        home = str(r.get("home")); away = str(r.get("away"))
        line_val = r.get("close_total_line_used")
        if line_val is None or (isinstance(line_val, float) and np.isnan(line_val)):
            line_val = r.get("total_line_used")
        total_line = _safe_float(line_val)
        # ML
        if include_ml:
            p_home = _safe_float(r.get("p_home_ml_sim_cal", r.get("p_home_ml_sim")))
            if not np.isnan(p_home):
                if p_home >= ml_thr:
                    out_rows.append({"date": date, "home": home, "away": away, "market": "ML", "side": "HOME", "line": None, "prob": p_home, "threshold": ml_thr})
                elif (1 - p_home) >= ml_thr:
                    out_rows.append({"date": date, "home": home, "away": away, "market": "ML", "side": "AWAY", "line": None, "prob": 1 - p_home, "threshold": ml_thr})
        # Totals
        if include_totals and total_line is not None and not np.isnan(total_line):
            p_over = _safe_float(r.get("p_over_sim_cal", r.get("p_over_sim")))
            if not np.isnan(p_over):
                p_under = 1.0 - p_over
                if p_over >= tot_thr:
                    out_rows.append({"date": date, "home": home, "away": away, "market": "TOTALS", "side": "OVER", "line": total_line, "prob": p_over, "threshold": tot_thr})
                elif p_under >= tot_thr:
                    out_rows.append({"date": date, "home": home, "away": away, "market": "TOTALS", "side": "UNDER", "line": total_line, "prob": p_under, "threshold": tot_thr})
        # Puckline
        if include_puckline:
            p_home_pl = _safe_float(r.get("p_home_pl_-1.5_sim_cal", r.get("p_home_pl_-1.5_sim")))
            if not np.isnan(p_home_pl):
                p_away_pl = 1.0 - p_home_pl
                if p_home_pl >= pl_thr:
                    out_rows.append({"date": date, "home": home, "away": away, "market": "PUCKLINE", "side": "HOME -1.5", "line": -1.5, "prob": p_home_pl, "threshold": pl_thr})
                elif p_away_pl >= pl_thr:
                    out_rows.append({"date": date, "home": home, "away": away, "market": "PUCKLINE", "side": "AWAY +1.5", "line": +1.5, "prob": p_away_pl, "threshold": pl_thr})

    out_df = pd.DataFrame(out_rows)
    out_path = PROC_DIR / f"sim_picks_{date}.csv"
    save_df(out_df, out_path)
    print(out_df.head())
    print(f"Saved recommendations to {out_path}")


@app.command(name="game-backtest-sim-picks-range")
def game_backtest_sim_picks_range(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
):
    """Backtest sim-backed picks written by game-recommendations-sim across a date range.

    Computes coverage and accuracy per market (ML, Totals, Puckline) using outcomes from data/raw/games.csv.

    Writes data/processed/sim_picks_backtest_{start}_to_{end}.json
    """
    from datetime import datetime as _dt, timedelta as _td
    games_raw = load_df(RAW_DIR / "games.csv")
    # Normalize ET date for easy joins
    try:
        games_raw["date_et"] = pd.to_datetime(games_raw["date"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    # Accumulators
    acc = {
        "ML": {"correct": 0, "total": 0},
        "Totals": {"correct": 0, "total": 0},
        "Puckline": {"correct": 0, "total": 0},
    }
    cur = _dt.strptime(start, "%Y-%m-%d"); end_dt = _dt.strptime(end, "%Y-%m-%d")
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        path = PROC_DIR / f"sim_picks_{d}.csv"
        if not (path.exists() and getattr(path.stat(), "st_size", 0) > 0):
            cur += _td(days=1)
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            market = str(r.get("market")).upper()
            home = str(r.get("home")); away = str(r.get("away"))
            # Join to actuals
            sub = games_raw[(games_raw["date_et"] == d) & (games_raw["home"] == home) & (games_raw["away"] == away)]
            if sub.empty:
                continue
            ah = sub.iloc[0]["home_goals"]; aa = sub.iloc[0]["away_goals"]
            try:
                ah = int(ah); aa = int(aa)
            except Exception:
                continue
            # Evaluate by market
            if market == "ML":
                side = str(r.get("side"))
                pick_home = side.upper() == "HOME"
                correct = (ah > aa) if pick_home else (aa > ah)
                acc["ML"]["total"] += 1
                acc["ML"]["correct"] += 1 if correct else 0
            elif market == "TOTALS":
                side = str(r.get("side"))  # OVER or UNDER
                line_val = r.get("line")
                try:
                    line = float(line_val)
                except Exception:
                    line = None
                total = ah + aa
                if line is not None and total != line:
                    pick_over = side.upper() == "OVER"
                    correct = (total > line) if pick_over else (total < line)
                    acc["Totals"]["total"] += 1
                    acc["Totals"]["correct"] += 1 if correct else 0
            elif market == "PUCKLINE":
                side = str(r.get("side"))  # HOME -1.5 or AWAY +1.5
                pick_home_minus = side.upper().startswith("HOME")
                diff = ah - aa
                correct = (diff > 1.5) if pick_home_minus else (diff < 1.5)
                acc["Puckline"]["total"] += 1
                acc["Puckline"]["correct"] += 1 if correct else 0
        cur += _td(days=1)

    def _summ(d):
        return {"n": int(d["total"]), "acc": (float(d["correct"]) / max(1, int(d["total"])))}
    res = {
        "range": {"start": start, "end": end},
        "ML": _summ(acc["ML"]),
        "Totals": _summ(acc["Totals"]),
        "Puckline": _summ(acc["Puckline"]),
    }
    out_path = PROC_DIR / f"sim_picks_backtest_{start}_to_{end}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(res)
    print(f"Saved sim picks backtest to {out_path}")


@app.command(name="game-backtest-sim-picks-roi-range")
def game_backtest_sim_picks_roi_range(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    stake: float = typer.Option(100.0, help="Stake per pick in dollars"),
    ml_default_odds: float = typer.Option(-115.0, help="Default American odds for ML when price missing"),
    totals_default_odds: float = typer.Option(-110.0, help="Default American odds for Totals when price missing"),
    puckline_default_odds: float = typer.Option(-110.0, help="Default American odds for Puckline when price missing"),
):
    """Compute baseline ROI for sim-backed picks across a date range.

    Uses outcomes from data/raw/games.csv. If prices are not present in sim picks, applies default American odds per market.

    Writes data/processed/sim_picks_roi_{start}_to_{end}.json
    """
    from datetime import datetime as _dt, timedelta as _td
    games_raw = load_df(RAW_DIR / "games.csv")
    try:
        games_raw["date_et"] = pd.to_datetime(games_raw["date"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    def _dec_from_american(a: float) -> float:
        a = float(a)
        return (1.0 + (a / 100.0)) if a > 0 else (1.0 + (100.0 / abs(a)))
    def _profit(p_win: bool, american: float, stake_amt: float) -> float:
        dec = _dec_from_american(american)
        return stake_amt * (dec - 1.0) if p_win else -stake_amt
    roi = {
        "ML": {"profit": 0.0, "count": 0},
        "Totals": {"profit": 0.0, "count": 0},
        "Puckline": {"profit": 0.0, "count": 0},
        "All": {"profit": 0.0, "count": 0},
    }
    cur = _dt.strptime(start, "%Y-%m-%d"); end_dt = _dt.strptime(end, "%Y-%m-%d")
    audit_rows = []
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        path = PROC_DIR / f"sim_picks_{d}.csv"
        if not (path.exists() and getattr(path.stat(), "st_size", 0) > 0):
            cur += _td(days=1)
            continue
        df = pd.read_csv(path)
        # Optional predictions for odds mapping
        pred_df = None
        try:
            ppath = PROC_DIR / f"predictions_{d}.csv"
            if ppath.exists() and getattr(ppath.stat(), "st_size", 0) > 0:
                pred_df = pd.read_csv(ppath)
        except Exception:
            pred_df = None
        for _, r in df.iterrows():
            market = str(r.get("market")).upper()
            home = str(r.get("home")); away = str(r.get("away"))
            sub = games_raw[(games_raw["date_et"] == d) & (games_raw["home"] == home) & (games_raw["away"] == away)]
            if sub.empty:
                continue
            try:
                ah = int(sub.iloc[0]["home_goals"]); aa = int(sub.iloc[0]["away_goals"])
            except Exception:
                continue
            win = False
            american = None
            # Try to map odds from predictions if available
            def _odds_from_predictions(side_key: str, totals_side: str | None = None, pl_side: str | None = None) -> float | None:
                try:
                    if pred_df is None:
                        return None
                    rowp = pred_df[(pred_df["date_et"] == d) & (pred_df["home"] == home) & (pred_df["away"] == away)]
                    if rowp is None or rowp.empty:
                        return None
                    rp = rowp.iloc[0]
                    if market == "ML":
                        # prefer close_*; fallback to base odds fields
                        val = rp.get("close_home_ml_odds") if side_key == "HOME" else rp.get("close_away_ml_odds")
                        if val is None or (isinstance(val, float) and pd.isna(val)):
                            val = rp.get("home_ml_odds") if side_key == "HOME" else rp.get("away_ml_odds")
                        v = float(val) if val is not None else None
                        return v if (v is not None and math.isfinite(v)) else None
                    elif market == "TOTALS" and totals_side is not None:
                        val = rp.get("close_over_odds") if totals_side == "OVER" else rp.get("close_under_odds")
                        if val is None or (isinstance(val, float) and pd.isna(val)):
                            val = rp.get("over_odds") if totals_side == "OVER" else rp.get("under_odds")
                        v = float(val) if val is not None else None
                        return v if (v is not None and math.isfinite(v)) else None
                    elif market == "PUCKLINE" and pl_side is not None:
                        val = rp.get("close_home_pl_-1.5_odds") if pl_side.startswith("HOME") else rp.get("close_away_pl_+1.5_odds")
                        if val is None or (isinstance(val, float) and pd.isna(val)):
                            val = rp.get("home_pl_-1.5_odds") if pl_side.startswith("HOME") else rp.get("away_pl_+1.5_odds")
                        v = float(val) if val is not None else None
                        return v if (v is not None and math.isfinite(v)) else None
                except Exception:
                    return None
                return None
            if market == "ML":
                side = str(r.get("side"))
                pick_home = side.upper() == "HOME"
                win = (ah > aa) if pick_home else (aa > ah)
                american = _odds_from_predictions("HOME" if pick_home else "AWAY") or ml_default_odds
            elif market == "TOTALS":
                side = str(r.get("side"))
                try:
                    line = float(r.get("line"))
                except Exception:
                    line = None
                total = ah + aa
                if line is None or total == line:
                    continue
                pick_over = side.upper() == "OVER"
                win = (total > line) if pick_over else (total < line)
                american = _odds_from_predictions(None, "OVER" if pick_over else "UNDER") or totals_default_odds
            elif market == "PUCKLINE":
                side = str(r.get("side"))
                pick_home_minus = side.upper().startswith("HOME")
                diff = ah - aa
                win = (diff > 1.5) if pick_home_minus else (diff < 1.5)
                american = _odds_from_predictions(None, None, "HOME -1.5" if pick_home_minus else "AWAY +1.5") or puckline_default_odds
            else:
                continue
            profit = _profit(win, american, stake)
            key = "ML" if market == "ML" else ("Totals" if market == "TOTALS" else "Puckline")
            roi[key]["profit"] += profit
            roi[key]["count"] += 1
            roi["All"]["profit"] += profit
            roi["All"]["count"] += 1
            audit_rows.append({
                "date": d,
                "home": home,
                "away": away,
                "market": market,
                "side": side,
                "line": r.get("line"),
                "prob": r.get("prob"),
                "american_odds": american,
                "win": bool(win),
                "profit": round(profit, 2),
            })
        cur += _td(days=1)

    def _summ(m):
        cnt = int(roi[m]["count"]);
        prof = float(roi[m]["profit"])
        return {"n": cnt, "profit": round(prof, 2), "roi": round(prof / max(1.0, cnt * stake), 4)}
    res = {
        "range": {"start": start, "end": end},
        "ML": _summ("ML"),
        "Totals": _summ("Totals"),
        "Puckline": _summ("Puckline"),
        "All": _summ("All"),
        "assumptions": {
            "stake": stake,
            "ml_default_odds": ml_default_odds,
            "totals_default_odds": totals_default_odds,
            "puckline_default_odds": puckline_default_odds,
        }
    }
    out_path = PROC_DIR / f"sim_picks_roi_{start}_to_{end}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(res)
    print(f"Saved sim picks ROI backtest to {out_path}")
    # Also emit CSV audit
    try:
        audit_df = pd.DataFrame(audit_rows)
        audit_csv = PROC_DIR / f"sim_picks_audit_{start}_to_{end}.csv"
        save_df(audit_df, audit_csv)
        print(f"Saved sim picks audit to {audit_csv}")
    except Exception:
        pass


@app.command(name="game-weekly-summary")
def game_weekly_summary(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
):
    """Consolidate weekly summaries: thresholds, picks accuracy, ROI, audit CSV path.

    Writes data/processed/weekly_summary_{start}_to_{end}.json
    """
    def _read_json(path: Path) -> dict:
        try:
            if path.exists() and getattr(path.stat(), "st_size", 0) > 0:
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return {}
    base = PROC_DIR
    th = _read_json(base / f"sim_thresholds_{start}_to_{end}.json")
    acc = _read_json(base / f"sim_picks_backtest_{start}_to_{end}.json")
    roi = _read_json(base / f"sim_picks_roi_{start}_to_{end}.json")
    audit_csv = str((base / f"sim_picks_audit_{start}_to_{end}.csv").resolve())
    out = {
        "range": {"start": start, "end": end},
        "thresholds": th or None,
        "picks_accuracy": acc or None,
        "roi": roi or None,
        "audit_csv": audit_csv,
    }
    out_path = PROC_DIR / f"weekly_summary_{start}_to_{end}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(out)
    print(f"Saved weekly summary to {out_path}")


@app.command(name="game-inject-odds-range")
def game_inject_odds_range(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    bookmaker: str = typer.Option("draftkings", help="Bookmaker key for OddsAPI (e.g., draftkings, pinnacle)"),
    backfill: bool = typer.Option(True, help="Only fill missing odds; do not overwrite existing prices"),
):
    """Inject The Odds API odds into predictions_{date}.csv across a range without running models.

    Avoids importing the web app to prevent circular imports by implementing injection here.
    """
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    import re, unicodedata, math
    import pandas as _pd
    from .utils.io import PROC_DIR
    from .data.odds_api import OddsAPIClient

    def _norm_team(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = s.lower()
        return re.sub(r"[^a-z0-9]+", "", s)

    def _extract_prices(markets):
        out = {}
        m_h2h = next((m for m in markets if m.get("key") == "h2h"), None)
        if m_h2h:
            for oc in m_h2h.get("outcomes", []):
                nm = str(oc.get("name"))
                out[f"ml::{nm}"] = oc.get("price")
        m_tot = next((m for m in markets if m.get("key") == "totals"), None)
        if m_tot:
            pts = None
            for oc in m_tot.get("outcomes", []):
                if oc.get("name") in ("Over", "Under"):
                    if pts is None:
                        pts = oc.get("point")
                    out[f"tot::{oc.get('name')}"] = oc.get("price")
                    out["tot::point"] = pts
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

    def _inject_for_date(d: str) -> dict:
        pred_path = PROC_DIR / f"predictions_{d}.csv"
        if not pred_path.exists():
            return {"status": "no-predictions", "date": d}
        try:
            df = _pd.read_csv(pred_path)
        except Exception as e:
            return {"status": "read-failed", "date": d, "error": str(e)}
        if df is None or df.empty:
            return {"status": "empty", "date": d}

        # Build ET day window
        try:
            from zoneinfo import ZoneInfo as _Z
            et = _Z("America/New_York")
            d0 = _dt.strptime(d, "%Y-%m-%d").replace(tzinfo=et)
            d1 = d0 + _td(days=1)
            start_iso = d0.astimezone(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_iso = d1.astimezone(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            start_iso = end_iso = None

        try:
            client = OddsAPIClient()
        except Exception as e:
            return {"status": "no-oddsapi", "date": d, "error": str(e)}

        try:
            events, _ = client.list_events("icehockey_nhl", commence_from_iso=start_iso, commence_to_iso=end_iso)
        except Exception:
            events = []

        rows = []
        for ev in events or []:
            try:
                eid = str(ev.get("id"))
                data, _ = client.event_odds(
                    sport="icehockey_nhl",
                    event_id=eid,
                    markets="h2h,totals,spreads",
                    regions="us",
                    bookmakers=bookmaker,
                    odds_format="american",
                )
                bks = data.get("bookmakers", []) if isinstance(data, dict) else []
                if not bks:
                    continue
                book = next((b for b in bks if b.get("key") == bookmaker), bks[0])
                markets = book.get("markets", [])
                prices = _extract_prices(markets)
                rows.append({
                    "home": ev.get("home_team"),
                    "away": ev.get("away_team"),
                    "home_ml": prices.get(f"ml::{ev.get('home_team')}"),
                    "away_ml": prices.get(f"ml::{ev.get('away_team')}"),
                    "over": prices.get("tot::Over"),
                    "under": prices.get("tot::Under"),
                    "total_line": prices.get("tot::point"),
                    "home_pl_-1.5": prices.get(f"pl::{ev.get('home_team')}::-1.5"),
                    "away_pl_+1.5": prices.get(f"pl::{ev.get('away_team')}::1.5"),
                    "home_ml_book": book.get("key"),
                    "away_ml_book": book.get("key"),
                    "over_book": book.get("key"),
                    "under_book": book.get("key"),
                    "home_pl_-1.5_book": book.get("key"),
                    "away_pl_+1.5_book": book.get("key"),
                })
            except Exception:
                continue
        if not rows:
            return {"status": "no-odds", "date": d}
        odds = _pd.DataFrame.from_records(rows)
        odds["home_norm"] = odds["home"].apply(_norm_team)
        odds["away_norm"] = odds["away"].apply(_norm_team)

        updated_rows = 0
        updated_fields = 0
        df = df.copy()
        df["home_norm"] = df["home"].apply(_norm_team)
        df["away_norm"] = df["away"].apply(_norm_team)

        for idx, r in df.iterrows():
            m = odds[(odds["home_norm"] == r.get("home_norm")) & (odds["away_norm"] == r.get("away_norm"))]
            if m.empty:
                m = odds[(odds["home_norm"] == r.get("away_norm")) & (odds["away_norm"] == r.get("home_norm"))]
            if m.empty:
                continue
            o = m.iloc[0]
            before = updated_fields
            def set_val(dst, val):
                nonlocal updated_fields
                if val is None or (isinstance(val, float) and _pd.isna(val)):
                    return
                cur = df.at[idx, dst] if dst in df.columns else None
                if backfill:
                    if cur is None or (isinstance(cur, float) and _pd.isna(cur)):
                        df.at[idx, dst] = val
                        updated_fields += 1
                else:
                    if str(cur) != str(val):
                        df.at[idx, dst] = val
                        updated_fields += 1
            for col, val in [
                ("home_ml_odds", o.get("home_ml")),
                ("away_ml_odds", o.get("away_ml")),
                ("over_odds", o.get("over")),
                ("under_odds", o.get("under")),
                ("total_line_used", o.get("total_line")),
                ("home_pl_-1.5_odds", o.get("home_pl_-1.5")),
                ("away_pl_+1.5_odds", o.get("away_pl_+1.5")),
                ("home_ml_book", o.get("home_ml_book")),
                ("away_ml_book", o.get("away_ml_book")),
                ("over_book", o.get("over_book")),
                ("under_book", o.get("under_book")),
                ("home_pl_-1.5_book", o.get("home_pl_-1.5_book")),
                ("away_pl_+1.5_book", o.get("away_pl_+1.5_book")),
            ]:
                set_val(col, val)
            if updated_fields > before:
                updated_rows += 1
        if updated_fields > 0:
            df.to_csv(pred_path, index=False)
        return {"status": "ok", "date": d, "updated_rows": int(updated_rows), "updated_fields": int(updated_fields)}

    cur = _dt.strptime(start, "%Y-%m-%d"); end_dt = _dt.strptime(end, "%Y-%m-%d")
    total_updates = {"rows": 0, "fields": 0}
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        summary = _inject_for_date(d)
        try:
            total_updates["rows"] += int(summary.get("updated_rows") or 0)
            total_updates["fields"] += int(summary.get("updated_fields") or 0)
        except Exception:
            pass
        print({"date": d, **(summary or {})})
        cur += _td(days=1)
    print({"range": {"start": start, "end": end}, "updates": total_updates})


@app.command(name="game-weekly-dashboard-csv")
def game_weekly_dashboard_csv(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
):
    """Aggregate per-day metrics (counts, accuracy, profits) into a CSV dashboard from audit data."""
    import math
    audit_csv = PROC_DIR / f"sim_picks_audit_{start}_to_{end}.csv"
    if not (audit_csv.exists() and getattr(audit_csv.stat(), "st_size", 0) > 0):
        print(f"Missing audit CSV: {audit_csv}. Run game-backtest-sim-picks-roi-range first.")
        raise typer.Exit(code=1)
    df = pd.read_csv(audit_csv)
    # Ensure grouping columns exist
    for c in ("date","market","win","profit"):
        if c not in df.columns:
            print(f"Audit CSV missing column: {c}")
            raise typer.Exit(code=1)
    # Compute per-day per-market metrics
    rows = []
    for (d, m), g in df.groupby(["date","market"]):
        n = int(len(g))
        wins = int(g["win"].sum()) if g["win"].dtype != object else int(g["win"].astype(bool).sum())
        acc = float(wins) / max(1, n)
        profit = float(g["profit"].sum()) if math.isfinite(g["profit"].sum()) else float(pd.to_numeric(g["profit"], errors="coerce").sum())
        rows.append({"date": d, "market": m, "n": n, "acc": round(acc,4), "profit": round(profit,2)})
    out = pd.DataFrame(rows).sort_values(["date","market"]) if rows else pd.DataFrame(columns=["date","market","n","acc","profit"])
    out_path = PROC_DIR / f"weekly_dashboard_{start}_to_{end}.csv"
    save_df(out, out_path)
    print(out.head(20))
    print(f"Saved weekly dashboard to {out_path}")
    
@app.command(name="game-sim-accuracy-range")
def game_sim_accuracy_range(
    start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD (ET)"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD (ET)"),
    use_calibrated: bool = typer.Option(True, help="Prefer calibrated sim probabilities if present"),
    bins: int = typer.Option(10, help="Calibration bins count"),
    output_prefix: str = typer.Option("game_sim_accuracy", help="Output JSON filename prefix"),
):
    """Compute simulation accuracy metrics (accuracy@0.5, Brier score, calibration bins)
    across Moneyline (home win), Totals Over, and Puckline Home -1.5 cover for a date range.

    Reads data/processed/simulations_{date}.csv and outcomes from data/raw/games.csv.
    Writes data/processed/{output_prefix}_{start}_to_{end}.json (or _all.json if no range specified).
    """
    def _read_outcomes() -> pd.DataFrame:
        path = RAW_DIR / "games.csv"
        if not (path.exists() and getattr(path.stat(), "st_size", 0) > 0):
            print(f"Missing outcomes: {path}")
            raise typer.Exit(code=1)
        df = pd.read_csv(path)
        # Normalize: create ET calendar 'date' from 'date_et' if present; else from 'date'
        if 'date_et' in df.columns:
            df['date'] = pd.to_datetime(df['date_et']).dt.strftime('%Y-%m-%d')
        else:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce').astype('Int64')
        df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce').astype('Int64')
        df['total_goals'] = (df['home_goals'].astype(int) + df['away_goals'].astype(int))
        df['home_win'] = (df['home_goals'] > df['away_goals']).astype(int)
        return df[['date','home','away','home_goals','away_goals','total_goals','home_win']]

    def _load_sim(path: Path, date_hint: Optional[str]) -> pd.DataFrame:
        df = pd.read_csv(path)
        # Ensure date column
        if 'date' not in df.columns:
            try:
                d = date_hint or path.stem.split('_')[-1]
                pd.to_datetime(d)
                df['date'] = d
            except Exception:
                pass
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df

    def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        y_true = y_true.astype(float)
        y_prob = y_prob.astype(float)
        return float(np.mean((y_prob - y_true) ** 2))

    def _cal_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(y_prob, edges, right=True)
        out = []
        for b in range(1, n_bins + 1):
            m = (idx == b)
            cnt = int(np.sum(m))
            if cnt == 0:
                out.append({'bin': b, 'mid': float((edges[b-1] + edges[b]) / 2), 'count': 0, 'avg_prob': None, 'emp_rate': None})
            else:
                out.append({'bin': b, 'mid': float((edges[b-1] + edges[b]) / 2), 'count': cnt, 'avg_prob': float(np.mean(y_prob[m])), 'emp_rate': float(np.mean(y_true[m]))})
        return out

    # Choose dates and simulation files
    sim_files: List[Path] = []
    date_list: List[str] = []
    if start and end:
        try:
            s_dt = datetime.strptime(start, '%Y-%m-%d').date(); e_dt = datetime.strptime(end, '%Y-%m-%d').date()
        except Exception:
            print('Invalid date format; use YYYY-MM-DD'); raise typer.Exit(code=1)
        if e_dt < s_dt:
            s_dt, e_dt = e_dt, s_dt
        cur = s_dt
        while cur <= e_dt:
            d = cur.strftime('%Y-%m-%d')
            p = PROC_DIR / f"simulations_{d}.csv"
            if p.exists() and getattr(p.stat(), 'st_size', 0) > 0:
                sim_files.append(p)
                date_list.append(d)
            cur = cur + timedelta(days=1)
    else:
        for p in sorted((PROC_DIR).glob('simulations_*.csv')):
            sim_files.append(p)
            try:
                date_list.append(pd.to_datetime(p.stem.split('_')[-1]).strftime('%Y-%m-%d'))
            except Exception:
                date_list.append(None)
    if not sim_files:
        print('No simulations files found for requested range.')
        raise typer.Exit(code=0)

    outcomes = _read_outcomes()

    ml_true: List[int] = []
    ml_prob: List[float] = []
    tot_true: List[int] = []
    tot_prob: List[float] = []
    pl_true: List[int] = []
    pl_prob: List[float] = []

    for p, d in zip(sim_files, date_list):
        df = _load_sim(p, d)
        if 'date' not in df.columns:
            # Cannot join without date; skip
            continue
        # Join on date/home/away
        cols_ok = all(c in df.columns for c in ('date','home','away'))
        if not cols_ok:
            continue
        merged = df.merge(outcomes, on=['date','home','away'], how='inner')
        if merged.empty:
            continue
        for _, r in merged.iterrows():
            # Moneyline
            p_ml = r.get('p_home_ml_sim_cal' if use_calibrated else 'p_home_ml_sim')
            if pd.isna(p_ml):
                p_ml = r.get('p_home_ml_sim' if use_calibrated else 'p_home_ml_sim_cal')
            if p_ml is not None and not pd.isna(p_ml):
                ml_prob.append(float(p_ml))
                ml_true.append(int(r['home_win']))
            # Totals (Over)
            # Use total line from simulations if present
            line = r.get('close_total_line_used')
            if line is None or (isinstance(line, float) and pd.isna(line)):
                line = r.get('total_line_used')
            try:
                tline = float(line) if line is not None else None
            except Exception:
                tline = None
            p_over = r.get('p_over_sim_cal' if use_calibrated else 'p_over_sim')
            if pd.isna(p_over):
                p_over = r.get('p_over_sim' if use_calibrated else 'p_over_sim_cal')
            if (p_over is not None and not pd.isna(p_over)) and (tline is not None):
                total = int(r['total_goals']) if 'total_goals' in r.index else (int(r['home_goals']) + int(r['away_goals']))
                if total != tline:  # exclude pushes
                    tot_prob.append(float(p_over))
                    tot_true.append(int(total > tline))
            # Puckline Home -1.5 cover
            p_pl = r.get('p_home_pl_-1.5_sim_cal' if use_calibrated else 'p_home_pl_-1.5_sim')
            if pd.isna(p_pl):
                p_pl = r.get('p_home_pl_-1.5_sim' if use_calibrated else 'p_home_pl_-1.5_sim_cal')
            if p_pl is not None and not pd.isna(p_pl):
                diff = int(r['home_goals']) - int(r['away_goals'])
                pl_prob.append(float(p_pl))
                pl_true.append(int(diff > 1.5))

    def _summ(y_true: List[int], y_prob: List[float]):
        if not y_true:
            return {'n': 0}
        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)
        acc = float(np.mean(((y_prob_arr >= 0.5).astype(int) == y_true_arr)))
        brier = _brier(y_true_arr, y_prob_arr)
        cbind = _cal_bins(y_true_arr, y_prob_arr, bins)
        return {'n': int(len(y_true)), 'accuracy_0p5': acc, 'brier': brier, 'calibration_bins': cbind}

    summary = {
        'range': {'start': start, 'end': end} if start and end else 'all',
        'moneyline_home_win': _summ(ml_true, ml_prob),
        'totals_over': _summ(tot_true, tot_prob),
        'puckline_home_minus_1p5_cover': _summ(pl_true, pl_prob),
    }

    out_name = f"{output_prefix}_{start}_to_{end}.json" if start and end else f"{output_prefix}_all.json"
    out_path = PROC_DIR / out_name
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(summary)
    print(f"Saved sim accuracy to {out_path}")
@app.command()
def predict_range(
    start: str = typer.Argument(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Argument(..., help="End date YYYY-MM-DD"),
    total_line: float = 6.0,
    odds_source: str = typer.Option("oddsapi", help="Odds source: csv|oddsapi"),
    bankroll: float = 0.0,
    kelly_fraction_part: float = 0.5,
):
    """Regenerate predictions_{date}.csv and edges_{date}.csv for each date in [start,end]."""
    from datetime import datetime as _dt, timedelta as _td
    try:
        s_dt = _dt.strptime(start, "%Y-%m-%d"); e_dt = _dt.strptime(end, "%Y-%m-%d")
    except Exception:
        print("Invalid date format; use YYYY-MM-DD"); raise typer.Exit(code=1)
    if e_dt < s_dt:
        s_dt, e_dt = e_dt, s_dt
    d = s_dt
    while d <= e_dt:
        day = d.strftime("%Y-%m-%d")
        try:
            # Load existing predictions (to preserve finals/closings where present)
            old_path = PROC_DIR / f"predictions_{day}.csv"
            old_df = None
            if old_path.exists():
                try:
                    old_df = pd.read_csv(old_path)
                except Exception:
                    old_df = None
            predict_core(
                date=day,
                total_line=total_line,
                odds_csv=None,
                source="web",
                odds_source=odds_source,
                snapshot=None,
                odds_regions="us",
                odds_markets="h2h,totals,spreads",
                odds_bookmaker=None,
                odds_best=False,
                bankroll=bankroll,
                kelly_fraction_part=kelly_fraction_part,
            )
            # After write, if we have an old_df with finals, merge them back in
            if old_df is not None:
                try:
                    new_df = pd.read_csv(old_path)
                    key_cols = [c for c in ("date","date_et") if c in new_df.columns]
                    key = [key_cols[0], "home", "away"] if key_cols else ["home", "away"]
                    if set(key).issubset(new_df.columns) and set(key).issubset(old_df.columns):
                        ndx = new_df.set_index(key)
                        odx = old_df.set_index(key)
                        for col in [
                            "final_home_goals","final_away_goals","actual_home_goals","actual_away_goals","actual_total",
                            "winner_actual","winner_model","winner_correct","result_total","total_diff",
                            "close_home_ml_odds","close_away_ml_odds","close_over_odds","close_under_odds","close_home_pl_-1.5_odds","close_away_pl_+1.5_odds",
                            "close_total_line_used","close_home_ml_book","close_away_ml_book","close_over_book","close_under_book","close_home_pl_-1.5_book","close_away_pl_+1.5_book","close_snapshot",
                        ]:
                            if col in odx.columns:
                                if col not in ndx.columns:
                                    ndx[col] = odx[col]
                                else:
                                    ndx[col] = ndx[col].combine_first(odx[col])
                        new_df = ndx.reset_index()
                        save_df(new_df, old_path)
                except Exception:
                    pass
        except SystemExit:
            pass
        except Exception as e:
            print(f"[warn] predict failed for {day}: {e}")
        d += _td(days=1)


@app.command()
def backfill_finals(
    start: str = typer.Argument(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Argument(..., help="End date YYYY-MM-DD"),
):
    """Backfill final scores and result fields into predictions_{date}.csv from data/raw/games.csv."""
    from datetime import datetime as _dt, timedelta as _td
    try:
        s_dt = _dt.strptime(start, "%Y-%m-%d"); e_dt = _dt.strptime(end, "%Y-%m-%d")
    except Exception:
        print("Invalid date format; use YYYY-MM-DD"); raise typer.Exit(code=1)
    if e_dt < s_dt:
        s_dt, e_dt = e_dt, s_dt
    games_path = RAW_DIR / "games.csv"
    if not games_path.exists():
        print("Missing data/raw/games.csv; cannot backfill finals.")
        raise typer.Exit(code=1)
    gdf = pd.read_csv(games_path)
    # Normalize date
    if "date" in gdf.columns:
        gdf["date"] = pd.to_datetime(gdf["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    need = {"home","away","home_goals","away_goals","date"}
    if not need.issubset(gdf.columns):
        print("games.csv missing required columns for backfill.")
        raise typer.Exit(code=1)
    d = s_dt
    while d <= e_dt:
        day = d.strftime("%Y-%m-%d")
        pred_path = PROC_DIR / f"predictions_{day}.csv"
        if pred_path.exists():
            try:
                pdf = pd.read_csv(pred_path)
                # Join on date_et (pred) to date (games), plus home/away names
                key_pred_date = "date_et" if "date_et" in pdf.columns else ("date" if "date" in pdf.columns else None)
                if key_pred_date is None:
                    d += _td(days=1); continue
                pdf[key_pred_date] = pd.to_datetime(pdf[key_pred_date], errors="coerce").dt.strftime("%Y-%m-%d")
                sub_games = gdf[gdf["date"] == day].copy()
                if sub_games.empty:
                    d += _td(days=1); continue
                # Merge in final scores from games.csv. Ensure goal columns are suffixed as *_g
                games_cols = ["home","away","home_goals","away_goals"]
                gsub = sub_games[games_cols].rename(columns={"home_goals": "home_goals_g", "away_goals": "away_goals_g"})
                merged = pdf.merge(gsub, on=["home","away"], how="left")
                # Fill finals where available
                has_suff = {"home_goals_g","away_goals_g"}.issubset(merged.columns)
                has_raw = {"home_goals","away_goals"}.issubset(merged.columns)
                if has_suff or has_raw:
                    # Ensure final_* columns exist before fill
                    if "final_home_goals" not in merged.columns:
                        merged["final_home_goals"] = pd.NA
                    if "final_away_goals" not in merged.columns:
                        merged["final_away_goals"] = pd.NA
                    src_h = "home_goals_g" if has_suff else "home_goals"
                    src_a = "away_goals_g" if has_suff else "away_goals"
                    merged["final_home_goals"] = merged["final_home_goals"].fillna(merged[src_h])
                    merged["final_away_goals"] = merged["final_away_goals"].fillna(merged[src_a])
                    # Derive extras
                    try:
                        merged["actual_home_goals"] = merged["final_home_goals"].astype(float)
                        merged["actual_away_goals"] = merged["final_away_goals"].astype(float)
                        merged["actual_total"] = merged["actual_home_goals"] + merged["actual_away_goals"]
                        merged["winner_actual"] = merged.apply(lambda r: r["home"] if float(r["actual_home_goals"]) > float(r["actual_away_goals"]) else (r["away"] if float(r["actual_home_goals"]) < float(r["actual_away_goals"]) else "Push"), axis=1)
                        if "p_home_ml" in merged.columns:
                            merged["winner_model"] = merged.apply(lambda r: (r["home"] if float(r["p_home_ml"]) >= 0.5 else r["away"]), axis=1)
                            merged["winner_correct"] = (merged["winner_actual"] == merged["winner_model"]) & (~merged["winner_actual"].isin(["Push"]))
                        if "total_line_used" in merged.columns:
                            def _tot_result(r):
                                try:
                                    at = float(r["actual_total"]); tl = float(r["total_line_used"]) if pd.notna(r["total_line_used"]) else np.nan
                                    if np.isnan(tl):
                                        return None
                                    if abs(at - tl) < 1e-9:
                                        return "Push"
                                    return "Over" if at > tl else "Under"
                                except Exception:
                                    return None
                            merged["result_total"] = merged.apply(_tot_result, axis=1)
                            merged["total_diff"] = merged.apply(lambda r: float(r["actual_total"]) - float(r["total_line_used"]) if pd.notna(r.get("total_line_used")) else np.nan, axis=1)
                    except Exception:
                        pass
                    # Drop helper cols and save
                    if "home_goals_g" in merged.columns:
                        merged = merged.drop(columns=[c for c in ["home_goals_g","away_goals_g"] if c in merged.columns])
                    save_df(merged, pred_path)
                    print(f"Backfilled finals into {pred_path}")
            except Exception as e:
                print(f"[warn] backfill {day}: {e}")
        d += _td(days=1)


@app.command()
def eval_segments(
    start: str = typer.Argument(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Argument(..., help="End date YYYY-MM-DD"),
    out_json: Optional[str] = typer.Option(None, help="Optional path to write JSON summary"),
):
    """Segmented diagnostics on moneyline predictions vs results and closings.

    Reports by:
      - side: favored by model (home p>=0.5) vs dog
      - prob buckets (5 bins)
      - team bias: avg(y - p) by team (home only)
      - line movement (if open/close available): moved toward home vs away
    """
    from datetime import datetime as _dt, timedelta as _td
    try:
        s_dt = _dt.strptime(start, "%Y-%m-%d"); e_dt = _dt.strptime(end, "%Y-%m-%d")
    except Exception:
        print("Invalid date format; use YYYY-MM-DD"); raise typer.Exit(code=1)
    if e_dt < s_dt:
        s_dt, e_dt = e_dt, s_dt
    rows = []
    d = s_dt
    while d <= e_dt:
        day = d.strftime("%Y-%m-%d")
        path = PROC_DIR / f"predictions_{day}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                need = {"home","away","p_home_ml","final_home_goals","final_away_goals"}
                if not need.issubset(df.columns):
                    d += _td(days=1); continue
                sub = df.dropna(subset=["p_home_ml","final_home_goals","final_away_goals"]).copy()
                if sub.empty:
                    d += _td(days=1); continue
                rows.append(sub)
            except Exception:
                pass
        d += _td(days=1)
    if not rows:
        print("No rows to evaluate.")
        return
    data = pd.concat(rows, ignore_index=True)
    # outcome 1 if home won
    y = (data["final_home_goals"].astype(float) > data["final_away_goals"].astype(float)).astype(int)
    p = data["p_home_ml"].astype(float).clip(1e-6, 1-1e-6)
    # side buckets
    favored = (p >= 0.5).astype(int)
    seg_side = data.assign(y=y, p=p).groupby(favored).apply(lambda g: pd.Series({
        "n": int(len(g)),
        "acc": float(((g["p"]>=0.5).astype(int) == g["y"]).mean()),
        "brier": float(((g["p"] - g["y"])**2).mean()),
    })).rename(index={0:"dog",1:"fav"}).to_dict(orient="index")
    # prob bins (5)
    q = pd.qcut(p, q=5, duplicates="drop")
    seg_bins = data.assign(y=y, p=p, bin=q.astype(str)).groupby("bin").apply(lambda g: pd.Series({
        "n": int(len(g)), "mean_p": float(g["p"].mean()), "obs": float(g["y"].mean())
    })).reset_index().to_dict(orient="records")
    # team bias (home side only aggregate)
    seg_team = data.assign(y=y, p=p).groupby("home").apply(lambda g: pd.Series({
        "n": int(len(g)), "bias": float((g["y"].mean() - g["p"].mean()))
    })).reset_index().sort_values("bias", ascending=False).to_dict(orient="records")
    # line movement direction if open/close available
    seg_move = {}
    if {"open_home_ml_odds","open_away_ml_odds","close_home_ml_odds","close_away_ml_odds"}.issubset(data.columns):
        tmp = data.dropna(subset=["open_home_ml_odds","open_away_ml_odds","close_home_ml_odds","close_away_ml_odds"]).copy()
        if not tmp.empty:
            # Convert American odds to implied (vig ignored) and compute movement toward home vs away
            def imp(o):
                o = float(o)
                return (100.0 / (o + 100.0)) if o > 0 else (abs(o) / (abs(o) + 100.0))
            tmp["open_home_imp"] = tmp["open_home_ml_odds"].apply(imp)
            tmp["open_away_imp"] = tmp["open_away_ml_odds"].apply(imp)
            tmp["close_home_imp"] = tmp["close_home_ml_odds"].apply(imp)
            tmp["close_away_imp"] = tmp["close_away_ml_odds"].apply(imp)
            tmp["move_to_home"] = (tmp["close_home_imp"] - tmp["open_home_imp"]) - (tmp["close_away_imp"] - tmp["open_away_imp"])  # positive favors home
            tmp["move_dir"] = tmp["move_to_home"].apply(lambda x: "to_home" if x>0 else ("to_away" if x<0 else "flat"))
            seg_move = tmp.assign(y=y.loc[tmp.index].values, p=p.loc[tmp.index].values).groupby("move_dir").apply(lambda g: pd.Series({
                "n": int(len(g)), "acc": float(((g["p"]>=0.5).astype(int) == g["y"]).mean()), "brier": float(((g["p"]-g["y"])**2).mean())
            })).to_dict(orient="index")
    res = {"range": {"start": start, "end": end}, "side": seg_side, "prob_bins": seg_bins, "team_bias": seg_team[:20], "line_move": seg_move}
    print(res)
    if out_json:
        out_path = Path(out_json); out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Wrote -> {out_path}")


@app.command()
def retune_elo(
    start: str = typer.Argument(..., help="Eval start date YYYY-MM-DD"),
    end: str = typer.Argument(..., help="Eval end date YYYY-MM-DD"),
    k_grid: str = typer.Option("14,18,20,22,26,30", help="Comma-separated K values to try"),
    home_adv_grid: str = typer.Option("35,45,50,55,65", help="Comma-separated home-adv Elo points to try"),
    metric: str = typer.Option("logloss", help="Optimization metric: logloss|brier"),
    save: bool = typer.Option(True, help="If true, save best Elo cfg and retrained ratings to model dir"),
):
    """Quick grid retune for Elo (K, home_adv):
    - Trains Elo on data/raw/games.csv for each candidate pair
    - Evaluates moneyline predictive quality over [start, end]
    - Optionally saves best config to MODEL_DIR/config.json and ratings to MODEL_DIR/elo_ratings.json
    """
    from datetime import datetime as _dt
    try:
        s_dt = _dt.strptime(start, "%Y-%m-%d"); e_dt = _dt.strptime(end, "%Y-%m-%d")
    except Exception:
        print("Invalid date format; use YYYY-MM-DD"); raise typer.Exit(code=1)
    path = RAW_DIR / "games.csv"
    if not path.exists():
        print("Missing data/raw/games.csv. Run 'fetch' or build_two_seasons first.")
        raise typer.Exit(code=1)
    df = pd.read_csv(path)
    # Normalize dates
    for col in ("date", "date_et"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    # Basic filter to completed games
    need = {"home","away","home_goals","away_goals"}
    if not need.issubset(df.columns):
        print("games.csv missing required columns")
        raise typer.Exit(code=1)
    eval_mask = df["date"].between(start, end) if "date" in df.columns else df["date_et"].between(start, end)
    eval_rows = df[eval_mask].copy()
    if eval_rows.empty:
        print("No games in eval range.")
        raise typer.Exit(code=1)
    ks = [float(x) for x in str(k_grid).split(",") if str(x).strip()]
    has = [float(x) for x in str(home_adv_grid).split(",") if str(x).strip()]
    best_cfg = None; best_score = float("inf")
    for k in ks:
        for ha in has:
            elo = Elo(EloConfig(k=float(k), home_adv=float(ha)))
            # Train on full history
            for _, r in df.dropna(subset=["home","away","home_goals","away_goals"]).iterrows():
                try:
                    elo.update_game(str(r["home"]), str(r["away"]), int(r["home_goals"]), int(r["away_goals"]))
                except Exception:
                    continue
            # Evaluate on window
            probs = []; ys = []
            eval_played = eval_rows.dropna(subset=["home_goals","away_goals"]).copy()
            for _, r in eval_played.iterrows():
                try:
                    ph, _ = elo.predict_moneyline_prob(str(r["home"]), str(r["away"]))
                    yv = 1 if int(r["home_goals"]) > int(r["away_goals"]) else 0
                    probs.append(ph); ys.append(yv)
                except Exception:
                    continue
            if not probs:
                continue
            p = np.clip(np.array(probs, dtype=float), 1e-6, 1-1e-6)
            y = np.array(ys, dtype=float)
            if metric == "brier":
                score = float(np.mean((p - y) ** 2))
            else:
                score = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            if score < best_score:
                best_score = score; best_cfg = (k, ha)
    if not best_cfg:
        print("No valid config found.")
        raise typer.Exit(code=1)
    print({"best": {"k": best_cfg[0], "home_adv": best_cfg[1], metric: best_score}})
    if save:
        # Persist config and retrained ratings for best cfg
        k, ha = best_cfg
        elo = Elo(EloConfig(k=float(k), home_adv=float(ha)))
        for _, r in df.dropna(subset=["home","away","home_goals","away_goals"]).iterrows():
            try:
                elo.update_game(str(r["home"]), str(r["away"]), int(r["home_goals"]), int(r["away_goals"]))
            except Exception:
                continue
        # Write ratings
        with (MODEL_DIR / "elo_ratings.json").open("w", encoding="utf-8") as f:
            json.dump(elo.ratings, f, indent=2)
        # Update config.json (merge with existing keys)
        cfg_path = MODEL_DIR / "config.json"
        obj = {}
        if cfg_path.exists():
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                obj = {}
        obj["elo_k"] = float(k); obj["elo_home_adv"] = float(ha)
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        print({"saved": {"elo_k": float(k), "elo_home_adv": float(ha)}})


@app.command()
def props_predict(odds_csv: str = typer.Option(..., help="CSV with columns: market,player,line,odds,team (optional)")):
    """
    Predict props using simple Poisson models from rolling means.
    - market: one of [SOG, SAVES]
    - player: player full name
    - line: numeric threshold (e.g., 2.5)
    - odds: American odds for OVER (we assume betting over line)
    - team: optional
    """
    stats_path = RAW_DIR / "player_game_stats.csv"
    if not stats_path.exists():
        print("Run collect_props first to build player stats history.")
        raise typer.Exit(code=1)
    try:
        hist = load_df(stats_path)
    except Exception:
        hist = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    req = pd.read_csv(odds_csv)
    shots = SkaterShotsModel()
    saves = GoalieSavesModel()
    goals = SkaterGoalsModel()
    assists = SkaterAssistsModel()
    points = SkaterPointsModel()
    out_rows = []
    for _, r in req.iterrows():
        market = str(r["market"]).upper()
@app.command()
def calibrate_models(
    start: str = typer.Argument(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Argument(..., help="End date YYYY-MM-DD"),
    metric: str = typer.Option("logloss", help="Metric to optimize: logloss|brier"),
):
    """
    Fit simple temperature+bias calibration for moneyline (home win) and totals (over) using predictions_{date}.csv across a date range.
    Writes data/processed/model_calibration.json.
    """
    from datetime import datetime as _dt, timedelta as _td
    from .utils.calibration import fit_temp_shift, save_calibration

    try:
        s_dt = _dt.strptime(start, "%Y-%m-%d"); e_dt = _dt.strptime(end, "%Y-%m-%d")
    except Exception:
        print("Invalid date format; use YYYY-MM-DD"); raise typer.Exit(code=1)
    if e_dt < s_dt:
        s_dt, e_dt = e_dt, s_dt

    p_ml, y_ml = [], []
    p_tot, y_tot = [], []
    d = s_dt
    while d <= e_dt:
        day = d.strftime("%Y-%m-%d")
        path = PROC_DIR / f"predictions_{day}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                # Moneyline: home win
                if {"p_home_ml","final_home_goals","final_away_goals"}.issubset(df.columns):
                    sub = df.dropna(subset=["p_home_ml","final_home_goals","final_away_goals"]).copy()
                    if not sub.empty:
                        p_ml.extend([float(x) for x in sub["p_home_ml"].values])
                        y_ml.extend([1 if (int(h) > int(a)) else 0 for h,a in zip(sub["final_home_goals"].values, sub["final_away_goals"].values)])
                # Totals: Over vs total_line_used; drop pushes
                if {"p_over","final_home_goals","final_away_goals","total_line_used"}.issubset(df.columns):
                    sub2 = df.dropna(subset=["p_over","final_home_goals","final_away_goals","total_line_used"]).copy()
                    if not sub2.empty:
                        totals = (sub2["final_home_goals"].astype(float) + sub2["final_away_goals"].astype(float)).values
                        lines = sub2["total_line_used"].astype(float).values
                        keep = totals != lines  # exclude pushes
                        if np.any(keep):
                            p_tot.extend([float(x) for x in sub2.loc[keep, "p_over"].values])
                            y_tot.extend([1 if (float(t) > float(l)) else 0 for t,l in zip(totals[keep], lines[keep])])
            except Exception as e:
                print(f"[warn] {day}: {e}")
        d += _td(days=1)

    if not p_ml and not p_tot:
        print("No data found for calibration.")
        raise typer.Exit(code=0)

    # Fit calibrations
    from .utils.calibration import summarize_binary, BinaryCalibration
    ml_cal = fit_temp_shift(p_ml, y_ml, metric=metric) if p_ml else BinaryCalibration()
    tot_cal = fit_temp_shift(p_tot, y_tot, metric=metric) if p_tot else BinaryCalibration()

    # Report
    p_ml_arr = np.array(p_ml, dtype=float) if p_ml else np.array([])
    y_ml_arr = np.array(y_ml, dtype=float) if y_ml else np.array([])
    p_ml_cal = ml_cal.apply(p_ml_arr) if p_ml else p_ml_arr
    p_tot_arr = np.array(p_tot, dtype=float) if p_tot else np.array([])
    y_tot_arr = np.array(y_tot, dtype=float) if y_tot else np.array([])
    p_tot_cal = tot_cal.apply(p_tot_arr) if p_tot else p_tot_arr
    ml_pre = summarize_binary(y_ml_arr, p_ml_arr) if p_ml else {"n":0}
    ml_post = summarize_binary(y_ml_arr, p_ml_cal) if p_ml else {"n":0}
    tt_pre = summarize_binary(y_tot_arr, p_tot_arr) if p_tot else {"n":0}
    tt_post = summarize_binary(y_tot_arr, p_tot_cal) if p_tot else {"n":0}
    print({"ml": {"n": ml_pre.get("n"), "pre": ml_pre, "post": ml_post, "cal": {"t": ml_cal.t, "b": ml_cal.b}},
           "totals": {"n": tt_pre.get("n"), "pre": tt_pre, "post": tt_post, "cal": {"t": tot_cal.t, "b": tot_cal.b}}})

    # Save
    out_path = PROC_DIR / "model_calibration.json"
    save_calibration(out_path, ml_cal, tot_cal, meta={"start": start, "end": end, "n_ml": len(p_ml), "n_totals": len(p_tot), "metric": metric})
    print(f"Saved calibration -> {out_path}")


@app.command()
def eval_predictions(
    start: str = typer.Argument(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Argument(..., help="End date YYYY-MM-DD"),
    out_json: Optional[str] = typer.Option(None, help="Optional path to write JSON summary")
):
    """
    Evaluate predictions across a date range, reporting accuracy/logloss/brier for moneyline (home win) and totals (over), with and without calibration.
    """
    from datetime import datetime as _dt, timedelta as _td
    from .utils.calibration import load_calibration, summarize_binary

    def _american_profit_per_1(odds: float | int | str) -> float:
        try:
            o = float(odds)
        except Exception:
            return float("nan")
        if np.isnan(o):
            return float("nan")
        return (o / 100.0) if o > 0 else (100.0 / abs(o))

    def _calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10):
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        m = (~np.isnan(y)) & (~np.isnan(p))
        y = y[m]
        p = np.clip(p[m], 1e-6, 1 - 1e-6)
        if y.size == 0:
            return []
        qs = np.quantile(p, np.linspace(0, 1, bins + 1))
        # Ensure strictly increasing bin edges
        for i in range(1, len(qs)):
            if qs[i] <= qs[i - 1]:
                qs[i] = min(qs[i - 1] + 1e-6, 1.0)
        out = []
        for i in range(bins):
            lo, hi = qs[i], qs[i + 1]
            if i == bins - 1:
                sel = (p >= lo) & (p <= hi)
            else:
                sel = (p >= lo) & (p < hi)
            if not np.any(sel):
                out.append({"bin": i + 1, "lo": float(lo), "hi": float(hi), "n": 0, "mean_p": float("nan"), "obs": float("nan")})
            else:
                pp = p[sel]; yy = y[sel]
                out.append({
                    "bin": i + 1,
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": int(pp.size),
                    "mean_p": float(np.mean(pp)),
                    "obs": float(np.mean(yy)),
                })
        return out

    try:
        s_dt = _dt.strptime(start, "%Y-%m-%d"); e_dt = _dt.strptime(end, "%Y-%m-%d")
    except Exception:
        print("Invalid date format; use YYYY-MM-DD"); raise typer.Exit(code=1)
    if e_dt < s_dt:
        s_dt, e_dt = e_dt, s_dt

    ml_cal, tot_cal = load_calibration(PROC_DIR / "model_calibration.json")
    p_ml_raw, y_ml = [], []
    p_tot_raw, y_tot = [], []
    # Track ROI vs closing odds using a naive "bet 1 unit on model-favored side" policy
    ml_roi_profits: list[float] = []
    tot_roi_profits: list[float] = []
    d = s_dt
    while d <= e_dt:
        day = d.strftime("%Y-%m-%d")
        path = PROC_DIR / f"predictions_{day}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                if {"p_home_ml","final_home_goals","final_away_goals"}.issubset(df.columns):
                    sub = df.dropna(subset=["p_home_ml","final_home_goals","final_away_goals"]).copy()
                    if not sub.empty:
                        p_ml_raw.extend([float(x) for x in sub["p_home_ml"].values])
                        y_ml.extend([1 if (int(h) > int(a)) else 0 for h,a in zip(sub["final_home_goals"].values, sub["final_away_goals"].values)])
                        # ROI vs closing for ML
                        # Choose side by p>=0.5; if closing odds missing for chosen side, skip row
                        if {"close_home_ml_odds","close_away_ml_odds"}.issubset(sub.columns):
                            for _, r in sub.iterrows():
                                try:
                                    ph = float(r["p_home_ml"]) if pd.notna(r["p_home_ml"]) else np.nan
                                    if np.isnan(ph):
                                        continue
                                    bet_home = bool(ph >= 0.5)
                                    close_home = r.get("close_home_ml_odds")
                                    close_away = r.get("close_away_ml_odds")
                                    # need final result
                                    h = r.get("final_home_goals"); a = r.get("final_away_goals")
                                    if pd.isna(h) or pd.isna(a):
                                        continue
                                    home_won = int(h) > int(a)
                                    if bet_home and pd.notna(close_home):
                                        ret = _american_profit_per_1(close_home)
                                        ml_roi_profits.append(ret if home_won else -1.0)
                                    elif (not bet_home) and pd.notna(close_away):
                                        ret = _american_profit_per_1(close_away)
                                        ml_roi_profits.append(ret if (not home_won) else -1.0)
                                except Exception:
                                    continue
                if {"p_over","final_home_goals","final_away_goals","total_line_used"}.issubset(df.columns):
                    sub2 = df.dropna(subset=["p_over","final_home_goals","final_away_goals","total_line_used"]).copy()
                    if not sub2.empty:
                        totals = (sub2["final_home_goals"].astype(float) + sub2["final_away_goals"].astype(float)).values
                        lines = sub2["total_line_used"].astype(float).values
                        keep = totals != lines
                        if np.any(keep):
                            p_tot_raw.extend([float(x) for x in sub2.loc[keep, "p_over"].values])
                            y_tot.extend([1 if (float(t) > float(l)) else 0 for t,l in zip(totals[keep], lines[keep])])
                            # ROI vs closing for totals (treat push as 0 profit)
                            if {"close_over_odds","close_under_odds","close_total_line_used"}.issubset(sub2.columns):
                                for _, r in sub2.loc[keep].iterrows():
                                    try:
                                        po = float(r["p_over"]) if pd.notna(r["p_over"]) else np.nan
                                        if np.isnan(po):
                                            continue
                                        bet_over = bool(po >= 0.5)
                                        close_over = r.get("close_over_odds")
                                        close_under = r.get("close_under_odds")
                                        # final vs line for push detection
                                        th = r.get("final_home_goals"); ta = r.get("final_away_goals"); tl = r.get("total_line_used")
                                        if pd.isna(th) or pd.isna(ta) or pd.isna(tl):
                                            continue
                                        total_goals = float(th) + float(ta)
                                        line = float(tl)
                                        if abs(total_goals - line) < 1e-9:
                                            tot_roi_profits.append(0.0)
                                            continue
                                        over_won = total_goals > line
                                        if bet_over and pd.notna(close_over):
                                            ret = _american_profit_per_1(close_over)
                                            tot_roi_profits.append(ret if over_won else -1.0)
                                        elif (not bet_over) and pd.notna(close_under):
                                            ret = _american_profit_per_1(close_under)
                                            tot_roi_profits.append(ret if (not over_won) else -1.0)
                                    except Exception:
                                        continue
            except Exception as e:
                print(f"[warn] {day}: {e}")
        d += _td(days=1)

    res = {"range": {"start": start, "end": end}}
    if p_ml_raw:
        p = np.array(p_ml_raw, dtype=float); y = np.array(y_ml, dtype=float)
        res["moneyline_raw"] = summarize_binary(y, p)
        res["moneyline_cal"] = summarize_binary(y, ml_cal.apply(p))
        res["moneyline_bins_raw"] = _calibration_bins(y, p)
        res["moneyline_bins_cal"] = _calibration_bins(y, ml_cal.apply(p))
    else:
        res["moneyline_raw"] = {"n": 0}
        res["moneyline_cal"] = {"n": 0}
        res["moneyline_bins_raw"] = []
        res["moneyline_bins_cal"] = []
    if p_tot_raw:
        p = np.array(p_tot_raw, dtype=float); y = np.array(y_tot, dtype=float)
        res["totals_raw"] = summarize_binary(y, p)
        res["totals_cal"] = summarize_binary(y, tot_cal.apply(p))
        res["totals_bins_raw"] = _calibration_bins(y, p)
        res["totals_bins_cal"] = _calibration_bins(y, tot_cal.apply(p))
    else:
        res["totals_raw"] = {"n": 0}
        res["totals_cal"] = {"n": 0}
        res["totals_bins_raw"] = []
        res["totals_bins_cal"] = []
    # ROI summaries
    if ml_roi_profits:
        total = float(np.sum(ml_roi_profits)); n_bets = int(len(ml_roi_profits))
        res["moneyline_roi"] = {"n": n_bets, "roi_per_bet": total / n_bets, "profit_total": total}
    else:
        res["moneyline_roi"] = {"n": 0}
    if tot_roi_profits:
        total = float(np.sum(tot_roi_profits)); n_bets = int(len(tot_roi_profits))
        res["totals_roi"] = {"n": n_bets, "roi_per_bet": total / n_bets, "profit_total": total}
    else:
        res["totals_roi"] = {"n": 0}
    print(res)
    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Wrote -> {out_path}")


@app.command()
def props_recommendations(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    min_ev: float = typer.Option(0.0, help="Minimum EV threshold for ev_over"),
    top: int = typer.Option(200, help="Top N to keep after sorting by EV desc"),
    market: str = typer.Option("", help="Optional filter: SOG,SAVES,GOALS,ASSISTS,POINTS"),
    min_ev_per_market: str = typer.Option("", help="Optional per-market EV thresholds, e.g., 'SOG=0.00,GOALS=0.05,ASSISTS=0.00,POINTS=0.12'"),
):
    """Build props recommendations_{date}.csv from canonical Parquet lines and simple Poisson projections.

    Reads data/props/player_props_lines/date=YYYY-MM-DD/*.parquet, computes p_over and EV for each OVER/UNDER pair
    using rolling-mean lambdas, and writes a denormalized recommendation list similar to NFL.
    """

    import os, time
    dbg = str(os.getenv("PROPS_DEBUG", "1")).strip().lower() in ("1","true","yes")
    def _dbg(msg: str):
        if dbg:
            print(f"[recs] {msg}", flush=True)
    from glob import glob
    # Read canonical lines for date (OddsAPI + Bovada if present)
    parts = []
    base = Path("data/props") / f"player_props_lines/date={date}"
    prefer = [base / "oddsapi.parquet", base / "bovada.parquet", base / "oddsapi.csv", base / "bovada.csv"]
    def _read_any(files):
        out = []
        for f in files:
            if not f.exists():
                continue
            try:
                if f.suffix == ".parquet":
                    # Try pyarrow first; if unavailable (e.g., Windows ARM64), fall back to DuckDB
                    try:
                        df_pa = pd.read_parquet(f, engine="pyarrow")
                        out.append(df_pa)
                        _dbg(f"read parquet via pyarrow: {f}")
                    except Exception as e_pa:
                        try:
                            import duckdb as _duckdb
                            # Use forward slashes for cross-platform compatibility in SQL string
                            f_posix = str(f).replace("\\\\", "/").replace("\\", "/")
                            df_duck = _duckdb.query(f"SELECT * FROM read_parquet('{f_posix}')").df()
                            out.append(df_duck)
                            _dbg(f"read parquet via duckdb: {f} rows={len(df_duck)}")
                        except Exception as e_duck:
                            _dbg(f"failed to read parquet {f} via pyarrow ({e_pa}) and duckdb ({e_duck})")
                            continue
                else:
                    df_csv = pd.read_csv(f)
                    out.append(df_csv)
                    _dbg(f"read csv: {f}")
            except Exception as e:
                _dbg(f"failed to read {f}: {e}")
                continue
        return out
    t0 = time.monotonic()
    parts = _read_any(prefer)
    _dbg(f"read preferred parts: {[str(p) for p in prefer if p.exists()]} -> {sum((0 if p is None else len(p)) for p in parts)} rows")
    if not parts:
        # Fast-fail: write empty output instead of aborting to avoid killing parent flows
        print("No props lines found for", date, "- writing empty recommendations")
        out = pd.DataFrame(columns=["date","player","team","market","line","proj","p_over","over_price","under_price","book","side","ev"])
        out_path = PROC_DIR / f"props_recommendations_{date}.csv"
        save_df(out, out_path)
        print(f"Wrote {out_path} with 0 rows")
        return
    try:
        lines = pd.concat(parts, ignore_index=True)
        _dbg(f"concat lines: {len(lines)} rows")
    except Exception:
        # Fallback: use first non-empty; otherwise write empty output and return
        lines = next((p for p in parts if p is not None and not p.empty), pd.DataFrame())
        if lines is None or lines.empty:
            print("No readable props lines for", date, "- writing empty recommendations")
            out = pd.DataFrame(columns=["date","player","team","market","line","proj","p_over","over_price","under_price","book","side","ev"])
            out_path = PROC_DIR / f"props_recommendations_{date}.csv"
            save_df(out, out_path)
            print(f"Wrote {out_path} with 0 rows")
            return

    # Build a player->team mapping to ensure team alignment in outputs
    from .web.teams import get_team_assets as _get_team_assets
    def _norm_name(x: str) -> str:
        try:
            s = str(x or "").strip()
            return " ".join(s.split())
        except Exception:
            return str(x)
    def _abbr_team(t: str | None) -> str | None:
        if not t:
            return None
        try:
            a = _get_team_assets(str(t)) or {}
            ab = a.get("abbr")
            return str(ab).upper() if ab else (str(t).strip().upper() if str(t).strip() else None)
        except Exception:
            return str(t).strip().upper() if isinstance(t, str) and t.strip() else None
    
    # Helper to create name variants for fuzzy matching (full name and abbreviated)
    def _create_name_variants(name: str) -> list[str]:
        """Create name variants for matching between full names (betting lines) and abbreviated names (projections)."""
        name = (name or "").strip()
        if not name:
            return []
        
        # Normalize whitespace
        norm = " ".join(name.split())
        variants = [norm.lower()]
        
        # If abbreviated (e.g., "A. Levshunov"), keep as-is and without dot
        if "." in norm:
            variants.append(norm.replace(".", "").lower())
        
        # If full name (e.g., "Artyom Levshunov"), create abbreviated variant
        parts = [p for p in norm.split() if p and not p.endswith(".")]
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            # Add "A. Levshunov" format
            abbreviated = f"{first[0].upper()}. {last}"
            variants.append(abbreviated.lower())
            variants.append(abbreviated.replace(".", "").lower())
            # Also add "A Levshunov" (no dot, space)
            abbreviated_nodot = f"{first[0].upper()} {last}"
            variants.append(abbreviated_nodot.lower())
        
        return list(set(variants))
    
    player_team_map: dict[str, str] = {}
    try:
        if not lines.empty and {"player_name","team"}.issubset(lines.columns):
            cur = lines
            try:
                if "is_current" in cur.columns:
                    cur = cur[cur["is_current"] == True]
            except Exception:
                pass
            cur = cur.dropna(subset=["player_name","team"]).copy()
            cur["_team_abbr"] = cur["team"].map(_abbr_team)
            last_team = cur.groupby("player_name")["_team_abbr"].agg(lambda s: s.dropna().astype(str).iloc[-1] if len(s.dropna()) else None)
            for nm, tm in last_team.items():
                if tm:
                    # Create variants for better matching
                    for variant in _create_name_variants(_norm_name(nm)):
                        player_team_map[variant.lower()] = str(tm)
        # Fallback: use historical player_game_stats.csv for last known team
        if True:
            from .utils.io import RAW_DIR as _RAW
            try:
                pstats = pd.read_csv(_RAW / "player_game_stats.csv")
            except Exception:
                pstats = pd.DataFrame()
            if pstats is not None and not pstats.empty and "player" in pstats.columns:
                try:
                    pstats = pstats.dropna(subset=["player"]).copy()
                    pstats["player"] = pstats["player"].astype(str).map(_norm_name)
                    if "team" in pstats.columns:
                        last_teams = pstats.dropna(subset=["team"]).groupby("player")["team"].last()
                        for nm, tm in last_teams.items():
                            ab = _abbr_team(tm)
                            if ab:
                                # Create variants for better matching
                                for variant in _create_name_variants(_norm_name(nm)):
                                    key = variant.lower()
                                    if key and key not in player_team_map:
                                        player_team_map[key] = ab
                except Exception:
                    pass
    except Exception:
        player_team_map = player_team_map
    # Enrich player_team_map with cached roster for the slate date if available (more authoritative)
    try:
        roster_cache = PROC_DIR / f"roster_{date}.csv"
        _dbg(f"roster_cache path: {roster_cache}, exists={roster_cache.exists()}")
        if roster_cache.exists():
            _r = pd.read_csv(roster_cache)
            _dbg(f"roster_cache rows: {len(_r) if _r is not None else 0}")
            if _r is not None and not _r.empty:
                # Accept columns: full_name or player, and team (abbr)
                name_col = "full_name" if "full_name" in _r.columns else ("player" if "player" in _r.columns else None)
                # support multiple team column variants
                team_col = None
                for cand in ("team", "team_abbr", "team_abbrev", "teamAbbrev", "team_abbreviation"):
                    if cand in _r.columns:
                        team_col = cand; break
                _dbg(f"roster_cache name_col={name_col}, team_col={team_col}")
                if name_col and team_col:
                    for _, rr in _r.dropna(subset=[name_col, team_col]).iterrows():
                        nm = _norm_name(str(rr.get(name_col)))
                        tm = str(rr.get(team_col)).strip().upper()
                        if nm and tm:
                            # Create variants for better matching
                            for variant in _create_name_variants(nm):
                                player_team_map[variant.lower()] = tm
    except Exception as e:
        _dbg(f"roster_cache error: {e}")
        pass
    # Enrich from roster_master.csv as an additional fallback
    try:
        rm_path = PROC_DIR / "roster_master.csv"
        if rm_path.exists():
            _rm = pd.read_csv(rm_path)
            if _rm is not None and not _rm.empty:
                name_col = "full_name" if "full_name" in _rm.columns else ("player" if "player" in _rm.columns else None)
                team_col = None
                for cand in ("team", "team_abbr", "team_abbrev", "teamAbbrev", "team_abbreviation"):
                    if cand in _rm.columns:
                        team_col = cand; break
                if name_col and team_col:
                    for _, rr in _rm.dropna(subset=[name_col, team_col]).iterrows():
                        nm = _norm_name(str(rr.get(name_col)))
                        tm = str(rr.get(team_col)).strip().upper()
                        if nm and tm:
                            # Create variants for better matching
                            for variant in _create_name_variants(nm):
                                key = variant.lower()
                                if key not in player_team_map:
                                    player_team_map[key] = tm
    except Exception:
        pass
    _dbg(f"player_team_map has {len(player_team_map)} entries")
    
    # Prefer precomputed per-player lambdas to avoid expensive history scans
    # data/processed/props_projections_all_{date}.csv: [date, player, team, position, market, proj_lambda]
    lam_map: dict[tuple[str, str], float] = {}
    try:
        proj_all_path = PROC_DIR / f"props_projections_all_{date}.csv"
        if proj_all_path.exists():
            _proj = pd.read_csv(proj_all_path)
            if _proj is not None and not _proj.empty and {"player","market","proj_lambda"}.issubset(_proj.columns):
                for _, rr in _proj.iterrows():
                    try:
                        player_name = _norm_name(rr.get("player"))
                        mkt = str(rr.get("market")).upper()
                        val = float(rr.get("proj_lambda")) if pd.notna(rr.get("proj_lambda")) else None
                        
                        if player_name and mkt and val is not None:
                            # Create all name variants for better matching
                            for variant in _create_name_variants(player_name):
                                key = (variant.lower(), mkt)
                                lam_map[key] = val
                    except Exception:
                        pass
    except Exception:
        lam_map = lam_map
    # Instantiate models (probabilities only). We'll only compute lambdas from history for missing players lazily.
    shots = SkaterShotsModel(); saves = GoalieSavesModel(); goals = SkaterGoalsModel(); assists = SkaterAssistsModel(); points = SkaterPointsModel(); blocks = SkaterBlocksModel()
    # Do NOT load or backfill history here; rely on precomputed projections_all. Use conservative fallbacks for misses.
    def _fallback_lambda(mk: str) -> float:
        m = (mk or '').upper()
        if m == 'SOG': return 2.4
        if m == 'GOALS': return 0.35
        if m == 'ASSISTS': return 0.45
        if m == 'POINTS': return 0.9
        if m == 'SAVES': return 27.0
        if m == 'BLOCKS': return 1.3
        return 1.0
    def _prob_over_for(mkt: str, lam: float, line: float) -> float:
        m = (mkt or '').upper()
        if m == "SOG":
            return shots.prob_over(lam, line)
        if m == "SAVES":
            return saves.prob_over(lam, line)
        if m == "GOALS":
            return goals.prob_over(lam, line)
        if m == "ASSISTS":
            return assists.prob_over(lam, line)
        if m == "POINTS":
            return points.prob_over(lam, line)
        if m == "BLOCKS":
            return blocks.prob_over(lam, line)
        return None
    def proj_and_prob(mkt: str, player: str, line: float) -> tuple[float, float]:
        m = (mkt or '').upper()
        key = (_norm_name(player).lower(), m)
        lam = lam_map.get(key)
        if lam is None:
            # Use conservative league-average fallback lambdas to avoid slow history scans
            lam = _fallback_lambda(m)
            lam_map[key] = lam
        if lam is None:
            return None, None
        return lam, _prob_over_for(m, lam, line)
    # Normalize player display and filter out team-level rows
    import ast as _ast
    def _norm_player(s):
        if s is None:
            return ""
        x = str(s).strip()
        if x.startswith('{') and x.endswith('}'):
            try:
                d = _ast.literal_eval(x)
                if isinstance(d, dict):
                    v = d.get('default') or d.get('name') or ''
                    if isinstance(v, str):
                        x = v.strip()
            except Exception:
                pass
        return " ".join(x.split())
    def _looks_like_player(x: str) -> bool:
        s = (x or '').strip().lower()
        if not s:
            return False
        bad = ['total shots on goal', 'team total', 'first period', 'second period', 'third period']
        return (any(ch.isalpha() for ch in s) and not any(b in s for b in bad))
    # Optionally load props probability calibration mapping (by market+line)
    cal_map: dict[tuple[str, float], object] = {}
    try:
        import os as _os
        if str(_os.getenv("SKIP_PROPS_CALIBRATION", "")).strip().lower() not in ("1","true","yes"):
            from .utils.calibration import load_props_stats_calibration_map
            # Prefer dated files if present; else generic
            pref = PROC_DIR / "props_stats_calibration.json"
            if pref.exists():
                cal_map = load_props_stats_calibration_map(pref)
            else:
                # Fallback: latest *_stats_calibration_*.json by mtime
                cands = sorted([p for p in PROC_DIR.glob("props_stats_calibration_*.json") if p.is_file()], key=lambda x: x.stat().st_mtime, reverse=True)
                if cands:
                    cal_map = load_props_stats_calibration_map(cands[0])
    except Exception:
        cal_map = {}
    def _round_half(x: float) -> float:
        try:
            import math as _math
            return _math.floor(x * 2.0 + 1e-6) / 2.0
        except Exception:
            return x
    def _apply_calibrated(mkt: str, line: float, p: float) -> float:
        try:
            if not (0.0 <= float(p) <= 1.0):
                return p
        except Exception:
            return p
        key = (str(mkt or '').upper(), float(_round_half(float(line))))
        cal = cal_map.get(key)
        if not cal:
            return float(p)
        try:
            # BinaryCalibration.apply expects numpy array; allow scalar via small array
            import numpy as _np
            arr = _np.asarray([float(p)], dtype=float)
            out = cal.apply(arr)
            return float(out[0]) if out is not None and len(out) > 0 else float(p)
        except Exception:
            return float(p)

    # Combine rows: lines contain over_price and under_price per (market,player,line,book)
    # Vectorized EV computation using precomputed lambdas; fallback row-wise for misses
    import numpy as _np
    from scipy.stats import poisson as _poisson
    # Prepare normalized working frame (explicit core columns to avoid accidental column loss)
    core_cols = [c for c in [
        "market","player_name","player","team","line","over_price","under_price","book"
    ] if c in lines.columns]
    work = lines[core_cols].copy()
    _dbg(f"work core cols kept={core_cols}; rows={len(work)}")
    # Uppercase market and optional filter
    if "market" in work.columns:
        work["market"] = work["market"].astype(str).str.upper()
        msel = (market or "").strip()
        _dbg(f"market parameter='{market}', msel='{msel}', will filter={bool(msel and msel.lower() not in ('all', ''))}")
        if msel and msel.lower() not in ("all", ""): 
            work = work.loc[work["market"] == msel.upper()].copy()
            _dbg(f"filtered to market={msel.upper()}: {len(work)} rows")
    # Normalize player display and filter likely players
    work["player_display"] = work.apply(lambda r: _norm_player(r.get("player_name") or r.get("player")), axis=1)
    work = work.loc[work["player_display"].map(_looks_like_player)].copy()
    _dbg(f"after looks_like_player filter: {len(work)} rows")
    # Parse numeric line and keep valid (preserve all columns using loc)
    work["line_num"] = pd.to_numeric(work.get("line"), errors="coerce")
    work = work.loc[work["line_num"].notna()].copy()
    # Attach normalized name for join - use abbreviated variant preferentially for matching
    def _norm_for_join(name: str) -> str:
        """Normalize name for joining - try all variants and pick most compact (abbreviated if available)."""
        variants = _create_name_variants(name)
        if not variants:
            return ""
        # Prefer abbreviated format (shortest with a dot or single letter + space)
        dot_variants = [v for v in variants if "." in v or (len(v.split()) == 2 and len(v.split()[0]) == 1)]
        if dot_variants:
            return min(dot_variants, key=len)
        return min(variants, key=len)
    
    work["player_norm"] = work["player_display"].astype(str).map(_norm_for_join)
    # Build lambda DataFrame from lam_map for merge
    lam_df = pd.DataFrame([{"player_norm": k[0], "market": k[1], "proj_lambda": v} for k, v in lam_map.items()]) if lam_map else pd.DataFrame(columns=["player_norm","market","proj_lambda"])
    merged = work.merge(lam_df, on=["player_norm", "market"], how="left")
    _dbg(f"merged with lam_df: {len(merged)} rows; lam_df={len(lam_df)} rows")
    # Restrict to tonight's slate teams (ET) to avoid cross-day or stale lines leaking in
    # Build slate team abbreviations via Web API
    slate_abbrs: set[str] = set()
    try:
        web = NHLWebClient()
        games = web.schedule_day(date)
        from .web.teams import get_team_assets as _assets
        names = set()
        for g in games:
            names.add(str(getattr(g, "home", "")))
            names.add(str(getattr(g, "away", "")))
        for nm in names:
            try:
                ab = (_assets(nm).get("abbr") or "").upper()
                if ab:
                    slate_abbrs.add(ab)
            except Exception:
                continue
    except Exception:
        slate_abbrs = set()
    # Build allowed player set for tonight's slate (from roster caches)
    allowed_names: set[str] = set()
    try:
        # helper to add names from a dataframe with name and team columns
        def _add_names(df: pd.DataFrame):
            nonlocal allowed_names
            if df is None or df.empty:
                return
            name_col = None
            for cand in ("full_name", "player", "name"):
                if cand in df.columns:
                    name_col = cand; break
            team_col = None
            for cand in ("team", "team_abbr", "teamAbbrev", "team_abbrev", "team_abbreviation"):
                if cand in df.columns:
                    team_col = cand; break
            if not name_col or not team_col:
                return
            tmp = df.dropna(subset=[name_col, team_col]).copy()
            tmp["_team_abbr"] = tmp[team_col].astype(str).str.upper()
            if slate_abbrs:
                tmp = tmp[tmp["_team_abbr"].isin(slate_abbrs)]
            vals = tmp[name_col].astype(str).map(lambda s: " ".join(str(s).split()).lower())
            allowed_names.update(vals.tolist())
        # roster_{date}.csv
        _rc = PROC_DIR / f"roster_{date}.csv"
        if _rc.exists():
            _add_names(pd.read_csv(_rc))
        # roster_master.csv
        _rm = PROC_DIR / "roster_master.csv"
        if _rm.exists():
            _add_names(pd.read_csv(_rm))
    except Exception:
        allowed_names = set()
    # Optional (disabled by default): live roster fallback via Stats API
    try:
        if (str(os.getenv('PROPS_ALLOW_LIVE_ROSTER','')).strip().lower() in ('1','true','yes')) and slate_abbrs and not allowed_names:
            from .data import rosters as _rosters_mod
            teams = _rosters_mod.list_teams()
            abbr_to_ids = {}
            for t in teams:
                try:
                    ab = str(t.get('abbreviation') or t.get('teamAbbrev') or '').upper()
                    tid = int(t.get('id')) if t.get('id') is not None else None
                    if ab and tid is not None:
                        abbr_to_ids.setdefault(ab, []).append(tid)
                except Exception:
                    continue
            for ab in slate_abbrs:
                for tid in abbr_to_ids.get(ab, []):
                    try:
                        rp = _rosters_mod.fetch_current_roster(int(tid))
                        for r in rp:
                            nm = str(getattr(r, 'full_name', '') or '').strip()
                            if nm:
                                allowed_names.add(' '.join(nm.split()).lower())
                    except Exception:
                        continue
    except Exception:
        pass
    # Resolve each row's best team guess (from line team or player mapping), then filter to slate_abbrs if available
    try:
        merged["_team_from_line"] = merged.get("team")
        try:
            merged["_team_from_line"] = merged["_team_from_line"].map(_abbr_team)
        except Exception:
            pass
        merged["_team_from_map"] = merged["player_norm"].map(lambda nm: player_team_map.get(nm))
        # Prefer roster-based mapping (player_team_map) over line-provided team to avoid opponent/market mis-tags
        merged["_team_final"] = merged["_team_from_map"].fillna(merged["_team_from_line"])
        if slate_abbrs:
            # Strict: only include rows where resolved team is in tonight's slate, but add a safety fallback
            _dbg(f"before slate filter: {len(merged)} rows, slate teams: {sorted(slate_abbrs)}")
            _dbg(f"_team_final values (unique): {sorted(merged['_team_final'].dropna().unique().tolist()[:20])}")
            pre_slate = merged.copy()
            merged = merged.loc[merged["_team_final"].isin(slate_abbrs)].copy()
            _dbg(f"after slate team filter: {len(merged)} rows")
            # If the slate filter drops too many rows (e.g., name mapping gaps), skip it to avoid empty outputs
            try:
                kept_frac = float(len(merged)) / float(len(pre_slate)) if len(pre_slate) else 0.0
            except Exception:
                kept_frac = 0.0
            if len(merged) == 0 or kept_frac < 0.3:
                merged = pre_slate
                _dbg("slate filter preserved <30% or zero rows; skipping slate filter for robustness")
            # Further restrict to names present on roster for slate teams if we have such a set.
            # However, roster caches can be stale or wrong (e.g., fallback build). To avoid
            # dropping valid slate players, only apply this filter if it retains a healthy
            # fraction of rows; otherwise, skip it.
            if allowed_names and not merged.empty and "player_norm" in merged.columns:
                msk = merged["player_norm"].isin(allowed_names)
                try:
                    frac = float(msk.sum()) / float(len(msk)) if len(msk) else 0.0
                except Exception:
                    frac = 0.0
                _dbg(f"allowed_names filter: {msk.sum()}/{len(msk)} = {frac:.2%}")
                if frac >= 0.6:  # keep filter only if it preserves at least 60% of rows
                    merged = merged.loc[msk].copy()
                    _dbg(f"after allowed_names filter: {len(merged)} rows")
    except Exception:
        # If anything goes wrong, proceed without slate filter
        pass
    _dbg(f"slate_abbrs={len(slate_abbrs)}; building allowed_names...")
    # Vectorized p_over for rows with proj_lambda available
    vec_mask = merged["proj_lambda"].notna()
    _dbg(f"vec rows with lambda: {int(vec_mask.sum())}; remain: {int((~vec_mask).sum())}")
    p_over_vec = pd.Series(_np.nan, index=merged.index)
    for mkt in ["SOG","SAVES","GOALS","ASSISTS","POINTS","BLOCKS"]:
        sel = vec_mask & (merged["market"] == mkt)
        if sel.any():
            lam_arr = merged.loc[sel, "proj_lambda"].astype(float).values
            line_arr = _np.floor(merged.loc[sel, "line_num"].astype(float).values + 1e-9).astype(int)
            raw_p = _poisson.sf(line_arr, mu=lam_arr)
            # Apply calibration per (market,line) if available
            try:
                if cal_map:
                    # Vectorized through Python loop per unique line for this market
                    idxs = merged.index[sel]
                    lines_unique = sorted(set(merged.loc[sel, "line_num"].map(_round_half).astype(float).tolist()))
                    for ln in lines_unique:
                        sub = merged.loc[sel & (merged["line_num"].map(_round_half).astype(float) == float(ln))]
                        if sub is None or sub.empty:
                            continue
                        key = (mkt, float(ln))
                        cal = cal_map.get(key)
                        if not cal:
                            # Assign raw for this subgroup
                            p_over_vec.loc[sub.index] = raw_p[[i for i, j in enumerate(idxs) if j in sub.index]]
                            continue
                        try:
                            vals = raw_p[[i for i, j in enumerate(idxs) if j in sub.index]]
                            out = cal.apply(_np.asarray(vals, dtype=float))
                            p_over_vec.loc[sub.index] = out
                        except Exception:
                            p_over_vec.loc[sub.index] = raw_p[[i for i, j in enumerate(idxs) if j in sub.index]]
                else:
                    p_over_vec.loc[sel] = raw_p
            except Exception:
                p_over_vec.loc[sel] = raw_p
    merged["p_over_vec"] = p_over_vec
    # Vectorized EVs
    def _american_to_decimal_series(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        pos = s[s > 0]
        neg = s[s <= 0]
        out = pd.Series(_np.nan, index=s.index)
        out.loc[pos.index] = 1.0 + (pos / 100.0)
        out.loc[neg.index] = 1.0 + (100.0 / _np.abs(neg))
        return out
    # Build vectorized output rows (only where we had lambda)
    vec_out = merged[vec_mask].copy()
    # Keep raw probability for diagnostics if calibration applied
    try:
        if cal_map and not vec_out.empty:
            vec_out["p_over_raw"] = vec_out.get("p_over_vec")
    except Exception:
        pass
    
    # Compute EV on the filtered data
    dec_over = _american_to_decimal_series(vec_out.get("over_price"))
    dec_under = _american_to_decimal_series(vec_out.get("under_price"))
    p_over_s = pd.to_numeric(vec_out["p_over_vec"], errors="coerce")
    ev_over_s = p_over_s * (dec_over - 1.0) - (1.0 - p_over_s)
    p_under_s = (1.0 - p_over_s).clip(lower=0.0, upper=1.0)
    ev_under_s = p_under_s * (dec_under - 1.0) - (1.0 - p_under_s)
    # Choose side with better EV, handling NaNs
    over_better = (ev_under_s.isna()) | (~ev_over_s.isna() & (ev_over_s >= ev_under_s))
    chosen_side = _np.where(over_better, "Over", "Under")
    chosen_price = _np.where(over_better, vec_out.get("over_price"), vec_out.get("under_price"))
    chosen_ev = _np.where(over_better, ev_over_s, ev_under_s)
    
    vec_out["side"] = chosen_side
    vec_out["ev"] = pd.to_numeric(chosen_ev, errors="coerce")
    # Parse per-market thresholds if provided
    def _parse_thresholds(s: str) -> dict:
        d = {}
        for part in (s or "").split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                d[str(k).strip().upper()] = float(str(v).strip())
            except Exception:
                continue
        return d
    _thr_map = _parse_thresholds(min_ev_per_market)
    if _thr_map:
        # Build a per-row threshold series with fallback to global min_ev
        thr_series = vec_out["market"].astype(str).str.upper().map(lambda m: _thr_map.get(m, float(min_ev))).astype(float)
        vec_out = vec_out[vec_out["ev"].notna() & (vec_out["ev"].astype(float) >= thr_series)]
    else:
        vec_out = vec_out[vec_out["ev"].notna() & (vec_out["ev"].astype(float) >= float(min_ev))]
    # Choose best team: prefer previously resolved _team_final; else fallback to input team or map
    if "_team_final" in vec_out.columns:
        vec_out["team_final"] = vec_out["_team_final"]
    else:
        vec_out["team_final"] = vec_out.get("team")
        try:
            missing_team = vec_out["team_final"].isna() | (vec_out["team_final"].astype(str).str.strip() == "")
            vec_out.loc[missing_team, "team_final"] = vec_out.loc[missing_team, "player_norm"].map(lambda nm: player_team_map.get(nm))
        except Exception:
            pass
    out_vec = vec_out.assign(
        date=date,
        player=lambda df: df["player_display"],
        market=lambda df: df["market"],
        line=lambda df: df["line_num"],
        proj=lambda df: df["proj_lambda"].astype(float).round(3),
        p_over=lambda df: df["p_over_vec"].astype(float).round(4),
        over_price=lambda df: df.get("over_price"),
        under_price=lambda df: df.get("under_price"),
        book=lambda df: df.get("book"),
        team=lambda df: df["team_final"],
    )[[
        "date","player","team","market","line","proj","p_over","over_price","under_price","book","side","ev"
    ]]
    # Fallback: row-wise compute for rows missing proj_lambda (should be small)
    remain = merged[~vec_mask]
    _dbg(f"fallback row-wise count: {len(remain)}")
    rows_fallback = []
    for _, rr in remain.iterrows():
        m = str(rr.get("market") or "").upper()
        player = rr.get("player_display")
        ln = rr.get("line_num")
        if pd.isna(ln):
            continue
        op = rr.get("over_price"); up = rr.get("under_price")
        if pd.isna(op) and pd.isna(up):
            continue
        # Skip non-slate teams if we resolved a team and have a slate set
        try:
            if slate_abbrs:
                tf = rr.get("_team_final") or rr.get("team") or player_team_map.get(_norm_name(str(player)).lower())
                if tf and str(tf).strip() and str(tf).strip().upper() not in slate_abbrs:
                    continue
                if allowed_names:
                    nm_ok = _norm_name(str(player)).lower() in allowed_names
                    if not nm_ok:
                        continue
        except Exception:
            pass
        lam, p_over = proj_and_prob(m, str(player), float(ln))
        if p_over is not None:
            try:
                p_over = _apply_calibrated(m, float(ln), float(p_over)) if cal_map else p_over
            except Exception:
                pass
        if lam is None or p_over is None:
            continue
        ev_o = ev_unit(float(p_over), american_to_decimal(float(op))) if pd.notna(op) else None
        p_under = max(0.0, 1.0 - float(p_over))
        ev_u = ev_unit(float(p_under), american_to_decimal(float(up))) if pd.notna(up) else None
        side = None; ev = None
        if ev_o is not None or ev_u is not None:
            if (ev_u is None) or (ev_o is not None and ev_o >= ev_u):
                side = "Over"; ev = ev_o
            else:
                side = "Under"; ev = ev_u
        # Apply per-market or global min_ev
        thr_map = _thr_map if _thr_map else {}
        thr_val = float(thr_map.get(m, float(min_ev)))
        if ev is None or float(ev) < thr_val:
            continue
        # Prefer map over line for consistency
        team_val = rr.get("_team_from_map") or rr.get("_team_final") or rr.get("team") or player_team_map.get(_norm_name(player).lower())
        rows_fallback.append({
            "date": date,
            "player": player,
            "team": team_val or None,
            "market": m,
            "line": float(ln),
            "proj": round(float(lam), 3),
            "p_over": round(float(p_over), 4),
            "over_price": op if pd.notna(op) else None,
            "under_price": up if pd.notna(up) else None,
            "book": rr.get("book"),
            "side": side,
            "ev": round(float(ev), 4) if ev is not None else None,
        })
    out = pd.concat([out_vec, pd.DataFrame(rows_fallback)], ignore_index=True) if not out_vec.empty or rows_fallback else pd.DataFrame()
    if not out.empty:
        # Normalize column names at write-time
        if 'proj' in out.columns and 'proj_lambda' not in out.columns:
            out['proj_lambda'] = out['proj']
        if 'ev' in out.columns and 'ev_over' not in out.columns:
            out['ev_over'] = out['ev']
        # Primary sort now by ev_over if present
        sort_col = 'ev_over' if 'ev_over' in out.columns else ('ev' if 'ev' in out.columns else None)
        if sort_col:
            out = out.sort_values(sort_col, ascending=False)
        out = out.head(top)
    out_path = PROC_DIR / f"props_recommendations_{date}.csv"
    save_df(out, out_path)
    dt = round(time.monotonic() - t0, 2)
    print(f"Wrote {out_path} with {len(out)} rows (normalized cols: {', '.join([c for c in ['proj_lambda','ev_over'] if c in out.columns])}); took {dt}s")


@app.command(name="props-collect")
def props_collect(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    source: str = typer.Option("oddsapi", help="Source to collect: oddsapi"),
):
    """Collect player props lines for a date from a single source and write canonical Parquet/CSV.

    Prints rows and output path, returns quickly. Use this to run step 1.
    """
    import time
    from .data import player_props as _pp
    t0 = time.monotonic()
    src = source.strip().lower()
    if src != "oddsapi":
        print("Unsupported source. Only 'oddsapi' is supported now."); raise typer.Exit(code=1)
    cfg = _pp.PropsCollectionConfig(output_root="data/props", book="oddsapi", source="oddsapi")
    res = _pp.collect_and_write(date, roster_df=None, cfg=cfg)
    dt = round(time.monotonic() - t0, 2)
    print(f"[collect:{src}] raw={res.get('raw_count')} combined={res.get('combined_count')} path={res.get('output_path')} ({dt}s)")


@app.command(name="props-verify")
def props_verify(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
):
    """Print counts for canonical lines files for the given date."""
    base = Path("data/props") / f"player_props_lines/date={date}"
    found = []
    for fn in ("oddsapi.parquet","bovada.parquet","oddsapi.csv","bovada.csv"):
        p = base / fn
        if p.exists():
            try:
                df = pd.read_parquet(p, engine="pyarrow") if p.suffix == ".parquet" else pd.read_csv(p)
                found.append((fn, len(df)))
            except Exception as e:
                found.append((fn, f"error: {e}"))
    if not found:
        print("No canonical files found for", date)
    else:
        for name, n in found:
            print(f"{name}: {n}")


@app.command(name="props-fast")
def props_fast(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    min_ev: float = typer.Option(0.0, help="Minimum EV threshold for recommendations"),
    top: int = typer.Option(400, help="Top N recommendations to keep"),
    market: str = typer.Option("", help="Optional market filter: SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS"),
):
    """Fast daily props pipeline: OddsAPI-only, projections_all, recommendations.

    Writes:
      - data/processed/props_projections_all_{date}.csv
      - data/processed/props_recommendations_{date}.csv
    """
    import time, json
    from .data import player_props as props_data
    t0 = time.monotonic(); timings = {}
    base = Path("data/props") / f"player_props_lines/date={date}"
    base.mkdir(parents=True, exist_ok=True)
    # Collect OddsAPI
    cnt = 0
    try:
        t1 = time.monotonic()
        cfg_odds = props_data.PropsCollectionConfig(output_root=str(base.parent.parent), book="oddsapi", source="oddsapi")
        res = props_data.collect_and_write(date, roster_df=None, cfg=cfg_odds)
        cnt = int(res.get("combined_count") or 0)
        timings['collect_oddsapi_sec'] = round(time.monotonic() - t1, 3)
    except Exception as e:
        print("[fast] oddsapi collection failed:", e)
    # Projections (all markets) — optional, default skip for speed unless forced
    import os as _os
    if str(_os.getenv("PROPS_SKIP_PROJECTIONS", "1")).strip().lower() not in ("1","true","yes"):
        try:
            t3 = time.monotonic()
            props_project_all.callback(date=date, ensure_history_days=365, include_goalies=True)
            timings['projections_all_sec'] = round(time.monotonic() - t3, 3)
        except Exception:
            try:
                t3b = time.monotonic()
                props_project_all(date=date, ensure_history_days=365, include_goalies=True)
                timings['projections_all_sec'] = round(time.monotonic() - t3b, 3)
            except Exception as e:
                print("[fast] projections failed:", e)
                timings['projections_error'] = str(e)
    else:
        timings['projections_skipped'] = True
    # Recommendations
    try:
        t4 = time.monotonic()
        props_recommendations.callback(date=date, min_ev=min_ev, top=top, market=market)
        timings['recommendations_sec'] = round(time.monotonic() - t4, 3)
    except BaseException as e:
        try:
            t4b = time.monotonic()
            props_recommendations(date=date, min_ev=min_ev, top=top, market=market)
            timings['recommendations_sec'] = round(time.monotonic() - t4b, 3)
        except BaseException as e2:
            import traceback
            print('[fast] recommendations failed:', e2)
            traceback.print_exc()
            timings['recommendations_error'] = str(e2)
    timings['total_sec'] = round(time.monotonic() - t0, 3)
    try:
        out = PROC_DIR / f"props_timing_{date}.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(timings, f, indent=2)
        print('[fast] timings:', timings, '\n[fast] wrote timings to', out)
    except Exception:
        print('[fast] timings:', timings)
    print('[fast] done')

# Alias
@app.command(name="props_fast")
def props_fast_alias(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    min_ev: float = typer.Option(0.0, help="Minimum EV threshold for recommendations"),
    top: int = typer.Option(400, help="Top N recommendations to keep"),
    market: str = typer.Option("", help="Optional market filter: SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS"),
):
    return props_fast(date=date, min_ev=min_ev, top=top, market=market)


@app.command(name="props-simulate")
def props_simulate(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    markets: str = typer.Option("SOG,GOALS,ASSISTS,POINTS,SAVES,BLOCKS", help="Comma-separated markets to simulate"),
    n_sims: int = typer.Option(20000, help="Number of Monte Carlo simulations"),
    sim_shared_k: float = typer.Option(1.0, help="Shared Gamma pace shape (mean=1, var=1/k)"),
    props_xg_gamma: float = typer.Option(0.02, help="Team xGF/60 impact on per-player lambda"),
    props_penalty_gamma: float = typer.Option(0.06, help="Opponent penalties committed per60 impact (PP exposure)"),
    props_goalie_form_gamma: float = typer.Option(0.02, help="Opponent goalie sv% (L10) dampening for GOALS/ASSISTS/POINTS"),
    props_refs_gamma: float = typer.Option(0.0, help="Referee penalty-rate impact (if assignments available) [reserved]"),
    props_strength_gamma: float = typer.Option(0.04, help="Strength-aware adjustment for SAVES using sim PP shot fraction vs league average"),
):
    """Monte Carlo simulate player props using model lambdas + shared game pace + team features.

    Outputs data/processed/props_simulations_{date}.csv with p_over_sim per (player, market, line, book).
    """
    import numpy as _np
    import pandas as pd
    from glob import glob as _glob
    from .web.teams import get_team_assets as _assets
    from .data.nhl_api_web import NHLWebClient as _Web

    def _norm(s: str | None) -> str:
        try:
            return " ".join(str(s or "").split())
        except Exception:
            return str(s or "")
    def _round_half(x: float) -> float:
        try:
            import math as _m
            return _m.floor(float(x) * 2.0 + 1e-6) / 2.0
        except Exception:
            return float(x)

    base_lines_dir = Path("data/props") / f"player_props_lines/date={date}"
    lines_path_parq = base_lines_dir / "oddsapi.parquet"
    lines_path_csv = base_lines_dir / "oddsapi.csv"
    if lines_path_parq.exists():
        try:
            lines = pd.read_parquet(lines_path_parq, engine="pyarrow")
        except Exception:
            import duckdb as _duck
            f_posix = str(lines_path_parq).replace("\\", "/")
            lines = _duck.query(f"SELECT * FROM read_parquet('{f_posix}')").df()
    elif lines_path_csv.exists():
        lines = pd.read_csv(lines_path_csv)
    else:
        print("No canonical props lines found for", date)
        raise typer.Exit(code=0)

    proj_path = PROC_DIR / f"props_projections_all_{date}.csv"
    if not proj_path.exists():
        print("No projections found:", proj_path)
        raise typer.Exit(code=1)
    proj = pd.read_csv(proj_path)
    if proj is None or proj.empty:
        print("Empty projections file:", proj_path)
        raise typer.Exit(code=1)

    def _create_name_variants(name: str) -> list[str]:
        name = (name or "").strip()
        if not name:
            return []
        norm = " ".join(name.split())
        out = [norm.lower()]
        if "." in norm:
            out.append(norm.replace(".", "").lower())
        parts = [p for p in norm.split() if p and not p.endswith(".")]
        if len(parts) >= 2:
            f, l = parts[0], parts[-1]
            ab = f"{f[0].upper()}. {l}"; out += [ab.lower(), ab.replace(".", "").lower(), f"{f[0].upper()} {l}".lower()]
        return list(set(out))

    def _abbr_team(n: str | None) -> str | None:
        if not n:
            return None
        try:
            a = _assets(str(n)) or {}
            return str(a.get("abbr") or "").upper() or None
        except Exception:
            return None

    markets_set = set([m.strip().upper() for m in (markets or "").split(",") if m.strip()])
    if not markets_set:
        markets_set = {"SOG","GOALS","ASSISTS","POINTS","SAVES","BLOCKS"}

    # Build player_norm in lines and attach abbr team guess
    import ast as _ast
    def _norm_player(x):
        s = str(x or "").strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = _ast.literal_eval(s)
                v = d.get("default") or d.get("name")
                if isinstance(v, str):
                    s = v.strip()
            except Exception:
                pass
        return " ".join(s.split())
    lines = lines.copy()
    name_cols = [c for c in ["player_name","player","name","display_name"] if c in lines.columns]
    if name_cols:
        s = lines[name_cols[0]].copy()
        for c in name_cols[1:]:
            try:
                s = s.where(s.notna() & (s.astype(str).str.strip() != ""), lines[c])
            except Exception:
                continue
        lines["player_display"] = s.map(_norm_player)
    else:
        lines["player_display"] = lines.index.map(lambda _: "")
    lines["market"] = lines.get("market").astype(str).str.upper()
    lines = lines[lines["market"].isin(markets_set)]
    lines["line_num"] = pd.to_numeric(lines.get("line"), errors="coerce")
    lines = lines[lines["line_num"].notna()].copy()
    def _for_join(name: str) -> str:
        v = _create_name_variants(name)
        if not v: return ""
        dv = [x for x in v if "." in x or (len(x.split())==2 and len(x.split()[0])==1)]
        return min(dv, key=len) if dv else min(v, key=len)
    lines["player_norm"] = lines["player_display"].map(_for_join)

    # Roster-based team abbr resolution similar to props_recommendations
    player_team_map: dict[str, str] = {}
    try:
        roster_cache = PROC_DIR / f"roster_{date}.csv"
        if roster_cache.exists():
            rc = pd.read_csv(roster_cache)
            if rc is not None and not rc.empty:
                name_col = "full_name" if "full_name" in rc.columns else ("player" if "player" in rc.columns else None)
                team_col = None
                for c in ("team","team_abbr","team_abbrev","teamAbbrev","team_abbreviation"):
                    if c in rc.columns: team_col=c; break
                if name_col and team_col:
                    for _, rr in rc.dropna(subset=[name_col, team_col]).iterrows():
                        nm = _norm(str(rr.get(name_col)))
                        tm = str(rr.get(team_col)).strip().upper()
                        if nm and tm:
                            for v in _create_name_variants(nm):
                                player_team_map[v.lower()] = tm
    except Exception:
        pass
    lines["team_abbr"] = lines.get("team")
    try:
        lines["team_abbr"] = lines["team_abbr"].map(lambda t: str(t).strip().upper() if pd.notna(t) else None)
        missing = lines["team_abbr"].isna() | (lines["team_abbr"].astype(str).str.strip()=="")
        lines.loc[missing, "team_abbr"] = lines.loc[missing, "player_norm"].map(lambda nm: player_team_map.get(str(nm).lower()))
    except Exception:
        pass

    # Opponent mapping via schedule
    abbr_to_opp: dict[str, str] = {}
    try:
        web = _Web(); sched = web.schedule_day(date)
        names=set(); games=[]
        for g in sched:
            h=str(getattr(g, "home", "")); a=str(getattr(g, "away", ""))
            games.append((_abbr_team(h), _abbr_team(a)))
        for h,a in games:
            if h and a:
                abbr_to_opp[h]=a; abbr_to_opp[a]=h
    except Exception:
        abbr_to_opp = {}
    lines["opp_abbr"] = lines["team_abbr"].map(lambda t: abbr_to_opp.get(str(t).upper()) if pd.notna(t) else None)

    # Load team features
    xg_path = PROC_DIR / "team_xg_latest.csv"
    xg_map = {}
    if xg_path.exists():
        try:
            _xg = pd.read_csv(xg_path)
            if not _xg.empty and {"abbr","xgf60"}.issubset(_xg.columns):
                xg_map = {str(r.abbr).upper(): float(r.xgf60) for _, r in _xg.iterrows()}
        except Exception:
            xg_map = {}
    league_xg = float(_np.mean(list(xg_map.values()))) if xg_map else 2.6
    pen_path = PROC_DIR / "team_penalty_rates.json"
    pen_comm = {}
    if pen_path.exists():
        try:
            pen_comm = json.loads(pen_path.read_text(encoding="utf-8"))
        except Exception:
            pen_comm = {}
    league_pen = float(_np.mean([float(v.get("committed_per60", 0.0)) for v in pen_comm.values()])) if pen_comm else 3.0
    # Goalie form (sv% L10) for opponent adjustment on scoring markets
    from datetime import date as _date
    gf_today = PROC_DIR / f"goalie_form_{_date.today().strftime('%Y-%m-%d')}.csv"
    gf_map = {}
    if gf_today.exists():
        try:
            _gf = pd.read_csv(gf_today)
            if not _gf.empty and {"team","sv_pct_l10"}.issubset(_gf.columns):
                gf_map = {str(r.team).upper(): float(r.sv_pct_l10) for _, r in _gf.iterrows()}
        except Exception:
            gf_map = {}
    league_sv = float(_np.mean(list(gf_map.values()))) if gf_map else 0.905

    # Load possession events to derive PP shot fractions (optional)
    # - opp_pp_frac_map: opponent PP fraction of shots (useful for SAVES, BLOCKS exposure)
    # - team_pp_frac_map: team PP fraction of own shots (useful for GOALS/ASSISTS/POINTS exposure)
    opp_pp_frac_map: dict[str, float] = {}
    team_pp_frac_map: dict[str, float] = {}
    league_pp_frac = 0.18
    try:
        ev_path = PROC_DIR / f"sim_events_pos_{date}.csv"
        if ev_path.exists():
            ev = pd.read_csv(ev_path)
            # Build abbr mapping via schedule
            web = _Web(); sched = web.schedule_day(date)
            def _abbr(n: str | None) -> str | None:
                try:
                    a = _assets(str(n)) or {}
                    return str(a.get("abbr") or "").upper() or None
                except Exception:
                    return None
            for _, r in ev.iterrows():
                h = str(r.get("home") or ""); a = str(r.get("away") or "")
                h_ab = _abbr(h); a_ab = _abbr(a)
                if not h_ab or not a_ab:
                    continue
                # Team PP fraction (own shots)
                sh_home_total = float(r.get("shots_ev_home", 0)) + float(r.get("shots_pp_home", 0)) + float(r.get("shots_pk_home", 0))
                sh_home_pp = float(r.get("shots_pp_home", 0))
                team_pp_home = (sh_home_pp / sh_home_total) if sh_home_total > 0 else None
                if team_pp_home is not None:
                    team_pp_frac_map[h_ab] = float(team_pp_home)
                sh_away_total = float(r.get("shots_ev_away", 0)) + float(r.get("shots_pp_away", 0)) + float(r.get("shots_pk_away", 0))
                sh_away_pp = float(r.get("shots_pp_away", 0))
                team_pp_away = (sh_away_pp / sh_away_total) if sh_away_total > 0 else None
                if team_pp_away is not None:
                    team_pp_frac_map[a_ab] = float(team_pp_away)
                # Opponent PP fraction (opponent shots)
                opp_pp_frac_home = (sh_away_pp / sh_away_total) if sh_away_total > 0 else None
                if opp_pp_frac_home is not None:
                    opp_pp_frac_map[h_ab] = float(opp_pp_frac_home)
                opp_pp_frac_away = (sh_home_pp / sh_home_total) if sh_home_total > 0 else None
                if opp_pp_frac_away is not None:
                    opp_pp_frac_map[a_ab] = float(opp_pp_frac_away)
            # Update league average from observed (use team PP fractions)
            vals = [v for v in team_pp_frac_map.values() if v is not None]
            if vals:
                league_pp_frac = float(_np.mean(vals))
    except Exception:
        opp_pp_frac_map = {}
        team_pp_frac_map = {}

    # Join projections (lambda per player+market)
    proj = proj.copy()
    proj["player_norm"] = proj["player"].astype(str).map(lambda s: min(_create_name_variants(s) or [s], key=len))
    proj["market"] = proj["market"].astype(str).str.upper()
    merged = lines.merge(proj[["player_norm","market","proj_lambda"]], on=["player_norm","market"], how="left")
    merged = merged[merged["proj_lambda"].notna()].copy()

    def _multiplier(row) -> float:
        return compute_props_lam_scale_mean(
            market=str(row.get("market")),
            team_abbr=row.get("team_abbr"),
            opp_abbr=row.get("opp_abbr"),
            league_xg=league_xg,
            xg_map=xg_map,
            league_pen=league_pen,
            pen_comm=pen_comm,
            league_sv=league_sv,
            gf_map=gf_map,
            league_pp_frac=league_pp_frac,
            opp_pp_frac_map=opp_pp_frac_map,
            team_pp_frac_map=team_pp_frac_map,
            props_xg_gamma=float(props_xg_gamma),
            props_penalty_gamma=float(props_penalty_gamma),
            props_goalie_form_gamma=float(props_goalie_form_gamma),
            props_strength_gamma=float(props_strength_gamma),
        )

    merged["lam_scale_mean"] = merged.apply(_multiplier, axis=1).astype(float)

    # Build game-level shared pace groups by (team,opp)
    merged["grp"] = merged.apply(lambda r: ":".join(sorted([str(r.get("team_abbr") or "").upper(), str(r.get("opp_abbr") or "").upper()])), axis=1)
    groups = {g: i for i, g in enumerate(sorted(merged["grp"].unique()))}
    merged["grp_id"] = merged["grp"].map(groups)

    rs = _np.random.RandomState(42)
    shape = float(sim_shared_k) if float(sim_shared_k) > 0 else 1.0
    scale = 1.0 / shape
    grp_ids = sorted(groups.values())
    pace_draws = _np.ones((len(grp_ids), n_sims), dtype=_np.float32)
    if shape > 0:
        pace_draws = rs.gamma(shape=shape, scale=scale, size=(len(grp_ids), n_sims)).astype(_np.float32)

    out_rows = []
    # Iterate row-wise to keep memory bounded; vectorization across sims via numpy arrays
    for idx, rr in merged.iterrows():
        mk = str(rr["market"]).upper(); ln = float(rr["line_num"]); lam0 = float(rr["proj_lambda"]); mmean = float(rr["lam_scale_mean"])
        if lam0 <= 0 or mmean <= 0:
            continue
        lam_eff = lam0 * mmean
        g = int(rr["grp_id"]) if pd.notna(rr.get("grp_id")) else 0
        lam_arr = lam_eff * pace_draws[g]
        # Discrete line handling
        k_line = int(_np.floor(ln + 1e-9))
        # Approximate via survival function of Poisson per draw using normal approximation is poor; sample directly
        # For speed, compute tail with CDF at k_line using vectorized poisson.sf
        try:
            p_over = _poisson.sf(k_line, mu=lam_arr).mean()
        except Exception:
            # Fallback sampling
            y = rs.poisson(lam_arr)
            p_over = float((y > k_line).mean())
        out_rows.append({
            "date": date,
            "player": rr.get("player_display"),
            "team": rr.get("team_abbr"),
            "opp": rr.get("opp_abbr"),
            "market": mk,
            "line": float(ln),
            "over_price": rr.get("over_price"),
            "under_price": rr.get("under_price"),
            "book": rr.get("book"),
            "proj_lambda": round(lam0, 4),
            "lam_scale_mean": round(mmean, 4),
            "p_over_sim": round(float(p_over), 6),
            "n_sims": int(n_sims),
        })
    out = pd.DataFrame(out_rows)
    out_path = PROC_DIR / f"props_simulations_{date}.csv"
    save_df(out, out_path)
    print(f"[props-sim] wrote {out_path} rows={len(out)}")


@app.command(name="props-simulate-boxscores")
def props_simulate_boxscores(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    n_sims: int = typer.Option(5000, help="Number of play-level simulations per game"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    write_samples: bool = typer.Option(True, help="Also write per-sim totals samples for p_over computation"),
):
    """Generate period and game-level simulated player boxscores from the play-level sim engine.

    Writes data/processed/props_boxscores_sim_{date}.csv (per-period and totals per player).
    """
    import pandas as pd
    from .data.nhl_api_web import NHLWebClient
    from .web.teams import get_team_assets
    from .sim.engine import GameSimulator, SimConfig
    from .sim.models import RateModels
    from .sim.props_boxscore import aggregate_events_to_boxscores
    # Build slate via schedule
    try:
        web = NHLWebClient()
        games = web.schedule_day(date)
    except Exception:
        games = []
    if not games:
        print("No games on slate for", date)
        raise typer.Exit(code=0)
    # Helper: fetch roster for team name
    from .data.rosters import fetch_current_roster as _fetch
    def _roster_for_name(nm: str) -> list[dict]:
        rows: list[dict] = []
        # Try Web API current roster via team abbreviation
        try:
            abbr = (get_team_assets(nm).get('abbr') or '').upper()
        except Exception:
            abbr = ''
        players = []
        if abbr:
            try:
                players = _fetch(abbr)
            except Exception:
                players = []
            for p in players or []:
                rows.append({
                    'player_id': int(getattr(p, 'player_id', 0) or 0),
                    'full_name': str(getattr(p, 'full_name', '') or ''),
                    'position': str(getattr(p, 'position', '') or ''),
                    'proj_toi': float(getattr(p, 'avg_toi', 0.0) or 0.0),
                })
        # Fallback: historical enrichment
        if not rows:
            try:
                from .data import player_props as _pp
                enrich = _pp._build_roster_enrichment()
            except Exception:
                enrich = None
            if enrich is not None and not enrich.empty:
                for _, rr in enrich.iterrows():
                    tm = str(rr.get('team') or rr.get('team_abbr') or '').upper()
                    if abbr and tm != abbr:
                        continue
                    pos_raw = str(rr.get('position') or '')
                    pos = 'G' if pos_raw.upper().startswith('G') else ('D' if pos_raw.upper().startswith('D') else 'F')
                    rows.append({
                        'player_id': int(rr.get('player_id') or 0),
                        'full_name': str(rr.get('full_name') or rr.get('player') or ''),
                        'position': pos,
                        'proj_toi': float(rr.get('avg_toi') or 0.0),
                    })
        return rows
    # Aggregate across games and sims
    agg_all: list[pd.DataFrame] = []
    samples_all: list[pd.DataFrame] = []
    for g in games:
        home = str(getattr(g, 'home', ''))
        away = str(getattr(g, 'away', ''))
        if not home or not away:
            continue
        roster_home = _roster_for_name(home)
        roster_away = _roster_for_name(away)
        if not roster_home or not roster_away:
            continue
        # Simple baseline rates per game; future: inject team-specific rates
        rates = RateModels.baseline()
        cfg = SimConfig(periods=3, seed=seed)
        sim = GameSimulator(cfg=cfg, rates=rates)
        # Run sims and aggregate per-sim boxscores
        df_parts = []
        for i in range(int(n_sims)):
            gs, ev = sim.simulate_with_lineups(home_name=home, away_name=away, roster_home=roster_home, roster_away=roster_away, lineup_home=[], lineup_away=[])
            df_i = aggregate_events_to_boxscores(gs, ev)
            # Attach player names for convenience
            name_map = { int(p['player_id']): str(p['full_name']) for p in (roster_home + roster_away) if p.get('player_id') }
            df_i['player'] = df_i['player_id'].map(lambda pid: name_map.get(int(pid)))
            df_i['game_home'] = home; df_i['game_away'] = away; df_i['date'] = date
            # Collect per-sim totals (period=0) in long format for markets
            if write_samples:
                d0 = df_i[df_i['period'] == 0].copy()
                if not d0.empty:
                    d0['sim_idx'] = int(i)
                    long = d0.melt(
                        id_vars=['team','player_id','player','game_home','game_away','date','period','sim_idx'],
                        value_vars=['shots','goals','assists','points','blocks','saves'],
                        var_name='market_raw', value_name='value'
                    )
                    long['market'] = long['market_raw'].astype(str).str.upper().map({
                        'SHOTS':'SOG', 'GOALS':'GOALS', 'ASSISTS':'ASSISTS', 'POINTS':'POINTS', 'BLOCKS':'BLOCKS', 'SAVES':'SAVES'
                    }).fillna(long['market_raw'].astype(str).str.upper())
                    samples_all.append(long[['team','player_id','player','market','value','sim_idx','game_home','game_away','date']])
            df_parts.append(df_i)
        # Average over sims
        if df_parts:
            df_sum = pd.concat(df_parts, ignore_index=True)
            grp_cols = ['team','player_id','period','player','game_home','game_away','date']
            df_sum = df_sum.groupby(grp_cols, as_index=False)[['shots','goals','assists','points','blocks','saves','toi_sec']].sum()
            df_sum[['shots','goals','assists','points','blocks','saves','toi_sec']] = df_sum[['shots','goals','assists','points','blocks','saves','toi_sec']].astype(float) / float(n_sims)
            agg_all.append(df_sum)
    if not agg_all:
        print("No aggregated boxscores generated.")
        raise typer.Exit(code=0)
    out = pd.concat(agg_all, ignore_index=True)
    out_path = PROC_DIR / f"props_boxscores_sim_{date}.csv"
    save_df(out, out_path)
    print(f"[props-sim-boxscores] wrote {out_path} rows={len(out)}")
    if write_samples and samples_all:
        samp = pd.concat(samples_all, ignore_index=True)
        samp_path = PROC_DIR / f"props_boxscores_sim_samples_{date}.parquet"
        try:
            import pyarrow as _pa  # noqa: F401
            samp.to_parquet(samp_path, engine='pyarrow')
        except Exception:
            # fallback to CSV if parquet fails
            samp_path = PROC_DIR / f"props_boxscores_sim_samples_{date}.csv"
            save_df(samp, samp_path)
        print(f"[props-sim-boxscores] wrote samples {samp_path}")


@app.command(name="props-precompute-all")
def props_precompute_all(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
):
    """Compute model-only projections (lambda) for all rostered players on the slate and write props_projections_all_{date}.csv.

    This mirrors web app behavior without requiring FastAPI. Uses historical player game stats and current rosters.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    from .data.nhl_api_web import NHLWebClient
    from .web.teams import get_team_assets
    from .models.props import SkaterShotsModel, GoalieSavesModel, SkaterGoalsModel, SkaterAssistsModel, SkaterPointsModel, SkaterBlocksModel
    # Ensure stats history exists (best effort)
    try:
        from .data.collect import collect_player_game_stats as _collect_stats
        start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
        stats_path = RAW_DIR / "player_game_stats.csv"
        need = (not stats_path.exists()) or (getattr(stats_path.stat(), "st_size", 0) == 0)
        if need:
            try:
                _collect_stats(start, date, source="web")
            except Exception:
                _collect_stats(start, date, source="stats")
        try:
            hist = pd.read_csv(stats_path)
        except Exception:
            hist = pd.DataFrame()
    except Exception:
        hist = pd.DataFrame()
    # Slate teams via Web API
    try:
        web = NHLWebClient()
        games = web.schedule_day(date)
    except Exception:
        games = []
    slate_names = set()
    for g in games or []:
        slate_names.add(str(g.home)); slate_names.add(str(g.away))
    slate_abbrs = set()
    for nm in slate_names:
        ab = (get_team_assets(str(nm)).get('abbr') or '').upper()
        if ab:
            slate_abbrs.add(ab)
    # Try live roster; fallback to historical enrichment
    roster_df = pd.DataFrame()
    try:
        from .data.rosters import list_teams as _list_teams, fetch_current_roster as _fetch
        teams = _list_teams()
        name_to_id = { str(t.get('name') or '').strip().lower(): int(t.get('id')) for t in teams }
        id_to_abbr = { int(t.get('id')): str(t.get('abbreviation') or '').upper() for t in teams }
        rows = []
        for nm in sorted(slate_names):
            tid = name_to_id.get(str(nm).strip().lower())
            if not tid:
                continue
            try:
                players = _fetch(tid)
            except Exception:
                players = []
            for p in players:
                rows.append({ 'player_id': p.player_id, 'player': p.full_name, 'position': p.position, 'team': id_to_abbr.get(tid) })
        roster_df = pd.DataFrame(rows)
    except Exception:
        roster_df = pd.DataFrame()
    if roster_df is None or roster_df.empty:
        # Historical enrichment
        try:
            from .data import player_props as _pp
            enrich = _pp._build_roster_enrichment()
        except Exception:
            enrich = pd.DataFrame()
        if enrich is None or enrich.empty:
            save_df(pd.DataFrame(columns=["date","player","team","position","market","proj_lambda"]), PROC_DIR / f"props_projections_all_{date}.csv")
            print(f"[props-precompute] wrote empty projections for {date}")
            return
        # Robust team abbreviation mapping: handle full names and existing abbreviations
        abbr_map_name: dict[str, str] = {}
        abbr_map_abbr: dict[str, str] = {}
        try:
            from .data.rosters import list_teams as _list_teams
            _teams = _list_teams()
            for t in _teams:
                try:
                    nm = str(t.get('name') or '').strip().lower()
                    ab = str(t.get('abbreviation') or '').strip().upper()
                    if nm:
                        abbr_map_name[nm] = ab
                    if ab:
                        abbr_map_abbr[ab] = ab
                except Exception:
                    pass
        except Exception:
            pass
        def _to_abbr(x):
            s = str(x or '').strip()
            if not s:
                return None
            try:
                # If already an abbreviation, normalize directly
                su = s.upper()
                if su in abbr_map_abbr:
                    return abbr_map_abbr[su]
                # Try name mapping
                sl = s.lower()
                if sl in abbr_map_name:
                    return abbr_map_name[sl]
                # Fallback to assets lookup
                a = get_team_assets(s).get('abbr')
                return str(a).upper() if a else None
            except Exception:
                return None
        enrich = enrich.copy(); enrich['team_abbr'] = enrich['team'].map(_to_abbr)
        # Infer position from historical if available
        pos_map = {}
        try:
            if hist is not None and not hist.empty and {'player','primary_position'}.issubset(hist.columns):
                tmp = hist.dropna(subset=['player']).copy()
                tmp['player'] = tmp['player'].astype(str)
                last_pos = tmp.dropna(subset=['primary_position']).groupby('player')['primary_position'].last()
                pos_map = {k: v for k, v in last_pos.items() if isinstance(k, str)}
        except Exception:
            pos_map = {}
        rows = []
        for _, rr in enrich.iterrows():
            ab = rr.get('team_abbr')
            # If slate teams are known, only filter when we have a known abbr; otherwise include
            if slate_abbrs and (ab is not None) and (ab not in slate_abbrs):
                continue
            nm = rr.get('full_name')
            pos_raw = pos_map.get(str(nm), '')
            pos = 'G' if str(pos_raw).upper().startswith('G') else ('D' if str(pos_raw).upper().startswith('D') else 'F')
            rows.append({'player_id': rr.get('player_id'), 'player': nm, 'position': pos, 'team': ab})
        roster_df = pd.DataFrame(rows)
    if roster_df is None or roster_df.empty:
        save_df(pd.DataFrame(columns=["date","player","team","position","market","proj_lambda"]), PROC_DIR / f"props_projections_all_{date}.csv")
        print(f"[props-precompute] wrote empty projections for {date}")
        return
    shots = SkaterShotsModel(); saves = GoalieSavesModel(); goals = SkaterGoalsModel(); assists = SkaterAssistsModel(); points = SkaterPointsModel(); blocks = SkaterBlocksModel()
    def _clean_player_display_name(s: str) -> str:
        try:
            x = str(s or '').strip()
            return ' '.join(x.split())
        except Exception:
            return str(s or '')
    out_rows = []
    for _, r in roster_df.iterrows():
        player = _clean_player_display_name(str(r.get('player') or ''))
        pos = str(r.get('position') or '').upper()
        team = r.get('team')
        if not player:
            continue
        try:
            if pos == 'G':
                lam = saves.player_lambda(hist, player)
                out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'SAVES', 'proj_lambda': float(lam) if lam is not None else None})
            else:
                lam = shots.player_lambda(hist, player); out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'SOG', 'proj_lambda': float(lam) if lam is not None else None})
                lam = goals.player_lambda(hist, player); out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'GOALS', 'proj_lambda': float(lam) if lam is not None else None})
                lam = assists.player_lambda(hist, player); out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'ASSISTS', 'proj_lambda': float(lam) if lam is not None else None})
                lam = points.player_lambda(hist, player); out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'POINTS', 'proj_lambda': float(lam) if lam is not None else None})
                lam = blocks.player_lambda(hist, player); out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'BLOCKS', 'proj_lambda': float(lam) if lam is not None else None})
        except Exception:
            continue
    df = pd.DataFrame(out_rows)
    try:
        if not df.empty:
            df = df.sort_values(['team','position','player','market'])
    except Exception:
        pass
    out_path = PROC_DIR / f"props_projections_all_{date}.csv"
    save_df(df, out_path)
    print(f"[props-precompute] wrote {out_path} rows={len(df)}")


@app.command(name="props-project-all")
def props_project_all(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    ensure_history_days: int = typer.Option(365, help="Minimum history window to ensure is collected"),
    include_goalies: bool = typer.Option(True, help="Include goalies (SAVES) in projections"),
):
    """Alias for props_precompute_all with compatible options used by scripts/daily_update.ps1."""
    # Currently props_precompute_all always includes goalies and uses ~365 days history; options provided for compatibility
    return props_precompute_all(date=date)


@app.command(name="props-recommendations-sim")
def props_recommendations_sim(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    min_ev: float = typer.Option(0.0, help="Minimum EV threshold for ev_over"),
    top: int = typer.Option(400, help="Top N to keep after sorting by EV desc"),
    min_ev_per_market: str = typer.Option("", help="Optional per-market EV thresholds"),
    min_prob: float = typer.Option(0.0, help="Minimum chosen-side probability threshold (0-1), e.g., 0.60"),
    min_prob_per_market: str = typer.Option("SOG=0.75,GOALS=0.60,ASSISTS=0.60,POINTS=0.60,SAVES=0.60,BLOCKS=0.60", help="Optional per-market probability thresholds, e.g., 'SOG=0.58,GOALS=0.60'"),
):
    """Generate recommendations using simulation-backed p_over if available; falls back to model-only if missing."""
    sim_path = PROC_DIR / f"props_simulations_{date}.csv"
    if not sim_path.exists():
        print("No simulations found; running props-simulate first…")
        try:
            props_simulate(date=date)
        except Exception:
            # If CLI invocation fails in-process, attempt a subprocess call
            import subprocess, sys
            try:
                subprocess.run([sys.executable, "-m", "nhl_betting.cli", "props-simulate", "--date", str(date)], check=False)
            except Exception:
                pass
    df = pd.read_csv(sim_path) if sim_path.exists() else pd.DataFrame()
    if df is None or df.empty:
        print("No simulation results present, aborting.")
        raise typer.Exit(code=0)
    import numpy as _np
    def _american_to_decimal(s):
        try:
            s = float(s)
            return 1.0 + (s/100.0) if s > 0 else 1.0 + (100.0/abs(s))
        except Exception:
            return _np.nan
    df["dec_over"] = df["over_price"].map(_american_to_decimal)
    df["dec_under"] = df["under_price"].map(_american_to_decimal)
    p = pd.to_numeric(df["p_over_sim"], errors="coerce")
    ev_over = p * (df["dec_over"] - 1.0) - (1.0 - p)
    p_under = (1.0 - p).clip(lower=0.0, upper=1.0)
    ev_under = p_under * (df["dec_under"] - 1.0) - (1.0 - p_under)
    over_better = ev_under.isna() | (~ev_over.isna() & (ev_over >= ev_under))
    side = _np.where(over_better, "Over", "Under")
    ev_chosen = _np.where(over_better, ev_over, ev_under)
    # chosen-side probability
    chosen_prob = _np.where(over_better, p, (1.0 - p))
    out = df.assign(
        ev_over=ev_over,
        side=side,
        ev=ev_chosen,
        chosen_prob=chosen_prob,
    )
    def _parse_thresholds(s: str) -> dict:
        d = {}
        for part in (s or "").split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                d[str(k).strip().upper()] = float(str(v).strip())
            except Exception:
                continue
        return d
    thr_map = _parse_thresholds(min_ev_per_market)
    prob_thr_map = _parse_thresholds(min_prob_per_market)
    if thr_map:
        thr_series = out["market"].astype(str).str.upper().map(lambda m: thr_map.get(m, float(min_ev))).astype(float)
        out = out[(out["ev"].notna()) & (out["ev"].astype(float) >= thr_series)]
    else:
        out = out[(out["ev"].notna()) & (out["ev"].astype(float) >= float(min_ev))]

    # Apply probability thresholds
    if prob_thr_map or (float(min_prob) > 0.0):
        prob_series = out["market"].astype(str).str.upper().map(lambda m: prob_thr_map.get(m, float(min_prob))).astype(float)
        out = out[(out["chosen_prob"].notna()) & (out["chosen_prob"].astype(float) >= prob_series)]
    out = out.sort_values("ev", ascending=False).head(int(top))
    final = out[["date","player","team","market","line","proj_lambda","p_over_sim","over_price","under_price","book","side","ev","chosen_prob"]].copy()
    final.rename(columns={"p_over_sim":"p_over"}, inplace=True)
    out_path = PROC_DIR / f"props_recommendations_{date}.csv"
    save_df(final, out_path)
    print(f"[props-recs-sim] wrote {out_path} with {len(final)} rows")


@app.command(name="props-recommendations-boxscores")
def props_recommendations_boxscores(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    min_ev: float = typer.Option(0.0, help="Minimum EV threshold for ev_over"),
    top: int = typer.Option(400, help="Top N to keep after sorting by EV desc"),
    min_ev_per_market: str = typer.Option("", help="Optional per-market EV thresholds"),
    min_prob: float = typer.Option(0.0, help="Minimum chosen-side probability threshold (0-1)"),
    min_prob_per_market: str = typer.Option("", help="Optional per-market probability thresholds, e.g., 'SOG=0.58,GOALS=0.60'"),
):
    """Generate player props recommendations using per-sim totals from play-level boxscore simulation.

    Expects props_boxscores_sim_samples_{date}.{parquet|csv} and canonical props lines under data/props.
    """
    import pandas as pd
    import numpy as _np
    from glob import glob as _glob
    from .web.teams import get_team_assets as _assets
    # Load samples
    samp_path_parq = PROC_DIR / f"props_boxscores_sim_samples_{date}.parquet"
    samp_path_csv = PROC_DIR / f"props_boxscores_sim_samples_{date}.csv"
    if samp_path_parq.exists():
        try:
            samples = pd.read_parquet(samp_path_parq, engine='pyarrow')
        except Exception:
            import duckdb as _duck
            f_posix = str(samp_path_parq).replace('\\', '/')
            samples = _duck.query(f"SELECT * FROM read_parquet('{f_posix}')").df()
    elif samp_path_csv.exists():
        samples = pd.read_csv(samp_path_csv)
    else:
        print("No boxscore samples found for", date)
        raise typer.Exit(code=0)
    if samples is None or samples.empty:
        print("Empty samples; aborting.")
        raise typer.Exit(code=0)
    # Lines load and normalization
    base_lines_dir = Path("data/props") / f"player_props_lines/date={date}"
    lines_path_parq = base_lines_dir / "oddsapi.parquet"
    lines_path_csv = base_lines_dir / "oddsapi.csv"
    if lines_path_parq.exists():
        try:
            lines = pd.read_parquet(lines_path_parq, engine="pyarrow")
        except Exception:
            import duckdb as _duck
            f_posix = str(lines_path_parq).replace("\\", "/")
            lines = _duck.query(f"SELECT * FROM read_parquet('{f_posix}')").df()
    elif lines_path_csv.exists():
        lines = pd.read_csv(lines_path_csv)
    else:
        print("No canonical props lines found for", date)
        raise typer.Exit(code=0)
    def _norm(s: str | None) -> str:
        try:
            return " ".join(str(s or "").split())
        except Exception:
            return str(s or "")
    import ast as _ast
    def _norm_player(x):
        s = str(x or "").strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = _ast.literal_eval(s)
                v = d.get("default") or d.get("name")
                if isinstance(v, str):
                    s = v.strip()
            except Exception:
                pass
        return " ".join(s.split())
    lines = lines.copy()
    name_cols = [c for c in ["player_name","player","name","display_name"] if c in lines.columns]
    if name_cols:
        s = lines[name_cols[0]].copy()
        for c in name_cols[1:]:
            try:
                s = s.where(s.notna() & (s.astype(str).str.strip() != ""), lines[c])
            except Exception:
                continue
        lines["player_display"] = s.map(_norm_player)
    else:
        lines["player_display"] = lines.index.map(lambda _: "")
    lines["market"] = lines.get("market").astype(str).str.upper()
    lines["line_num"] = pd.to_numeric(lines.get("line"), errors="coerce")
    lines = lines[lines["line_num"].notna()].copy()
    # Create simple join on normalized name
    def _create_name_variants(name: str) -> list[str]:
        name = (name or "").strip()
        if not name:
            return []
        norm = " ".join(name.split())
        out = [norm.lower()]
        if "." in norm:
            out.append(norm.replace(".", "").lower())
        parts = [p for p in norm.split() if p and not p.endswith(".")]
        if len(parts) >= 2:
            f, l = parts[0], parts[-1]
            ab = f"{f[0].upper()}. {l}"; out += [ab.lower(), ab.replace(".", "").lower(), f"{f[0].upper()} {l}".lower()]
        return list(set(out))
    def _for_join(name: str) -> str:
        v = _create_name_variants(name)
        return (min(v, key=len) if v else "")
    lines["player_norm"] = lines["player_display"].map(_for_join)
    samples = samples.copy()
    samples["player_norm"] = samples["player"].astype(str).map(lambda s: min(_create_name_variants(s) or [s], key=len))
    # Team abbr for samples
    def _abbr_team(n: str | None) -> str | None:
        if not n:
            return None
        try:
            a = _assets(str(n)) or {}
            return str(a.get("abbr") or "").upper() or None
        except Exception:
            return None
    samples["team_abbr"] = samples["team"].map(_abbr_team)
    # Join samples to lines
    merged = lines.merge(samples, on=["player_norm","market"], how="inner")
    if merged is None or merged.empty:
        print("No joinable samples/lines; aborting.")
        raise typer.Exit(code=0)
    # Compute p_over from sample counts per (player, market, line, book)
    # For each group, proportion of sims with value > line_num
    merged["over_win"] = (merged["value"].astype(float) > merged["line_num"].astype(float)).astype(int)
    grp_cols = ["date","player","team_abbr","market","line_num","book","over_price","under_price"]
    prob = merged.groupby(grp_cols, as_index=False)["over_win"].mean().rename(columns={"over_win":"p_over_sim"})
    # Prices to decimal
    def _american_to_decimal(s):
        try:
            s = float(s)
            return 1.0 + (s/100.0) if s > 0 else 1.0 + (100.0/abs(s))
        except Exception:
            return _np.nan
    prob["dec_over"] = prob["over_price"].map(_american_to_decimal)
    prob["dec_under"] = prob["under_price"].map(_american_to_decimal)
    p = pd.to_numeric(prob["p_over_sim"], errors="coerce")
    ev_over = p * (prob["dec_over"] - 1.0) - (1.0 - p)
    p_under = (1.0 - p).clip(lower=0.0, upper=1.0)
    ev_under = p_under * (prob["dec_under"] - 1.0) - (1.0 - p_under)
    over_better = ev_under.isna() | (~ev_over.isna() & (ev_over >= ev_under))
    side = _np.where(over_better, "Over", "Under")
    ev_chosen = _np.where(over_better, ev_over, ev_under)
    chosen_prob = _np.where(over_better, p, (1.0 - p))
    out = prob.assign(side=side, ev=ev_chosen, chosen_prob=chosen_prob)
    # Thresholds
    def _parse_thresholds(s: str) -> dict:
        d = {}
        for part in (s or "").split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                d[str(k).strip().upper()] = float(str(v).strip())
            except Exception:
                continue
        return d
    thr_map = _parse_thresholds(min_ev_per_market)
    prob_thr_map = _parse_thresholds(min_prob_per_market)
    if thr_map:
        thr_series = out["market"].astype(str).str.upper().map(lambda m: thr_map.get(m, float(min_ev))).astype(float)
        out = out[(out["ev"].notna()) & (out["ev"].astype(float) >= thr_series)]
    else:
        out = out[(out["ev"].notna()) & (out["ev"].astype(float) >= float(min_ev))]
    if prob_thr_map or (float(min_prob) > 0.0):
        prob_series = out["market"].astype(str).str.upper().map(lambda m: prob_thr_map.get(m, float(min_prob))).astype(float)
        out = out[(out["chosen_prob"].notna()) & (out["chosen_prob"].astype(float) >= prob_series)]
    out = out.sort_values("ev", ascending=False).head(int(top))
    final = out.rename(columns={"line_num":"line"})[["date","player","team_abbr","market","line","p_over_sim","over_price","under_price","book","side","ev","chosen_prob"]].copy()
    final.rename(columns={"team_abbr":"team","p_over_sim":"p_over"}, inplace=True)
    out_path = PROC_DIR / f"props_recommendations_{date}.csv"
    save_df(final, out_path)
    print(f"[props-recs-boxscores] wrote {out_path} with {len(final)} rows")

@app.command()
def odds_fetch_historical(
    snapshot: str = typer.Option(..., help="ISO timestamp for snapshot, e.g., 2024-03-01T12:00:00Z"),
    sport: str = typer.Option("icehockey_nhl", help="The Odds API sport key, e.g., icehockey_nhl"),
    regions: str = typer.Option("us", help="Regions to include, e.g., us or us,us2"),
    markets: str = typer.Option("h2h,totals,spreads", help="Comma-separated markets"),
    bookmaker: str = typer.Option("", help="Optional bookmaker key to prefer (e.g., pinnacle, draftkings)"),
    best: bool = typer.Option(False, help="If true, aggregate best odds across all bookmakers in snapshot"),
    out_csv: str = typer.Option("", help="Optional output CSV path; defaults under data/raw/odds_YYYY-MM-DD.csv"),
):
    client = OddsAPIClient()
    snap, headers = client.historical_odds_snapshot(
        sport=sport, snapshot_iso=snapshot, regions=regions, markets=markets, odds_format="american"
    )
    df = normalize_snapshot_to_rows(snap, bookmaker=bookmaker or None, best_of_all=best)
    if df.empty:
        print("No odds found for the given snapshot.")
        return
    date_key = df.iloc[0]["date"] if "date" in df.columns and pd.notna(df.iloc[0]["date"]) else "snapshot"
    out_path = Path(out_csv) if out_csv else (RAW_DIR / f"odds_{date_key}.csv")
    save_df(df, out_path)
    print(df.head())
    print("Saved historical odds to", out_path)


@app.command()
def props_project_all(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    ensure_history_days: int = typer.Option(365, help="Days of player history to ensure before date"),
    include_goalies: bool = typer.Option(True, help="Include goalie SAVES projections"),
    use_nn: bool = typer.Option(True, help="Use neural network models (default: True)"),
):
    """Compute model-only props projections (lambdas) for all rostered players on teams with a game.

    Writes data/processed/props_projections_all_{date}.csv with columns:
      [date, player, team, position, market, proj_lambda]
    
    Uses trained neural network models by default. Set --no-use-nn for rolling averages.
    """
    from datetime import datetime as _dt, timedelta as _td
    # Ensure history (fast check). Allow opt-out via env PROPS_SKIP_HISTORY=1
    stats_path = RAW_DIR / "player_game_stats.csv"
    def _needs_history(p: Path) -> bool:
        try:
            if not p.exists() or p.stat().st_size <= 64:  # tiny/empty file
                return True
            # Read just one data row to avoid loading entire CSV
            sample = pd.read_csv(p, nrows=1)
            return sample is None or sample.empty
        except Exception:
            return True
    import os as _os
    # Default to skipping history backfill for speed unless explicitly opted-in
    if (str(_os.getenv("PROPS_FORCE_HISTORY", "")).strip().lower() in ("1","true","yes")) and _needs_history(stats_path):
        try:
            start = (_dt.strptime(date, "%Y-%m-%d") - _td(days=int(ensure_history_days))).strftime("%Y-%m-%d")
            try:
                collect_player_game_stats(start, date, source="web")
            except Exception:
                collect_player_game_stats(start, date, source="stats")
        except Exception as e:
            print(f"[history] ensure failed: {e}")
    # We'll build any needed last-known positions lazily, filtered to slate roster only, using chunked reads
    # Load stats history now (we ensured presence above). This is critical so all markets get non-null lambdas.
    try:
        hist = load_df(stats_path) if stats_path.exists() else pd.DataFrame()
    except Exception:
        try:
            hist = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
        except Exception:
            hist = pd.DataFrame()
    
    # Pre-parse player names ONCE if using NN models (huge performance boost)
    if use_nn and not hist.empty and "player_name" not in hist.columns:
        import json
        print(f"[nn] Pre-parsing {len(hist)} player names for fast lookups...")
        
        def parse_player_name(p):
            if pd.isna(p):
                return ""
            p_str = str(p)
            if p_str.startswith("{"):
                try:
                    p_dict = json.loads(p_str.replace("'", '"'))
                    return p_dict.get("default", "")
                except:
                    return p_str
            return p_str
        
        hist["player_name"] = hist["player"].apply(parse_player_name)
        print(f"[nn] Player names parsed successfully")
    
    # Get slate teams (Web API)
    web = NHLWebClient()
    games = web.schedule_day(date)
    slate_team_names = set()
    for g in games:
        slate_team_names.add(str(g.home))
        slate_team_names.add(str(g.away))
    # Normalize slate teams to abbreviations
    from .web.teams import get_team_assets as _assets
    slate_abbrs = set()
    for nm in slate_team_names:
        ab = (_assets(str(nm)).get('abbr') or '').upper()
        if ab:
            slate_abbrs.add(ab)
    # Prepare a tiny cache to avoid rebuilding roster repeatedly within a day
    roster_cache = PROC_DIR / f"roster_{date}.csv"
    roster_df = pd.DataFrame()
    if roster_cache.exists():
        try:
            tmp = pd.read_csv(roster_cache)
            # Basic sanity: required columns present and within slate
            if not tmp.empty and {"player","team","position"}.issubset(tmp.columns):
                if slate_abbrs:
                    tmp = tmp[tmp["team"].astype(str).str.upper().isin(slate_abbrs)]
                roster_df = tmp.copy()
        except Exception:
            roster_df = pd.DataFrame()
    # Fast path: derive roster from canonical lines if present (avoids slow Stats API calls)
    if roster_df.empty:
        try:
            base = Path("data/props") / f"player_props_lines/date={date}"
            parts = []
            for fn in ("bovada.parquet", "oddsapi.parquet"):
                p = base / fn
                if p.exists():
                    try:
                        parts.append(pd.read_parquet(p, engine="pyarrow"))
                    except Exception:
                        pass
            # Also support CSV if parquet not available
            if not parts:
                for fn in ("bovada.csv", "oddsapi.csv"):
                    p = base / fn
                    if p.exists():
                        try:
                            parts.append(pd.read_csv(p))
                        except Exception:
                            pass
            if parts:
                lines_df = pd.concat(parts, ignore_index=True)
                # Map to minimal roster columns
                def _abbr_team(t: str | None) -> str | None:
                    if not t:
                        return None
                    try:
                        a = _assets(str(t)) or {}
                        ab = a.get("abbr")
                        return str(ab).upper() if ab else (str(t).strip().upper() if str(t).strip() else None)
                    except Exception:
                        return str(t).strip().upper() if isinstance(t, str) and t.strip() else None
                cur = lines_df.copy()
                # Prefer current rows when marked; otherwise use all
                try:
                    if "is_current" in cur.columns:
                        cur = cur[cur["is_current"] == True]
                except Exception:
                    pass
                cur = cur.dropna(subset=["player_name"]).copy()
                cur["_team_abbr"] = cur.get("team", pd.Series(index=cur.index)).map(_abbr_team)
                # Position inference: use cached positions map to avoid repeated scans
                from .utils.positions_cache import get_positions_map
                pos_cache = PROC_DIR / "props_positions_cache.json"
                targets: set[str] = set()
                try:
                    targets = set(lines_df.dropna(subset=["player_name"])['player_name'].astype(str))
                except Exception:
                    targets = set()
                pos_map = get_positions_map(targets, stats_path, pos_cache) if targets else {}
                # Identify goalies from lines explicitly
                goalie_names = set()
                try:
                    if {"market","player_name"}.issubset(cur.columns):
                        goalie_names = set(cur[cur["market"].astype(str).str.upper() == "SAVES"]["player_name"].astype(str))
                except Exception:
                    goalie_names = set()
                rows_fast = []
                for _, rr in cur.iterrows():
                    nm = str(rr.get("player_name") or "").strip()
                    if not nm:
                        continue
                    tm = rr.get("_team_abbr")
                    if tm and slate_abbrs and str(tm).upper() not in slate_abbrs:
                        # Keep only slate teams
                        continue
                    raw_pos = pos_map.get(nm, "")
                    pos = "G" if (include_goalies and (str(raw_pos).upper().startswith("G") or nm in goalie_names)) else ("D" if str(raw_pos).upper().startswith("D") else "F")
                    rows_fast.append({"player": nm, "position": pos, "team": (str(tm).upper() if isinstance(tm, str) else tm)})
                roster_df = pd.DataFrame(rows_fast).drop_duplicates()
                # Persist cache for next calls
                try:
                    if not roster_df.empty:
                        roster_df.to_csv(roster_cache, index=False)
                except Exception:
                    pass
        except Exception:
            pass
    # Supplement roster with live Stats API rosters (union) to cover players without lines
    try:
        from .data.rosters import list_teams as _list_teams, fetch_current_roster as _fetch_current_roster
        teams = _list_teams()
        name_to_id = { str(t.get('name') or '').strip().lower(): int(t.get('id')) for t in teams }
        id_to_abbr = { int(t.get('id')): str(t.get('abbreviation') or '').upper() for t in teams }
        rows_live = []
        for nm in sorted(slate_team_names):
            tid = name_to_id.get(str(nm).strip().lower())
            if not tid:
                continue
            try:
                players = _fetch_current_roster(tid)
            except Exception:
                players = []
            for p in players:
                rows_live.append({
                    'player_id': p.player_id,
                    'player': p.full_name,
                    'position': p.position,
                    'team': id_to_abbr.get(tid),
                })
        live_df = pd.DataFrame(rows_live)
        if roster_df.empty:
            roster_df = live_df
        else:
            try:
                roster_df = pd.concat([roster_df, live_df], ignore_index=True)
                # Normalize and drop duplicates by player+team
                if 'player' in roster_df.columns:
                    roster_df['player'] = roster_df['player'].astype(str).str.strip()
                if 'team' in roster_df.columns:
                    roster_df['team'] = roster_df['team'].astype(str).str.upper()
                roster_df = roster_df.drop_duplicates(subset=['player','team'])
            except Exception:
                pass
    except Exception:
        # Silently use historical fallback if live roster fetch fails
        try:
            from .data import player_props as _pp
            enrich = _pp._build_roster_enrichment()
        except Exception:
            enrich = pd.DataFrame()
        if enrich is None or enrich.empty:
            roster_df = pd.DataFrame()
        else:
            # Map team to abbreviation and filter to slate teams
            def _to_abbr(x):
                try:
                    a = _assets(str(x)).get('abbr')
                    return str(a).upper() if a else None
                except Exception:
                    return None
            enrich = enrich.copy()
            enrich['team_abbr'] = enrich['team'].map(_to_abbr)
            # Infer position using cached positions map
            from .utils.positions_cache import get_positions_map
            pos_cache = PROC_DIR / "props_positions_cache.json"
            targets = set(enrich['full_name'].astype(str)) if {'full_name'}.issubset(enrich.columns) else set()
            pos_map = get_positions_map(targets, stats_path, pos_cache) if targets else {}
            rows_fb = []
            for _, rr in enrich.iterrows():
                ab = rr.get('team_abbr')
                if not ab or ab not in slate_abbrs:
                    continue
                nm = rr.get('full_name')
                pos_raw = pos_map.get(str(nm), '')
                pos = 'G' if str(pos_raw).upper().startswith('G') else ('D' if str(pos_raw).upper().startswith('D') else 'F')
                rows_fb.append({'player_id': rr.get('player_id'), 'player': nm, 'position': pos, 'team': ab})
            roster_df = pd.DataFrame(rows_fb)
    if roster_df.empty:
        # Last-chance fallback: try loading cached roster directly if previous steps failed silently
        try:
            if roster_cache.exists():
                tmp = pd.read_csv(roster_cache)
                if tmp is not None and not tmp.empty:
                    cols = set(tmp.columns)
                    if {"player","position"}.issubset(cols) and ("team" in cols or "team_abbr" in cols):
                        roster_df = tmp.copy()
                        if "team" not in roster_df.columns and "team_abbr" in roster_df.columns:
                            roster_df = roster_df.rename(columns={"team_abbr":"team"})
        except Exception:
            pass
    if roster_df.empty:
        print("No roster players found for slate. Writing empty projections file for fast exit.")
        out = pd.DataFrame(columns=["date","player","team","position","market","proj_lambda"])
        out_path = PROC_DIR / f"props_projections_all_{date}.csv"
        save_df(out, out_path)
        print(f"Wrote {out_path} with 0 rows")
        return
    # Normalize player display strings to avoid dict-string artifacts and stray whitespace
    import ast as _ast
    def _norm_player(s):
        if s is None:
            return ""
        x = str(s).strip()
        if x.startswith('{') and x.endswith('}'):
            try:
                d = _ast.literal_eval(x)
                if isinstance(d, dict):
                    v = d.get('default') or d.get('name') or ''
                    if isinstance(v, str):
                        x = v.strip()
            except Exception:
                pass
        return " ".join(x.split())
    # Models - use neural networks if requested
    # Always initialize traditional models for per-player fallback
    shots = SkaterShotsModel()
    saves = GoalieSavesModel()
    goals = SkaterGoalsModel()
    assists = SkaterAssistsModel()
    points = SkaterPointsModel()
    blocks = SkaterBlocksModel()

    # Optionally initialize NN models
    if use_nn:
        print(f"[nn] Using neural network models for projections (NPU-accelerated)...")
        from .models.nn_props import NNPropsModel
        nn_shots = NNPropsModel("SOG", model_dir=MODEL_DIR / "nn_props", use_npu=True)
        nn_saves = NNPropsModel("SAVES", model_dir=MODEL_DIR / "nn_props", use_npu=True)
        nn_goals = NNPropsModel("GOALS", model_dir=MODEL_DIR / "nn_props", use_npu=True)
        nn_assists = NNPropsModel("ASSISTS", model_dir=MODEL_DIR / "nn_props", use_npu=True)
        nn_points = NNPropsModel("POINTS", model_dir=MODEL_DIR / "nn_props", use_npu=True)
        nn_blocks = NNPropsModel("BLOCKS", model_dir=MODEL_DIR / "nn_props", use_npu=True)

        # Check if models loaded (either ONNX or PyTorch)
        models_loaded = all([
            nn_shots.onnx_session is not None or nn_shots.model is not None,
            nn_saves.onnx_session is not None or nn_saves.model is not None,
            nn_goals.onnx_session is not None or nn_goals.model is not None,
            nn_assists.onnx_session is not None or nn_assists.model is not None,
            nn_points.onnx_session is not None or nn_points.model is not None,
            nn_blocks.onnx_session is not None or nn_blocks.model is not None,
        ])

        if not models_loaded:
            print(f"[warn] Not all NN models loaded, projections will fall back per-player to traditional models")
            # Keep use_nn True to still try per-player; fallback will engage case-by-case
    # Conservative league-average fallbacks to avoid missing projections
    def _fallback_lambda(mk: str) -> float:
        m = (mk or '').upper()
        if m == 'SOG':
            return 2.4
        if m == 'GOALS':
            return 0.35
        if m == 'ASSISTS':
            return 0.45
        if m == 'POINTS':
            return 0.9
        if m == 'SAVES':
            return 27.0
        if m == 'BLOCKS':
            return 1.3
        return 1.0
    # Helper: robust per-market lambda prediction with NN->traditional->fallback cascade
    def _predict_lambda(market: str, player_name: str, team_abbr: str | None, is_goalie: bool) -> tuple[float,str]:
        mk = (market or '').upper()
        lam = None
        src = 'unknown'
        # Try NN first if enabled
        if use_nn:
            try:
                if mk == 'SOG' and not is_goalie:
                    lam = nn_shots.predict_lambda(hist, player_name, team_abbr)
                elif mk == 'GOALS' and not is_goalie:
                    lam = nn_goals.predict_lambda(hist, player_name, team_abbr)
                elif mk == 'ASSISTS' and not is_goalie:
                    lam = nn_assists.predict_lambda(hist, player_name, team_abbr)
                elif mk == 'POINTS' and not is_goalie:
                    lam = nn_points.predict_lambda(hist, player_name, team_abbr)
                elif mk == 'BLOCKS' and not is_goalie:
                    lam = nn_blocks.predict_lambda(hist, player_name, team_abbr)
                elif mk == 'SAVES' and is_goalie:
                    lam = nn_saves.predict_lambda(hist, player_name, team_abbr)
                if lam is not None:
                    src = 'nn'
            except Exception:
                lam = None
        # Traditional fallback
        if lam is None:
            try:
                if mk == 'SOG' and not is_goalie:
                    lam = shots.player_lambda(hist, player_name)
                elif mk == 'GOALS' and not is_goalie:
                    lam = goals.player_lambda(hist, player_name)
                elif mk == 'ASSISTS' and not is_goalie:
                    lam = assists.player_lambda(hist, player_name)
                elif mk == 'POINTS' and not is_goalie:
                    lam = points.player_lambda(hist, player_name)
                elif mk == 'BLOCKS' and not is_goalie:
                    lam = blocks.player_lambda(hist, player_name)
                elif mk == 'SAVES' and is_goalie:
                    lam = saves.player_lambda(hist, player_name)
                if lam is not None:
                    src = 'trad'
            except Exception:
                lam = None
        if lam is None:
            lam = _fallback_lambda(mk)
            src = 'fallback'
        try:
            return float(lam), src
        except Exception:
            return _fallback_lambda(mk), 'fallback'

    rows = []
    for _, r in roster_df.iterrows():
        player = _norm_player(r.get('player'))
        pos = str(r.get('position') or '').upper()
        team = r.get('team')
        if not player:
            continue
        try:
            if pos == 'G':
                if include_goalies:
                    lam, src = _predict_lambda('SAVES', player, team, is_goalie=True)
                    rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'SAVES', 'proj_lambda': lam, 'source': src})
            else:
                # SOG
                lam, src = _predict_lambda('SOG', player, team, is_goalie=False)
                rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'SOG', 'proj_lambda': lam, 'source': src})
                # GOALS
                lam, src = _predict_lambda('GOALS', player, team, is_goalie=False)
                rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'GOALS', 'proj_lambda': lam, 'source': src})
                # ASSISTS
                lam, src = _predict_lambda('ASSISTS', player, team, is_goalie=False)
                rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'ASSISTS', 'proj_lambda': lam, 'source': src})
                # POINTS
                lam, src = _predict_lambda('POINTS', player, team, is_goalie=False)
                rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'POINTS', 'proj_lambda': lam, 'source': src})
                # BLOCKS
                lam, src = _predict_lambda('BLOCKS', player, team, is_goalie=False)
                rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'BLOCKS', 'proj_lambda': lam, 'source': src})
        except Exception:
            continue
    out = pd.DataFrame(rows)
    # Optional: apply per-player reconciliation bias if available for date
    try:
        bias_path = PROC_DIR / f"player_props_bias_{date}.csv"
        if out is not None and not out.empty and bias_path.exists():
            try:
                bias = pd.read_csv(bias_path)
            except Exception:
                bias = pd.DataFrame()
            if bias is not None and not bias.empty and {"player","market","bias"}.issubset(bias.columns):
                # Normalize join keys
                def _n(x: str) -> str:
                    try:
                        return " ".join(str(x or "").split())
                    except Exception:
                        return str(x)
                b = bias.copy()
                b["player"] = b["player"].astype(str).map(_n)
                b["market"] = b["market"].astype(str).str.upper()
                out["player"] = out["player"].astype(str).map(_n)
                out["market"] = out["market"].astype(str).str.upper()
                out = out.merge(b[["player","market","bias"]], on=["player","market"], how="left")
                # Apply bounded bias multiplicatively
                try:
                    msk = out["bias"].notna()
                    out.loc[msk, "proj_lambda"] = (out.loc[msk, "proj_lambda"].astype(float) * out.loc[msk, "bias"].astype(float)).clip(lower=0.0)
                    out.loc[msk, "source"] = out.loc[msk, "source"].astype(str) + "+bias"
                except Exception:
                    pass
                try:
                    out = out.drop(columns=[c for c in ["bias"] if c in out.columns])
                except Exception:
                    pass
    except Exception:
        pass
    if not out.empty:
        # Final pass normalization on player column
        try:
            out['player'] = out['player'].astype(str).map(_norm_player)
        except Exception:
            pass
        try:
            out = out.sort_values(['team','position','player','market'])
        except Exception:
            pass
    out_path = PROC_DIR / f"props_projections_all_{date}.csv"
    save_df(out, out_path)
    print(f"Wrote {out_path} with {0 if out is None or out.empty else len(out)} rows")
    try:
        if not out.empty and 'source' in out.columns:
            summary = out.groupby(['market','source']).size().reset_index(name='count')
            print('[summary] projection source counts:')
            for _, rr in summary.iterrows():
                print(f"  {rr['market']:<7} {rr['source']:<9} {rr['count']}")
            # Basic dispersion stats per market for quick uniformity check
            disp = out.groupby('market')['proj_lambda'].agg(['mean','std','min','max','count']).reset_index()
            print('[dispersion] per-market lambda stats:')
            for _, rr in disp.iterrows():
                print(f"  {rr['market']:<7} mean={rr['mean']:.2f} std={rr['std']:.2f} min={rr['min']:.2f} max={rr['max']:.2f} n={int(rr['count'])}")
    except Exception as e:
        print('[warn] summary failed:', e)


@app.command()
def odds_fetch_bovada(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD"),
    out_csv: str = typer.Option("", help="Optional output CSV path; defaults under data/raw/bovada_odds_YYYY-MM-DD.csv"),
):
    """Deprecated: Bovada support removed. Use The Odds API-based commands instead."""
    print("This command has been removed. Use oddsapi flows (predict, daily_update, closings) instead.")
    raise typer.Exit(code=1)


@app.command()
def daily_update(days_ahead: int = typer.Option(2, help="How many days ahead to update (including today)")):
    """Refresh predictions with odds for today (+N days) using The Odds API; ensure CSV exists if odds missing."""
    from datetime import datetime, timedelta, timezone
    base = datetime.now(timezone.utc)
    for i in range(0, max(1, days_ahead)):
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            predict_core(date=d, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings")
        except Exception:
            pass
        # Only do the CSV ensure step if the predictions file doesn't exist yet
        try:
            path = PROC_DIR / f"predictions_{d}.csv"
            if not path.exists():
                predict_core(date=d, source="web", odds_source="csv")
        except Exception:
            pass

def build_range_core(
    start: str,
    end: str,
    source: str = "web",
    bankroll: float = 0.0,
    kelly_fraction_part: float = 0.5,
):
    """Core implementation to build predictions with odds for all dates with NHL games in [start, end].

    Strategy per date: The Odds API -> ensure predictions exist without odds.
    This function is safe to call from non-CLI contexts and expects plain Python types.
    """
    # Collect schedule once, then iterate unique dates with NHL vs NHL games
    if source == "web":
        client = NHLWebClient()
        games = client.schedule_range(start, end)
    elif source == "stats":
        client = NHLClient()
        games = client.schedule(start, end)
    elif source == "nhlpy":
        try:
            from .data.nhl_api_nhlpy import NHLNhlPyClient  # lazy import
        except Exception as e:
            print("nhl-api-py adapter not available:", e)
            raise typer.Exit(code=1)
        client = NHLNhlPyClient()
        games = client.schedule_range(start, end)
    else:
        print(f"Unknown source '{source}'. Use one of: web, stats, nhlpy")
        raise typer.Exit(code=1)
    # Filter to NHL teams via assets (avoid non-NHL exhibitions)
    try:
        from .web.teams import get_team_assets as _assets
        games = [g for g in games if (_assets(getattr(g, 'home', '')).get('abbr') and _assets(getattr(g, 'away', '')).get('abbr'))]
    except Exception:
        pass
    # Group build days by US/Eastern calendar date to avoid cross-midnight UTC duplications
    def _et_day(iso_utc: str) -> str:
        try:
            s = str(iso_utc).replace("Z", "+00:00")
            dt_utc = datetime.fromisoformat(s)
            et = ZoneInfo("America/New_York")
            return dt_utc.astimezone(et).strftime('%Y-%m-%d')
        except Exception:
            try:
                dt_utc = datetime.fromisoformat(str(iso_utc)[:19]).replace(tzinfo=timezone.utc)
                et = ZoneInfo("America/New_York")
                return dt_utc.astimezone(et).strftime('%Y-%m-%d')
            except Exception:
                return pd.to_datetime(iso_utc, errors='coerce').strftime('%Y-%m-%d')
    dates = sorted({_et_day(getattr(g, 'gameDate')) for g in games})
    print(f"Found {len(dates)} dates with NHL games between {start} and {end}.")
    for d in dates:
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n=== Building {d} ===")
        # Try Odds API
        try:
            path = PROC_DIR / f"predictions_{d}.csv"
            df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
        try:
            predict_core(date=d, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings", bankroll=bankroll, kelly_fraction_part=kelly_fraction_part)
        except Exception as e:
            print("Odds API step failed:", e)
        # Ensure file exists even without odds
        try:
            path = PROC_DIR / f"predictions_{d}.csv"
            if not path.exists():
                predict_core(date=d, source="web", odds_source="csv")
        except Exception:
            pass


@app.command()
def build_range(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    source: str = typer.Option("web", help="Data source for schedule: web | stats | nhlpy"),
    bankroll: float = typer.Option(0.0, help="Bankroll for Kelly sizing; 0 disables"),
    kelly_fraction_part: float = typer.Option(0.5, help="Kelly fraction (0-1)"),
):
    """Build predictions with odds for all dates with NHL games in [start, end].

    Strategy per date: The Odds API -> ensure predictions exist without odds.
    """
    build_range_core(start=start, end=end, source=source, bankroll=bankroll, kelly_fraction_part=kelly_fraction_part)

@app.command()
def closings(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    prefer_book: Optional[str] = typer.Option(None, help="Prefer a bookmaker key (e.g., draftkings)"),
    best_of_all: bool = typer.Option(True, help="Use best price across all bookmakers"),
    verbose: bool = typer.Option(False, help="Verbose output")
):
    """Capture closing odds into predictions_{date}.csv (close_* columns)."""
    from .scripts.daily_update import capture_closing_for_date
    res = capture_closing_for_date(date, prefer_book=prefer_book, best_of_all=best_of_all, verbose=verbose)
    print(res)

@app.command()
def recompute_ev(
    date: str = typer.Option(..., help="Date YYYY-MM-DD to recompute EVs for"),
    prefer_closing: bool = typer.Option(True, help="If regular odds are missing, use close_* odds to compute EVs and write them back to main columns"),
):
    """Recompute EV and edge columns for predictions_{date}.csv using available odds or closings.

    If prefer_closing=True and *_odds columns are missing, copy close_* odds into main odds columns first.
    """
    from .utils.io import PROC_DIR
    path = PROC_DIR / f"predictions_{date}.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if df.empty:
        print("No predictions to recompute.")
        return
    # Helper to pick price source
    def pick(col_main: str, col_close: str):
        if col_main in df.columns and df[col_main].notna().any():
            return col_main
        if prefer_closing and col_close in df.columns and df[col_close].notna().any():
            # also copy into main for UI consistency
            if col_main not in df.columns:
                df[col_main] = pd.NA
            df[col_main] = df[col_main].fillna(df[col_close])
            return col_main
        return None
    from .utils.odds import american_to_decimal, decimal_to_implied_prob, remove_vig_two_way, ev_unit
    # Moneyline
    ml_h_src = pick("home_ml_odds", "close_home_ml_odds")
    ml_a_src = pick("away_ml_odds", "close_away_ml_odds")
    # Totals
    tot_o_src = pick("over_odds", "close_over_odds")
    tot_u_src = pick("under_odds", "close_under_odds")
    # Puckline
    hpl_src = pick("home_pl_-1.5_odds", "close_home_pl_-1.5_odds")
    apl_src = pick("away_pl_+1.5_odds", "close_away_pl_+1.5_odds")
    # Recompute
    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return None
    for idx, r in df.iterrows():
        # Moneyline EV
        if ml_h_src and ml_a_src and pd.notna(r.get(ml_h_src)) and pd.notna(r.get(ml_a_src)):
            try:
                dec_h = american_to_decimal(_safe_float(r.get(ml_h_src)))
                dec_a = american_to_decimal(_safe_float(r.get(ml_a_src)))
                imp_h = decimal_to_implied_prob(dec_h); imp_a = decimal_to_implied_prob(dec_a)
                nv_h, nv_a = remove_vig_two_way(imp_h, imp_a)
                df.at[idx, "ev_home_ml"] = round(ev_unit(float(r.get("p_home_ml")), dec_h), 4)
                df.at[idx, "ev_away_ml"] = round(ev_unit(float(r.get("p_away_ml")), dec_a), 4)
                df.at[idx, "edge_home_ml"] = round(float(r.get("p_home_ml")) - nv_h, 4)
                df.at[idx, "edge_away_ml"] = round(float(r.get("p_away_ml")) - nv_a, 4)
            except Exception:
                pass
        # Totals EV
        if tot_o_src and tot_u_src and pd.notna(r.get(tot_o_src)) and pd.notna(r.get(tot_u_src)):
            try:
                dec_o = american_to_decimal(_safe_float(r.get(tot_o_src)))
                dec_u = american_to_decimal(_safe_float(r.get(tot_u_src)))
                imp_o = decimal_to_implied_prob(dec_o); imp_u = decimal_to_implied_prob(dec_u)
                nv_o, nv_u = remove_vig_two_way(imp_o, imp_u)
                df.at[idx, "ev_over"] = round(ev_unit(float(r.get("p_over")), dec_o), 4)
                df.at[idx, "ev_under"] = round(ev_unit(float(r.get("p_under")), dec_u), 4)
                df.at[idx, "edge_over"] = round(float(r.get("p_over")) - nv_o, 4)
                df.at[idx, "edge_under"] = round(float(r.get("p_under")) - nv_u, 4)
            except Exception:
                pass
        # Puckline EV
        if hpl_src and apl_src and pd.notna(r.get(hpl_src)) and pd.notna(r.get(apl_src)):
            try:
                dec_hpl = american_to_decimal(_safe_float(r.get(hpl_src)))
                dec_apl = american_to_decimal(_safe_float(r.get(apl_src)))
                imp_hpl = decimal_to_implied_prob(dec_hpl); imp_apl = decimal_to_implied_prob(dec_apl)
                nv_hpl, nv_apl = remove_vig_two_way(imp_hpl, imp_apl)
                df.at[idx, "ev_home_pl_-1.5"] = round(ev_unit(float(r.get("p_home_pl_-1.5")), dec_hpl), 4)
                df.at[idx, "ev_away_pl_+1.5"] = round(ev_unit(float(r.get("p_away_pl_+1.5")), dec_apl), 4)
                df.at[idx, "edge_home_pl_-1.5"] = round(float(r.get("p_home_pl_-1.5")) - nv_hpl, 4)
                df.at[idx, "edge_away_pl_+1.5"] = round(float(r.get("p_away_pl_+1.5")) - nv_apl, 4)
            except Exception:
                pass
    # Save + edges file
    df.to_csv(path, index=False)
    ev_cols = [c for c in df.columns if c.startswith("ev_")]
    if ev_cols:
        edges_long = df.melt(id_vars=["date", "home", "away"], value_vars=ev_cols, var_name="market", value_name="ev").dropna()
        edges_long = edges_long.sort_values("ev", ascending=False)
        from .utils.io import save_df
        from .utils.io import PROC_DIR as _P
        save_df(edges_long, _P / f"edges_{date}.csv")
    print({"status": "ok", "ev_cols": len([c for c in df.columns if c.startswith('ev_')])})

@app.command()
def build_season(
    season: int = typer.Option(..., help="Season start year, e.g., 2025"),
    include_preseason: bool = typer.Option(True, help="Include preseason (Sep)"),
    include_playoffs: bool = typer.Option(False, help="Include playoffs (Apr-Jun)"),
    bankroll: float = typer.Option(0.0, help="Bankroll for Kelly sizing; 0 disables"),
    kelly_fraction_part: float = typer.Option(0.5, help="Kelly fraction (0-1)"),
):
    """Build predictions across a season window (preseason + regular by default)."""
    start = f"{season}-09-01" if include_preseason else f"{season}-10-01"
    # End at Aug 1 next year to include entire season window safely
    end = f"{season+1}-08-01"
    build_range_core(start=start, end=end, source="web", bankroll=bankroll, kelly_fraction_part=kelly_fraction_part)

@app.command()
def props_fetch_bovada(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD"),
    out_csv: str = typer.Option("", help="Optional output CSV path; defaults under data/raw/bovada_props_YYYY-MM-DD.csv"),
    over_only: bool = typer.Option(True, help="If true, write only OVER rows compatible with props_predict CLI"),
):
    """Deprecated: Bovada support removed. Use props-collect/props-fast with OddsAPI instead."""
    print("This command has been removed. Use 'props-collect' or 'props-fast' (OddsAPI) instead.")
    raise typer.Exit(code=1)


@app.command()
def props_collect(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD"),
    output_root: str = typer.Option("data/props", help="Output root directory for Parquet files"),
    source: str = typer.Option("oddsapi", help="Source: oddsapi (requires ODDS_API_KEY) | bovada"),
):
    """Collect & normalize player props and write canonical Parquet under data/props/player_props_lines/date=YYYY-MM-DD.

    - oddsapi: use The Odds API historical snapshot for player markets (requires ODDS_API_KEY)
    """
    src = source.lower().strip()
    if src not in ("oddsapi", "bovada"):
        print("Unsupported source. Use 'oddsapi' or 'bovada'."); raise typer.Exit(code=1)
    cfg = props_data.PropsCollectionConfig(output_root=output_root, book=src, source=src)
    # Optional: roster mapping could be passed; for now, None
    res = props_data.collect_and_write(date, roster_df=None, cfg=cfg)
    print(json.dumps(res, indent=2))


@app.command()
def props_backfill(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    output_root: str = typer.Option("data/props", help="Output root directory for Parquet files"),
):
    """Backfill player props lines (SOG, GOALS, SAVES, ASSISTS, POINTS) by day and store Parquet partitions.

    Strategy per day: Use The Odds API only (requires ODDS_API_KEY).
    """
    from datetime import datetime, timedelta
    def to_dt(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")
    cur = to_dt(start)
    end_dt = to_dt(end)
    cfg_o = props_data.PropsCollectionConfig(output_root=output_root, book="oddsapi", source="oddsapi")
    total = 0
    days = 0
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        try:
            res = props_data.collect_and_write(d, roster_df=None, cfg=cfg_o)
            cnt = int(res.get("combined_count") or 0)
            total += cnt
        except Exception as e:
            print(f"[backfill] {d} failed: {e}")
        days += 1
        cur += timedelta(days=1)
    print(f"Backfill complete: days={days}, rows={total}")


@app.command()
def props_build_dataset(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    output_csv: str = typer.Option("data/props/props_modeling_dataset.csv", help="Output CSV path for modeling dataset"),
):
    """Build a modeling dataset by joining canonical props lines with actual player results.

    Supports markets: SOG, GOALS, SAVES, ASSISTS, POINTS, BLOCKS.
    """
    import os
    from glob import glob
    # Read canonical lines across date partition range
    lines = []
    from datetime import datetime, timedelta
    def to_dt(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")
    cur = to_dt(start)
    end_dt = to_dt(end)
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        for fname in ("bovada.parquet", "oddsapi.parquet"):
            path = Path(f"data/props/player_props_lines/date={d}/{fname}")
            if path.exists():
                try:
                    lines.append(pd.read_parquet(path))
                except Exception:
                    pass
        cur += timedelta(days=1)
    if not lines:
        print("No props lines found in the given range.")
        raise typer.Exit(code=1)
    lines_df = pd.concat(lines, ignore_index=True)
    # Ensure player stats exist for only the dates present in lines (avoid scanning whole season)
    stats_path = RAW_DIR / "player_game_stats.csv"
    needed_dates = sorted(list(set(pd.to_datetime(lines_df["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().tolist())))
    have_dates: set[str] = set()
    if stats_path.exists():
        try:
            _cur = pd.read_csv(stats_path)
            if not _cur.empty:
                have_dates = set(pd.to_datetime(_cur.get("date"), errors="coerce").dt.strftime("%Y-%m-%d").dropna().tolist())
        except Exception:
            have_dates = set()
    missing_dates = [d for d in needed_dates if d not in have_dates]
    if missing_dates:
        # Collect only missing span using faster NHL Web API first, fallback to Stats API
        try:
            d0, d1 = missing_dates[0], missing_dates[-1]
            collect_player_game_stats(d0, d1, source="web")
        except Exception:
            try:
                collect_player_game_stats(missing_dates[0], missing_dates[-1], source="stats")
            except Exception as e:
                print(f"player_game_stats collection failed for {missing_dates[0]}..{missing_dates[-1]}: {e}")
    # Load what we have; proceed even if partial to avoid blocking daily run
    if stats_path.exists():
        try:
            stats = pd.read_csv(stats_path)
        except Exception:
            stats = pd.DataFrame()
    else:
        stats = pd.DataFrame()
    # Filter stats to date window and build per-player per-date outcome columns
    stats["date_key"] = pd.to_datetime(stats["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if not stats.empty:
        stats = stats[(stats["date_key"] >= start) & (stats["date_key"] <= end)]
    # Keep relevant columns
    keep = ["date_key","player","shots","goals","assists","saves","blocked"]
    stats_small = stats[keep].copy()
    # Join lines to stats on player name + date
    df = lines_df.copy()
    df.rename(columns={"date": "date_key", "player_name": "player"}, inplace=True)
    merged = df.merge(stats_small, on=["date_key","player"], how="left")
    # Compute actual value per market
    def actual_for_market(row):
        m = str(row.get("market")).upper()
        if m == "SOG":
            return row.get("shots")
        if m == "GOALS":
            return row.get("goals")
        if m == "SAVES":
            return row.get("saves")
        if m == "ASSISTS":
            return row.get("assists")
        if m == "POINTS":
            try:
                g = float(row.get("goals") or 0)
                a = float(row.get("assists") or 0)
                return g + a
            except Exception:
                return None
        if m == "BLOCKS":
            return row.get("blocked")
        return None
    merged["actual"] = merged.apply(actual_for_market, axis=1)
    # Classify result for OVER/UNDER if actual is present
    def classify(row):
        try:
            if pd.isna(row.get("actual")) or pd.isna(row.get("line")):
                return None
            av = float(row.get("actual")); ln = float(row.get("line"))
            if av > ln:
                return "win"
            if av < ln:
                return "loss"
            return "push"
        except Exception:
            return None
    merged["result"] = merged.apply(classify, axis=1)
    # Save dataset
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Saved modeling dataset to {out_path} with {len(merged)} rows")


@app.command()
def props_watch(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    tries: int = typer.Option(30, help="Max polling attempts"),
    interval: int = typer.Option(300, help="Seconds between attempts"),
    output_root: str = typer.Option("data/props", help="Output root for Parquet"),
    source: str = typer.Option("bovada", help="Source: bovada | oddsapi"),
):
    """Poll for player props lines and write canonical Parquet when available.

    Strategy per attempt:
      - Try Bovada (or selected source) collection. If combined rows > 0, stop.
      - If 0 and source=bovada, try Odds API fallback (if configured).
      - Sleep between attempts.
    """
    import time as _time
    cfg_b = props_data.PropsCollectionConfig(output_root=output_root, book="bovada", source="bovada")
    cfg_o = props_data.PropsCollectionConfig(output_root=output_root, book="oddsapi", source="oddsapi")
    # Build roster snapshot once (best-effort) to aid player_id normalization
    roster_df = None
    try:
        roster_df = build_all_team_roster_snapshots()
    except Exception as e:
        print(f"[props_watch] roster snapshot failed: {e}")
    for i in range(max(1, int(tries))):
        sel = source.lower().strip()
        print(f"[props_watch] {date} attempt {i+1}/{tries} via {sel}…")
        try:
            cfg = cfg_b if sel == "bovada" else cfg_o
            res = props_data.collect_and_write(date, roster_df=roster_df, cfg=cfg)
            cnt = int(res.get("combined_count") or 0)
            if cnt > 0:
                print(f"[props_watch] success: rows={cnt}, path={res.get('output_path')}")
                return
            if sel == "bovada":
                try:
                    res2 = props_data.collect_and_write(date, roster_df=roster_df, cfg=cfg_o)
                    cnt2 = int(res2.get("combined_count") or 0)
                    if cnt2 > 0:
                        print(f"[props_watch] fallback success: rows={cnt2}, path={res2.get('output_path')}")
                        return
                except Exception as e2:
                    print(f"[props_watch] oddsapi fallback failed: {e2}")
        except Exception as e:

            @app.command()
            def roster_update(date: Optional[str] = typer.Option(None, help="ET date YYYY-MM-DD to stamp output")):
                """Build and save all-team roster snapshot (Web API)."""
                d = date or ymd(today_utc())
                df = build_all_team_roster_snapshots()
                out_path = PROC_DIR / f"roster_snapshot_{d}.csv"
                save_df(df, out_path)
                print(f"Saved roster snapshot to {out_path} ({len(df)} players)")

            @app.command()
            def lineup_update(
                date: Optional[str] = typer.Option(None, help="ET date YYYY-MM-DD to stamp output"),
                prefer_source: Optional[str] = typer.Option("dailyfaceoff", help="Optional external source to try first (dailyfaceoff|none)")
            ):
                """Build expected lineups per team using external source when available; fallback to TOI-based inference."""
                d = date or ymd(today_utc())
                frames = []
                # Use aliased team abbrevs from roster module
                for ab in TEAM_ABBRS:
                    try:
                        snap = None
                        if prefer_source and prefer_source.lower() == "dailyfaceoff":
                            try:
                                from .data.lineups import build_lineup_snapshot_from_source
                                snap_src = build_lineup_snapshot_from_source(ab, d)
                                if snap_src is not None and not snap_src.empty:
                                    snap = snap_src
                            except Exception as e_src:
                                print({"team": ab, "source": prefer_source, "error": str(e_src)})
                        if snap is None or snap.empty:
                            snap = build_lineup_snapshot(ab)
                        snap["team"] = ab
                        frames.append(snap)
                    except Exception as e:
                        print({"team": ab, "error": str(e)})
                out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["player_id","full_name","position","line_slot","pp_unit","pk_unit","proj_toi","confidence","team"])
                out_path = PROC_DIR / f"lineups_{d}.csv"
                save_df(out, out_path)
                print(f"Saved lineup snapshot to {out_path} ({len(out)} rows)")
                # Also write starting goalies snapshot (best-effort)
                try:
                    rows_g = fetch_dailyfaceoff_starting_goalies(d)
                    df_g = pd.DataFrame(rows_g, columns=["team","goalie","status","confidence","source"]) if rows_g else pd.DataFrame(columns=["team","goalie","status","confidence","source"])
                    # Fallback: derive likely starters from lineup snapshot by highest projected TOI among goalies
                    if df_g.empty and not out.empty:
                        df_goalies = out.copy()
                        # Normalize position filter
                        df_goalies["position"] = df_goalies["position"].astype(str)
                        df_goalies = df_goalies[df_goalies["position"].str.upper().str.contains("G")]
                        if not df_goalies.empty:
                            # Ensure proj_toi numeric
                            def _to_num(x):
                                try:
                                    import math
                                    v = float(x)
                                    return v if math.isfinite(v) else None
                                except Exception:
                                    return None
                            df_goalies["_toi"] = df_goalies["proj_toi"].apply(_to_num)
                            # Group by team and pick max TOI, fallback to highest confidence
                            starters = []
                            for team, grp in df_goalies.groupby("team"):
                                g1 = grp.copy()
                                g1 = g1.sort_values(["_toi","confidence"], ascending=[False, False])
                                if not g1.empty:
                                    r = g1.iloc[0]
                                    starters.append({
                                        "team": str(team),
                                        "goalie": str(r.get("full_name","")),
                                        "status": "Derived",
                                        "confidence": float(r.get("confidence", 0.5)),
                                        "source": "lineup_snapshot",
                                    })
                            if starters:
                                df_g = pd.DataFrame(starters, columns=["team","goalie","status","confidence","source"])
                    out_g = PROC_DIR / f"starting_goalies_{d}.csv"
                    save_df(df_g, out_g)
                    print(f"Saved starting goalies snapshot to {out_g} ({len(df_g)} rows)")
                except Exception as e_g:
                    print(f"[lineup_update] starting goalies fetch failed: {e_g}")


            @app.command()
            def injury_update(date: Optional[str] = typer.Option(None, help="ET date YYYY-MM-DD to stamp output"), overrides_path: Optional[str] = typer.Option(None, help="Optional CSV with manual injury entries")):
                """Save a normalized injury snapshot for the date (baseline manual/empty)."""
                d = date or ymd(today_utc())
                manual = []
                if overrides_path:
                    try:
                        dfm = pd.read_csv(overrides_path)
                        manual = dfm.to_dict(orient="records")
                    except Exception as e:
                        print({"overrides": "error", "path": overrides_path, "error": str(e)})
                df = build_injury_snapshot(d, manual=manual)
                out_path = PROC_DIR / f"injuries_{d}.csv"
                save_df(df, out_path)
                print(f"Saved injury snapshot to {out_path} ({len(df)} rows)")

            @app.command(name="game-simulate-baseline")
            def game_simulate_baseline(date: Optional[str] = typer.Option(None, help="ET date YYYY-MM-DD to simulate schedule"), seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility")):
                """Simulate all scheduled games for a date using baseline period-level engine and output game and box score CSVs."""
                d = date or ymd(today_utc())
                client = NHLWebClient()
                games = client.schedule_day(d)
                # Base goals rate from config if present
                base_mu = 3.0
                try:
                    cfg_path = MODEL_DIR / "config.json"
                    if cfg_path.exists():
                        obj = json.loads(cfg_path.read_text(encoding="utf-8"))
                        base_mu = float(obj.get("base_mu", base_mu))
                except Exception:
                    pass
                sim_cfg = SimConfig(seed=seed)
                sim_rates = RateModels.baseline(base_mu=base_mu)
                sim = GameSimulator(cfg=sim_cfg, rates=sim_rates)
                game_rows = []
                box_rows = []
                for g in games:
                    try:
                        # Map full team names to abbreviations
                        h_ab = get_team_assets(g.home).get("abbr") or g.home
                        a_ab = get_team_assets(g.away).get("abbr") or g.away
                        roster_home = build_roster_snapshot(h_ab).to_dict(orient="records")
                        roster_away = build_roster_snapshot(a_ab).to_dict(orient="records")
                        gs, _events = sim.simulate(home_name=g.home, away_name=g.away, roster_home=roster_home, roster_away=roster_away)
                        game_rows.append({
                            "gamePk": g.gamePk,
                            "date": d,
                            "home": g.home,
                            "away": g.away,
                            "home_goals_sim": gs.home.score,
                            "away_goals_sim": gs.away.score,
                        })
                        # Box scores
                        for p in list(gs.home.players.values()) + list(gs.away.players.values()):
                            box_rows.append({
                                "gamePk": g.gamePk,
                                "date": d,
                                "team": p.team,
                                "player_id": p.player_id,
                                "full_name": p.full_name,
                                "position": p.position,
                                "shots": int(p.stats.get("shots", 0.0)),
                                "goals": int(p.stats.get("goals", 0.0)),
                                "saves": int(p.stats.get("saves", 0.0)),
                            })
                    except Exception as e:
                        print({"simulate": "error", "gamePk": g.gamePk, "error": str(e)})
                # Save outputs
                games_df = pd.DataFrame(game_rows)
                boxes_df = pd.DataFrame(box_rows)
                out_g = PROC_DIR / f"sim_games_{d}.csv"
                out_b = PROC_DIR / f"sim_boxscores_{d}.csv"
                save_df(games_df, out_g)
                save_df(boxes_df, out_b)
                print({"saved": {"games": str(out_g), "boxscores": str(out_b)}, "counts": {"games": len(games_df), "box": len(boxes_df)}})

            print(f"[props_watch] attempt failed: {e}")
        if i < tries - 1:
            _time.sleep(max(1, int(interval)))


@app.command()
def props_backtest(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    window: int = typer.Option(10, help="Rolling window (games) for lambda"),
    stake: float = typer.Option(100.0, help="Flat stake per play for ROI calc"),
    markets: str = typer.Option("SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS", help="Comma list of markets to include"),
    min_ev: float = typer.Option(-1.0, help="Filter to plays with EV >= min_ev; set -1 to include all"),
    out_prefix: str = typer.Option("", help="Optional output filename prefix under data/processed/"),
):
    """Backtest props models on canonical lines without lookahead and compute ROI + calibration.

    For each day in [start, end]:
      - Load canonical props lines for that date.
      - Compute rolling lambda using player history strictly BEFORE the date.
      - Compute p_over and EV for Over/Under.
      - Choose side by higher EV (at that day's lines) and evaluate result using actuals.
      - Aggregate ROI and calibration stats by market and overall.
    Writes rows and summary to data/processed.
    """
    from datetime import datetime, timedelta
    from .utils.io import RAW_DIR, PROC_DIR
    from .models.props import (
        PropsConfig,
        SkaterShotsModel, GoalieSavesModel,
        SkaterGoalsModel, SkaterAssistsModel, SkaterPointsModel, SkaterBlocksModel,
    )
    # Ensure player stats exist over the full window before end
    try:
        collect_player_game_stats(start, end, source="stats")
    except Exception:
        pass
    stats_path = RAW_DIR / "player_game_stats.csv"
    if not stats_path.exists():
        print("player_game_stats.csv missing; cannot backtest.")
        raise typer.Exit(code=1)
    try:
        stats_all = pd.read_csv(stats_path)
    except Exception:
        print("Failed to read player_game_stats.csv; is the file empty or malformed?")
        raise typer.Exit(code=1)
    # Normalize dates to strings YYYY-MM-DD for comparisons (UTC and ET)
    stats_all["date_key"] = pd.to_datetime(stats_all["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    # Build ET calendar day to align with props line partitions
    from zoneinfo import ZoneInfo as _Z
    def _iso_to_et(iso_utc: str) -> str:
        if not isinstance(iso_utc, str):
            try:
                iso_utc = str(iso_utc)
            except Exception:
                return None
        if not iso_utc:
            return None
        try:
            s = iso_utc.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
        except Exception:
            try:
                dt = datetime.fromisoformat(str(iso_utc)[:19]).replace(tzinfo=timezone.utc)
            except Exception:
                return None
        try:
            return dt.astimezone(_Z("America/New_York")).strftime("%Y-%m-%d")
        except Exception:
            return None
    stats_all["date_et"] = stats_all["date"].apply(_iso_to_et)
    # Extract a plain-text player name from possible dict-like entries, then normalize
    import ast, re, unicodedata
    def _extract_player_text(v) -> str:
        if v is None:
            return ""
        try:
            # Some rows serialize dicts like "{'default': 'N. Schmaltz'}"
            if isinstance(v, str) and v.strip().startswith("{") and v.strip().endswith("}"):
                d = ast.literal_eval(v)
                if isinstance(d, dict):
                    for k in ("default", "en", "name", "fullName", "full_name"):
                        if d.get(k):
                            return str(d.get(k))
                return str(v)
            if isinstance(v, dict):
                for k in ("default", "en", "name", "fullName", "full_name"):
                    if v.get(k):
                        return str(v.get(k))
                return str(v)
            return str(v)
        except Exception:
            return str(v)
    def _norm_name(s: str) -> str:
        s = (s or "").strip()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = re.sub(r"\s+", " ", s)
        return s.lower()
    stats_all["player_text_raw"] = stats_all["player"].apply(_extract_player_text)
    # Normalized forms
    stats_all["player_text_norm"] = stats_all["player_text_raw"].apply(_norm_name)
    # Also provide a no-dot variant for initial matching (e.g., "N. Schmaltz" -> "n schmaltz")
    stats_all["player_text_nodot"] = stats_all["player_text_norm"].str.replace(".", "", regex=False)

    # Helper: name variants for joining outcomes (exact, no-dot, initial-last)
    def _name_variants(full: str):
        full = (full or "").strip()
        parts = [p for p in full.split(" ") if p]
        vars = set()
        if full:
            nm = _norm_name(full)
            vars.add(nm)
            vars.add(nm.replace(".", ""))
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            init_last = f"{first[0]}. {last}"
            nm2 = _norm_name(init_last)
            vars.add(nm2)
            vars.add(nm2.replace(".", ""))
        return vars

    def _name_variants(full: str):
        full = (full or "").strip()
        parts = [p for p in full.split(" ") if p]
        vars = set()
        if full:
            nm = _norm_name(full)
            vars.add(nm)
            vars.add(nm.replace(".", ""))
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            init_last = f"{first[0]}. {last}"
            nm2 = _norm_name(init_last)
            vars.add(nm2)
            vars.add(nm2.replace(".", ""))
        return vars
    # Models with configured window
    cfg = PropsConfig(window=window)
    shots = SkaterShotsModel(cfg); saves = GoalieSavesModel(cfg); goals = SkaterGoalsModel(cfg); assists = SkaterAssistsModel(cfg); points = SkaterPointsModel(cfg); blocks = SkaterBlocksModel(cfg)
    allowed_markets = [m.strip().upper() for m in (markets or "").split(",") if m.strip()]
    # Iterate dates
    def to_dt(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")
    cur = to_dt(start); end_dt = to_dt(end)
    rows = []
    calib = []  # over-probability calibration
    total_lines = 0
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        # Load lines for date d
        base = Path("data/props") / f"player_props_lines/date={d}"
        parts = []
        # Prefer parquet, but fall back to CSV if parquet not available
        for prov in ("bovada", "oddsapi"):
            p_parq = base / f"{prov}.parquet"
            p_csv = base / f"{prov}.csv"
            if p_parq.exists():
                try:
                    parts.append(pd.read_parquet(p_parq))
                except Exception:
                    pass
            elif p_csv.exists():
                try:
                    parts.append(pd.read_csv(p_csv))
                except Exception:
                    pass
        if parts:
            lines_df = pd.concat(parts, ignore_index=True)
            total_lines += len(lines_df)
        else:
            lines_df = pd.DataFrame()
        if lines_df.empty:
            cur += timedelta(days=1)
            continue
        # History strictly before date d
        hist = stats_all[stats_all["date_key"] < d].copy()
        # Helper for price to decimal
        def _dec(a):
            try:
                a = float(a)
                return american_to_decimal(a)
            except Exception:
                return None
        # Compute projections and outcomes
        for _, r in lines_df.iterrows():
            m = str(r.get("market") or "").upper()
            if allowed_markets and m not in allowed_markets:
                continue
            player = r.get("player_name") or r.get("player")
            if not player:
                continue
            try:
                ln = float(r.get("line"))
            except Exception:
                continue
            op = r.get("over_price"); up = r.get("under_price")
            if pd.isna(op) and pd.isna(up):
                continue
            # Lambda & prob using only history BEFORE date
            lam = None; p_over = None
            if m == "SOG":
                lam = shots.player_lambda(hist, str(player)); p_over = shots.prob_over(lam, ln)
            elif m == "SAVES":
                lam = saves.player_lambda(hist, str(player)); p_over = saves.prob_over(lam, ln)
            elif m == "GOALS":
                lam = goals.player_lambda(hist, str(player)); p_over = goals.prob_over(lam, ln)
            elif m == "ASSISTS":
                lam = assists.player_lambda(hist, str(player)); p_over = assists.prob_over(lam, ln)
            elif m == "POINTS":
                lam = points.player_lambda(hist, str(player)); p_over = points.prob_over(lam, ln)
            elif m == "BLOCKS":
                lam = blocks.player_lambda(hist, str(player)); p_over = blocks.prob_over(lam, ln)
            else:
                continue
            if lam is None or p_over is None:
                continue
            # EVs
            ev_o = None; ev_u = None
            dec_o = _dec(op); dec_u = _dec(up)
            if dec_o is not None:
                ev_o = ev_unit(float(p_over), dec_o)
            p_under = max(0.0, 1.0 - float(p_over))
            if dec_u is not None:
                ev_u = ev_unit(float(p_under), dec_u)
            # Choose side by higher EV
            side = None; price = None; ev = None
            if ev_o is not None or ev_u is not None:
                if (ev_u is None) or (ev_o is not None and ev_o >= ev_u):
                    side = "Over"; price = op; ev = ev_o
                else:
                    side = "Under"; price = up; ev = ev_u
            if ev is None:
                continue
            if min_ev is not None and float(min_ev) > -1.0 and float(ev) < float(min_ev):
                continue
            # Actual outcome
            # Use stats for date d and player
            # Flexible name matching: try full name, initial+lastname, and no-dot version
            def _name_variants(full: str):
                full = (full or "").strip()
                parts = [p for p in full.split(" ") if p]
                vars = set()
                if full:
                    vars.add(_norm_name(full))
                    vars.add(_norm_name(full).replace(".", ""))
                if len(parts) >= 2:
                    first, last = parts[0], parts[-1]
                    init_last = f"{first[0]}. {last}"
                    vars.add(_norm_name(init_last))
                    vars.add(_norm_name(init_last).replace(".", ""))
                return vars
            variants = _name_variants(str(player))
            # Use ET calendar date to align with props partition date
            day_stats = stats_all[stats_all["date_et"] == d]
            ps = day_stats[day_stats["player_text_norm"].isin(variants) | day_stats["player_text_nodot"].isin(variants)]
            actual = None
            if not ps.empty:
                row = ps.iloc[0]
                if m == "SOG":
                    actual = row.get("shots")
                elif m == "SAVES":
                    actual = row.get("saves")
                elif m == "GOALS":
                    actual = row.get("goals")
                elif m == "ASSISTS":
                    actual = row.get("assists")
                elif m == "POINTS":
                    try:
                        actual = float((row.get("goals") or 0)) + float((row.get("assists") or 0))
                    except Exception:
                        actual = None
                elif m == "BLOCKS":
                    actual = row.get("blocked")
            result = None
            if actual is not None and pd.notna(actual):
                try:
                    av = float(actual)
                    if av > ln:
                        over_res = "win"
                    elif av < ln:
                        over_res = "loss"
                    else:
                        over_res = "push"
                    # Map to chosen side
                    if side == "Over":
                        result = over_res
                    elif side == "Under":
                        if over_res == "win":
                            result = "loss"
                        elif over_res == "loss":
                            result = "win"
                        else:
                            result = "push"
                except Exception:
                    result = None
            # Payout calc
            payout = None
            if result is not None:
                dec = _dec(price)
                if result == "win" and dec is not None:
                    payout = stake * (dec - 1.0)
                elif result == "loss":
                    payout = -stake
                elif result == "push":
                    payout = 0.0
            # Record row
            rows.append({
                "date": d,
                "market": m,
                "player": player,
                "line": ln,
                "book": r.get("book"),
                "over_price": op if pd.notna(op) else None,
                "under_price": up if pd.notna(up) else None,
                "proj": float(lam),
                "p_over": float(p_over),
                "side": side,
                "ev": float(ev) if ev is not None else None,
                "actual": actual,
                "result": result,
                "stake": stake,
                "payout": payout,
            })
            # Calibration for over-side outcome regardless of chosen bet
            if actual is not None and pd.notna(actual):
                calib.append({
                    "date": d,
                    "market": m,
                    "p_over": float(p_over),
                    "over_won": bool(float(actual) > ln),
                })
        cur += timedelta(days=1)
    # Summaries
    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        print("No backtest rows generated. Ensure props lines exist under data/props/player_props_lines/date=YYYY-MM-DD.")
        raise typer.Exit(code=0)
    # Overall and by-market performance
    def summarize(df: pd.DataFrame) -> dict:
        d = {
            "picks": int(len(df)),
            "decided": int(df["result"].isin(["win","loss"]).sum()),
            "wins": int((df["result"] == "win").sum()),
            "losses": int((df["result"] == "loss").sum()),
            "pushes": int((df["result"] == "push").sum()),
        }
        # Accuracy (exclude pushes)
        try:
            dec = df[df["result"].isin(["win","loss"])].copy()
            d["accuracy"] = float((dec["result"] == "win").mean()) if len(dec) > 0 else None
        except Exception:
            d["accuracy"] = None
        # Brier score for chosen side if probabilities present
        try:
            p_chosen = np.where(df["side"].astype(str)=="Over", df["p_over"].astype(float), (1.0 - df["p_over"].astype(float)))
            y = np.where(df["result"].astype(str)=="win", 1.0, np.where(df["result"].astype(str)=="loss", 0.0, np.nan))
            mask = ~np.isnan(p_chosen) & ~np.isnan(y)
            d["brier"] = float(np.mean((p_chosen[mask] - y[mask])**2)) if np.any(mask) else None
        except Exception:
            d["brier"] = None
        return d
    overall = summarize(rows_df)
    by_market = {}
    for mkt, g in rows_df.groupby("market"):
        by_market[mkt] = summarize(g)
    # Calibration bins
    calib_df = pd.DataFrame(calib)
    calib_bins = []
    if not calib_df.empty:
        bins = np.linspace(0.0, 1.0, 11)
        calib_df["bin"] = pd.cut(calib_df["p_over"], bins=bins, include_lowest=True)
        for b, g in calib_df.groupby("bin"):
            try:
                exp = float(g["p_over"].mean())
                obs = float(g["over_won"].mean())
                cnt = int(len(g))
                calib_bins.append({"bin": str(b), "expected": exp, "observed": obs, "count": cnt})
            except Exception:
                continue
    # Outputs
    pref = (out_prefix.strip() + "_") if out_prefix.strip() else ""
    rows_path = PROC_DIR / f"{pref}props_backtest_rows_{start}_to_{end}.csv"
    summ_path = PROC_DIR / f"{pref}props_backtest_summary_{start}_to_{end}.json"
    rows_df.to_csv(rows_path, index=False)
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "by_market": by_market, "calibration": calib_bins}, f, indent=2)
    print(json.dumps({"overall": overall, "by_market": by_market}, indent=2))
    print(f"Saved rows to {rows_path} and summary to {summ_path}")


@app.command(name="props-recommendations-nolines")
def props_recommendations_nolines(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    markets: str = typer.Option("SAVES,BLOCKS", help="Comma list of markets to include"),
    top: int = typer.Option(400, help="Top N to keep after sorting by chosen_prob desc"),
    min_prob: float = typer.Option(0.0, help="Global minimum chosen-side probability (0-1)"),
    min_prob_per_market: str = typer.Option("SAVES=0.60,BLOCKS=0.85", help="Per-market minimum chosen-side probability thresholds"),
):
    """Generate recommendations from nolines simulations by probability gating.

    Reads data/processed/props_simulations_nolines_{date}.csv and filters picks by chosen-side probability.
    Writes data/processed/props_recommendations_nolines_{date}.csv
    """
    import pandas as pd
    import numpy as _np
    from .utils.io import PROC_DIR, save_df

    path = PROC_DIR / f"props_simulations_nolines_{date}.csv"
    if not path.exists():
        print("No nolines simulations found:", path)
        raise typer.Exit(code=0)
    df = pd.read_csv(path)
    if df is None or df.empty:
        print("Empty nolines simulations file:", path)
        raise typer.Exit(code=0)
    df["market"] = df["market"].astype(str).str.upper()
    allowed = [m.strip().upper() for m in (markets or "").split(",") if m.strip()]
    if allowed:
        df = df[df["market"].isin(allowed)]
    if df.empty:
        print("No rows after market filter.")
        raise typer.Exit(code=0)

    p_over = pd.to_numeric(df["p_over_sim"], errors="coerce").astype(float)
    over_better = p_over >= 0.5
    chosen_prob = _np.where(over_better, p_over, 1.0 - p_over)
    side = _np.where(over_better, "Over", "Under")
    out = df.assign(chosen_prob=chosen_prob, side=side)

    def _parse_thresholds(s: str) -> dict[str, float]:
        d: dict[str, float] = {}
        for part in (s or "").split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                d[str(k).strip().upper()] = float(str(v).strip())
            except Exception:
                continue
        return d
    thr_map = _parse_thresholds(min_prob_per_market)
    if thr_map or (float(min_prob) > 0.0):
        thr_series = out["market"].astype(str).str.upper().map(lambda m: thr_map.get(m, float(min_prob))).astype(float)
        out = out[(out["chosen_prob"].notna()) & (out["chosen_prob"].astype(float) >= thr_series)]
    out = out.sort_values("chosen_prob", ascending=False).head(int(top))
    final = out[["date","player","team","opp","market","line","proj_lambda","lam_scale_mean","p_over_sim","side","chosen_prob"]].copy()
    final.rename(columns={"p_over_sim":"p_over"}, inplace=True)
    out_path = PROC_DIR / f"props_recommendations_nolines_{date}.csv"
    save_df(final, out_path)
    print(f"[props-recs-nolines] wrote {out_path} with {len(final)} rows")


@app.command(name="props-nolines-monitor")
def props_nolines_monitor(
    window_days: int = typer.Option(7, help="Rolling window in days for monitor"),
    markets: str = typer.Option("SAVES,BLOCKS", help="Markets to include"),
    min_prob_per_market: str = typer.Option("SAVES=0.60,BLOCKS=0.85", help="Per-market probability gates for backtest"),
):
    """Generate a rolling monitor JSON for nolines simulations over the last N days."""
    from datetime import date, timedelta
    import json
    from .utils.io import PROC_DIR
    # Derive start/end
    end = date.today()
    start = end - timedelta(days=max(1, int(window_days)))
    start_s = start.strftime("%Y-%m-%d"); end_s = end.strftime("%Y-%m-%d")
    # Run backtest to compute metrics
    try:
        props_backtest_nolines(start=start_s, end=end_s, markets=markets, min_prob=0.0, min_prob_per_market=min_prob_per_market, out_prefix="monitor")
    except SystemExit:
        pass
    # Read the produced summary and write standard monitor path
    summ_path = PROC_DIR / f"monitor_props_backtest_nolines_summary_{start_s}_to_{end_s}.json"
    if not summ_path.exists():
        print("No monitor summary found:", summ_path)
        raise typer.Exit(code=0)
    try:
        with open(summ_path, "r", encoding="utf-8") as f:
            summ = json.load(f)
        out_path = PROC_DIR / "props_nolines_monitor.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"window_days": window_days, "start": start_s, "end": end_s, **summ}, f, indent=2)
        print("[nolines-monitor] wrote", out_path)
    except Exception as e:
        print("Failed to write props_nolines_monitor.json:", e)


@app.command(name="props-recommendations-combined")
def props_recommendations_combined(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    include_nolines: bool = typer.Option(True, help="Include nolines recommendations if present"),
    out_json: bool = typer.Option(True, help="Also write a JSON alongside CSV"),
):
    """Merge EV-based sim recommendations and nolines recommendations into one output.

    Sources:
      - data/processed/props_recommendations_{date}.csv (EV-based)
      - data/processed/props_recommendations_nolines_{date}.csv (probability-gated)
    Writes: data/processed/props_recommendations_combined_{date}.csv (+.json)
    """
    import pandas as pd
    import json
    from .utils.io import PROC_DIR, save_df

    rows = []
    ev_path = PROC_DIR / f"props_recommendations_{date}.csv"
    if ev_path.exists():
        try:
            ev = pd.read_csv(ev_path)
            if not ev.empty:
                ev = ev.copy(); ev["source"] = "ev"
                rows.append(ev)
        except Exception:
            pass
    no_path = PROC_DIR / f"props_recommendations_nolines_{date}.csv"
    if include_nolines and no_path.exists():
        try:
            no = pd.read_csv(no_path)
            if not no.empty:
                no = no.copy(); no["source"] = "nolines"
                rows.append(no)
        except Exception:
            pass
    if not rows:
        print("No recommendation sources found for", date)
        raise typer.Exit(code=0)
    df = pd.concat(rows, ignore_index=True)
    # Standardize columns
    have_ev = {c for c in df.columns}
    for col in ["ev","chosen_prob"]:
        if col not in have_ev:
            df[col] = None
    # Preferred ordering
    cols = [c for c in ["date","player","team","opp","market","line","book","over_price","under_price","proj_lambda","p_over","side","ev","chosen_prob","source"] if c in df.columns]
    final = df[cols].copy()
    out_csv = PROC_DIR / f"props_recommendations_combined_{date}.csv"
    save_df(final, out_csv)
    print(f"[props-recs-combined] wrote {out_csv} with {len(final)} rows")
    if out_json:
        try:
            out_json_path = PROC_DIR / f"props_recommendations_combined_{date}.json"
            final.to_json(out_json_path, orient="records")
            print("[props-recs-combined] wrote", out_json_path)
        except Exception:
            pass


@app.command(name="props-simulate-unlined")
def props_simulate_unlined(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    markets: str = typer.Option("SAVES,BLOCKS", help="Comma-separated markets to simulate without relying on lines"),
    candidate_lines: str = typer.Option("SAVES=24.5,26.5,28.5,30.5;BLOCKS=1.5,2.5,3.5", help="Per-market candidate lines; ';' between markets, ',' between values"),
    n_sims: int = typer.Option(20000, help="Number of Monte Carlo simulations"),
    sim_shared_k: float = typer.Option(1.0, help="Shared Gamma pace shape (mean=1, var=1/k)"),
    props_xg_gamma: float = typer.Option(0.02, help="Team xGF/60 impact on per-player lambda"),
    props_penalty_gamma: float = typer.Option(0.06, help="Opponent penalties committed per60 impact (PP exposure)"),
    props_goalie_form_gamma: float = typer.Option(0.02, help="Opponent goalie sv% (L10) dampening for GOALS/ASSISTS/POINTS"),
):
    """Simulate selected player props using model lambdas + shared game pace + team features without needing canonical lines.

    Writes data/processed/props_simulations_nolines_{date}.csv with p_over_sim per (player, market, candidate_line).
    """
    import numpy as _np
    import pandas as _pd
    from glob import glob as _glob
    from datetime import datetime as _dt
    from .web.teams import get_team_assets as _assets
    from .data.nhl_api_web import NHLWebClient as _Web
    from .utils.io import PROC_DIR
    from scipy.stats import poisson as _poisson
    from .data.rosters import fetch_current_roster as _fetch_roster

    # Parse markets and candidate lines mapping
    markets_set = set([m.strip().upper() for m in (markets or "").split(",") if m.strip()])
    def _parse_candidate_lines(s: str) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for part in (s or "").split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            kk = str(k).strip().upper()
            vals = []
            for x in v.split(","):
                x = x.strip()
                if not x:
                    continue
                try:
                    vals.append(float(x))
                except Exception:
                    continue
            if vals:
                out[kk] = vals
        return out
    cand_map = _parse_candidate_lines(candidate_lines)
    if not markets_set:
        markets_set = {"SAVES","BLOCKS"}

    # Ensure projections exist (attempt on-demand precompute if missing/empty)
    proj_path = PROC_DIR / f"props_projections_all_{date}.csv"
    def _load_proj(path):
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            return None
    proj = _load_proj(proj_path) if proj_path.exists() else None
    if proj is None or proj.empty:
        try:
            # On-demand compute projections for this date
            props_precompute_all(date=date)  # type: ignore[name-defined]
        except Exception:
            pass
        proj = _load_proj(proj_path)
    if proj is None or proj.empty:
        # Fallback: build minimal projections from current rosters with baseline lambdas
        try:
            slate_teams = []
            try:
                web = _Web(); sched = web.schedule_day(date)
                for g in sched:
                    h = str(getattr(g, "home", "")); a = str(getattr(g, "away", ""))
                    slate_teams.extend([_assets(h).get("abbr"), _assets(a).get("abbr")])
                slate_teams = [str(t or "").upper() for t in slate_teams if t]
            except Exception:
                slate_teams = []
            rows = []
            for t in set(slate_teams):
                try:
                    ros = _fetch_roster(str(t))
                except Exception:
                    ros = []
                for rp in (ros or []):
                    pname = str(getattr(rp, "full_name", "")).strip()
                    pos = str(getattr(rp, "position", "")).upper()
                    if not pname:
                        continue
                    if pos.startswith("G"):
                        rows.append({"date": date, "player": pname, "team": str(t), "market": "SAVES", "proj_lambda": 27.0})
                    else:
                        rows.append({"date": date, "player": pname, "team": str(t), "market": "SOG", "proj_lambda": 2.5})
                        rows.append({"date": date, "player": pname, "team": str(t), "market": "GOALS", "proj_lambda": 0.4})
                        rows.append({"date": date, "player": pname, "team": str(t), "market": "ASSISTS", "proj_lambda": 0.4})
                        rows.append({"date": date, "player": pname, "team": str(t), "market": "POINTS", "proj_lambda": 0.8})
                        rows.append({"date": date, "player": pname, "team": str(t), "market": "BLOCKS", "proj_lambda": 1.8})
            proj = _pd.DataFrame(rows)
        except Exception:
            proj = _pd.DataFrame()
        if proj is None or proj.empty:
            print("Empty projections file:", proj_path)
            raise typer.Exit(code=1)


    # Opponent mapping via schedule
    def _abbr_team(n: str | None) -> str | None:
        if not n:
            return None
        try:
            a = _assets(str(n)) or {}
            return str(a.get("abbr") or "").upper() or None
        except Exception:
            return None
    abbr_to_opp: dict[str, str] = {}
    try:
        web = _Web(); sched = web.schedule_day(date)
        games=[]
        for g in sched:
            h=str(getattr(g, "home", "")); a=str(getattr(g, "away", ""))
            games.append((_abbr_team(h), _abbr_team(a)))
        for h,a in games:
            if h and a:
                abbr_to_opp[h]=a; abbr_to_opp[a]=h
    except Exception:
        abbr_to_opp = {}

    # If SAVES requested but missing in projections, supplement with web rosters (goalies) at fallback lambda
    if ("SAVES" in markets_set) and ("SAVES" not in set(proj["market"].astype(str).str.upper().unique())):
        goalie_rows = []
        slate_teams = list(set([k for k in abbr_to_opp.keys()]))
        for t in slate_teams:
            try:
                ros = _fetch_roster(str(t))
                for rp in (ros or []):
                    if str(getattr(rp, "position", "")).upper() == "G" and str(getattr(rp, "full_name", "")).strip():
                        goalie_rows.append({
                            "date": date,
                            "player": str(rp.full_name),
                            "team": str(t),
                            "market": "SAVES",
                            "proj_lambda": 27.0,  # conservative fallback; will be scaled by team features
                        })
            except Exception:
                continue
        if goalie_rows:
            proj = pd.concat([proj, pd.DataFrame(goalie_rows)], ignore_index=True)

    # Load team features
    xg_path = PROC_DIR / "team_xg_latest.csv"
    xg_map = {}
    if xg_path.exists():
        try:
            _xg = pd.read_csv(xg_path)
            if not _xg.empty and {"abbr","xgf60"}.issubset(_xg.columns):
                xg_map = {str(r.abbr).upper(): float(r.xgf60) for _, r in _xg.iterrows()}
        except Exception:
            xg_map = {}
    league_xg = float(_np.mean(list(xg_map.values()))) if xg_map else 2.6
    pen_path = PROC_DIR / "team_penalty_rates.json"
    pen_comm = {}
    if pen_path.exists():
        try:
            import json
            pen_comm = json.loads(pen_path.read_text(encoding="utf-8"))
        except Exception:
            pen_comm = {}
    league_pen = float(_np.mean([float(v.get("committed_per60", 0.0)) for v in pen_comm.values()])) if pen_comm else 3.0
    # Goalie form (sv% L10) for opponent adjustment on scoring markets (not used for SAVES directly)
    from datetime import date as _date
    gf_today = PROC_DIR / f"goalie_form_{_date.today().strftime('%Y-%m-%d')}.csv"
    gf_map = {}
    if gf_today.exists():
        try:
            _gf = pd.read_csv(gf_today)
            if not _gf.empty and {"team","sv_pct_l10"}.issubset(_gf.columns):
                gf_map = {str(r.team).upper(): float(r.sv_pct_l10) for _, r in _gf.iterrows()}
        except Exception:
            gf_map = {}
    league_sv = float(_np.mean(list(gf_map.values()))) if gf_map else 0.905

    # Prep projections subset
    proj = proj.copy()
    proj["market"] = proj["market"].astype(str).str.upper()
    proj = proj[proj["market"].isin(markets_set)].copy()
    if proj.empty:
        print("No projections for requested markets; ensure include_goalies for SAVES.")
        raise typer.Exit(code=0)
    # Normalize team abbr and opponent
    proj["team_abbr"] = proj["team"].astype(str).str.upper()
    proj["opp_abbr"] = proj["team_abbr"].map(lambda t: abbr_to_opp.get(str(t).upper()))

    def _multiplier(row) -> float:
        mk = str(row.get("market")).upper(); team=str(row.get("team_abbr") or "").upper(); opp=str(row.get("opp_abbr") or "").upper()
        lam_scale = 1.0
        try:
            txg = xg_map.get(team); oxg = xg_map.get(opp)
            if mk == "SAVES":
                if oxg:
                    lam_scale *= (1.0 + props_xg_gamma * ((oxg / league_xg) - 1.0))
                pc = pen_comm.get(team, {}).get("committed_per60")
                if pc is not None:
                    lam_scale *= (1.0 + props_penalty_gamma * ((float(pc) / league_pen) - 1.0))
            else:
                if txg:
                    lam_scale *= (1.0 + props_xg_gamma * ((txg / league_xg) - 1.0))
                pc = pen_comm.get(opp, {}).get("committed_per60")
                if pc is not None and mk in {"GOALS","ASSISTS","POINTS","SOG","BLOCKS"}:
                    lam_scale *= (1.0 + props_penalty_gamma * ((float(pc) / league_pen) - 1.0))
                if mk in {"GOALS","ASSISTS","POINTS"}:
                    sv = gf_map.get(opp)
                    if sv is not None:
                        lam_scale *= (1.0 - props_goalie_form_gamma * (float(sv) - league_sv))
        except Exception:
            lam_scale = lam_scale
        return max(0.0, lam_scale)

    proj["lam_scale_mean"] = proj.apply(_multiplier, axis=1).astype(float)
    # Group by game for shared pace
    proj["grp"] = proj.apply(lambda r: ":".join(sorted([str(r.get("team_abbr") or "").upper(), str(r.get("opp_abbr") or "").upper()])), axis=1)
    groups = {g: i for i, g in enumerate(sorted(proj["grp"].unique()))}
    proj["grp_id"] = proj["grp"].map(groups)

    rs = _np.random.RandomState(42)
    shape = float(sim_shared_k) if float(sim_shared_k) > 0 else 1.0
    scale = 1.0 / shape
    grp_ids = sorted(groups.values())
    pace_draws = _np.ones((len(grp_ids), n_sims), dtype=_np.float32)
    if shape > 0:
        pace_draws = rs.gamma(shape=shape, scale=scale, size=(len(grp_ids), n_sims)).astype(_np.float32)

    out_rows = []
    for _, rr in proj.iterrows():
        mk = str(rr["market"]).upper(); lam0 = float(rr["proj_lambda"]); mmean = float(rr["lam_scale_mean"])
        if lam0 <= 0 or mmean <= 0:
            continue
        lines = cand_map.get(mk, [])
        if not lines:
            continue
        lam_eff = lam0 * mmean
        g = int(rr["grp_id"]) if pd.notna(rr.get("grp_id")) else 0
        lam_arr = lam_eff * pace_draws[g]
        for ln in lines:
            k_line = int(_np.floor(float(ln) + 1e-9))
            try:
                p_over = _poisson.sf(k_line, mu=lam_arr).mean()
            except Exception:
                y = rs.poisson(lam_arr)
                p_over = float((y > k_line).mean())
            out_rows.append({
                "date": date,
                "player": rr.get("player"),
                "team": rr.get("team_abbr"),
                "opp": rr.get("opp_abbr"),
                "market": mk,
                "line": float(ln),
                "book": "NA",
                "proj_lambda": round(lam0, 4),
                "lam_scale_mean": round(mmean, 4),
                "p_over_sim": round(float(p_over), 6),
                "n_sims": int(n_sims),
            })
    out = pd.DataFrame(out_rows)
    out_path = PROC_DIR / f"props_simulations_nolines_{date}.csv"
    save_df(out, out_path)
    print(f"[props-sim-nolines] wrote {out_path} rows={len(out)}")


@app.command(name="props-backtest-nolines")
def props_backtest_nolines(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    markets: str = typer.Option("SAVES,BLOCKS", help="Comma list of markets to include"),
    min_prob: float = typer.Option(0.0, help="Minimum chosen-side probability threshold (0-1)"),
    min_prob_per_market: str = typer.Option("", help="Optional per-market probability thresholds, e.g., 'SAVES=0.60,BLOCKS=0.60'"),
    out_prefix: str = typer.Option("nolines", help="Output filename prefix under data/processed/"),
):
    """Backtest 'nolines' simulations (no odds/EV) using chosen probability and outcomes.

    For each day in [start, end]:
      - Load data/processed/props_simulations_nolines_{date}.csv
      - Choose side: Over if p_over_sim >= 0.5 else Under
      - Apply probability thresholds (global or per-market) to filter picks
      - Evaluate outcomes from data/raw/player_game_stats.csv
      - Aggregate accuracy, Brier, and avg chosen probability
    Writes rows and summary to data/processed with a prefix.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo as _Z
    import numpy as _np
    import pandas as pd
    import json
    from .utils.io import RAW_DIR, PROC_DIR

    # Load realized player stats for outcomes
    stats_path = RAW_DIR / "player_game_stats.csv"
    if not stats_path.exists():
        print("player_game_stats.csv missing; run props_stats_backfill first.")
        raise typer.Exit(code=1)
    try:
        stats_all = pd.read_csv(stats_path)
    except Exception:
        print("Failed to read player_game_stats.csv; is the file empty or malformed?")
        raise typer.Exit(code=1)

    def _iso_to_et(iso_utc: str) -> str | None:
        try:
            s = str(iso_utc or "").replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
        except Exception:
            try:
                dt = datetime.fromisoformat(str(iso_utc)[:19]).replace(tzinfo=datetime.timezone.utc)  # type: ignore
            except Exception:
                return None
        try:
            return dt.astimezone(_Z("America/New_York")).strftime("%Y-%m-%d")
        except Exception:
            return None

    import ast, re, unicodedata
    def _extract_player_text(v) -> str:
        if v is None:
            return ""
        try:
            if isinstance(v, str) and v.strip().startswith("{") and v.strip().endswith("}"):
                d = ast.literal_eval(v)
                if isinstance(d, dict):
                    for k in ("default", "en", "name", "fullName", "full_name"):
                        if d.get(k):
                            return str(d.get(k))
                return str(v)
            if isinstance(v, dict):
                for k in ("default", "en", "name", "fullName", "full_name"):
                    if v.get(k):
                        return str(v.get(k))
                return str(v)
            return str(v)
        except Exception:
            return str(v)

    def _norm_name(s: str) -> str:
        s = (s or "").strip()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = re.sub(r"\s+", " ", s)
        return s.lower()

    stats_all["date_et"] = stats_all["date"].apply(_iso_to_et)
    stats_all["player_text_raw"] = stats_all["player"].apply(_extract_player_text)
    stats_all["player_text_norm"] = stats_all["player_text_raw"].apply(_norm_name)
    stats_all["player_text_nodot"] = stats_all["player_text_norm"].str.replace(".", "", regex=False)

    allowed_markets = [m.strip().upper() for m in (markets or "").split(",") if m.strip()]

    def _parse_thresholds(s: str) -> dict[str, float]:
        d: dict[str, float] = {}
        for part in (s or "").split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                d[str(k).strip().upper()] = float(str(v).strip())
            except Exception:
                continue
        return d
    prob_thr_map = _parse_thresholds(min_prob_per_market)

    def to_dt(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")
    cur = to_dt(start); end_dt = to_dt(end)
    rows = []; calib = []

    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        sim_path = PROC_DIR / f"props_simulations_nolines_{d}.csv"
        if not sim_path.exists():
            cur += timedelta(days=1)
            continue
        try:
            sim_df = pd.read_csv(sim_path)
        except Exception:
            cur += timedelta(days=1)
            continue
        if sim_df.empty:
            cur += timedelta(days=1)
            continue
        sim_df["market"] = sim_df["market"].astype(str).str.upper()
        if allowed_markets:
            sim_df = sim_df[sim_df["market"].isin(allowed_markets)]
        if sim_df.empty:
            cur += timedelta(days=1)
            continue
        sim_df["player_norm"] = sim_df["player"].astype(str).apply(_norm_name)
        # Evaluate outcomes across all candidate lines
        for _, r in sim_df.iterrows():
            m = str(r.get("market") or "").upper()
            if allowed_markets and m not in allowed_markets:
                continue
            player_disp = r.get("player")
            ln = r.get("line")
            try:
                ln = float(ln)
            except Exception:
                continue
            p_over = pd.to_numeric(r.get("p_over_sim"), errors="coerce")
            if pd.isna(p_over):
                continue
            p_over = float(p_over)
            # chosen side by probability
            over_better = bool(p_over >= 0.5)
            side = "Over" if over_better else "Under"
            chosen_prob = float(p_over if over_better else (1.0 - p_over))
            # Apply probability thresholds
            thr = prob_thr_map.get(m, float(min_prob)) if prob_thr_map else float(min_prob)
            if thr > 0.0 and chosen_prob < thr:
                continue

            # Actual outcome from stats on ET date d
            day_stats = stats_all[stats_all["date_et"] == d]
            _full = str(player_disp or "").strip()
            _parts = [p for p in _full.split(" ") if p]
            _vars = set()
            if _full:
                _nm = _norm_name(_full)
                _vars.add(_nm); _vars.add(_nm.replace(".", ""))
            if len(_parts) >= 2:
                _first, _last = _parts[0], _parts[-1]
                _init_last = f"{_first[0]}. {_last}"
                _nm2 = _norm_name(_init_last)
                _vars.add(_nm2); _vars.add(_nm2.replace(".", ""))
            ps = day_stats[day_stats["player_text_norm"].isin(_vars) | day_stats["player_text_nodot"].isin(_vars)]
            actual = None
            if not ps.empty:
                row = ps.iloc[0]
                if m == "SOG":
                    actual = row.get("shots")
                elif m == "SAVES":
                    actual = row.get("saves")
                elif m == "GOALS":
                    actual = row.get("goals")
                elif m == "ASSISTS":
                    actual = row.get("assists")
                elif m == "POINTS":
                    try:
                        actual = float((row.get("goals") or 0)) + float((row.get("assists") or 0))
                    except Exception:
                        actual = None
                elif m == "BLOCKS":
                    actual = row.get("blocked")
            result = None
            if actual is not None and pd.notna(actual):
                try:
                    av = float(actual)
                    if av > ln:
                        over_res = "win"
                    elif av < ln:
                        over_res = "loss"
                    else:
                        over_res = "push"
                    if side == "Over":
                        result = over_res
                    elif side == "Under":
                        if over_res == "win":
                            result = "loss"
                        elif over_res == "loss":
                            result = "win"
                        else:
                            result = "push"
                except Exception:
                    result = None

            rows.append({
                "date": d,
                "market": m,
                "player": player_disp,
                "line": float(ln),
                "p_over": float(p_over),
                "side": side,
                "chosen_prob": float(chosen_prob),
                "actual": actual,
                "result": result,
            })
            # Calibration record for over outcome
            if actual is not None and pd.notna(actual) and (abs(ln - round(ln)) > 1e-9 or float(actual) != float(round(ln))):
                calib.append({
                    "date": d,
                    "market": m,
                    "p_over": float(p_over),
                    "over_won": bool(float(actual) > float(ln)),
                })
        cur += timedelta(days=1)

    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        print("No backtest rows generated. Ensure nolines simulations exist for the range.")
        raise typer.Exit(code=0)

    def summarize(df: pd.DataFrame) -> dict:
        out: dict[str, object] = {
            "picks": int(len(df)),
            "decided": int(df["result"].isin(["win", "loss"]).sum()),
            "wins": int((df["result"] == "win").sum()),
            "losses": int((df["result"] == "loss").sum()),
            "pushes": int((df["result"] == "push").sum()),
        }
        try:
            decided = df[df["result"].isin(["win", "loss"])].copy()
            if not decided.empty:
                p_sel = decided["chosen_prob"].astype(float).to_numpy()
                y = (decided["result"] == "win").astype(float).to_numpy()
                p_sel = _np.clip(p_sel, 0.0, 1.0)
                out["accuracy"] = float((y == 1.0).mean()) if len(y) > 0 else None
                out["brier"] = float(_np.mean((p_sel - y) ** 2)) if len(y) > 0 else None
                out["avg_prob"] = float(_np.mean(p_sel))
            else:
                out["accuracy"] = None; out["brier"] = None; out["avg_prob"] = None
        except Exception:
            out["accuracy"] = None; out["brier"] = None; out["avg_prob"] = None
        return out

    overall = summarize(rows_df)
    by_market = {mkt: summarize(g) for mkt, g in rows_df.groupby("market")}

    def acc_at_threshold(df: pd.DataFrame, thr: float) -> dict:
        try:
            decided = df[df["result"].isin(["win","loss"])].copy()
            mask = decided["chosen_prob"].astype(float) >= thr
            sub = decided[mask]
            acc = float((sub["result"] == "win").mean()) if len(sub) > 0 else None
            return {"picks": int(len(sub)), "accuracy": acc}
        except Exception:
            return {"picks": 0, "accuracy": None}
    thresholds = [0.55, 0.60, 0.65]
    by_market_thresholds = {mkt: {str(t): acc_at_threshold(g, t) for t in thresholds} for mkt, g in rows_df.groupby("market")}

    calib_df = pd.DataFrame(calib)
    calib_bins = []
    if not calib_df.empty:
        bins = _np.linspace(0.0, 1.0, 11)
        calib_df["bin"] = pd.cut(calib_df["p_over"], bins=bins, include_lowest=True)
        for b, g in calib_df.groupby("bin"):
            try:
                exp = float(g["p_over"].mean()); obs = float(g["over_won"].mean()); cnt = int(len(g))
                calib_bins.append({"bin": str(b), "expected": exp, "observed": obs, "count": cnt})
            except Exception:
                continue

    pref = (out_prefix.strip() + "_") if out_prefix.strip() else ""
    rows_path = PROC_DIR / f"{pref}props_backtest_nolines_rows_{start}_to_{end}.csv"
    summ_path = PROC_DIR / f"{pref}props_backtest_nolines_summary_{start}_to_{end}.json"
    rows_df.to_csv(rows_path, index=False)
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall": overall,
            "by_market": by_market,
            "by_market_thresholds": by_market_thresholds,
            "calibration": calib_bins,
            "filters": {"markets": allowed_markets, "min_prob": min_prob}
        }, f, indent=2)
    print(json.dumps({"overall": overall, "by_market": by_market}, indent=2))
    print(f"Saved rows to {rows_path} and summary to {summ_path}")


@app.command()
def game_backtest_dc_anchor(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    stake: float = typer.Option(100.0, help="Flat stake per play for ROI calc"),
    min_ev_ml: float = typer.Option(0.0, help="Minimum EV threshold for Moneyline plays"),
    min_ev_totals: float = typer.Option(0.0, help="Minimum EV threshold for Totals plays"),
    min_ev_pl: float = typer.Option(0.0, help="Minimum EV threshold for Puckline plays"),
    min_ev_first10: float = typer.Option(0.0, help="Minimum EV threshold for First-10 plays"),
    min_ev_periods: float = typer.Option(0.0, help="Minimum EV threshold for Period totals (P1/P2/P3)"),
    totals_temp: float = typer.Option(1.0, help="Temperature scaling for totals probs (>1 flattens toward 0.5)"),
    use_close: bool = typer.Option(True, help="Prefer closing odds/lines when available"),
    out_prefix: str = typer.Option("dc", help="Output filename prefix under data/processed/"),
):
    """Backtest game markets (Moneyline, Totals) using DC+market-anchored probabilities in predictions_{date}.csv.

    For each day in [start, end]:
      - Load data/processed/predictions_{date}.csv (populated by web layer)
      - Ensure final scores are available; if missing, backfill from data/raw/games.csv by date_et+teams
      - Use p_home_ml/p_away_ml and p_over/p_under (already DC+anchored) with preferred odds
      - Compute EV and choose side if EV >= threshold
      - Evaluate result; aggregate ROI by market and overall
    Writes rows and summary to data/processed with a prefix.
    """
    from datetime import datetime as _dt, timedelta as _td
    import json
    import math
    import numpy as np
    import pandas as pd
    from .utils.io import RAW_DIR, PROC_DIR
    from .utils.odds import american_to_decimal

    # Default totals_temp and EV gates from model_calibration.json if available and caller didn't override
    try:
        cal_path = PROC_DIR / "model_calibration.json"
        if cal_path.exists() and abs(float(totals_temp) - 1.0) < 1e-9:
            with open(cal_path, "r", encoding="utf-8") as f:
                _obj = json.load(f) or {}
            if _obj.get("totals_temp") is not None:
                totals_temp = float(_obj["totals_temp"])  # use learned default
            # EV gates: adopt learned thresholds only when caller left zero (meaning auto)
            try:
                if abs(float(min_ev_ml)) < 1e-12 and _obj.get("min_ev_ml") is not None:
                    min_ev_ml = float(_obj.get("min_ev_ml"))
                if abs(float(min_ev_totals)) < 1e-12 and _obj.get("min_ev_totals") is not None:
                    min_ev_totals = float(_obj.get("min_ev_totals"))
                if abs(float(min_ev_pl)) < 1e-12 and _obj.get("min_ev_pl") is not None:
                    min_ev_pl = float(_obj.get("min_ev_pl"))
                if abs(float(min_ev_first10)) < 1e-12 and _obj.get("min_ev_first10") is not None:
                    min_ev_first10 = float(_obj.get("min_ev_first10"))
                if abs(float(min_ev_periods)) < 1e-12 and _obj.get("min_ev_periods") is not None:
                    min_ev_periods = float(_obj.get("min_ev_periods"))
            except Exception:
                pass
    except Exception:
        pass

    def _parse_date(s: str) -> _dt:
        try:
            return _dt.strptime(s, "%Y-%m-%d")
        except Exception:
            print("Invalid date format; use YYYY-MM-DD"); raise typer.Exit(code=1)

    s_dt = _parse_date(start); e_dt = _parse_date(end)
    if e_dt < s_dt:
        s_dt, e_dt = e_dt, s_dt

    # Load raw finals once (date_et, home, away, home_goals, away_goals)
    games_path = RAW_DIR / "games.csv"
    if not games_path.exists():
        print("Missing data/raw/games.csv. Run data collection first.")
        raise typer.Exit(code=1)
    gdf = pd.read_csv(games_path)
    # Normalize dates to YYYY-MM-DD
    for col in ("date_et", "date"):
        if col in gdf.columns:
            gdf[col] = pd.to_datetime(gdf[col], errors="coerce").dt.strftime("%Y-%m-%d")
    gdf = gdf.rename(columns={"home_goals": "final_home_goals", "away_goals": "final_away_goals"})
    finals_idx = gdf[["date_et","home","away","final_home_goals","final_away_goals"]].copy()

    def _num(v):
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                f = float(v)
                return f if math.isfinite(f) else None
            s = str(v).strip().replace(",", "")
            if s == "":
                return None
            return float(s)
        except Exception:
            return None

    def _ev(p: float, american: float, p_push: float = 0.0) -> float | None:
        try:
            if p is None or american is None:
                return None
            dec = american_to_decimal(float(american))
            if dec is None or not math.isfinite(dec):
                return None
            p = float(p); p_push = float(p_push)
            if not (0.0 <= p <= 1.0):
                return None
            p_loss = max(0.0, 1.0 - p - p_push)
            return float(round(p * (dec - 1.0) - p_loss, 4))
        except Exception:
            return None

    rows = []
    # helpers for totals temp scaling
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except Exception:
            return 0.5
    def _logit(p: float) -> float:
        p = min(max(p, 1e-9), 1 - 1e-9)
        return math.log(p / (1.0 - p))
    d = s_dt
    while d <= e_dt:
        day = d.strftime("%Y-%m-%d")
        path = PROC_DIR / f"predictions_{day}.csv"
        if not path.exists():
            d += _td(days=1); continue
        try:
            df = pd.read_csv(path)
        except Exception:
            d += _td(days=1); continue
        # Backfill finals if missing
        need_finals = {"final_home_goals","final_away_goals"}
        if not need_finals.issubset(df.columns) or df["final_home_goals"].isna().any() or df["final_away_goals"].isna().any():
            try:
                merged = df.merge(finals_idx, left_on=["date","home","away"], right_on=["date_et","home","away"], how="left", suffixes=("","_g"))
                for col in ("final_home_goals","final_away_goals"):
                    if col not in merged.columns or merged[col].isna().any():
                        if f"{col}_g" in merged.columns:
                            merged[col] = merged[col].fillna(merged[f"{col}_g"])  # prefer CSV finals
                df = merged.drop(columns=[c for c in merged.columns if c.endswith("_g") or c=="date_et" or c=="date_y"], errors="ignore")
            except Exception:
                pass
        # Evaluate ML and Totals
        for _, r in df.iterrows():
            home = r.get("home"); away = r.get("away")
            # which odds to use
            h_ml = _num(r.get("close_home_ml_odds")) if use_close else _num(r.get("home_ml_odds"))
            a_ml = _num(r.get("close_away_ml_odds")) if use_close else _num(r.get("away_ml_odds"))
            o_odds = _num(r.get("close_over_odds")) if use_close else _num(r.get("over_odds"))
            u_odds = _num(r.get("close_under_odds")) if use_close else _num(r.get("under_odds"))
            # probabilities (already DC+anchored by web layer)
            ph = _num(r.get("p_home_ml")); pa = _num(r.get("p_away_ml"))
            po = _num(r.get("p_over")); pu = _num(r.get("p_under"))
            # ML best side
            if ph is not None and pa is not None and h_ml is not None and a_ml is not None:
                # choose side with greater EV
                ev_h = _ev(ph, h_ml)
                ev_a = _ev(pa, a_ml)
                if ev_h is not None and ev_h >= min_ev_ml or ev_a is not None and ev_a >= min_ev_ml:
                    if (ev_h or -999) >= (ev_a or -999):
                        pick = "home_ml"; ev = ev_h; price = h_ml
                        won = None
                        try:
                            if r.get("final_home_goals") is not None and r.get("final_away_goals") is not None:
                                won = (float(r.get("final_home_goals")) > float(r.get("final_away_goals")))
                        except Exception:
                            won = None
                    else:
                        pick = "away_ml"; ev = ev_a; price = a_ml
                        won = None
                        try:
                            if r.get("final_home_goals") is not None and r.get("final_away_goals") is not None:
                                won = (float(r.get("final_home_goals")) < float(r.get("final_away_goals")))
                        except Exception:
                            won = None
                    if ev is not None and ev >= min_ev_ml:
                        # CLV metrics (open vs close)
                        ml_open = _num(r.get("home_ml_odds")) if pick=="home_ml" else _num(r.get("away_ml_odds"))
                        ml_close = _num(r.get("close_home_ml_odds")) if pick=="home_ml" else _num(r.get("close_away_ml_odds"))
                        p_pick = float(ph if pick=="home_ml" else pa)
                        ev_open = _ev(p_pick, ml_open) if ml_open is not None else None
                        ev_close = _ev(p_pick, ml_close) if ml_close is not None else None
                        def _imp(od):
                            try:
                                d = american_to_decimal(float(od)); return (1.0/d) if (d and d>0) else None
                            except Exception:
                                return None
                        imp_open = _imp(ml_open); imp_close = _imp(ml_close)
                        rows.append({
                            "date": day, "home": home, "away": away, "market": "moneyline", "pick": pick,
                            "prob": float(ph if pick=="home_ml" else pa), "odds": price, "ev": ev,
                            "won": won, "stake": stake if won is not None else 0.0,
                            "payout": (stake* (american_to_decimal(price)-1.0)) if won else (0.0 if won is None else -stake),
                            "open_odds": ml_open, "close_odds": ml_close,
                            "clv_ev_delta": (None if (ev_open is None or ev_close is None) else round(ev_close - ev_open, 4)),
                            "clv_implied_delta": (None if (imp_open is None or imp_close is None) else round(imp_close - imp_open, 6)),
                        })
            # Totals best side (with push prob when integer line)
            tl = r.get("close_total_line_used") if use_close else r.get("total_line_used")
            tl_val = _num(tl)
            p_push = 0.0
            if tl_val is not None and abs(tl_val - round(tl_val)) < 1e-9:
                try:
                    k = int(round(tl_val))
                    mt = _num(r.get("model_total"))
                    if mt is None:
                        mt = _num(r.get("proj_home_goals"))
                        mt = (mt or 0.0) + (_num(r.get("proj_away_goals")) or 0.0)
                    if mt is not None and math.isfinite(mt) and k >= 0:
                        from math import exp, factorial
                        p_push = float(exp(-mt) * (mt ** k) / factorial(k))
                except Exception:
                    p_push = 0.0
            if po is not None and pu is not None and (o_odds is not None and u_odds is not None):
                # Optional totals temperature scaling (around 0.5), respecting push mass on integer lines
                if totals_temp is not None and abs(float(totals_temp) - 1.0) > 1e-6:
                    try:
                        T = max(0.5, min(3.0, float(totals_temp)))
                        if tl_val is not None and abs(tl_val - round(tl_val)) < 1e-9:
                            S = max(1e-9, 1.0 - float(p_push))
                            if S > 0:
                                po_c = min(max(po / S, 1e-9), 1 - 1e-9)
                                lo = _logit(po_c) / T
                                po = _sigmoid(lo) * S
                                pu = max(0.0, S - po)
                        else:
                            lo = _logit(po) / T
                            po = _sigmoid(lo)
                            pu = max(0.0, 1.0 - po)
                    except Exception:
                        pass
                ev_o = _ev(po, o_odds, p_push)
                ev_u = _ev(pu, u_odds, p_push)
                if ev_o is not None and ev_o >= min_ev_totals or ev_u is not None and ev_u >= min_ev_totals:
                    if (ev_o or -999) >= (ev_u or -999):
                        pick = "over"; ev = ev_o; price = o_odds
                    else:
                        pick = "under"; ev = ev_u; price = u_odds
                    won = None
                    try:
                        if r.get("final_home_goals") is not None and r.get("final_away_goals") is not None and tl_val is not None:
                            at = float(r.get("final_home_goals")) + float(r.get("final_away_goals"))
                            if abs(at - tl_val) < 1e-9:
                                won = None  # push, ignore for PnL
                            else:
                                won = (at > tl_val) if pick=="over" else (at < tl_val)
                    except Exception:
                        won = None
                    if ev is not None and ev >= min_ev_totals:
                        # CLV metrics for totals: open/close odds and lines; EV delta uses corresponding push mass
                        to_open = _num(r.get("over_odds")) if pick=="over" else _num(r.get("under_odds"))
                        to_close = _num(r.get("close_over_odds")) if pick=="over" else _num(r.get("close_under_odds"))
                        tl_open = _num(r.get("total_line_used"))
                        tl_close = _num(r.get("close_total_line_used"))
                        # compute push mass for open/close lines
                        def _p_push_at_line(mu_tot: float, line: float | None) -> float:
                            try:
                                if line is None or not math.isfinite(line):
                                    return 0.0
                                if abs(line - round(line)) < 1e-9:
                                    k = int(round(line))
                                    from math import exp, factorial
                                    return float(exp(-mu_tot) * (mu_tot ** k) / factorial(k)) if k >= 0 else 0.0
                                return 0.0
                            except Exception:
                                return 0.0
                        mu_tot = None
                        try:
                            mt = _num(r.get("model_total"))
                            if mt is None:
                                mt = (_num(r.get("proj_home_goals")) or 0.0) + (_num(r.get("proj_away_goals")) or 0.0)
                            mu_tot = float(mt) if mt is not None else None
                        except Exception:
                            mu_tot = None
                        p_push_open = _p_push_at_line(mu_tot, tl_open) if mu_tot is not None else 0.0
                        p_push_close = _p_push_at_line(mu_tot, tl_close) if mu_tot is not None else 0.0
                        p_pick = float(po if pick=="over" else pu)
                        ev_open = _ev(p_pick, to_open, p_push_open) if to_open is not None else None
                        ev_close = _ev(p_pick, to_close, p_push_close) if to_close is not None else None
                        def _imp(od):
                            try:
                                d = american_to_decimal(float(od)); return (1.0/d) if (d and d>0) else None
                            except Exception:
                                return None
                        imp_open = _imp(to_open); imp_close = _imp(to_close)
                        rows.append({
                            "date": day, "home": home, "away": away, "market": "totals", "pick": pick,
                            "prob": float(po if pick=="over" else pu), "odds": price, "line": tl_val, "ev": ev,
                            "won": won, "stake": stake if won is not None else 0.0,
                            "payout": (stake* (american_to_decimal(price)-1.0)) if won else (0.0 if won is None else -stake),
                            "open_odds": to_open, "close_odds": to_close, "open_line": tl_open, "close_line": tl_close,
                            "clv_ev_delta": (None if (ev_open is None or ev_close is None) else round(ev_close - ev_open, 4)),
                            "clv_implied_delta": (None if (imp_open is None or imp_close is None) else round(imp_close - imp_open, 6)),
                        })
            # Period totals P1..P3
            for pn in (1, 2, 3):
                try:
                    p_over_key = f"p{pn}_over_prob"; p_under_key = f"p{pn}_under_prob"; p_push_key = f"p{pn}_push_prob"
                    line_key = f"close_p{pn}_total_line" if use_close else f"p{pn}_total_line"
                    over_odds = _num(r.get(f"close_p{pn}_over_odds")) if use_close else _num(r.get(f"p{pn}_over_odds"))
                    under_odds = _num(r.get(f"close_p{pn}_under_odds")) if use_close else _num(r.get(f"p{pn}_under_odds"))
                    po = _num(r.get(p_over_key)); pu = _num(r.get(p_under_key)); ln = _num(r.get(line_key))
                    if po is None or pu is None or ln is None or over_odds is None or under_odds is None:
                        continue
                    # push probability if provided; else approximate via Poisson with period projections
                    p_push = _num(r.get(p_push_key))
                    if p_push is None:
                        try:
                            mu = _num(r.get(f"period{pn}_home_proj"))
                            mu = (mu or 0.0) + (_num(r.get(f"period{pn}_away_proj")) or 0.0)
                            if mu is not None and abs(ln - round(ln)) < 1e-9:
                                from math import exp, factorial
                                k = int(round(ln)); p_push = float(exp(-mu) * (mu ** k) / factorial(k)) if k >= 0 else 0.0
                            else:
                                p_push = 0.0
                        except Exception:
                            p_push = 0.0
                    ev_o = _ev(po, over_odds, p_push)
                    ev_u = _ev(pu, under_odds, p_push)
                    if (ev_o is not None and ev_o >= min_ev_periods) or (ev_u is not None and ev_u >= min_ev_periods):
                        if (ev_o or -999) >= (ev_u or -999):
                            pick = f"p{pn}_over"; ev = ev_o; price = over_odds; p_pick = po
                        else:
                            pick = f"p{pn}_under"; ev = ev_u; price = under_odds; p_pick = pu
                        won = None
                        # use precomputed result if available
                        res_key = f"result_p{pn}_total"
                        rk = r.get(res_key)
                        if isinstance(rk, str) and rk:
                            if rk == "Push": won = None
                            elif rk == "Over": won = (pick.endswith("over"))
                            elif rk == "Under": won = (pick.endswith("under"))
                        if ev is not None and ev >= min_ev_periods:
                            # CLV: open/close odds and line
                            o_open = _num(r.get(f"p{pn}_over_odds")); o_close = _num(r.get(f"close_p{pn}_over_odds"))
                            u_open = _num(r.get(f"p{pn}_under_odds")); u_close = _num(r.get(f"close_p{pn}_under_odds"))
                            l_open = _num(r.get(f"p{pn}_total_line")); l_close = _num(r.get(f"close_p{pn}_total_line"))
                            # push mass at open/close
                            def _p_push(mu_tot: float|None, line_val: float|None) -> float:
                                try:
                                    if mu_tot is None or line_val is None: return 0.0
                                    if abs(line_val - round(line_val)) < 1e-9:
                                        from math import exp, factorial
                                        k = int(round(line_val)); return float(exp(-mu_tot) * (mu_tot ** k) / factorial(k)) if k>=0 else 0.0
                                    return 0.0
                                except Exception: return 0.0
                            mu_tot = None
                            try:
                                mu_tot = (_num(r.get(f"period{pn}_home_proj")) or 0.0) + (_num(r.get(f"period{pn}_away_proj")) or 0.0)
                            except Exception:
                                mu_tot = None
                            ppo = _p_push(mu_tot, l_open); ppc = _p_push(mu_tot, l_close)
                            ev_open = _ev(p_pick, o_open if pick.endswith("over") else u_open, ppo) if (o_open or u_open) is not None else None
                            ev_close = _ev(p_pick, o_close if pick.endswith("over") else u_close, ppc) if (o_close or u_close) is not None else None
                            def _impX(od):
                                try:
                                    d = american_to_decimal(float(od)); return (1.0/d) if (d and d>0) else None
                                except Exception:
                                    return None
                            imp_open = _impX(o_open if pick.endswith("over") else u_open)
                            imp_close = _impX(o_close if pick.endswith("over") else u_close)
                            rows.append({
                                "date": day, "home": home, "away": away, "market": "periods", "pick": pick,
                                "prob": float(p_pick), "odds": price, "line": ln, "ev": ev,
                                "won": won, "stake": stake if won is not None else 0.0,
                                "payout": (stake* (american_to_decimal(price)-1.0)) if won else (0.0 if won is None else -stake),
                                "open_odds": (o_open if pick.endswith("over") else u_open),
                                "close_odds": (o_close if pick.endswith("over") else u_close),
                                "open_line": l_open, "close_line": l_close,
                                "clv_ev_delta": (None if (ev_open is None or ev_close is None) else round(ev_close - ev_open, 4)),
                                "clv_implied_delta": (None if (imp_open is None or imp_close is None) else round(imp_close - imp_open, 6)),
                            })
                except Exception:
                    continue
            # Puckline (-1.5 home, +1.5 away)
            try:
                php = _num(r.get("p_home_pl_-1.5")); pap = _num(r.get("p_away_pl_+1.5"))
            except Exception:
                php = pap = None
            hpl_odds = _num(r.get("close_home_pl_-1.5_odds")) if use_close else _num(r.get("home_pl_-1.5_odds"))
            apl_odds = _num(r.get("close_away_pl_+1.5_odds")) if use_close else _num(r.get("away_pl_+1.5_odds"))
            if php is not None and pap is not None and (hpl_odds is not None and apl_odds is not None):
                ev_hpl = _ev(php, hpl_odds)
                ev_apl = _ev(pap, apl_odds)
                if (ev_hpl is not None and ev_hpl >= min_ev_pl) or (ev_apl is not None and ev_apl >= min_ev_pl):
                    if (ev_hpl or -999) >= (ev_apl or -999):
                        pick = "home_-1.5"; ev = ev_hpl; price = hpl_odds; p_pick = php; line = -1.5
                    else:
                        pick = "away_+1.5"; ev = ev_apl; price = apl_odds; p_pick = pap; line = 1.5
                    won = None
                    try:
                        if r.get("final_home_goals") is not None and r.get("final_away_goals") is not None:
                            diff = float(r.get("final_home_goals")) - float(r.get("final_away_goals"))
                            won = (diff > 1.5) if pick == "home_-1.5" else (diff < 1.5)
                    except Exception:
                        won = None
                    if ev is not None and ev >= min_ev_pl:
                        # CLV: open vs close odds only
                        pl_open = _num(r.get("home_pl_-1.5_odds")) if pick=="home_-1.5" else _num(r.get("away_pl_+1.5_odds"))
                        pl_close = _num(r.get("close_home_pl_-1.5_odds")) if pick=="home_-1.5" else _num(r.get("close_away_pl_+1.5_odds"))
                        ev_open = _ev(p_pick, pl_open) if pl_open is not None else None
                        ev_close = _ev(p_pick, pl_close) if pl_close is not None else None
                        def _imp2(od):
                            try:
                                d = american_to_decimal(float(od)); return (1.0/d) if (d and d>0) else None
                            except Exception:
                                return None
                        imp_open = _imp2(pl_open); imp_close = _imp2(pl_close)
                        rows.append({
                            "date": day, "home": home, "away": away, "market": "puckline", "pick": pick,
                            "prob": float(p_pick), "odds": price, "line": line, "ev": ev,
                            "won": won, "stake": stake if won is not None else 0.0,
                            "payout": (stake* (american_to_decimal(price)-1.0)) if won else (0.0 if won is None else -stake),
                            "open_odds": pl_open, "close_odds": pl_close,
                            "clv_ev_delta": (None if (ev_open is None or ev_close is None) else round(ev_close - ev_open, 4)),
                            "clv_implied_delta": (None if (imp_open is None or imp_close is None) else round(imp_close - imp_open, 6)),
                        })
            # First-10 Yes/No
            try:
                p_yes = _num(r.get("p_f10_yes")); p_no = _num(r.get("p_f10_no"))
            except Exception:
                p_yes = p_no = None
            y_odds = _num(r.get("close_f10_yes_odds")) if use_close else _num(r.get("f10_yes_odds"))
            n_odds = _num(r.get("close_f10_no_odds")) if use_close else _num(r.get("f10_no_odds"))
            if p_yes is not None and p_no is not None and (y_odds is not None and n_odds is not None):
                ev_yes = _ev(p_yes, y_odds)
                ev_no = _ev(p_no, n_odds)
                if (ev_yes is not None and ev_yes >= min_ev_first10) or (ev_no is not None and ev_no >= min_ev_first10):
                    if (ev_yes or -999) >= (ev_no or -999):
                        pick = "f10_yes"; ev = ev_yes; price = y_odds; p_pick = p_yes
                    else:
                        pick = "f10_no"; ev = ev_no; price = n_odds; p_pick = p_no
                    won = None
                    try:
                        rf = r.get("result_first10")
                        if isinstance(rf, str) and rf:
                            won = (rf.strip().lower() == ("yes" if pick=="f10_yes" else "no"))
                    except Exception:
                        won = None
                    if ev is not None and ev >= min_ev_first10:
                        # CLV odds only
                        f_open = _num(r.get("f10_yes_odds")) if pick=="f10_yes" else _num(r.get("f10_no_odds"))
                        f_close = _num(r.get("close_f10_yes_odds")) if pick=="f10_yes" else _num(r.get("close_f10_no_odds"))
                        ev_open = _ev(p_pick, f_open) if f_open is not None else None
                        ev_close = _ev(p_pick, f_close) if f_close is not None else None
                        def _imp3(od):
                            try:
                                d = american_to_decimal(float(od)); return (1.0/d) if (d and d>0) else None
                            except Exception:
                                return None
                        imp_open = _imp3(f_open); imp_close = _imp3(f_close)
                        rows.append({
                            "date": day, "home": home, "away": away, "market": "first10", "pick": pick,
                            "prob": float(p_pick), "odds": price, "ev": ev,
                            "won": won, "stake": stake if won is not None else 0.0,
                            "payout": (stake* (american_to_decimal(price)-1.0)) if won else (0.0 if won is None else -stake),
                            "open_odds": f_open, "close_odds": f_close,
                            "clv_ev_delta": (None if (ev_open is None or ev_close is None) else round(ev_close - ev_open, 4)),
                            "clv_implied_delta": (None if (imp_open is None or imp_close is None) else round(imp_close - imp_open, 6)),
                        })
        d += _td(days=1)

    if not rows:
        print("No rows generated in range.")
        return
    rdf = pd.DataFrame(rows)
    # Summaries
    def _summ(df):
        st = float(df["stake"].fillna(0).sum()); pnl = float(df["payout"].fillna(0).sum())
        roi = (pnl / st) if st > 0 else None
        n = int(len(df))
        decided = int(df[~df["won"].isna()].shape[0])
        acc = None
        try:
            acc = float(df.dropna(subset=["won"]).assign(w=lambda x: x["won"].astype(bool)).w.mean()) if decided>0 else None
        except Exception:
            acc = None
        # CLV summaries
        clv_ev_mean = None; clv_ev_pos = None; clv_imp_mean = None
        try:
            if "clv_ev_delta" in df.columns:
                ce = pd.to_numeric(df["clv_ev_delta"], errors="coerce").dropna()
                if not ce.empty:
                    clv_ev_mean = float(round(ce.mean(), 6))
                    clv_ev_pos = float(round((ce > 0).mean(), 6))
        except Exception:
            pass
        try:
            if "clv_implied_delta" in df.columns:
                ci = pd.to_numeric(df["clv_implied_delta"], errors="coerce").dropna()
                if not ci.empty:
                    clv_imp_mean = float(round(ci.mean(), 6))
        except Exception:
            pass
        return {
            "n": n, "decided": decided, "staked": st, "pnl": pnl, "roi": roi, "acc": acc,
            "clv_ev_delta_mean": clv_ev_mean, "clv_ev_delta_pos_rate": clv_ev_pos,
            "clv_implied_delta_mean": clv_imp_mean,
        }

    overall = _summ(rdf)
    by_market = {k: _summ(g) for k, g in rdf.groupby("market")}

    pref = (out_prefix.strip() + "_") if out_prefix.strip() else ""
    rows_path = PROC_DIR / f"{pref}games_backtest_rows_{start}_to_{end}.csv"
    summ_path = PROC_DIR / f"{pref}games_backtest_summary_{start}_to_{end}.json"
    rdf.to_csv(rows_path, index=False)
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "by_market": by_market}, f, indent=2)
    print(json.dumps({"overall": overall, "by_market": by_market}, indent=2))
    print(f"Saved rows to {rows_path} and summary to {summ_path}")


@app.command()
def game_calibration(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    bins: int = typer.Option(10, help="Number of probability bins for calibration"),
    use_close: bool = typer.Option(True, help="Prefer closing lines/odds if available for totals"),
    out_json: Optional[str] = typer.Option(None, help="Optional explicit output path for JSON; default under processed"),
):
    """Calibration for ML and Totals probabilities written in predictions CSVs.

    Outputs expected vs observed win rates per probability bin for:
      - Moneyline: p_home_ml
      - Totals: p_over and p_under separately (excluding pushes)
    """
    from datetime import datetime as _dt, timedelta as _td
    import json
    import math
    import numpy as np
    import pandas as pd
    from .utils.io import RAW_DIR, PROC_DIR

    # Parse dates
    try:
        s_dt = _dt.strptime(start, "%Y-%m-%d"); e_dt = _dt.strptime(end, "%Y-%m-%d")
    except Exception:
        print("Invalid date format; use YYYY-MM-DD"); raise typer.Exit(code=1)
    if e_dt < s_dt:
        s_dt, e_dt = e_dt, s_dt

    # finals lookup
    games_path = RAW_DIR / "games.csv"
    if not games_path.exists():
        print("Missing data/raw/games.csv. Run data collection first.")
        raise typer.Exit(code=1)
    gdf = pd.read_csv(games_path)
    for col in ("date_et","date"):
        if col in gdf.columns:
            gdf[col] = pd.to_datetime(gdf[col], errors="coerce").dt.strftime("%Y-%m-%d")
    finals = gdf.rename(columns={"home_goals":"final_home_goals","away_goals":"final_away_goals"})[["date_et","home","away","final_home_goals","final_away_goals"]]

    ml_rows = []
    to_rows = []
    d = s_dt
    while d <= e_dt:
        day = d.strftime("%Y-%m-%d")
        path = PROC_DIR / f"predictions_{day}.csv"
        if not path.exists():
            d += _td(days=1); continue
        try:
            df = pd.read_csv(path)
        except Exception:
            d += _td(days=1); continue
        # Join finals if missing
        if ("final_home_goals" not in df.columns) or df["final_home_goals"].isna().any():
            try:
                df = df.merge(finals, left_on=["date","home","away"], right_on=["date_et","home","away"], how="left", suffixes=("","_g"))
                for col in ("final_home_goals","final_away_goals"):
                    if col not in df.columns or df[col].isna().any():
                        if f"{col}_g" in df.columns:
                            df[col] = df[col].fillna(df[f"{col}_g"])  # prefer joined finals
                df = df.drop(columns=[c for c in df.columns if c.endswith("_g") or c=="date_et_y" or c=="date_et"], errors="ignore")
            except Exception:
                pass
        # ML calibration
        try:
            sub = df.dropna(subset=["p_home_ml","final_home_goals","final_away_goals"]).copy()
            if not sub.empty:
                sub["y_home"] = (sub["final_home_goals"].astype(float) > sub["final_away_goals"].astype(float)).astype(int)
                sub["p"] = sub["p_home_ml"].astype(float).clip(1e-6, 1-1e-6)
                sub["date"] = day
                ml_rows.append(sub[["date","home","away","p","y_home"]])
        except Exception:
            pass
        # Totals calibration (exclude pushes, bin p_over and p_under separately)
        try:
            tl_col = "close_total_line_used" if use_close else "total_line_used"
            sub2 = df.dropna(subset=["p_over","p_under",tl_col,"final_home_goals","final_away_goals"]).copy()
            if not sub2.empty:
                sub2["line"] = sub2[tl_col].astype(float)
                sub2["at"] = sub2["final_home_goals"].astype(float) + sub2["final_away_goals"].astype(float)
                # exclude pushes (at == line when integer)
                def _is_push(row):
                    try:
                        ln = float(row["line"]); at = float(row["at"]) 
                        return abs(ln - round(ln)) < 1e-9 and abs(at - ln) < 1e-9
                    except Exception:
                        return False
                sub2 = sub2[~sub2.apply(_is_push, axis=1)]
                if not sub2.empty:
                    # over side
                    o = sub2.copy()
                    o["p"] = o["p_over"].astype(float).clip(1e-6, 1-1e-6)
                    o["y"] = (o["at"] > o["line"]).astype(int)
                    o["side"] = "over"
                    # under side
                    u = sub2.copy()
                    u["p"] = u["p_under"].astype(float).clip(1e-6, 1-1e-6)
                    u["y"] = (u["at"] < u["line"]).astype(int)
                    u["side"] = "under"
                    to_rows.append(o[["p","y","side"]])
                    to_rows.append(u[["p","y","side"]])
        except Exception:
            pass
        d += _td(days=1)

    res = {"range": {"start": start, "end": end}, "ml": {}, "totals": {}}
    if ml_rows:
        mld = pd.concat(ml_rows, ignore_index=True)
        try:
            q = pd.qcut(mld["p"], q=bins, duplicates="drop")
            ml_bins = mld.assign(bin=q.astype(str)).groupby("bin").apply(lambda g: pd.Series({
                "n": int(len(g)), "mean_p": float(g["p"].mean()), "obs": float(g["y_home"].mean())
            })).reset_index().to_dict(orient="records")
        except Exception:
            # fallback: equal-width bins
            edges = np.linspace(0.0, 1.0, num=bins+1)
            mld["bin"] = pd.cut(mld["p"], bins=edges, include_lowest=True).astype(str)
            ml_bins = mld.groupby("bin").apply(lambda g: pd.Series({
                "n": int(len(g)), "mean_p": float(g["p"].mean()), "obs": float(g["y_home"].mean())
            })).reset_index().to_dict(orient="records")
        res["ml"] = {"bins": ml_bins, "n": int(len(mld))}
    if to_rows:
        tod = pd.concat(to_rows, ignore_index=True)
        out = {}
        for side, g in tod.groupby("side"):
            try:
                q = pd.qcut(g["p"], q=bins, duplicates="drop")
                bins_list = g.assign(bin=q.astype(str)).groupby("bin").apply(lambda s: pd.Series({
                    "n": int(len(s)), "mean_p": float(s["p"].mean()), "obs": float(s["y"].mean())
                })).reset_index().to_dict(orient="records")
            except Exception:
                edges = np.linspace(0.0, 1.0, num=bins+1)
                gg = g.copy(); gg["bin"] = pd.cut(gg["p"], bins=edges, include_lowest=True).astype(str)
                bins_list = gg.groupby("bin").apply(lambda s: pd.Series({
                    "n": int(len(s)), "mean_p": float(s["p"].mean()), "obs": float(s["y"].mean())
                })).reset_index().to_dict(orient="records")
            out[side] = {"bins": bins_list, "n": int(len(g))}
        res["totals"] = out

    # Write JSON
    if out_json:
        out_path = Path(out_json)
    else:
        out_path = PROC_DIR / f"game_calibration_{start}_{end}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v if k=="range" else {"n": (v.get("n") if isinstance(v, dict) else None)} for k, v in res.items()}, indent=2))
    print(f"Wrote -> {out_path}")


@app.command()
def game_learn_ev_gates(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    use_close: bool = typer.Option(True, help="Use closing odds/lines to compute EV and result"),
    out_json: Optional[str] = typer.Option(None, help="Output JSON file (default: data/processed/model_calibration.json)"),
    alpha: float = typer.Option(0.35, help="Smoothing factor 0..1 for gate updates (EMA): new = old*(1-alpha) + alpha*learned"),
    min_change: float = typer.Option(0.005, help="Minimum absolute change required to update a gate; else keep old"),
):
    """Learn per-market minimum EV thresholds to maximize ROI using predictions CSVs.

    Markets: moneyline, totals (others can be added later). For each day in [start,end],
    read predictions_{date}.csv, build candidate picks by choosing the best side per market,
    then sweep EV thresholds and select the threshold with maximum ROI.
    """
    import math, json
    from datetime import datetime as _dt, timedelta as _td
    import pandas as pd
    from .utils.io import PROC_DIR

    # Parse dates
    try:
        s_dt = _dt.strptime(start, "%Y-%m-%d"); e_dt = _dt.strptime(end, "%Y-%m-%d")
    except Exception:
        print("Invalid date format; use YYYY-MM-DD"); raise typer.Exit(code=1)
    if e_dt < s_dt:
        s_dt, e_dt = e_dt, s_dt

    def _num(v):
        try:
            if v is None: return None
            f = float(v); return f if math.isfinite(f) else None
        except Exception:
            return None

    def _dec(american: float | None):
        try:
            if american is None: return None
            a = float(american)
            return 1.0 + (a/100.0) if a > 0 else 1.0 + (100.0/abs(a))
        except Exception:
            return None

    rows = []
    d = s_dt
    while d <= e_dt:
        day = d.strftime('%Y-%m-%d')
        p = PROC_DIR / f"predictions_{day}.csv"
        if not p.exists():
            d += _td(days=1); continue
        try:
            df = pd.read_csv(p)
        except Exception:
            d += _td(days=1); continue
        for _, r in df.iterrows():
            # Moneyline pick
            ph = _num(r.get('p_home_ml')); pa = _num(r.get('p_away_ml'))
            if ph is not None and pa is not None:
                # choose higher EV side, compute EV with chosen price (open/close)
                h_od = _num(r.get('close_home_ml_odds') if use_close else r.get('home_ml_odds'))
                a_od = _num(r.get('close_away_ml_odds') if use_close else r.get('away_ml_odds'))
                def _ev(p, od):
                    dec = _dec(od); return (p * (dec-1.0) - (1.0 - p)) if (p is not None and dec is not None) else None
                ev_h = _ev(ph, h_od); ev_a = _ev(pa, a_od)
                if ev_h is not None or ev_a is not None:
                    if (ev_h or -999) >= (ev_a or -999):
                        pick='home_ml'; ev=ev_h; price=h_od
                    else:
                        pick='away_ml'; ev=ev_a; price=a_od
                    # outcome
                    try:
                        fh=_num(r.get('final_home_goals')); fa=_num(r.get('final_away_goals'))
                        won=None
                        if fh is not None and fa is not None:
                            won = (fh>fa) if pick=='home_ml' else (fh<fa)
                    except Exception:
                        won=None
                    rows.append({'date': day,'market':'moneyline','pick':pick,'ev':ev,'price':price,'won':won})
            # Totals pick
            po = _num(r.get('p_over')); pu = _num(r.get('p_under'))
            tl = _num(r.get('close_total_line_used') if use_close else r.get('total_line_used'))
            if po is not None and pu is not None:
                o_od = _num(r.get('close_over_odds') if use_close else r.get('over_odds'))
                u_od = _num(r.get('close_under_odds') if use_close else r.get('under_odds'))
                def _ev_to(p, od):
                    dec=_dec(od); return (p*(dec-1.0) - (1.0 - p)) if (p is not None and dec is not None) else None
                ev_o=_ev_to(po, o_od); ev_u=_ev_to(pu, u_od)
                if ev_o is not None or ev_u is not None:
                    if (ev_o or -999) >= (ev_u or -999):
                        pick='over'; ev=ev_o; price=o_od
                    else:
                        pick='under'; ev=ev_u; price=u_od
                    # outcome (push ignored)
                    won=None
                    try:
                        fh=_num(r.get('final_home_goals')); fa=_num(r.get('final_away_goals'))
                        if fh is not None and fa is not None and tl is not None:
                            at=fh+fa; 
                            if abs(tl - round(tl)) < 1e-9 and int(round(tl)) == int(at):
                                won=None
                            else:
                                won = (at>tl) if pick=='over' else (at<tl)
                    except Exception:
                        won=None
                    rows.append({'date': day,'market':'totals','pick':pick,'ev':ev,'price':price,'won':won})
            # Puckline pick (+/-1.5)
            try:
                phc = _num(r.get('p_home_pl_-1.5')); pac = _num(r.get('p_away_pl_+1.5'))
            except Exception:
                phc = pac = None
            if phc is not None and pac is not None:
                hpl = _num(r.get('close_home_pl_-1.5_odds') if use_close else r.get('home_pl_-1.5_odds'))
                apl = _num(r.get('close_away_pl_+1.5_odds') if use_close else r.get('away_pl_+1.5_odds'))
                def _ev_pl(p, od):
                    dec=_dec(od); return (p*(dec-1.0) - (1.0 - p)) if (p is not None and dec is not None) else None
                ev_h = _ev_pl(phc, hpl); ev_a = _ev_pl(pac, apl)
                if ev_h is not None or ev_a is not None:
                    if (ev_h or -999) >= (ev_a or -999):
                        pick='home_-1.5'; ev=ev_h; price=hpl
                    else:
                        pick='away_+1.5'; ev=ev_a; price=apl
                    won=None
                    try:
                        fh=_num(r.get('final_home_goals')); fa=_num(r.get('final_away_goals'))
                        if fh is not None and fa is not None:
                            diff = (fh - fa)
                            won = (diff > 1.5) if pick=='home_-1.5' else (diff < 1.5)
                    except Exception:
                        won=None
                    rows.append({'date': day,'market':'puckline','pick':pick,'ev':ev,'price':price,'won':won})
            # First-10 pick (Yes/No)
            try:
                py = _num(r.get('p_f10_yes')); pn = _num(r.get('p_f10_no'))
            except Exception:
                py = pn = None
            if py is not None and pn is not None:
                y_od = _num(r.get('close_f10_yes_odds') if use_close else r.get('f10_yes_odds'))
                n_od = _num(r.get('close_f10_no_odds') if use_close else r.get('f10_no_odds'))
                def _ev_f10(p, od):
                    dec=_dec(od); return (p*(dec-1.0) - (1.0 - p)) if (p is not None and dec is not None) else None
                ev_y = _ev_f10(py, y_od); ev_n = _ev_f10(pn, n_od)
                if ev_y is not None or ev_n is not None:
                    if (ev_y or -999) >= (ev_n or -999):
                        pick='f10_yes'; ev=ev_y; price=y_od
                    else:
                        pick='f10_no'; ev=ev_n; price=n_od
                    won=None
                    try:
                        rf = r.get('result_first10')
                        if isinstance(rf, str) and rf:
                            won = (rf.strip().lower()==('yes' if pick=='f10_yes' else 'no'))
                    except Exception:
                        won=None
                    rows.append({'date': day,'market':'first10','pick':pick,'ev':ev,'price':price,'won':won})
        d += _td(days=1)

    if not rows:
        print('No candidate rows in range.'); return
    rdf = pd.DataFrame(rows)
    # Sweep thresholds 0.00..0.10 step 0.005
    def best_thresh(sub: pd.DataFrame) -> float:
        if sub.empty: return 0.0
        # pandas.np was removed; use numpy directly
        thresholds = [round(x,3) for x in list(np.arange(0.0, 0.1001, 0.005))]
        best_t = 0.0; best_roi = -1e9; best_staked = 0.0
        for t in thresholds:
            g = sub[pd.to_numeric(sub['ev'], errors='coerce') >= t]
            if g.empty: continue
            # stake=100 flat; payouts require decimal
            def _decx(od):
                return 1.0 + (float(od)/100.0) if float(od) > 0 else 1.0 + (100.0/abs(float(od)))
            staked = 100.0 * g['won'].notna().sum()
            pnl = 0.0
            for _, rr in g.iterrows():
                if pd.isna(rr.get('won')):
                    continue
                if bool(rr.get('won')):
                    pnl += 100.0 * (_decx(rr.get('price')) - 1.0)
                else:
                    pnl -= 100.0
            roi = (pnl/staked) if staked>0 else -1e9
            # prefer higher staked when ROI ties
            if (roi > best_roi) or (abs(roi - best_roi) < 1e-9 and staked > best_staked):
                best_roi = roi; best_t = t; best_staked = staked
        return float(best_t)

    ml_t = best_thresh(rdf[rdf['market']=='moneyline'])
    to_t = best_thresh(rdf[rdf['market']=='totals'])
    pl_t = best_thresh(rdf[rdf['market']=='puckline']) if 'puckline' in set(rdf['market'].unique()) else 0.0
    f10_t = best_thresh(rdf[rdf['market']=='first10']) if 'first10' in set(rdf['market'].unique()) else 0.0
    per_t = best_thresh(rdf[rdf['market']=='periods']) if 'periods' in set(rdf['market'].unique()) else 0.0

    # Write into model_calibration.json (merge with existing)
    cal_path = Path(out_json) if out_json else PROC_DIR / 'model_calibration.json'
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {}
    if cal_path.exists():
        try:
            with open(cal_path, 'r', encoding='utf-8') as f:
                obj = json.load(f) or {}
        except Exception:
            obj = {}
    # Smoothing helper
    def _smooth(old_val: float | None, learned: float) -> float:
        try:
            a = max(0.0, min(1.0, float(alpha)))
            mc = max(0.0, float(min_change))
            ov = float(old_val) if (old_val is not None) else learned
            # Only update if change exceeds min_change
            if abs(learned - ov) < mc:
                return float(ov)
            return float(ov*(1.0 - a) + learned*a)
        except Exception:
            return float(learned)

    obj['min_ev_ml'] = _smooth(obj.get('min_ev_ml'), float(ml_t))
    obj['min_ev_totals'] = _smooth(obj.get('min_ev_totals'), float(to_t))
    obj['min_ev_pl'] = _smooth(obj.get('min_ev_pl'), float(pl_t))
    obj['min_ev_first10'] = _smooth(obj.get('min_ev_first10'), float(f10_t))
    obj['min_ev_periods'] = _smooth(obj.get('min_ev_periods'), float(per_t))
    obj['ev_gates_last_learned_utc'] = _dt.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    obj['ev_gates_range'] = {'start': start, 'end': end}
    with open(cal_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    print({"min_ev_ml": obj['min_ev_ml'], "min_ev_totals": obj['min_ev_totals'], "min_ev_pl": obj['min_ev_pl'], "min_ev_first10": obj['min_ev_first10'], "min_ev_periods": obj['min_ev_periods'], "path": str(cal_path), "alpha": alpha, "min_change": min_change})

@app.command()
def game_daily_monitor(
    window_days: int = typer.Option(30, help="Lookback window in days ending today (ET)"),
    use_close: bool = typer.Option(True, help="Prefer closing odds/lines when available"),
    out_json: str = typer.Option("", help="Optional output path; default data/processed/game_daily_monitor.json"),
):
    """Summarize recent performance (ROI, accuracy, volatility, CLV) and gate thresholds.

    Markets: moneyline, totals, puckline, first10, periods (P1..P3 grouped).
    Sources: predictions_{date}.csv for each day in lookback.
    EV gating: min_ev_* thresholds from model_calibration.json.
    Metrics per market: n, decided, staked, pnl, roi, acc, ret_std, clv_mean/med/std.
    """
    from datetime import datetime as _dt, timedelta as _td
    from pathlib import Path
    import json, math, statistics as _stats
    import pandas as pd
    from .utils.io import PROC_DIR
    from .utils.odds import american_to_decimal

    today = _dt.utcnow().strftime('%Y-%m-%d')
    end_dt = _dt.strptime(today, '%Y-%m-%d')
    start_dt = end_dt - _td(days=int(window_days) - 1)

    gates = {"min_ev_ml":0.0,"min_ev_totals":0.0,"min_ev_pl":0.0,"min_ev_first10":0.0,"min_ev_periods":0.0}
    cal_path = PROC_DIR / 'model_calibration.json'
    if cal_path.exists():
        try:
            obj = json.load(open(cal_path, 'r', encoding='utf-8')) or {}
            for k in gates:
                if obj.get(k) is not None:
                    gates[k] = float(obj.get(k))
        except Exception:
            pass

    def _num(v):
        try:
            if v is None: return None
            f = float(v)
            return f if math.isfinite(f) else None
        except Exception:
            return None

    def _ev(p: float, american: float) -> float | None:
        try:
            if p is None or american is None: return None
            dec = american_to_decimal(float(american))
            if dec is None or not math.isfinite(dec): return None
            p = float(p)
            if not (0.0 <= p <= 1.0): return None
            return round(p*(dec-1.0) - (1.0 - p), 4)
        except Exception:
            return None

    plays = []
    d = start_dt
    while d <= end_dt:
        day = d.strftime('%Y-%m-%d')
        pred_path = PROC_DIR / f'predictions_{day}.csv'
        if not pred_path.exists():
            d += _td(days=1); continue
        try:
            df = pd.read_csv(pred_path)
        except Exception:
            d += _td(days=1); continue
        for _, r in df.iterrows():
            fh = _num(r.get('final_home_goals')); fa = _num(r.get('final_away_goals'))
            # Moneyline
            ph = _num(r.get('p_home_ml')); pa = _num(r.get('p_away_ml'))
            h_od = _num(r.get('close_home_ml_odds')) if use_close else _num(r.get('home_ml_odds'))
            a_od = _num(r.get('close_away_ml_odds')) if use_close else _num(r.get('away_ml_odds'))
            if ph is not None and pa is not None and h_od is not None and a_od is not None:
                ev_h = _ev(ph, h_od); ev_a = _ev(pa, a_od)
                if (ev_h is not None and ev_h >= gates['min_ev_ml']) or (ev_a is not None and ev_a >= gates['min_ev_ml']):
                    if (ev_h or -999) >= (ev_a or -999): pick='home_ml'; ev=ev_h; price=h_od; p_pick=ph; won=(fh>fa) if (fh is not None and fa is not None) else None; close_price=_num(r.get('close_home_ml_odds'))
                    else: pick='away_ml'; ev=ev_a; price=a_od; p_pick=pa; won=(fa>fh) if (fh is not None and fa is not None) else None; close_price=_num(r.get('close_away_ml_odds'))
                    if ev is not None and ev >= gates['min_ev_ml']:
                        clv=None
                        if close_price is not None:
                            dec_cp = american_to_decimal(close_price)
                            if dec_cp is not None and math.isfinite(dec_cp):
                                imp_cp = 1.0/dec_cp; clv = round(p_pick - imp_cp,4)
                        plays.append({'date':day,'market':'moneyline','ev':ev,'odds':price,'won':won,'prob':p_pick,'close_odds':close_price,'clv':clv})
            # Totals
            po = _num(r.get('p_over')); pu = _num(r.get('p_under'))
            o_od = _num(r.get('close_over_odds')) if use_close else _num(r.get('over_odds'))
            u_od = _num(r.get('close_under_odds')) if use_close else _num(r.get('under_odds'))
            tl = _num(r.get('close_total_line_used')) if use_close else _num(r.get('total_line_used'))
            if po is not None and pu is not None and o_od is not None and u_od is not None and tl is not None:
                ev_o=_ev(po,o_od); ev_u=_ev(pu,u_od)
                if (ev_o is not None and ev_o >= gates['min_ev_totals']) or (ev_u is not None and ev_u >= gates['min_ev_totals']):
                    if (ev_o or -999) >= (ev_u or -999): pick='over'; ev=ev_o; price=o_od; p_pick=po; close_price=_num(r.get('close_over_odds'))
                    else: pick='under'; ev=ev_u; price=u_od; p_pick=pu; close_price=_num(r.get('close_under_odds'))
                    won=None
                    if fh is not None and fa is not None:
                        at = fh+fa
                        if abs(tl - round(tl)) < 1e-9 and abs(at - tl) < 1e-9: won=None
                        else: won = (at>tl) if pick=='over' else (at<tl)
                    if ev is not None and ev >= gates['min_ev_totals']:
                        clv=None
                        if close_price is not None:
                            dec_cp = american_to_decimal(close_price)
                            if dec_cp is not None and math.isfinite(dec_cp):
                                imp_cp = 1.0/dec_cp; clv = round(p_pick - imp_cp,4)
                        plays.append({'date':day,'market':'totals','ev':ev,'odds':price,'won':won,'prob':p_pick,'close_odds':close_price,'clv':clv})
            # Puckline
            php = _num(r.get('p_home_pl_-1.5')); pap = _num(r.get('p_away_pl_+1.5'))
            hpl = _num(r.get('close_home_pl_-1.5_odds')) if use_close else _num(r.get('home_pl_-1.5_odds'))
            apl = _num(r.get('close_away_pl_+1.5_odds')) if use_close else _num(r.get('away_pl_+1.5_odds'))
            if php is not None and pap is not None and hpl is not None and apl is not None:
                ev_hpl=_ev(php,hpl); ev_apl=_ev(pap,apl)
                if (ev_hpl is not None and ev_hpl >= gates['min_ev_pl']) or (ev_apl is not None and ev_apl >= gates['min_ev_pl']):
                    if (ev_hpl or -999) >= (ev_apl or -999): pick='home_-1.5'; ev=ev_hpl; price=hpl; p_pick=php; won=None; close_price=_num(r.get('close_home_pl_-1.5_odds'))
                    else: pick='away_+1.5'; ev=ev_apl; price=apl; p_pick=pap; won=None; close_price=_num(r.get('close_away_pl_+1.5_odds'))
                    if fh is not None and fa is not None:
                        diff = fh - fa; won = (diff>1.5) if pick=='home_-1.5' else (diff<1.5)
                    if ev is not None and ev >= gates['min_ev_pl']:
                        clv=None
                        if close_price is not None:
                            dec_cp = american_to_decimal(close_price)
                            if dec_cp is not None and math.isfinite(dec_cp):
                                imp_cp = 1.0/dec_cp; clv = round(p_pick - imp_cp,4)
                        plays.append({'date':day,'market':'puckline','ev':ev,'odds':price,'won':won,'prob':p_pick,'close_odds':close_price,'clv':clv})
            # First10
            py = _num(r.get('p_f10_yes')); pn = _num(r.get('p_f10_no'))
            y_od = _num(r.get('close_f10_yes_odds')) if use_close else _num(r.get('f10_yes_odds'))
            n_od = _num(r.get('close_f10_no_odds')) if use_close else _num(r.get('f10_no_odds'))
            rf = str(r.get('result_first10') or '').strip().lower()
            if py is not None and pn is not None and y_od is not None and n_od is not None:
                ev_y=_ev(py,y_od); ev_n=_ev(pn,n_od)
                if (ev_y is not None and ev_y >= gates['min_ev_first10']) or (ev_n is not None and ev_n >= gates['min_ev_first10']):
                    if (ev_y or -999) >= (ev_n or -999): pick='f10_yes'; ev=ev_y; price=y_od; p_pick=py; close_price=_num(r.get('close_f10_yes_odds'))
                    else: pick='f10_no'; ev=ev_n; price=n_od; p_pick=pn; close_price=_num(r.get('close_f10_no_odds'))
                    won=None
                    if rf in ('yes','no'): won = (rf == ('yes' if pick=='f10_yes' else 'no'))
                    if ev is not None and ev >= gates['min_ev_first10']:
                        clv=None
                        if close_price is not None:
                            dec_cp = american_to_decimal(close_price)
                            if dec_cp is not None and math.isfinite(dec_cp):
                                imp_cp = 1.0/dec_cp; clv = round(p_pick - imp_cp,4)
                        plays.append({'date':day,'market':'first10','ev':ev,'odds':price,'won':won,'prob':p_pick,'close_odds':close_price,'clv':clv})
            # Period totals (aggregate P1..P3 as one market label 'periods')
            for pn_ in (1,2,3):
                po = _num(r.get(f'p{pn_}_over_prob')); pu = _num(r.get(f'p{pn_}_under_prob'))
                o_od = _num(r.get(f'close_p{pn_}_over_odds')) if use_close else _num(r.get(f'p{pn_}_over_odds'))
                u_od = _num(r.get(f'close_p{pn_}_under_odds')) if use_close else _num(r.get(f'p{pn_}_under_odds'))
                ln = _num(r.get(f'close_p{pn_}_total_line')) if use_close else _num(r.get(f'p{pn_}_total_line'))
                rk = str(r.get(f'result_p{pn_}_total') or '')
                if po is None or pu is None or o_od is None or u_od is None or ln is None: continue
                ev_o=_ev(po,o_od); ev_u=_ev(pu,u_od)
                if (ev_o is not None and ev_o >= gates['min_ev_periods']) or (ev_u is not None and ev_u >= gates['min_ev_periods']):
                    if (ev_o or -999) >= (ev_u or -999): pick='over'; ev=ev_o; price=o_od; p_pick=po; close_price=_num(r.get(f'close_p{pn_}_over_odds'))
                    else: pick='under'; ev=ev_u; price=u_od; p_pick=pu; close_price=_num(r.get(f'close_p{pn_}_under_odds'))
                    won=None
                    if rk in ('Over','Under','Push'):
                        if rk=='Push': won=None
                        else: won = (rk==('Over' if pick=='over' else 'Under'))
                    if ev is not None and ev >= gates['min_ev_periods']:
                        clv=None
                        if close_price is not None:
                            dec_cp = american_to_decimal(close_price)
                            if dec_cp is not None and math.isfinite(dec_cp):
                                imp_cp = 1.0/dec_cp; clv = round(p_pick - imp_cp,4)
                        plays.append({'date':day,'market':'periods','ev':ev,'odds':price,'won':won,'prob':p_pick,'close_odds':close_price,'clv':clv})
        d += _td(days=1)

    if not plays:
        print('{"empty": true}'); return
    pdf = pd.DataFrame(plays)

    def _summ(sub: pd.DataFrame) -> dict:
        st = 100.0 * sub['won'].notna().sum(); pnl=0.0; rets=[]; clvs=[]
        for _, rr in sub.iterrows():
            if rr.get('won') is None: continue
            dec = american_to_decimal(rr.get('odds'))
            if dec is None or not math.isfinite(dec): continue
            ret = (dec - 1.0) if rr.get('won') else -1.0
            rets.append(ret)
            pnl += (100.0 * (dec - 1.0)) if rr.get('won') else -100.0
            c = rr.get('clv');
            if c is not None: clvs.append(c)
        roi = (pnl/st) if st>0 else None
        acc = None
        try:
            decided = sub.dropna(subset=['won'])
            acc = float(decided['won'].astype(bool).mean()) if decided.shape[0] else None
        except Exception: acc=None
        ret_std = round(_stats.pstdev(rets),4) if len(rets)>1 else None
        clv_mean = round(sum(clvs)/len(clvs),4) if clvs else None
        clv_med = round(sorted(clvs)[len(clvs)//2],4) if clvs else None
        clv_std = round(_stats.pstdev(clvs),4) if len(clvs)>1 else None
        return {"n": int(len(sub)), "decided": int(sub['won'].notna().sum()), "staked": st, "pnl": pnl, "roi": roi, "acc": acc,
                "ret_std": ret_std, "clv_mean": clv_mean, "clv_med": clv_med, "clv_std": clv_std}

    markets = {m: _summ(g) for m, g in pdf.groupby('market')}
    overall = _summ(pdf)
    out_obj = {"range": {"start": start_dt.strftime('%Y-%m-%d'), "end": end_dt.strftime('%Y-%m-%d')}, "gates": gates, "markets": markets, "overall": overall}
    out_path = Path(out_json) if out_json else PROC_DIR / 'game_daily_monitor.json'
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(out_obj, f, indent=2)
    except Exception:
        pass
    print(json.dumps(out_obj, indent=2))

@app.command()
def game_monitor_anomalies(
    monitor_json: str = typer.Option("", help="Path to existing game_daily_monitor.json; default = processed dir"),
    out_json: str = typer.Option("", help="Output alerts path; default = data/processed/monitor_alerts_{today}.json"),
    roi_drop: float = typer.Option(-0.10, help="Overall ROI anomaly threshold"),
    market_roi_drop: float = typer.Option(-0.20, help="Per-market ROI anomaly threshold"),
    clv_floor: float = typer.Option(0.05, help="CLV mean floor; below -> anomaly"),
    acc_floor_ml: float = typer.Option(0.44, help="Moneyline accuracy floor"),
    acc_floor_pl: float = typer.Option(0.54, help="Puckline accuracy floor"),
):
    """Detect performance anomalies from the latest game_daily_monitor metrics and write alerts JSON."""
    import json, math
    from datetime import datetime as _dt
    from pathlib import Path
    from .utils.io import PROC_DIR
    today = _dt.utcnow().strftime('%Y-%m-%d')
    mon_path = Path(monitor_json) if monitor_json else PROC_DIR / 'game_daily_monitor.json'
    if not mon_path.exists():
        print(json.dumps({"ok": False, "reason": "monitor_missing", "path": str(mon_path)})); return
    try:
        data = json.loads(mon_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(json.dumps({"ok": False, "reason": "read_error", "error": str(e)})); return
    anomalies = []
    overall = data.get('overall') or {}
    o_roi = overall.get('roi')
    if isinstance(o_roi, (int,float)) and math.isfinite(o_roi) and o_roi < roi_drop:
        anomalies.append({"scope":"overall","type":"roi_drop","value":o_roi,"threshold":roi_drop})
    markets = data.get('markets') or {}
    for mk, stats in markets.items():
        if not isinstance(stats, dict): continue
        m_roi = stats.get('roi'); m_acc = stats.get('acc'); m_clv = stats.get('clv_mean')
        if isinstance(m_roi,(int,float)) and math.isfinite(m_roi) and m_roi < market_roi_drop:
            anomalies.append({"scope":mk,"type":"roi_drop","value":m_roi,"threshold":market_roi_drop})
        if mk == 'moneyline' and isinstance(m_acc,(int,float)) and math.isfinite(m_acc) and m_acc < acc_floor_ml:
            anomalies.append({"scope":mk,"type":"accuracy_low","value":m_acc,"threshold":acc_floor_ml})
        if mk == 'puckline' and isinstance(m_acc,(int,float)) and math.isfinite(m_acc) and m_acc < acc_floor_pl:
            anomalies.append({"scope":mk,"type":"accuracy_low","value":m_acc,"threshold":acc_floor_pl})
        if isinstance(m_clv,(int,float)) and math.isfinite(m_clv) and m_clv < clv_floor:
            anomalies.append({"scope":mk,"type":"clv_low","value":m_clv,"threshold":clv_floor})
    out_path = Path(out_json) if out_json else PROC_DIR / f'monitor_alerts_{today}.json'
    alert_obj = {"date": today, "source": str(mon_path), "anomalies": anomalies, "counts": {"total": len(anomalies)}}
    try:
        out_path.write_text(json.dumps(alert_obj, indent=2), encoding='utf-8')
    except Exception:
        pass
    print(json.dumps({"ok": True, "alerts_path": str(out_path), "anomalies": anomalies}))

@app.command()
def game_recompute_edges(
    date: Optional[str] = typer.Option(None, help="Slate date YYYY-MM-DD (ET); default = today (ET)"),
    write_recommendations: bool = typer.Option(False, help="If true, also regenerate recommendations_{date}.csv using existing logic"),
):
    """Recompute EVs/edges for team markets from predictions_{date}.csv and write edges_{date}.csv.

    - Reads data/processed/predictions_{date}.csv
    - Ensures EV fields exist for: ML (home/away), Totals (over/under with push), PL (-1.5/+1.5), First10 (Yes/No), Period totals (P1..P3 Over/Under)
    - Writes back updated predictions and edges long-form CSV
    """
    import math
    import json
    import pandas as pd
    from datetime import datetime as _dt
    from .utils.io import PROC_DIR, save_df
    from .utils.odds import american_to_decimal, decimal_to_implied_prob, remove_vig_two_way

    def _today_et() -> str:
        try:
            from zoneinfo import ZoneInfo as _Z
            return _dt.now(_Z("America/New_York")).strftime("%Y-%m-%d")
        except Exception:
            return _dt.utcnow().strftime("%Y-%m-%d")

    d = date or _today_et()
    pred_path = PROC_DIR / f"predictions_{d}.csv"
    if not pred_path.exists():
        print(json.dumps({"ok": False, "date": d, "reason": "no_predictions"})); return
    try:
        df = pd.read_csv(pred_path)
    except Exception as e:
        print(json.dumps({"ok": False, "date": d, "reason": "read_failed", "error": str(e)})); return
    if df is None or df.empty:
        print(json.dumps({"ok": False, "date": d, "reason": "empty"})); return

    def _num(v):
        try:
            if v is None: return None
            if isinstance(v, (int, float)):
                f = float(v)
                return f if math.isfinite(f) else None
            s = str(v).strip().replace(',', '')
            if s == '': return None
            f = float(s)
            return f if math.isfinite(f) else None
        except Exception:
            return None

    # Poisson helpers
    from math import exp, factorial, floor
    def _pois_pmf(mu: float, k: int) -> float:
        try:
            if k < 0 or mu is None or not math.isfinite(mu) or mu < 0: return 0.0
            return float(exp(-mu) * (mu ** k) / factorial(k))
        except Exception:
            return 0.0
    def _pois_cdf(mu: float, k: int) -> float:
        try:
            if k < 0: return 0.0
            s = 0.0
            for i in range(0, k+1): s += _pois_pmf(mu, i)
            return float(min(1.0, max(0.0, s)))
        except Exception:
            return 0.0

    # Compute First10 prob if missing, and Period totals probs
    for i, r in df.iterrows():
        # First 10 min Yes/No from period1 projections or legacy fields
        try:
            if pd.isna(r.get('p_f10_yes')):
                p_yes = None
                h1 = _num(r.get('period1_home_proj')); a1 = _num(r.get('period1_away_proj'))
                if h1 is not None and a1 is not None and h1>=0 and a1>=0:
                    # Resolve factor from calibration or default 0.55
                    f = 0.55
                    try:
                        cal = json.load(open(PROC_DIR / 'model_calibration.json', 'r', encoding='utf-8'))
                        if cal.get('f10_early_factor') is not None:
                            f = float(cal.get('f10_early_factor'))
                    except Exception:
                        pass
                    lam10 = f * (h1 + a1)
                    if math.isfinite(lam10) and lam10 >= 0:
                        p_yes = 1.0 - exp(-lam10)
            if p_yes is None:
                p10 = _num(r.get('first_10min_prob'))
                if p10 is not None: p_yes = p10
                else:
                    lam = _num(r.get('first_10min_proj'))
                    if lam is not None and lam>=0: p_yes = 1.0 - exp(-lam)
            if p_yes is not None:
                df.at[i, 'p_f10_yes'] = max(0.0, min(1.0, float(p_yes)))
                df.at[i, 'p_f10_no'] = 1.0 - float(df.at[i, 'p_f10_yes'])
        except Exception:
            pass
        # Period totals probabilities P1..P3 given projections and lines
        for pn in (1,2,3):
            try:
                mu = _num(r.get(f'period{pn}_home_proj'))
                mu2 = _num(r.get(f'period{pn}_away_proj'))
                ln = _num(r.get(f'p{pn}_total_line'))
                if mu is None or mu2 is None or ln is None: continue
                tot = mu + mu2
                if not math.isfinite(tot): continue
                if abs(ln - round(ln)) < 1e-9:
                    k = int(round(ln)); p_push = _pois_pmf(tot, k)
                    p_under = _pois_cdf(tot, k-1); p_over = max(0.0, 1.0 - _pois_cdf(tot, k))
                else:
                    k = floor(ln); p_push = 0.0
                    p_under = _pois_cdf(tot, k); p_over = max(0.0, 1.0 - p_under)
                df.at[i, f'p{pn}_over_prob'] = max(0.0, min(1.0, float(p_over)))
                df.at[i, f'p{pn}_under_prob'] = max(0.0, min(1.0, float(p_under)))
                df.at[i, f'p{pn}_push_prob'] = max(0.0, min(1.0, float(p_push)))
            except Exception:
                continue

    # EV ensuring utility
    def _ensure_ev(row: pd.Series, prob_key: str, odds_key: str, ev_key: str, edge_key: Optional[str] = None) -> pd.Series:
        try:
            ev_present = (ev_key in row) and (row.get(ev_key) is not None) and not (isinstance(row.get(ev_key), float) and pd.isna(row.get(ev_key)))
            if ev_present and (edge_key is None or (edge_key in row and pd.notna(row.get(edge_key)))):
                return row
            p = None
            if prob_key in row and pd.notna(row.get(prob_key)):
                p = float(row.get(prob_key))
                if not (0.0 <= p <= 1.0) or not math.isfinite(p): p = None
            price = _num(row.get(odds_key)) if odds_key in row else None
            if price is None:
                close_map = {
                    'home_ml_odds': 'close_home_ml_odds', 'away_ml_odds': 'close_away_ml_odds',
                    'over_odds': 'close_over_odds', 'under_odds': 'close_under_odds',
                    'home_pl_-1.5_odds': 'close_home_pl_-1.5_odds', 'away_pl_+1.5_odds': 'close_away_pl_+1.5_odds',
                }
                ck = close_map.get(odds_key)
                if ck and (ck in row): price = _num(row.get(ck))
            if price is None:
                if odds_key in ('f10_yes_odds','f10_no_odds'): price = -150.0
                elif odds_key in ('over_odds','under_odds','home_pl_-1.5_odds','away_pl_+1.5_odds','p1_over_odds','p1_under_odds','p2_over_odds','p2_under_odds','p3_over_odds','p3_under_odds'):
                    price = -110.0
            if (p is not None) and (price is not None):
                dec = american_to_decimal(price)
                if dec is not None and math.isfinite(dec):
                    p_push = 0.0
                    try:
                        if prob_key in ('p_over','p_under'):
                            tl = row.get('total_line_used') if 'total_line_used' in row else None
                            if tl is None: tl = row.get('close_total_line_used') if 'close_total_line_used' in row else None
                            mt = row.get('model_total') if 'model_total' in row else None
                            if tl is not None and mt is not None:
                                tl_f = float(tl); mt_f = float(mt)
                                if math.isfinite(tl_f) and math.isfinite(mt_f) and abs(tl_f - round(tl_f)) < 1e-9:
                                    k = int(round(tl_f)); p_push = float(exp(-mt_f) * (mt_f ** k) / factorial(k)) if k >= 0 else 0.0
                    except Exception:
                        p_push = 0.0
                    p_loss = max(0.0, 1.0 - float(p) - float(p_push)) if prob_key in ('p_over','p_under') else max(0.0, 1.0 - float(p))
                    row[ev_key] = round(float(p) * (dec - 1.0) - p_loss, 4)
                    if edge_key:
                        counterpart_map = {
                            ('p_home_ml','home_ml_odds'): ('p_away_ml','away_ml_odds'), ('p_away_ml','away_ml_odds'): ('p_home_ml','home_ml_odds'),
                            ('p_over','over_odds'): ('p_under','under_odds'), ('p_under','under_odds'): ('p_over','over_odds'),
                            ('p_home_pl_-1.5','home_pl_-1.5_odds'): ('p_away_pl_+1.5','away_pl_+1.5_odds'), ('p_away_pl_+1.5','away_pl_+1.5_odds'): ('p_home_pl_-1.5','home_pl_-1.5_odds'),
                            ('p_f10_yes','f10_yes_odds'): ('p_f10_no','f10_no_odds'), ('p_f10_no','f10_no_odds'): ('p_f10_yes','f10_yes_odds'),
                            ('p1_over_prob','p1_over_odds'): ('p1_under_prob','p1_under_odds'), ('p1_under_prob','p1_under_odds'): ('p1_over_prob','p1_over_odds'),
                            ('p2_over_prob','p2_over_odds'): ('p2_under_prob','p2_under_odds'), ('p2_under_prob','p2_under_odds'): ('p2_over_prob','p2_over_odds'),
                            ('p3_over_prob','p3_over_odds'): ('p3_under_prob','p3_under_odds'), ('p3_under_prob','p3_under_odds'): ('p3_over_prob','p3_over_odds'),
                        }
                        other = counterpart_map.get((prob_key, odds_key))
                        if other:
                            p2 = None
                            if other[0] in row and pd.notna(row.get(other[0])):
                                try:
                                    p2 = float(row.get(other[0]));
                                    if not (0.0 <= p2 <= 1.0) or not math.isfinite(p2): p2 = None
                                except Exception:
                                    p2 = None
                            price2 = _num(row.get(other[1])) if other[1] in row else None
                            if price2 is not None:
                                dec1 = american_to_decimal(price); dec2 = american_to_decimal(price2)
                                if dec1 is not None and dec2 is not None:
                                    imp1 = decimal_to_implied_prob(dec1); imp2 = decimal_to_implied_prob(dec2)
                                    nv1, nv2 = remove_vig_two_way(imp1, imp2)
                                    nv = nv1 if prob_key in ('p_home_ml','p_over','p_home_pl_-1.5') else nv2
                                    row[edge_key] = round(p - nv, 4)
        except Exception:
            return row
        return row

    # Apply EV/edge computation across rows
    for i, r in df.iterrows():
        r = _ensure_ev(r, 'p_home_ml', 'home_ml_odds', 'ev_home_ml', 'edge_home_ml')
        r = _ensure_ev(r, 'p_away_ml', 'away_ml_odds', 'ev_away_ml', 'edge_away_ml')
        r = _ensure_ev(r, 'p_over', 'over_odds', 'ev_over', 'edge_over')
        r = _ensure_ev(r, 'p_under', 'under_odds', 'ev_under', 'edge_under')
        r = _ensure_ev(r, 'p_home_pl_-1.5', 'home_pl_-1.5_odds', 'ev_home_pl_-1.5', 'edge_home_pl_-1.5')
        r = _ensure_ev(r, 'p_away_pl_+1.5', 'away_pl_+1.5_odds', 'ev_away_pl_+1.5', 'edge_away_pl_+1.5')
        r = _ensure_ev(r, 'p_f10_yes', 'f10_yes_odds', 'ev_f10_yes', 'edge_f10_yes')
        r = _ensure_ev(r, 'p_f10_no', 'f10_no_odds', 'ev_f10_no', 'edge_f10_no')
        r = _ensure_ev(r, 'p1_over_prob', 'p1_over_odds', 'ev_p1_over', None)
        r = _ensure_ev(r, 'p1_under_prob', 'p1_under_odds', 'ev_p1_under', None)
        r = _ensure_ev(r, 'p2_over_prob', 'p2_over_odds', 'ev_p2_over', None)
        r = _ensure_ev(r, 'p2_under_prob', 'p2_under_odds', 'ev_p2_under', None)
        r = _ensure_ev(r, 'p3_over_prob', 'p3_over_odds', 'ev_p3_over', None)
        r = _ensure_ev(r, 'p3_under_prob', 'p3_under_odds', 'ev_p3_under', None)
    df.loc[i, r.index] = r

    # Persist predictions and edges
    save_df(df, pred_path)
    try:
        ev_cols = [c for c in df.columns if c.startswith('ev_')]
        if ev_cols:
            edges = df.melt(id_vars=['date','home','away'], value_vars=ev_cols, var_name='market', value_name='ev').dropna()
            edges = edges.sort_values('ev', ascending=False)
            edges_path = PROC_DIR / f"edges_{d}.csv"
            save_df(edges, edges_path)
    except Exception:
        pass

    # Optional: regenerate recommendations via existing CLI endpoint wrapper
    if write_recommendations:
        try:
            # Call the existing recommendations command if present
            try:
                props_recommendations.callback(date=d, min_ev=0.0, top=1000)
            except Exception:
                pass
        except Exception:
            pass

    print(json.dumps({"ok": True, "date": d, "predictions": str(pred_path), "edges": str(PROC_DIR / f'edges_{d}.csv')}))


@app.command()
def game_auto_calibrate(
    start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD (ET); default = season to date"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD (ET); default = today"),
    use_close: bool = typer.Option(True, help="Prefer closing lines/odds when available for market implied probs"),
    out_json: Optional[str] = typer.Option(None, help="Write calibration to this JSON path; default data/processed/model_calibration.json"),
):
    """Automatically calibrate team-level model hyperparameters from historical predictions and outcomes.

    Learns and persists:
      - dc_rho: Dixon–Coles low-score correlation (by maximizing exact score likelihood)
      - market_anchor_w_ml: blend weight toward no‑vig market for Moneyline (minimizes log loss)
      - market_anchor_w_totals: blend weight toward no‑vig market for Totals (minimizes log loss)
      - totals_temp: temperature scaling T (>1 flattens toward 0.5) for Totals (minimizes log loss)

    Output JSON is read by the web layer; environment toggles are no longer needed.
    """
    import math, json
    from datetime import datetime as _dt, timedelta as _td
    import numpy as np
    import pandas as pd
    from .utils.io import RAW_DIR, PROC_DIR
    from .utils.odds import american_to_decimal, remove_vig_two_way
    from .models.dixon_coles import dc_score_matrix

    # Date range defaults: season to date in ET
    def _et_today() -> str:
        try:
            from zoneinfo import ZoneInfo as _Z
            return _dt.now(_Z("America/New_York")).strftime("%Y-%m-%d")
        except Exception:
            return _dt.utcnow().strftime("%Y-%m-%d")
    end = end or _et_today()
    # Derive season start from RAW games if no start given
    if not start:
        gpath = RAW_DIR / "games.csv"
        if gpath.exists():
            try:
                gdf = pd.read_csv(gpath)
                # choose first ET date of current season boundary (July separation)
                gdf["date_et"] = pd.to_datetime(gdf.get("date_et") or gdf.get("date"), errors="coerce").dt.tz_localize(None)
                # pick rows within the season containing 'end'
                e_dt = _dt.strptime(end, "%Y-%m-%d")
                season_start_year = e_dt.year if e_dt.month >= 7 else (e_dt.year - 1)
                season_start = _dt(season_start_year, 9, 1)
                start = season_start.strftime("%Y-%m-%d")
            except Exception:
                start = end
        else:
            start = end

    def _parse_date(s: str) -> _dt:
        return _dt.strptime(s, "%Y-%m-%d")

    s_dt = _parse_date(start); e_dt = _parse_date(end)
    if e_dt < s_dt:
        s_dt, e_dt = e_dt, s_dt

    # Collect datasets
    ml_samples = []  # (p_model, p_mkt_home, y_home)
    to_samples = []  # (p_model_over, p_mkt_over, y_over, line, actual_total)
    dc_samples = []  # (lam_h, lam_a, fh, fa)

    # Finals lookup
    games_path = RAW_DIR / "games.csv"
    finals_idx = None
    if games_path.exists():
        try:
            gdf = pd.read_csv(games_path)
            for col in ("date_et","date"):
                if col in gdf.columns:
                    gdf[col] = pd.to_datetime(gdf[col], errors="coerce").dt.strftime("%Y-%m-%d")
            finals_idx = gdf.rename(columns={"home_goals":"final_home_goals","away_goals":"final_away_goals"})[["date_et","home","away","final_home_goals","final_away_goals"]]
        except Exception:
            finals_idx = None

    def _num(v):
        try:
            if v is None: return None
            if isinstance(v, (int, float)):
                f = float(v); return f if math.isfinite(f) else None
            s = str(v).strip();
            if s == "": return None
            return float(s)
        except Exception:
            return None

    d = s_dt
    while d <= e_dt:
        day = d.strftime("%Y-%m-%d")
        p = PROC_DIR / f"predictions_{day}.csv"
        if not p.exists():
            d += _td(days=1); continue
        try:
            df = pd.read_csv(p)
        except Exception:
            d += _td(days=1); continue
        # Join finals if missing
        if finals_idx is not None and ("final_home_goals" not in df.columns or df["final_home_goals"].isna().any()):
            try:
                df = df.merge(finals_idx, left_on=["date","home","away"], right_on=["date_et","home","away"], how="left", suffixes=("","_g"))
                for col in ("final_home_goals","final_away_goals"):
                    if col not in df.columns or df[col].isna().any():
                        if f"{col}_g" in df.columns:
                            df[col] = df[col].fillna(df[f"{col}_g"])  # prefer joined finals
                df = df.drop(columns=[c for c in df.columns if c.endswith("_g") or c=="date_et"], errors="ignore")
            except Exception:
                pass
        # ML samples
        for _, r in df.iterrows():
            try:
                ph_mod = _num(r.get("p_home_ml_model"))
                # market implied (no‑vig) from odds
                if use_close:
                    odds_h = _num(r.get("close_home_ml_odds")); odds_a = _num(r.get("close_away_ml_odds"))
                else:
                    odds_h = _num(r.get("home_ml_odds")); odds_a = _num(r.get("away_ml_odds"))
                p_mkt = None
                if odds_h is not None and odds_a is not None:
                    a = american_to_decimal(odds_h); b = american_to_decimal(odds_a)
                    if a and b and a>1 and b>1:
                        # no-vig pair
                        inv_a = 1.0 / a; inv_b = 1.0 / b; s = inv_a + inv_b
                        if s > 0:
                            p_mkt = inv_a / s  # home
                y_home = None
                if r.get("final_home_goals") is not None and r.get("final_away_goals") is not None:
                    y_home = 1 if float(r.get("final_home_goals")) > float(r.get("final_away_goals")) else 0
                if ph_mod is not None and p_mkt is not None and y_home is not None:
                    ml_samples.append((float(ph_mod), float(p_mkt), int(y_home)))
            except Exception:
                continue
        # Totals samples
        for _, r in df.iterrows():
            try:
                po_mod = _num(r.get("p_over_model")); pu_mod = _num(r.get("p_under_model"))
                if use_close:
                    o_odds = _num(r.get("close_over_odds")); u_odds = _num(r.get("close_under_odds"))
                    tl = _num(r.get("close_total_line_used"))
                else:
                    o_odds = _num(r.get("over_odds")); u_odds = _num(r.get("under_odds"))
                    tl = _num(r.get("total_line_used"))
                # no‑vig implied pair (may be None if odds missing)
                p_mkt_over = p_mkt_under = None
                if o_odds is not None and u_odds is not None:
                    try:
                        # convert to decimal, then remove-vig on two-sided
                        a = american_to_decimal(o_odds); b = american_to_decimal(u_odds)
                        if a and b and a>1 and b>1:
                            inv_a = 1.0 / a; inv_b = 1.0 / b; s = inv_a + inv_b
                            if s > 0:
                                p_mkt_over = inv_a / s; p_mkt_under = inv_b / s
                    except Exception:
                        pass
                # observed outcome excluding pushes
                y_over = None
                if r.get("final_home_goals") is not None and r.get("final_away_goals") is not None and tl is not None:
                    at = float(r.get("final_home_goals")) + float(r.get("final_away_goals"))
                    if not (abs(tl - round(tl)) < 1e-9 and abs(at - tl) < 1e-9):
                        y_over = 1 if at > tl else 0
                if po_mod is not None and p_mkt_over is not None and y_over is not None:
                    to_samples.append((float(po_mod), float(p_mkt_over), int(y_over), float(tl), float(at)))
            except Exception:
                continue
        # DC rho samples
        for _, r in df.iterrows():
            try:
                lam_h = _num(r.get("proj_home_goals")); lam_a = _num(r.get("proj_away_goals"))
                fh = _num(r.get("final_home_goals")); fa = _num(r.get("final_away_goals"))
                if lam_h is not None and lam_a is not None and fh is not None and fa is not None:
                    dc_samples.append((float(lam_h), float(lam_a), int(round(fh)), int(round(fa))))
            except Exception:
                continue
        d += _td(days=1)

    # Optimization helpers
    def _fit_anchor(samples):
        # samples: list of (p_mod, p_mkt, y)
        if not samples:
            return 0.25  # sensible default
        ps_mod = np.array([s[0] for s in samples], dtype=float)
        ps_mkt = np.array([s[1] for s in samples], dtype=float)
        ys = np.array([s[2] for s in samples], dtype=int)
        def nll(w):
            w = float(w)
            p = np.clip((1.0 - w) * ps_mod + w * ps_mkt, 1e-9, 1 - 1e-9)
            return float(- (ys * np.log(p) + (1 - ys) * np.log(1 - p)).sum())
        grid = np.linspace(0.0, 1.0, 51)
        vals = [nll(w) for w in grid]
        w0 = float(grid[int(np.argmin(vals))])
        return w0

    def _fit_totals_temp(samples, w_totals):
        # samples: list of (p_mod_over, p_mkt_over, y_over, line, actual_total)
        if not samples:
            return 1.0
        ps_mod = np.array([s[0] for s in samples], dtype=float)
        ps_mkt = np.array([s[1] for s in samples], dtype=float)
        ys = np.array([s[2] for s in samples], dtype=int)
        lines = np.array([s[3] for s in samples], dtype=float)
        ats = np.array([s[4] for s in samples], dtype=float)
        def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
        def _logit(p):
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return np.log(p / (1.0 - p))
        def nll(T):
            T = float(max(0.5, min(3.0, T)))
            # Blend then temperature-scale around 0.5; exclude pushes already
            p_blend = (1.0 - w_totals) * ps_mod + w_totals * ps_mkt
            lo = _logit(p_blend) / T
            p_t = _sigmoid(lo)
            p_t = np.clip(p_t, 1e-9, 1 - 1e-9)
            return float(- (ys * np.log(p_t) + (1 - ys) * np.log(1 - p_t)).sum())
        grid = np.linspace(0.8, 1.6, 41)
        vals = [nll(T) for T in grid]
        T0 = float(grid[int(np.argmin(vals))])
        return T0

    def _fit_dc_rho(samples):
        # samples: list of (lam_h, lam_a, fh, fa)
        if not samples:
            return -0.05
        rhos = np.linspace(-0.2, 0.2, 41)
        def nll(rho):
            ll = 0.0
            for (lh, la, fh, fa) in samples:
                try:
                    mat = dc_score_matrix(lh, la, rho=rho, max_goals=10)
                    p = float(mat[fh if fh<=10 else 10, fa if fa<=10 else 10])
                    p = max(p, 1e-12)
                    ll -= math.log(p)
                except Exception:
                    continue
            return float(ll)
        vals = [nll(r) for r in rhos]
        rho0 = float(rhos[int(np.argmin(vals))])
        return rho0

    w_ml = _fit_anchor(ml_samples)
    w_tot = _fit_anchor([(pm, pk, y) for (pm, pk, y, _, _) in to_samples])
    T = _fit_totals_temp(to_samples, w_tot)
    rho = _fit_dc_rho(dc_samples)

    # Write calibration JSON
    out_path = Path(out_json) if out_json else PROC_DIR / "model_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "last_calibrated_utc": _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "range": {"start": start, "end": end},
        "dc_rho": float(rho),
        # Back-compat generic key (use ML weight)
        "market_anchor_w": float(w_ml),
        "market_anchor_w_ml": float(w_ml),
        "market_anchor_w_totals": float(w_tot),
        "totals_temp": float(T),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(json.dumps({k: obj[k] for k in ("dc_rho","market_anchor_w_ml","market_anchor_w_totals","totals_temp")}, indent=2))
    print(f"Wrote calibration -> {out_path}")


@app.command()
def props_stats_backfill(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    source: str = typer.Option("stats", help="Data source: stats | web"),
):
    """Backfill player game stats (skaters + goalies) into data/raw/player_game_stats.csv.

    This pulls NHL boxscore stats for each day in [start, end] via the selected API and persists a single CSV.
    """
    try:
        collect_player_game_stats(start, end, source=source)
        print({"status": "ok", "start": start, "end": end, "path": str(RAW_DIR / 'player_game_stats.csv')})
    except Exception as e:
        print({"status": "error", "error": str(e)})


@app.command()
def props_backtest_from_projections(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    stake: float = typer.Option(100.0, help="Flat stake per play for ROI calc"),
    markets: str = typer.Option("SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS", help="Comma list of markets to include"),
    min_ev: float = typer.Option(-1.0, help="Filter to plays with EV >= min_ev; set -1 to include all"),
    source_filter: str = typer.Option("", help="If set, only use projections with this source label (e.g., 'nn', 'trad')"),
    out_prefix: str = typer.Option("nn", help="Output filename prefix under data/processed/"),
    min_ev_per_market: str = typer.Option("", help="Optional per-market EV thresholds, e.g., 'SOG=0.00,GOALS=0.04,ASSISTS=0.00,POINTS=0.08'"),
):
    """Backtest using daily projections_all (NN/trad/bias cascade) joined to canonical lines.

    For each day in [start, end]:
      - Load data/processed/props_projections_all_{date}.csv
      - Optionally filter to a specific source (nn|trad|fallback)
    - Load canonical player props lines for that date (bovada/oddsapi)
    - Join on player+market (normalized names), compute Poisson P(Over) using proj_lambda_eff when available (falls back to proj_lambda)
      - Compute EV for Over and Under (with push-prob handling); choose higher EV side
      - Evaluate outcome from data/raw/player_game_stats.csv (ET date alignment)
      - Aggregate ROI and calibration stats by market and overall
    Writes rows and summary to data/processed with a prefix.
    """
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo as _Z
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    from scipy.stats import poisson as _poisson
    from .utils.io import RAW_DIR, PROC_DIR
    from .utils.odds import american_to_decimal

    # Load realized player stats for outcomes
    stats_path = RAW_DIR / "player_game_stats.csv"
    if not stats_path.exists():
        print("player_game_stats.csv missing; run props_stats_backfill first.")
        raise typer.Exit(code=1)
    try:
        stats_all = pd.read_csv(stats_path)
    except Exception:
        print("Failed to read player_game_stats.csv; is the file empty or malformed?")
        raise typer.Exit(code=1)

    # Normalize stats date and player fields
    def _iso_to_et(iso_utc: str) -> str | None:
        if not isinstance(iso_utc, str):
            try:
                iso_utc = str(iso_utc)
            except Exception:
                return None
        if not iso_utc:
            return None
        try:
            s = iso_utc.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
        except Exception:
            try:
                dt = datetime.fromisoformat(str(iso_utc)[:19]).replace(tzinfo=timezone.utc)
            except Exception:
                return None
        try:
            return dt.astimezone(_Z("America/New_York")).strftime("%Y-%m-%d")
        except Exception:
            return None

    import ast, re, unicodedata
    def _extract_player_text(v) -> str:
        if v is None:
            return ""
        try:
            if isinstance(v, str) and v.strip().startswith("{") and v.strip().endswith("}"):
                d = ast.literal_eval(v)
                if isinstance(d, dict):
                    for k in ("default", "en", "name", "fullName", "full_name"):
                        if d.get(k):
                            return str(d.get(k))
                return str(v)
            if isinstance(v, dict):
                for k in ("default", "en", "name", "fullName", "full_name"):
                    if v.get(k):
                        return str(v.get(k))
                return str(v)
            return str(v)
        except Exception:
            return str(v)

    def _norm_name(s: str) -> str:
        s = (s or "").strip()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = re.sub(r"\s+", " ", s)
        return s.lower()

    stats_all["date_et"] = stats_all["date"].apply(_iso_to_et)
    stats_all["player_text_raw"] = stats_all["player"].apply(_extract_player_text)
    stats_all["player_text_norm"] = stats_all["player_text_raw"].apply(_norm_name)
    stats_all["player_text_nodot"] = stats_all["player_text_norm"].str.replace(".", "", regex=False)

    def _name_variants(full: str):
        full = (full or "").strip()
        parts = [p for p in full.split(" ") if p]
        vars = set()
        if full:
            nm = _norm_name(full)
            vars.add(nm)
            vars.add(nm.replace(".", ""))
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            init_last = f"{first[0]}. {last}"
            nm2 = _norm_name(init_last)
            vars.add(nm2)
            vars.add(nm2.replace(".", ""))
        return vars

    def _name_variants(full: str):
        full = (full or "").strip()
        parts = [p for p in full.split(" ") if p]
        vars = set()
        if full:
            nm = _norm_name(full)
            vars.add(nm)
            vars.add(nm.replace(".", ""))
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            init_last = f"{first[0]}. {last}"
            nm2 = _norm_name(init_last)
            vars.add(nm2)
            vars.add(nm2.replace(".", ""))
        return vars

    allowed_markets = [m.strip().upper() for m in (markets or "").split(",") if m.strip()]
    src_filter = (source_filter or "").strip().lower()

    # Parse per-market thresholds if provided
    def _parse_thresholds(s: str) -> dict[str, float]:
        d: dict[str, float] = {}
        for part in (s or "").split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip().upper()
            try:
                d[k] = float(v.strip())
            except Exception:
                continue
        return d
    per_thr = _parse_thresholds(min_ev_per_market)

    def _name_variants(full: str):
        full = (full or "").strip()
        parts = [p for p in full.split(" ") if p]
        vars = set()
        if full:
            nm = _norm_name(full)
            vars.add(nm)
            vars.add(nm.replace(".", ""))
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            init_last = f"{first[0]}. {last}"
            nm2 = _norm_name(init_last)
            vars.add(nm2)
            vars.add(nm2.replace(".", ""))
        return vars

    def _poisson_probs_over_under(lambda_, line_):
        try:
            lam = float(lambda_)
            ln = float(line_)
        except Exception:
            return None, None, None
        # Over wins when X > line; Under when X < line; Push when X == integer line
        if abs(ln - round(ln)) < 1e-9:  # integer line
            k = int(round(ln))
            p_push = float(_poisson.pmf(k, mu=lam))
            p_over = float(_poisson.sf(k, mu=lam))  # P(X >= k+1)
            # For numerical stability, compute under as 1 - p_over - p_push
            p_under = max(0.0, 1.0 - p_over - p_push)
        else:
            # Half lines: no push; P(X > ln) == P(X >= floor(ln)+1)
            k = int(np.floor(ln))
            p_over = float(_poisson.sf(k, mu=lam))
            p_push = 0.0
            p_under = max(0.0, 1.0 - p_over)
        return p_over, p_under, p_push

    def _ev(dec_odds: float | None, p_win: float, p_lose: float) -> float | None:
        if dec_odds is None:
            return None
        try:
            return p_win * (float(dec_odds) - 1.0) - p_lose
        except Exception:
            return None

    # Iterate dates
    def to_dt(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")
    cur = to_dt(start); end_dt = to_dt(end)
    rows = []
    calib = []

    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        proj_path = PROC_DIR / f"props_projections_all_{d}.csv"
        if not proj_path.exists():
            cur += timedelta(days=1)
            continue
        try:
            proj_df = pd.read_csv(proj_path)
        except Exception:
            cur += timedelta(days=1)
            continue
        if proj_df.empty:
            cur += timedelta(days=1)
            continue
        # Normalize projection fields
        proj_df["market"] = proj_df["market"].astype(str).str.upper()
        proj_df["player_norm"] = proj_df["player"].astype(str).apply(_norm_name)
        # Older files may not have 'source'; ensure it exists for grouping
        if "source" not in proj_df.columns:
            proj_df["source"] = "unknown"
        if src_filter:
            proj_df = proj_df[proj_df["source"].astype(str).str.lower() == src_filter]
        if allowed_markets:
            proj_df = proj_df[proj_df["market"].isin(allowed_markets)]
        if proj_df.empty:
            cur += timedelta(days=1)
            continue

        # Load lines for date d (bovada/oddsapi; parquet preferred)
        base = Path("data/props") / f"player_props_lines/date={d}"
        parts = []
        for prov in ("bovada", "oddsapi"):
            p_parq = base / f"{prov}.parquet"
            p_csv = base / f"{prov}.csv"
            if p_parq.exists():
                try:
                    parts.append(pd.read_parquet(p_parq))
                except Exception:
                    pass
            elif p_csv.exists():
                try:
                    parts.append(pd.read_csv(p_csv))
                except Exception:
                    pass
        if parts:
            lines_df = pd.concat(parts, ignore_index=True)
        else:
            cur += timedelta(days=1)
            continue
        if lines_df.empty:
            cur += timedelta(days=1)
            continue

        # Normalize lines fields
        lines_df["market"] = lines_df["market"].astype(str).str.upper()
        name_col = "player_name" if "player_name" in lines_df.columns else ("player" if "player" in lines_df.columns else None)
        if not name_col:
            cur += timedelta(days=1)
            continue
        lines_df["player_norm"] = lines_df[name_col].astype(str).apply(_norm_name)

        # Join projections to lines on player+market
        # Build projection columns to merge, preferring effective lambda when present
        _proj_cols = ["player_norm", "market", "proj_lambda", "source"]
        if "proj_lambda_eff" in proj_df.columns:
            _proj_cols.append("proj_lambda_eff")
        merged = lines_df.merge(
            proj_df[_proj_cols],
            on=["player_norm", "market"], how="inner",
        )
        if merged.empty:
            cur += timedelta(days=1)
            continue

        # Compute probabilities and EV, pick sides, and evaluate outcome
        for _, r in merged.iterrows():
            m = str(r.get("market") or "").upper()
            if allowed_markets and m not in allowed_markets:
                continue
            player_disp = r.get(name_col)
            ln = r.get("line")
            try:
                ln = float(ln)
            except Exception:
                continue
            # Prefer effective lambda if available; otherwise use base lambda
            lam = None
            if "proj_lambda_eff" in merged.columns and pd.notna(r.get("proj_lambda_eff")):
                try:
                    lam = float(r.get("proj_lambda_eff"))
                except Exception:
                    lam = None
            if lam is None:
                try:
                    lam = float(r.get("proj_lambda"))
                except Exception:
                    continue
            # Compute P(over), P(under), P(push)
            p_over, p_under, p_push = _poisson_probs_over_under(lam, ln)
            if p_over is None:
                continue
            # EVs
            dec_o = american_to_decimal(r.get("over_price")) if pd.notna(r.get("over_price")) else None
            dec_u = american_to_decimal(r.get("under_price")) if pd.notna(r.get("under_price")) else None
            ev_o = _ev(dec_o, p_over, p_under)
            ev_u = _ev(dec_u, p_under, p_over)
            side = None; price = None; ev = None
            if ev_o is not None or ev_u is not None:
                if (ev_u is None) or (ev_o is not None and ev_o >= ev_u):
                    side = "Over"; price = r.get("over_price"); ev = ev_o
                else:
                    side = "Under"; price = r.get("under_price"); ev = ev_u
            if ev is None:
                continue
            # Determine threshold: specific market threshold overrides global min_ev
            thr = None
            if per_thr:
                thr = per_thr.get(m)
            if thr is None:
                thr = float(min_ev) if min_ev is not None else -1.0
            if thr is not None and float(thr) > -1.0 and float(ev) < float(thr):
                continue

            # Actual outcome from stats on ET date d
            day_stats = stats_all[stats_all["date_et"] == d]
            variants = _name_variants(str(player_disp))
            ps = day_stats[day_stats["player_text_norm"].isin(variants) | day_stats["player_text_nodot"].isin(variants)]
            actual = None
            if not ps.empty:
                row = ps.iloc[0]
                if m == "SOG":
                    actual = row.get("shots")
                elif m == "SAVES":
                    actual = row.get("saves")
                elif m == "GOALS":
                    actual = row.get("goals")
                elif m == "ASSISTS":
                    actual = row.get("assists")
                elif m == "POINTS":
                    try:
                        actual = float((row.get("goals") or 0)) + float((row.get("assists") or 0))
                    except Exception:
                        actual = None
                elif m == "BLOCKS":
                    actual = row.get("blocked")
            result = None
            if actual is not None and pd.notna(actual):
                try:
                    av = float(actual)
                    if av > ln:
                        over_res = "win"
                    elif av < ln:
                        over_res = "loss"
                    else:
                        over_res = "push"
                    if side == "Over":
                        result = over_res
                    elif side == "Under":
                        if over_res == "win":
                            result = "loss"
                        elif over_res == "loss":
                            result = "win"
                        else:
                            result = "push"
                except Exception:
                    result = None

            # Payout calc
            payout = None
            if result is not None:
                dec = american_to_decimal(price)
                if result == "win" and dec is not None:
                    payout = stake * (dec - 1.0)
                elif result == "loss":
                    payout = -stake
                elif result == "push":
                    payout = 0.0

            rows.append({
                "date": d,
                "market": m,
                "player": player_disp,
                "line": float(ln),
                "book": r.get("book"),
                "over_price": r.get("over_price") if pd.notna(r.get("over_price")) else None,
                "under_price": r.get("under_price") if pd.notna(r.get("under_price")) else None,
                "proj_lambda": float(r.get("proj_lambda")) if pd.notna(r.get("proj_lambda")) else None,
                "proj_lambda_eff": float(r.get("proj_lambda_eff")) if ("proj_lambda_eff" in merged.columns and pd.notna(r.get("proj_lambda_eff"))) else None,
                "p_over": float(p_over),
                "p_under": float(p_under),
                "p_push": float(p_push),
                "side": side,
                "ev": float(ev) if ev is not None else None,
                "actual": actual,
                "result": result,
                "stake": stake,
                "payout": payout,
                "source": r.get("source"),
            })
            # Calibration record for over outcome
            if actual is not None and pd.notna(actual):
                calib.append({
                    "date": d,
                    "market": m,
                    "p_over": float(p_over),
                    "over_won": bool(float(actual) > float(ln)),
                })
        cur += timedelta(days=1)

    # Summaries
    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        print("No backtest rows generated. Ensure projections and lines exist for the range.")
        raise typer.Exit(code=0)

    def summarize(df: pd.DataFrame) -> dict:
        d = {
            "picks": int(len(df)),
            "decided": int(df["result"].isin(["win","loss"]).sum()),
            "wins": int((df["result"] == "win").sum()),
            "losses": int((df["result"] == "loss").sum()),
            "pushes": int((df["result"] == "push").sum()),
        }
        # Accuracy (exclude pushes)
        try:
            dec = df[df["result"].isin(["win","loss"])].copy()
            d["accuracy"] = float((dec["result"] == "win").mean()) if len(dec) > 0 else None
        except Exception:
            d["accuracy"] = None
        # Brier score for chosen side if probabilities present
        try:
            p_chosen = np.where(df["side"].astype(str)=="Over", df["p_over"].astype(float), (1.0 - df["p_over"].astype(float)))
            y = np.where(df["result"].astype(str)=="win", 1.0, np.where(df["result"].astype(str)=="loss", 0.0, np.nan))
            mask = ~np.isnan(p_chosen) & ~np.isnan(y)
            d["brier"] = float(np.mean((p_chosen[mask] - y[mask])**2)) if np.any(mask) else None
        except Exception:
            d["brier"] = None
        return d

    overall = summarize(rows_df)
    by_market = {mkt: summarize(g) for mkt, g in rows_df.groupby("market")}
    by_source = {src: summarize(g) for src, g in rows_df.groupby("source")}

    calib_df = pd.DataFrame(calib)
    calib_bins = []
    if not calib_df.empty:
        bins = np.linspace(0.0, 1.0, 11)
        calib_df["bin"] = pd.cut(calib_df["p_over"], bins=bins, include_lowest=True)
        for b, g in calib_df.groupby("bin"):
            try:
                exp = float(g["p_over"].mean())
                obs = float(g["over_won"].mean())
                cnt = int(len(g))
                calib_bins.append({"bin": str(b), "expected": exp, "observed": obs, "count": cnt})
            except Exception:
                continue

    pref = (out_prefix.strip() + "_") if out_prefix.strip() else ""
    rows_path = PROC_DIR / f"{pref}props_backtest_rows_{start}_to_{end}.csv"
    summ_path = PROC_DIR / f"{pref}props_backtest_summary_{start}_to_{end}.json"
    rows_df.to_csv(rows_path, index=False)
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall": overall,
            "by_market": by_market,
            "by_source": by_source,
            "calibration": calib_bins,
            "filters": {"markets": allowed_markets, "source_filter": src_filter or None, "min_ev": min_ev}
        }, f, indent=2)
    print(json.dumps({"overall": overall, "by_market": by_market, "by_source": by_source}, indent=2))
    print(f"Saved rows to {rows_path} and summary to {summ_path}")


@app.command()
def props_backtest_from_simulations(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    stake: float = typer.Option(100.0, help="Flat stake per play for ROI calc"),
    markets: str = typer.Option("SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS", help="Comma list of markets to include"),
    min_ev: float = typer.Option(-1.0, help="Filter to plays with EV >= min_ev; set -1 to include all"),
    out_prefix: str = typer.Option("sim", help="Output filename prefix under data/processed/"),
    min_ev_per_market: str = typer.Option("", help="Optional per-market EV thresholds, e.g., 'SOG=0.00,GOALS=0.04,ASSISTS=0.00,POINTS=0.08'"),
):
    """Backtest using simulation-backed p_over from props_simulations_{date}.csv joined to canonical lines.

    For each day in [start, end]:
      - Load data/processed/props_simulations_{date}.csv
      - Compute EV for Over and Under from p_over_sim and odds; choose higher EV side
      - Evaluate outcome from data/raw/player_game_stats.csv (ET date alignment)
      - Aggregate ROI and calibration stats by market and overall
    Writes rows and summary to data/processed with a prefix.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo as _Z
    import json
    import numpy as np
    import pandas as pd
    from .utils.io import RAW_DIR, PROC_DIR
    from .utils.odds import american_to_decimal

    # Load realized player stats for outcomes
    stats_path = RAW_DIR / "player_game_stats.csv"
    if not stats_path.exists():
        print("player_game_stats.csv missing; run props_stats_backfill first.")
        raise typer.Exit(code=1)
    try:
        stats_all = pd.read_csv(stats_path)
    except Exception:
        print("Failed to read player_game_stats.csv; is the file empty or malformed?")
        raise typer.Exit(code=1)

    # Normalize stats date and player fields (reuse logic from projections backtest)
    def _iso_to_et(iso_utc: str) -> str | None:
        try:
            s = str(iso_utc or "").replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
        except Exception:
            try:
                dt = datetime.fromisoformat(str(iso_utc)[:19]).replace(tzinfo=datetime.timezone.utc)  # type: ignore
            except Exception:
                return None
        try:
            return dt.astimezone(_Z("America/New_York")).strftime("%Y-%m-%d")
        except Exception:
            return None

    import ast, re, unicodedata
    def _extract_player_text(v) -> str:
        if v is None:
            return ""
        try:
            if isinstance(v, str) and v.strip().startswith("{") and v.strip().endswith("}"):
                d = ast.literal_eval(v)
                if isinstance(d, dict):
                    for k in ("default", "en", "name", "fullName", "full_name"):
                        if d.get(k):
                            return str(d.get(k))
                return str(v)
            if isinstance(v, dict):
                for k in ("default", "en", "name", "fullName", "full_name"):
                    if v.get(k):
                        return str(v.get(k))
                return str(v)
            return str(v)
        except Exception:
            return str(v)

    def _norm_name(s: str) -> str:
        s = (s or "").strip()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = re.sub(r"\s+", " ", s)
        return s.lower()

    stats_all["date_et"] = stats_all["date"].apply(_iso_to_et)
    stats_all["player_text_raw"] = stats_all["player"].apply(_extract_player_text)
    stats_all["player_text_norm"] = stats_all["player_text_raw"].apply(_norm_name)
    stats_all["player_text_nodot"] = stats_all["player_text_norm"].str.replace(".", "", regex=False)

    allowed_markets = [m.strip().upper() for m in (markets or "").split(",") if m.strip()]

    # Parse per-market thresholds if provided
    def _parse_thresholds(s: str) -> dict[str, float]:
        d: dict[str, float] = {}
        for part in (s or "").split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                d[str(k).strip().upper()] = float(str(v).strip())
            except Exception:
                continue
        return d
    per_thr = _parse_thresholds(min_ev_per_market)

    # Iterate dates
    def to_dt(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")
    cur = to_dt(start); end_dt = to_dt(end)
    rows = []
    calib = []

    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        sim_path = PROC_DIR / f"props_simulations_{d}.csv"
        if not sim_path.exists():
            cur += timedelta(days=1)
            continue
        try:
            sim_df = pd.read_csv(sim_path)
        except Exception:
            cur += timedelta(days=1)
            continue
        if sim_df.empty:
            cur += timedelta(days=1)
            continue
        sim_df["market"] = sim_df["market"].astype(str).str.upper()
        if allowed_markets:
            sim_df = sim_df[sim_df["market"].isin(allowed_markets)]
        if sim_df.empty:
            cur += timedelta(days=1)
            continue
        # Normalize player display for join to stats
        name_col = "player" if "player" in sim_df.columns else None
        if not name_col:
            cur += timedelta(days=1)
            continue
        sim_df["player_norm"] = sim_df[name_col].astype(str).apply(_norm_name)
        # Compute EVs and evaluate outcomes
        for _, r in sim_df.iterrows():
            m = str(r.get("market") or "").upper()
            if allowed_markets and m not in allowed_markets:
                continue
            player_disp = r.get(name_col)
            ln = r.get("line")
            try:
                ln = float(ln)
            except Exception:
                continue
            p_over = pd.to_numeric(r.get("p_over_sim"), errors="coerce")
            if pd.isna(p_over):
                continue
            p_over = float(p_over)
            p_under = max(0.0, 1.0 - p_over)
            dec_o = american_to_decimal(r.get("over_price")) if pd.notna(r.get("over_price")) else None
            dec_u = american_to_decimal(r.get("under_price")) if pd.notna(r.get("under_price")) else None
            ev_o = (p_over * (dec_o - 1.0) - (1.0 - p_over)) if (dec_o is not None) else None
            ev_u = (p_under * (dec_u - 1.0) - (1.0 - p_under)) if (dec_u is not None) else None
            side = None; price = None; ev = None
            if ev_o is not None or ev_u is not None:
                if (ev_u is None) or (ev_o is not None and ev_o >= ev_u):
                    side = "Over"; price = r.get("over_price"); ev = ev_o
                else:
                    side = "Under"; price = r.get("under_price"); ev = ev_u
            if ev is None:
                continue
            thr = per_thr.get(m, float(min_ev)) if per_thr else float(min_ev)
            try:
                if thr is not None and float(thr) > -1.0 and float(ev) < float(thr):
                    continue
            except Exception:
                pass

            # Actual outcome from stats on ET date d
            day_stats = stats_all[stats_all["date_et"] == d]
            # Name variants inlined: exact, no-dot, initial-last
            _full = str(player_disp or "").strip()
            _parts = [p for p in _full.split(" ") if p]
            _vars = set()
            if _full:
                _nm = _norm_name(_full)
                _vars.add(_nm)
                _vars.add(_nm.replace(".", ""))
            if len(_parts) >= 2:
                _first, _last = _parts[0], _parts[-1]
                _init_last = f"{_first[0]}. {_last}"
                _nm2 = _norm_name(_init_last)
                _vars.add(_nm2)
                _vars.add(_nm2.replace(".", ""))
            ps = day_stats[day_stats["player_text_norm"].isin(_vars) | day_stats["player_text_nodot"].isin(_vars)]
            actual = None
            if not ps.empty:
                row = ps.iloc[0]
                if m == "SOG":
                    actual = row.get("shots")
                elif m == "SAVES":
                    actual = row.get("saves")
                elif m == "GOALS":
                    actual = row.get("goals")
                elif m == "ASSISTS":
                    actual = row.get("assists")
                elif m == "POINTS":
                    try:
                        actual = float((row.get("goals") or 0)) + float((row.get("assists") or 0))
                    except Exception:
                        actual = None
                elif m == "BLOCKS":
                    actual = row.get("blocked")
            result = None
            if actual is not None and pd.notna(actual):
                try:
                    av = float(actual)
                    if av > ln:
                        over_res = "win"
                    elif av < ln:
                        over_res = "loss"
                    else:
                        over_res = "push"
                    if side == "Over":
                        result = over_res
                    elif side == "Under":
                        if over_res == "win":
                            result = "loss"
                        elif over_res == "loss":
                            result = "win"
                        else:
                            result = "push"
                except Exception:
                    result = None

            payout = None
            if result is not None:
                dec = american_to_decimal(price)
                if result == "win" and dec is not None:
                    payout = stake * (dec - 1.0)
                elif result == "loss":
                    payout = -stake
                elif result == "push":
                    payout = 0.0

            rows.append({
                "date": d,
                "market": m,
                "player": player_disp,
                "line": float(ln),
                "book": r.get("book"),
                "over_price": r.get("over_price") if pd.notna(r.get("over_price")) else None,
                "under_price": r.get("under_price") if pd.notna(r.get("under_price")) else None,
                "p_over": float(p_over),
                "side": side,
                "ev": float(ev) if ev is not None else None,
                "actual": actual,
                "result": result,
                "stake": stake,
                "payout": payout,
            })
            # Calibration record for over outcome
            if actual is not None and pd.notna(actual) and (abs(ln - round(ln)) > 1e-9 or float(actual) != float(round(ln))):
                calib.append({
                    "date": d,
                    "market": m,
                    "p_over": float(p_over),
                    "over_won": bool(float(actual) > float(ln)),
                })
        cur += timedelta(days=1)

    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        print("No backtest rows generated. Ensure simulations exist for the range.")
        raise typer.Exit(code=0)

    def summarize(df: pd.DataFrame) -> dict:
        out: dict[str, object] = {
            "picks": int(len(df)),
            "decided": int(df["result"].isin(["win", "loss"]).sum()),
            "wins": int((df["result"] == "win").sum()),
            "losses": int((df["result"] == "loss").sum()),
            "pushes": int((df["result"] == "push").sum()),
        }
        try:
            decided = df[df["result"].isin(["win", "loss"])].copy()
            if not decided.empty:
                # chosen probability for the selected side
                p_sel = np.where(
                    decided["side"].astype(str) == "Over",
                    pd.to_numeric(decided["p_over"], errors="coerce").astype(float),
                    1.0 - pd.to_numeric(decided["p_over"], errors="coerce").astype(float),
                )
                y = (decided["result"] == "win").astype(float).to_numpy()
                p_sel = np.clip(p_sel, 0.0, 1.0)
                # accuracy and Brier score
                acc = float((y == 1.0).mean()) if len(y) > 0 else None
                brier = float(np.mean((p_sel - y) ** 2)) if len(y) > 0 else None
                out["accuracy"] = acc
                out["brier"] = brier
                out["avg_prob"] = float(np.mean(p_sel))
            else:
                out["accuracy"] = None
                out["brier"] = None
                out["avg_prob"] = None
        except Exception:
            out["accuracy"] = None
            out["brier"] = None
            out["avg_prob"] = None
        return out

    overall = summarize(rows_df)
    by_market = {mkt: summarize(g) for mkt, g in rows_df.groupby("market")}
    # Also report accuracy at probability cut thresholds for quick targeting
    def acc_at_threshold(df: pd.DataFrame, thr: float) -> dict:
        try:
            p_chosen = np.where(df["side"].astype(str)=="Over", df["p_over"].astype(float), (1.0 - df["p_over"].astype(float)))
            decided = df[df["result"].isin(["win","loss"])].copy()
            p_sel = np.where(decided["side"].astype(str)=="Over", decided["p_over"].astype(float), (1.0 - decided["p_over"].astype(float)))
            mask = p_sel >= thr
            sub = decided[mask]
            acc = float((sub["result"] == "win").mean()) if len(sub) > 0 else None
            return {"picks": int(len(sub)), "accuracy": acc}
        except Exception:
            return {"picks": 0, "accuracy": None}
    thresholds = [0.55, 0.60, 0.65]
    by_market_thresholds = {
        mkt: {str(t): acc_at_threshold(g, t) for t in thresholds}
        for mkt, g in rows_df.groupby("market")
    }

    calib_df = pd.DataFrame(calib)
    calib_bins = []
    if not calib_df.empty:
        bins = np.linspace(0.0, 1.0, 11)
        calib_df["bin"] = pd.cut(calib_df["p_over"], bins=bins, include_lowest=True)
        for b, g in calib_df.groupby("bin"):
            try:
                exp = float(g["p_over"].mean())
                obs = float(g["over_won"].mean())
                cnt = int(len(g))
                calib_bins.append({"bin": str(b), "expected": exp, "observed": obs, "count": cnt})
            except Exception:
                continue

    pref = (out_prefix.strip() + "_") if out_prefix.strip() else ""
    rows_path = PROC_DIR / f"{pref}props_backtest_sim_rows_{start}_to_{end}.csv"
    summ_path = PROC_DIR / f"{pref}props_backtest_sim_summary_{start}_to_{end}.json"
    rows_df.to_csv(rows_path, index=False)
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall": overall,
            "by_market": by_market,
            "by_market_thresholds": by_market_thresholds,
            "calibration": calib_bins,
            "filters": {"markets": allowed_markets, "min_ev": min_ev}
        }, f, indent=2)
    print(json.dumps({"overall": overall, "by_market": by_market}, indent=2))
    print(f"Saved rows to {rows_path} and summary to {summ_path}")


@app.command()
def props_stats_features(
    windows: str = typer.Option("5,10,20", help="Comma-separated rolling windows (games), e.g., 5,10,20"),
    output_csv: str = typer.Option("data/props/props_stats_features.csv", help="Output CSV path for features"),
):
    """Compute rolling per-player features from player_game_stats.csv without using odds.

    Outputs features like shots_mean_{w}, goals_mean_{w}, assists_mean_{w}, points_mean_{w} for skaters,
    and saves_mean_{w} for goalies, using strictly prior games (shifted rolling mean).
    """
    stats_path = RAW_DIR / "player_game_stats.csv"
    if not stats_path.exists():
        print("player_game_stats.csv missing; run props_stats_backfill first.")
        raise typer.Exit(code=1)
    try:
        df = pd.read_csv(stats_path)
    except Exception:
        print("player_game_stats.csv exists but could not be parsed; aborting features build.")
        raise typer.Exit(code=1)
    if df.empty:
        print("player_game_stats.csv is empty.")
        raise typer.Exit(code=1)
    df["date_key"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    # Normalize role strings
    df["role"] = df["role"].apply(lambda x: str(x).lower() if pd.notna(x) else "")
    # Sort within each player
    df = df.sort_values(["player", "date_key"])  # stable order per player
    # Compute rolling means using shifted values to avoid lookahead
    w_list = [int(w.strip()) for w in windows.split(",") if w.strip()]
    feature_rows = []
    # Build per-player sequences
    for player, g in df.groupby("player"):
        g = g.copy().sort_values("date_key")
        # Prepare base columns
        skater_mask = (g["role"] == "skater")
        goalie_mask = (g["role"] == "goalie")
        # Derived points
        try:
            pts_series = (g.get("goals").astype(float).fillna(0) + g.get("assists").astype(float).fillna(0))
        except Exception:
            pts_series = pd.Series([None]*len(g), index=g.index)
        for w in w_list:
            # Skater features
            if skater_mask.any():
                g.loc[skater_mask, f"shots_mean_{w}"] = g.loc[skater_mask, "shots"].astype(float).shift(1).rolling(w, min_periods=1).mean()
                g.loc[skater_mask, f"goals_mean_{w}"] = g.loc[skater_mask, "goals"].astype(float).shift(1).rolling(w, min_periods=1).mean()
                g.loc[skater_mask, f"assists_mean_{w}"] = g.loc[skater_mask, "assists"].astype(float).shift(1).rolling(w, min_periods=1).mean()
                g.loc[skater_mask, f"points_mean_{w}"] = pts_series.shift(1).rolling(w, min_periods=1).mean()
            # Goalie features
            if goalie_mask.any():
                g.loc[goalie_mask, f"saves_mean_{w}"] = g.loc[goalie_mask, "saves"].astype(float).shift(1).rolling(w, min_periods=1).mean()
        # Append rows with features for this player's games
        feature_rows.append(g)
    out = pd.concat(feature_rows, ignore_index=True)
    # Keep a concise set of columns
    keep_cols = [c for c in out.columns if (
        c in ("date_key","date","player","team","role","shots","goals","assists","saves")
        or c.startswith("shots_mean_") or c.startswith("goals_mean_") or c.startswith("assists_mean_") or c.startswith("points_mean_") or c.startswith("saves_mean_")
    )]
    out_small = out[keep_cols]
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_small.to_csv(out_path, index=False)
    print(f"Saved features to {out_path} with {len(out_small)} rows and {len(out_small.columns)} columns")


@app.command()
def props_stats_calibration(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (ET)"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD (ET)"),
    windows: str = typer.Option("5,10,20", help="Comma-separated rolling windows (games)"),
    bins: int = typer.Option(10, help="Calibration bins (equal-width) for probability buckets"),
    output_json: str = typer.Option("data/processed/props_stats_calibration.json", help="Output JSON summary path"),
):
    """Calibrate stats-only Poisson probabilities using rolling means vs actual outcomes.

    For each game in [start, end], for each player and market, compute lambda as a rolling mean of the last W games (per window).
    Evaluate typical thresholds:
      - SOG: 2.5, 3.5
      - GOALS: 0.5
      - ASSISTS: 0.5
      - POINTS: 0.5, 1.5
      - SAVES: 24.5, 26.5
    For each (market, threshold, window), compute calibration bins of predicted P(Over) vs observed Over, plus accuracy and Brier score.
    """
    from datetime import datetime
    from .models.props import (
        PropsConfig,
        SkaterShotsModel, GoalieSavesModel,
        SkaterGoalsModel, SkaterAssistsModel, SkaterPointsModel,
    )
    stats_path = RAW_DIR / "player_game_stats.csv"
    if not stats_path.exists():
        print("player_game_stats.csv missing; run props_stats_backfill first.")
        raise typer.Exit(code=1)
    try:
        df = pd.read_csv(stats_path)
    except Exception:
        print("player_game_stats.csv exists but could not be parsed; aborting calibration.")
        raise typer.Exit(code=1)
    if df.empty:
        print("player_game_stats.csv is empty.")
        raise typer.Exit(code=1)
    df["date_key"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    # Filter to date range
    df = df[(df["date_key"] >= start) & (df["date_key"] <= end)].copy()
    if df.empty:
        print("No stats in the given date range.")
        raise typer.Exit(code=0)
    df["role"] = df["role"].apply(lambda x: str(x).lower() if pd.notna(x) else "")
    windows_list = [int(w.strip()) for w in windows.split(",") if w.strip()]
    # models (probability only); we will compute lambdas via rolling means for speed
    models_by_w = {}
    for w in windows_list:
        cfg = PropsConfig(window=w)
        models_by_w[w] = {
            "SOG": SkaterShotsModel(cfg),
            "SAVES": GoalieSavesModel(cfg),
            "GOALS": SkaterGoalsModel(cfg),
            "ASSISTS": SkaterAssistsModel(cfg),
            "POINTS": SkaterPointsModel(cfg),
        }
    # thresholds per market
    thresholds = {
        "SOG": [2.5, 3.5],
        "GOALS": [0.5],
        "ASSISTS": [0.5],
        "POINTS": [0.5, 1.5],
        "SAVES": [24.5, 26.5],
    }
    # Precompute per-player rolling lambdas (shifted to exclude current row)
    df = df.sort_values(["player", "date_key"])  # ensure chronological per player
    for metric in ("shots", "goals", "assists", "saves"):
        df[metric] = pd.to_numeric(df.get(metric), errors="coerce")
    # Points as goals+assists
    df["points_val_cal"] = pd.to_numeric(df.get("goals"), errors="coerce").fillna(0) + pd.to_numeric(df.get("assists"), errors="coerce").fillna(0)
    for w in windows_list:
        df[f"lam_shots_{w}"] = df.groupby("player")["shots"].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
        df[f"lam_goals_{w}"] = df.groupby("player")["goals"].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
        df[f"lam_assists_{w}"] = df.groupby("player")["assists"].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
        df[f"lam_points_{w}"] = df.groupby("player")["points_val_cal"].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
        df[f"lam_saves_{w}"] = df.groupby("player")["saves"].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)

    results = []  # rows of {date, player, market, window, line, p_over, over_actual}
    for idx, row in df.iterrows():
        is_skater = (str(row.get("role")).lower() == "skater")
        is_goalie = (str(row.get("role")).lower() == "goalie")
        shots = row.get("shots"); goals = row.get("goals"); assists = row.get("assists"); saves = row.get("saves")
        points_val = row.get("points_val_cal")
        for w in windows_list:
            ms = models_by_w[w]
            if is_skater:
                # SOG
                try:
                    lam = float(row.get(f"lam_shots_{w}")) if pd.notna(row.get(f"lam_shots_{w}")) else None
                    if lam is not None:
                        for ln in thresholds["SOG"]:
                            p = ms["SOG"].prob_over(lam, ln)
                            ov = (float(shots) > ln) if pd.notna(shots) else None
                            results.append({"date": row["date_key"], "player": row.get("player"), "market": "SOG", "window": w, "line": ln, "p_over": float(p), "over_actual": (1 if ov else (0 if ov is not None else None))})
                except Exception:
                    pass
                # GOALS
                try:
                    lam = float(row.get(f"lam_goals_{w}")) if pd.notna(row.get(f"lam_goals_{w}")) else None
                    if lam is not None:
                        for ln in thresholds["GOALS"]:
                            p = ms["GOALS"].prob_over(lam, ln)
                            ov = (float(goals) > ln) if pd.notna(goals) else None
                            results.append({"date": row["date_key"], "player": row.get("player"), "market": "GOALS", "window": w, "line": ln, "p_over": float(p), "over_actual": (1 if ov else (0 if ov is not None else None))})
                except Exception:
                    pass
                # ASSISTS
                try:
                    lam = float(row.get(f"lam_assists_{w}")) if pd.notna(row.get(f"lam_assists_{w}")) else None
                    if lam is not None:
                        for ln in thresholds["ASSISTS"]:
                            p = ms["ASSISTS"].prob_over(lam, ln)
                            ov = (float(assists) > ln) if pd.notna(assists) else None
                            results.append({"date": row["date_key"], "player": row.get("player"), "market": "ASSISTS", "window": w, "line": ln, "p_over": float(p), "over_actual": (1 if ov else (0 if ov is not None else None))})
                except Exception:
                    pass
                # POINTS
                try:
                    lam = float(row.get(f"lam_points_{w}")) if pd.notna(row.get(f"lam_points_{w}")) else None
                    if lam is not None:
                        for ln in thresholds["POINTS"]:
                            p = ms["POINTS"].prob_over(lam, ln)
                            ov = (float(points_val) > ln) if (points_val is not None) else None
                            results.append({"date": row["date_key"], "player": row.get("player"), "market": "POINTS", "window": w, "line": ln, "p_over": float(p), "over_actual": (1 if ov else (0 if ov is not None else None))})
                except Exception:
                    pass
            if is_goalie:
                # SAVES
                try:
                    lam = float(row.get(f"lam_saves_{w}")) if pd.notna(row.get(f"lam_saves_{w}")) else None
                    if lam is not None:
                        for ln in thresholds["SAVES"]:
                            p = ms["SAVES"].prob_over(lam, ln)
                            ov = (float(saves) > ln) if pd.notna(saves) else None
                            results.append({"date": row["date_key"], "player": row.get("player"), "market": "SAVES", "window": w, "line": ln, "p_over": float(p), "over_actual": (1 if ov else (0 if ov is not None else None))})
                except Exception:
                    pass
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No calibration rows produced.")
        raise typer.Exit(code=0)
    # Drop rows without actual outcome
    res_df = res_df[pd.notna(res_df["over_actual"])].copy()
    # Calibration stats by (market, line, window)
    out = {"start": start, "end": end, "groups": []}
    # Binning function
    def make_bins(n):
        edges = np.linspace(0.0, 1.0, n+1)
        return edges
    for (mkt, ln, w), g in res_df.groupby(["market","line","window"]):
        try:
            # Accuracy and Brier score
            y = g["over_actual"].astype(int).values
            p = g["p_over"].astype(float).values
            acc = float((y == (p >= 0.5)).mean())
            brier = float(np.mean((p - y) ** 2))
            # Also fit simple temperature+bias calibration for this group
            try:
                from .utils.calibration import fit_temp_shift, BinaryCalibration
                cal = fit_temp_shift(p, y, metric="brier")
                cal_t = float(getattr(cal, 't', 1.0))
                cal_b = float(getattr(cal, 'b', 0.0))
                # Post-calibration Brier for reporting
                p_cal = cal.apply(np.asarray(p, dtype=float))
                brier_cal = float(np.mean((p_cal - y) ** 2))
            except Exception:
                cal_t = 1.0
                cal_b = 0.0
                brier_cal = brier
            # Calibration bins
            edges = make_bins(bins)
            inds = np.digitize(p, edges, right=True)
            cal = []
            for b in range(1, len(edges)+1):
                mask = (inds == b)
                if not np.any(mask):
                    continue
                p_mean = float(np.mean(p[mask]))
                y_mean = float(np.mean(y[mask]))
                cnt = int(np.sum(mask))
                cal.append({"bin": b, "expected": p_mean, "observed": y_mean, "count": cnt})
            out["groups"].append({
                "market": mkt,
                "line": float(ln),
                "window": int(w),
                "count": int(len(g)),
                "accuracy": acc,
                "brier": brier,
                "calibration_params": {"t": cal_t, "b": cal_b, "brier_post": brier_cal},
                "calibration": cal,
            })
        except Exception:
            continue
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved calibration summary to {out_path} with {len(out['groups'])} groups")
    print("[props_watch] no props found; finished attempts")


@app.command()
def periods_recommend(
    date: str = typer.Option(..., help="Date to analyze (YYYY-MM-DD or 'today')"),
    min_prob: float = typer.Option(0.55, help="Minimum probability threshold (0-1)"),
    top: int = typer.Option(50, help="Maximum number of recommendations"),
    output: Optional[str] = typer.Option(None, help="Output CSV path"),
):
    """Generate period-specific betting recommendations.
    
    Analyzes period 1/2/3 totals, first 10 minute goals, and period winners.
    Requires predictions with period projections.
    """
    from .models.period_betting import analyze_period_bets, format_period_bets_table
    
    # Parse date
    if date == "today":
        date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    
    # Load predictions with period data
    pred_path = PROC_DIR / f"predictions_{date}.csv"
    if not pred_path.exists():
        print(f"[error] Predictions not found for {date}")
        print(f"[hint] Run: python -m nhl_betting.cli predict --date {date} --source web")
        raise typer.Exit(code=1)
    
    try:
        df = pd.read_csv(pred_path)
    except Exception as e:
        print(f"[error] Failed to load predictions: {e}")
        raise typer.Exit(code=1)
    
    # Check for period columns
    required_cols = ["period1_home_proj", "period1_away_proj"]
    if not all(col in df.columns for col in required_cols):
        print(f"[error] Predictions missing period projections")
        print(f"[hint] Predictions must include period-by-period data")
        raise typer.Exit(code=1)
    
    print(f"[analyze] Finding period betting opportunities for {date}")
    print(f"  Min probability: {min_prob*100:.1f}%")
    print(f"  Games: {len(df)}")
    
    # Analyze period bets
    bets = analyze_period_bets(df, min_prob=min_prob)
    
    if not bets:
        print(f"\n[result] No period bets found meeting criteria")
        return
    
    # Limit to top N
    bets = bets[:top]
    
    # Format as table
    bets_df = format_period_bets_table(bets)
    
    # Display
    print(f"\n[result] Found {len(bets)} period betting opportunities:\n")
    print(bets_df.to_string(index=False))
    
    # Save to file
    if output:
        out_path = Path(output)
    else:
        out_path = PROC_DIR / f"period_bets_{date}.csv"
    
    save_df(bets_df, out_path)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    def _backfill_recommendations_for_date(date: str, min_ev: float = 0.0, top: int = 100, markets: str = "all") -> Optional[Path]:
        """Generate recommendations_{date}.csv by reusing predictions and on-the-fly EV calc (similar to web api)."""
        from .web.app import PROC_DIR as _PROC_DIR  # reuse same directory
        path = _PROC_DIR / f"predictions_{date}.csv"
        if not path.exists():
            print(f"[skip] predictions missing for {date} -> {path}")
            return None
        import pandas as _pd
        df = _pd.read_csv(path)
        # Local helpers copied from API for EV/odds parsing
        def _num(v):
            if v is None:
                return None
            try:
                # Unwrap simple containers like list/tuple/numpy array
                import numpy as _np
                if isinstance(v, (list, tuple, _np.ndarray)):
                    if len(v) == 0:
                        return None
                    v = v[0]
                if isinstance(v, (int, float)):
                    import math as _math
                    _fv = float(v)
                    return _fv if _math.isfinite(_fv) else None
                s = str(v).strip()
                if s == "":
                    return None
                import re, math as _math
                if re.fullmatch(r"[a-zA-Z_\-]+", s):
                    return None
                _fv2 = float(s)
                return _fv2 if _math.isfinite(_fv2) else None
            except Exception:
                return None
        def _american_to_decimal(american):
            try:
                a = float(american)
                if a > 0:
                    return 1.0 + (a / 100.0)
                else:
                    return 1.0 + (100.0 / abs(a))
            except Exception:
                return None
        def _add_rec(row, market_key, label, prob_key, ev_key, edge_key, odds_key, book_key=None, out_list=None):
            raw_price = row.get(odds_key) if odds_key in row else None
            price_val = _num(raw_price)
            if price_val is None:
                close_map = {
                    "home_ml_odds": "close_home_ml_odds",
                    "away_ml_odds": "close_away_ml_odds",
                    "over_odds": "close_over_odds",
                    "under_odds": "close_under_odds",
                    "home_pl_-1.5_odds": "close_home_pl_-1.5_odds",
                    "away_pl_+1.5_odds": "close_away_pl_+1.5_odds",
                }
                ck = close_map.get(odds_key)
                if ck and ck in row:
                    price_val = _num(row.get(ck))
            if price_val is None and market_key in ("totals", "puckline"):
                price_val = -110.0
            prob_val = None
            try:
                if prob_key in row and _pd.notna(row.get(prob_key)):
                    import math as _math
                    _pv = float(row.get(prob_key))
                    if _math.isfinite(_pv) and 0.0 <= _pv <= 1.0:
                        prob_val = _pv
            except Exception:
                prob_val = None
            ev_val = None
            try:
                if ev_key in row and _pd.notna(row[ev_key]):
                    import math as _math
                    _ev = float(row[ev_key])
                    ev_val = _ev if _math.isfinite(_ev) else None
            except Exception:
                ev_val = None
            if ev_val is None and (prob_val is not None) and (price_val is not None):
                dec = _american_to_decimal(price_val)
                if dec is not None:
                    import math as _math
                    _ev2 = prob_val * (dec - 1.0) - (1.0 - prob_val)
                    ev_val = _ev2 if _math.isfinite(_ev2) else None
            try:
                if (ev_val is None) or not (float(ev_val) >= float(min_ev)):
                    return
            except Exception:
                return
            edge_val = None
            try:
                if edge_key in row and _pd.notna(row.get(edge_key)):
                    import math as _math
                    _edge = float(row.get(edge_key))
                    edge_val = _edge if _math.isfinite(_edge) else None
            except Exception:
                edge_val = None
            total_line_used_val = None
            try:
                if "total_line_used" in row and _pd.notna(row.get("total_line_used")):
                    import math as _math
                    _tlu = float(row.get("total_line_used"))
                    total_line_used_val = _tlu if _math.isfinite(_tlu) else None
            except Exception:
                total_line_used_val = None
            rec = {
                "date": row.get("date"),
                "home": row.get("home"),
                "away": row.get("away"),
                "market": market_key,
                "bet": label,
                "model_prob": prob_val,
                "ev": ev_val,
                "edge": edge_val,
                "price": price_val,
                "book": row.get(book_key) if book_key and (book_key in row) and _pd.notna(row.get(book_key)) else None,
                "total_line_used": total_line_used_val,
                "stake": None,
            }
            out_list.append(rec)

        recs = []
        # Markets filter
        try:
            f_markets = set([m.strip().lower() for m in str(markets).split(",")]) if markets and str(markets) != "all" else {"moneyline", "totals", "puckline"}
        except Exception:
            f_markets = {"moneyline", "totals", "puckline"}
        for _, r in df.iterrows():
            if "moneyline" in f_markets:
                _add_rec(r, "moneyline", "home_ml", "p_home_ml", "ev_home_ml", "edge_home_ml", "home_ml_odds", book_key="home_ml_book", out_list=recs)
                _add_rec(r, "moneyline", "away_ml", "p_away_ml", "ev_away_ml", "edge_away_ml", "away_ml_odds", book_key="away_ml_book", out_list=recs)
            if "totals" in f_markets:
                _add_rec(r, "totals", "over", "p_over", "ev_over", "edge_over", "over_odds", book_key="over_book", out_list=recs)
                _add_rec(r, "totals", "under", "p_under", "ev_under", "edge_under", "under_odds", book_key="under_book", out_list=recs)
            if "puckline" in f_markets:
                _add_rec(r, "puckline", "home_pl_-1.5", "p_home_pl_-1.5", "ev_home_pl_-1.5", "edge_home_pl_-1.5", "home_pl_-1.5_odds", book_key="home_pl_-1.5_book", out_list=recs)
                _add_rec(r, "puckline", "away_pl_+1.5", "p_away_pl_+1.5", "ev_away_pl_+1.5", "edge_away_pl_+1.5", "away_pl_+1.5_odds", book_key="away_pl_+1.5_book", out_list=recs)
        recs_sorted = sorted(recs, key=lambda x: x["ev"], reverse=True)[: top if top and top > 0 else len(recs)]
        out_cols = ["date","home","away","market","bet","price","model_prob","ev","edge","book","total_line_used"]
        out_df = _pd.DataFrame([{k: r.get(k) for k in out_cols} for r in recs_sorted])
        out_path = _PROC_DIR / f"recommendations_{date}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"wrote {len(out_df)} recs -> {out_path}")
        return out_path


    @app.command()
    def backfill_recommendations(
        start: str = typer.Argument(..., help="Start date YYYY-MM-DD"),
        end: str = typer.Argument(..., help="End date YYYY-MM-DD"),
        min_ev: float = typer.Option(0.0, help="Minimum EV threshold"),
        top: int = typer.Option(100, help="Top N per day"),
        markets: str = typer.Option("all", help="moneyline,totals,puckline or 'all'"),
    ):
        """Backfill recommendations_{date}.csv across a date range using saved predictions."""
        from datetime import datetime as _dt, timedelta as _td
        try:
            start_dt = _dt.strptime(start, "%Y-%m-%d")
            end_dt = _dt.strptime(end, "%Y-%m-%d")
        except Exception:
            print("Invalid date format; use YYYY-MM-DD")
            raise typer.Exit(code=1)
        if end_dt < start_dt:
            start_dt, end_dt = end_dt, start_dt
        d = start_dt
        count = 0
        while d <= end_dt:
            day = d.strftime("%Y-%m-%d")
            try:
                _backfill_recommendations_for_date(day, min_ev=min_ev, top=top, markets=markets)
                count += 1
            except Exception as e:
                print(f"[error] {day}: {e}")
            d += _td(days=1)
        print(f"Backfill complete for {count} days.")

    @app.command()
    def props_postgame(
        date: str = typer.Option("yesterday", help="Target ET date YYYY-MM-DD or 'today'/'yesterday'"),
        stats_source: str = typer.Option("stats", help="Data source for player stats: stats | web"),
        window: int = typer.Option(10, help="Backtest rolling window (games)"),
        stake: float = typer.Option(100.0, help="Flat stake for backtest/recon PnL"),
    ):
        """Run postgame pipeline for most recent slate: stats backfill -> props reconciliation -> backtest (ALL props)."""
        # Resolve ET date tokens
        from zoneinfo import ZoneInfo
        d_in = (date or "").strip().lower()
        if d_in in ("today", "yesterday"):
            now_et = datetime.now(ZoneInfo("America/New_York")).date()
            if d_in == "yesterday":
                from datetime import timedelta as _td
                d = (now_et - _td(days=1)).strftime("%Y-%m-%d")
            else:
                d = now_et.strftime("%Y-%m-%d")
        else:
            d = date
        print({"step": "props_postgame", "date": d})
        # 1) Ensure player game stats for date
        try:
            collect_player_game_stats(d, d, source=stats_source)
            print({"stats": "ok", "date": d})
        except Exception as e:
            print({"stats": "error", "date": d, "error": str(e)})
        # 2) Ensure recommendations exist (ALL markets)
        try:
            rec_path = PROC_DIR / f"props_recommendations_{d}.csv"
            if not rec_path.exists():
                print({"recs": "building", "date": d})
                props_recommendations(date=d, min_ev=0.0, top=200, market="")
        except Exception as e:
            print({"recs": "error", "date": d, "error": str(e)})
        # 3) Reconcile props for date (ALL markets)
        try:
            from .scripts.daily_update import reconcile_props_date as _recon_props
            res = _recon_props(d, flat_stake=stake, verbose=False)
            print({"reconcile_props": res})
        except Exception as e:
            print({"reconcile_props": "error", "date": d, "error": str(e)})
        # 4) Backtest for date across all markets including BLOCKS
        try:
            # Directly call the Typer command function with kwargs
            props_backtest(start=d, end=d, window=window, stake=stake, markets="SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS", min_ev=-1.0, out_prefix="postgame")
            print({"backtest": "done", "date": d})
        except Exception as e:
            print({"backtest": "error", "date": d, "error": str(e)})


    @app.command(name="roster-update")
    def roster_update_cmd(date: Optional[str] = typer.Option(None, help="ET date YYYY-MM-DD to stamp output")):
        d = date or ymd(today_utc())
        df = build_all_team_roster_snapshots()
        out_path = PROC_DIR / f"roster_snapshot_{d}.csv"
        save_df(df, out_path)
        print(f"Saved roster snapshot to {out_path} ({len(df)} players)")

    @app.command(name="lineup-update")
    def lineup_update_cmd(date: Optional[str] = typer.Option(None, help="ET date YYYY-MM-DD to stamp output"), prefer_source: Optional[str] = typer.Option("dailyfaceoff", help="Optional external source to try first (dailyfaceoff|none)")):
        d = date or ymd(today_utc())
        frames = []
        for ab in TEAM_ABBRS:
            try:
                snap = None
                if prefer_source and str(prefer_source).lower() == "dailyfaceoff":
                    try:
                        snap_src = build_lineup_snapshot_from_source(ab, d)
                        if snap_src is not None and not snap_src.empty:
                            snap = snap_src
                    except Exception as e_src:
                        print({"team": ab, "source": prefer_source, "error": str(e_src)})
                if snap is None or snap.empty:
                    snap = build_lineup_snapshot(ab)
                snap["team"] = ab
                frames.append(snap)
            except Exception as e:
                print({"team": ab, "error": str(e)})
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["player_id","full_name","position","line_slot","pp_unit","pk_unit","proj_toi","confidence","team"])
        out_path = PROC_DIR / f"lineups_{d}.csv"
        save_df(out, out_path)
        print(f"Saved lineup snapshot to {out_path} ({len(out)} rows)")

        try:
            # Build co-TOI pairs from lineups and save alongside
            co = build_co_toi_from_lineups(out)
            co_path = PROC_DIR / f"lineups_co_toi_{d}.csv"
            save_df(co, co_path)
            print(f"Saved co-TOI snapshot to {co_path} ({len(co)} rows)")
        except Exception as e:
            print({"co_toi": "error", "date": d, "error": str(e)})

        # Also write starting goalies snapshot (best-effort)
        try:
            rows_g = fetch_dailyfaceoff_starting_goalies(d)
            df_g = pd.DataFrame(rows_g, columns=["team","goalie","status","confidence","source"]) if rows_g else pd.DataFrame(columns=["team","goalie","status","confidence","source"])
            # Fallback: derive likely starters from lineup snapshot by highest projected TOI among goalies
            if df_g.empty and not out.empty:
                df_goalies = out.copy()
                df_goalies["position"] = df_goalies["position"].astype(str)
                df_goalies = df_goalies[df_goalies["position"].str.upper().str.contains("G")]
                if not df_goalies.empty:
                    def _to_num(x):
                        try:
                            import math
                            v = float(x)
                            return v if math.isfinite(v) else None
                        except Exception:
                            return None
                    df_goalies["_toi"] = df_goalies["proj_toi"].apply(_to_num)
                    starters = []
                    for team, grp in df_goalies.groupby("team"):
                        g1 = grp.copy().sort_values(["_toi","confidence"], ascending=[False, False])
                        if not g1.empty:
                            r = g1.iloc[0]
                            starters.append({
                                "team": str(team),
                                "goalie": str(r.get("full_name","")),
                                "status": "Derived",
                                "confidence": float(r.get("confidence", 0.5)),
                                "source": "lineup_snapshot",
                            })
                    if starters:
                        df_g = pd.DataFrame(starters, columns=["team","goalie","status","confidence","source"])
            out_g = PROC_DIR / f"starting_goalies_{d}.csv"
            save_df(df_g, out_g)
            print(f"Saved starting goalies snapshot to {out_g} ({len(df_g)} rows)")
        except Exception as e_g:
            print({"starting_goalies": "error", "date": d, "error": str(e_g)})

    @app.command(name="shifts-update")
    def shifts_update_cmd(date: Optional[str] = typer.Option(None, help="ET date YYYY-MM-DD to fetch shiftcharts and compute co-TOI")):
        d = date or ymd(today_utc())
        client = NHLWebClient()
        games = client.schedule_day(d)
        frames = []
        for g in games:
            try:
                df = shifts_frame(g.gamePk)
                if df is not None and not df.empty:
                    df["gamePk"] = g.gamePk
                    frames.append(df)
            except Exception as e:
                print({"shifts": "error", "gamePk": g.gamePk, "error": str(e)})
        all_shifts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["team","player_id","period","start_s","end_s","gamePk"])
        out_s = PROC_DIR / f"shifts_{d}.csv"
        save_df(all_shifts, out_s)
        print(f"Saved shifts to {out_s} ({len(all_shifts)} rows)")
        try:
            co = co_toi_from_shifts(all_shifts)
            out_c = PROC_DIR / f"co_toi_shifts_{d}.csv"
            save_df(co, out_c)
            print(f"Saved co-TOI from shifts to {out_c} ({len(co)} rows)")
        except Exception as e:
            print({"co_toi_shifts": "error", "date": d, "error": str(e)})

    @app.command(name="injury-update")
    def injury_update_cmd(date: Optional[str] = typer.Option(None, help="ET date YYYY-MM-DD to stamp output"), overrides_path: Optional[str] = typer.Option(None, help="Optional CSV with manual injury entries")):
        d = date or ymd(today_utc())
        manual = []
        if overrides_path:
            try:
                dfm = pd.read_csv(overrides_path)
                manual = dfm.to_dict(orient="records")
            except Exception as e:
                print({"overrides": "error", "path": overrides_path, "error": str(e)})
        df = build_injury_snapshot(d, manual=manual)
        out_path = PROC_DIR / f"injuries_{d}.csv"
        save_df(df, out_path)
        print(f"Saved injury snapshot to {out_path} ({len(df)} rows)")

    @app.command(name="game-simulate-baseline")
    def game_simulate_baseline_cmd(date: Optional[str] = typer.Option(None, help="ET date YYYY-MM-DD to simulate schedule"), seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility")):
        d = date or ymd(today_utc())
        client = NHLWebClient()
        games = client.schedule_day(d)
        base_mu = 3.0
        try:
            cfg_path = MODEL_DIR / "config.json"
            if cfg_path.exists():
                obj = json.loads(cfg_path.read_text(encoding="utf-8"))
                base_mu = float(obj.get("base_mu", base_mu))
        except Exception:
            pass
        sim_cfg = SimConfig(seed=seed)
        sim_rates = RateModels.baseline(base_mu=base_mu)
        sim = GameSimulator(cfg=sim_cfg, rates=sim_rates)
        game_rows = []
        box_rows = []
        evt_rows = []
        for g in games:
            try:
                h_ab = get_team_assets(g.home).get("abbr") or g.home
                a_ab = get_team_assets(g.away).get("abbr") or g.away
                roster_home = build_roster_snapshot(h_ab).to_dict(orient="records")
                roster_away = build_roster_snapshot(a_ab).to_dict(orient="records")
                gs, _events = sim.simulate(home_name=g.home, away_name=g.away, roster_home=roster_home, roster_away=roster_away)
                game_rows.append({
                    "gamePk": g.gamePk,
                    "date": d,
                    "home": g.home,
                    "away": g.away,
                    "home_goals_sim": gs.home.score,
                    "away_goals_sim": gs.away.score,
                })
                for p in list(gs.home.players.values()) + list(gs.away.players.values()):
                    box_rows.append({
                        "gamePk": g.gamePk,
                        "date": d,
                        "team": p.team,
                        "player_id": p.player_id,
                        "full_name": p.full_name,
                        "position": p.position,
                        "shots": int(p.stats.get("shots", 0.0)),
                        "goals": int(p.stats.get("goals", 0.0)),
                        "assists": int(p.stats.get("assists", 0.0)),
                        "points": int(p.stats.get("goals", 0.0)) + int(p.stats.get("assists", 0.0)),
                        "saves": int(p.stats.get("saves", 0.0)),
                    })
            except Exception as e:
                print({"simulate": "error", "gamePk": g.gamePk, "error": str(e)})
        games_df = pd.DataFrame(game_rows)
        boxes_df = pd.DataFrame(box_rows)
        out_g = PROC_DIR / f"sim_games_{d}.csv"
        out_b = PROC_DIR / f"sim_boxscores_{d}.csv"
        save_df(games_df, out_g)
        save_df(boxes_df, out_b)
        print({"saved": {"games": str(out_g), "boxscores": str(out_b)}, "counts": {"games": len(games_df), "box": len(boxes_df)}})

    @app.command(name="game-simulate-possession")
    def game_simulate_possession_cmd(date: Optional[str] = typer.Option(None, help="ET date YYYY-MM-DD to simulate schedule using lineups and simple score effects"), seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility")):
        d = date or ymd(today_utc())
        client = NHLWebClient()
        games = client.schedule_day(d)
        base_mu = 3.0
        try:
            cfg_path = MODEL_DIR / "config.json"
            if cfg_path.exists():
                obj = json.loads(cfg_path.read_text(encoding="utf-8"))
                base_mu = float(obj.get("base_mu", base_mu))
        except Exception:
            pass
        sim_cfg = SimConfig(seed=seed)
        sim_rates = RateModels.baseline(base_mu=base_mu)
        sim = GameSimulator(cfg=sim_cfg, rates=sim_rates)
        # Load calibrated special-teams multipliers (optional)
        st_cal: Dict[str, float] = {}
        try:
            mc_path = PROC_DIR / "model_calibration.json"
            if mc_path.exists() and getattr(mc_path.stat(), "st_size", 0) > 0:
                import json
                mc = json.loads(mc_path.read_text(encoding="utf-8"))
                st_cal = dict(mc.get("special_teams", {}) or {})
        except Exception:
            st_cal = {}
        # Load team special teams for PP/PK effects
        try:
            from .data.team_stats import load_team_special_teams
            team_st = load_team_special_teams(d) or {}
        except Exception:
            team_st = {}
        # Load team penalty rates (committed/drawn per game) to refine PP frequency
        try:
            from .data.penalty_rates import load_team_penalty_rates
            team_pr = load_team_penalty_rates(d) or {}
        except Exception:
            team_pr = {}
        # Load lineup snapshot for date
        lineup_path = PROC_DIR / f"lineups_{d}.csv"
        if not lineup_path.exists():
            print(f"Missing {lineup_path}; run lineup-update first.")
            raise typer.Exit(code=1)
        lineups_all = pd.read_csv(lineup_path)
        # Optional: enrich TOI projections from shiftcharts if available
        shifts_path = PROC_DIR / f"shifts_{d}.csv"
        toi_map_home = {}; toi_map_away = {}
        if shifts_path.exists():
            try:
                sh = pd.read_csv(shifts_path)
                ptoi = player_toi_from_shifts(sh)
                # Build maps per team for quick lookup
                for _, r in ptoi.iterrows():
                    tm = str(r["team"]).upper(); pid = int(r["player_id"]); m = float(r["toi_ev_minutes"]) if pd.notna(r["toi_ev_minutes"]) else 0.0
                    # Assign sets after we know home/away abbrevs
                    # We'll map by abbr later when we know h_ab/a_ab
                    pass
                # We'll re-compute per game below once h_ab/a_ab known
                ptoi_all = ptoi
            except Exception:
                ptoi_all = pd.DataFrame()
        else:
            ptoi_all = pd.DataFrame()
        game_rows = []
        box_rows = []
        evt_rows = []
        for g in games:
            try:
                h_ab = get_team_assets(g.home).get("abbr") or g.home
                a_ab = get_team_assets(g.away).get("abbr") or g.away
                roster_home_df = build_roster_snapshot(h_ab)
                roster_away_df = build_roster_snapshot(a_ab)
                # Merge shift-based TOI if available
                if not ptoi_all.empty:
                    try:
                        p_h = ptoi_all[ptoi_all["team"].eq(h_ab)].rename(columns={"toi_ev_minutes":"proj_toi"})[["player_id","proj_toi"]]
                        p_a = ptoi_all[ptoi_all["team"].eq(a_ab)].rename(columns={"toi_ev_minutes":"proj_toi"})[["player_id","proj_toi"]]
                        roster_home_df = roster_home_df.merge(p_h, on="player_id", how="left")
                        roster_away_df = roster_away_df.merge(p_a, on="player_id", how="left")
                        roster_home_df["proj_toi"] = roster_home_df["proj_toi"].fillna(15.0)
                        roster_away_df["proj_toi"] = roster_away_df["proj_toi"].fillna(15.0)
                    except Exception:
                        roster_home_df["proj_toi"] = 15.0
                        roster_away_df["proj_toi"] = 15.0
                else:
                    roster_home_df["proj_toi"] = 15.0
                    roster_away_df["proj_toi"] = 15.0
                roster_home = roster_home_df.to_dict(orient="records")
                roster_away = roster_away_df.to_dict(orient="records")
                l_home = lineups_all[lineups_all["team"].eq(h_ab)].to_dict(orient="records")
                l_away = lineups_all[lineups_all["team"].eq(a_ab)].to_dict(orient="records")
                st_h = team_st.get(h_ab) or {"pp_pct": 0.2, "pk_pct": 0.8, "drawn_per_game": 3.0, "committed_per_game": 3.0}
                st_a = team_st.get(a_ab) or {"pp_pct": 0.2, "pk_pct": 0.8, "drawn_per_game": 3.0, "committed_per_game": 3.0}
                # Refine with penalty rates if available
                try:
                    pr_h = team_pr.get(h_ab) or {}
                    pr_a = team_pr.get(a_ab) or {}
                    if pr_h:
                        if pr_h.get("drawn_per60") is not None:
                            st_h["drawn_per_game"] = float(pr_h.get("drawn_per60"))
                        if pr_h.get("committed_per60") is not None:
                            st_h["committed_per_game"] = float(pr_h.get("committed_per60"))
                    if pr_a:
                        if pr_a.get("drawn_per60") is not None:
                            st_a["drawn_per_game"] = float(pr_a.get("drawn_per60"))
                        if pr_a.get("committed_per60") is not None:
                            st_a["committed_per_game"] = float(pr_a.get("committed_per60"))
                except Exception:
                    pass
                gs, _events = sim.simulate_with_lineups(home_name=g.home, away_name=g.away, roster_home=roster_home, roster_away=roster_away, lineup_home=l_home, lineup_away=l_away, st_home=st_h, st_away=st_a, special_teams_cal=st_cal)
                game_rows.append({
                    "gamePk": g.gamePk,
                    "date": d,
                    "home": g.home,
                    "away": g.away,
                    "home_goals_sim": gs.home.score,
                    "away_goals_sim": gs.away.score,
                })
                for p in list(gs.home.players.values()) + list(gs.away.players.values()):
                    box_rows.append({
                        "gamePk": g.gamePk,
                        "date": d,
                        "team": p.team,
                        "player_id": p.player_id,
                        "full_name": p.full_name,
                        "position": p.position,
                        "shots": int(p.stats.get("shots", 0.0)),
                        "goals": int(p.stats.get("goals", 0.0)),
                        "saves": int(p.stats.get("saves", 0.0)),
                        "blocks": int(p.stats.get("blocks", 0.0)),
                    })
                # Event strength summary per game (shots, goals, saves by strength)
                try:
                    def _cnt(kind: str, strength: str, team_name: str) -> int:
                        return sum(1 for e in _events if (e.kind == kind and e.meta.get("strength") == strength and e.team == team_name))
                    # Saves by strength = opponent shots - opponent goals for that strength
                    sh_ev_home = _cnt("shot", "EV", g.home); sh_pp_home = _cnt("shot", "PP", g.home); sh_pk_home = _cnt("shot", "PK", g.home)
                    gl_ev_home = _cnt("goal", "EV", g.home); gl_pp_home = _cnt("goal", "PP", g.home); gl_pk_home = _cnt("goal", "PK", g.home)
                    bl_ev_home = _cnt("block", "EV", g.home); bl_pp_home = _cnt("block", "PP", g.home); bl_pk_home = _cnt("block", "PK", g.home)
                    sh_ev_away = _cnt("shot", "EV", g.away); sh_pp_away = _cnt("shot", "PP", g.away); sh_pk_away = _cnt("shot", "PK", g.away)
                    gl_ev_away = _cnt("goal", "EV", g.away); gl_pp_away = _cnt("goal", "PP", g.away); gl_pk_away = _cnt("goal", "PK", g.away)
                    bl_ev_away = _cnt("block", "EV", g.away); bl_pp_away = _cnt("block", "PP", g.away); bl_pk_away = _cnt("block", "PK", g.away)
                    evt_rows.append({
                        "gamePk": g.gamePk,
                        "date": d,
                        "home": g.home,
                        "away": g.away,
                        # shots
                        "shots_ev_home": sh_ev_home,
                        "shots_pp_home": sh_pp_home,
                        "shots_pk_home": sh_pk_home,
                        "shots_ev_away": sh_ev_away,
                        "shots_pp_away": sh_pp_away,
                        "shots_pk_away": sh_pk_away,
                        # goals
                        "goals_ev_home": gl_ev_home,
                        "goals_pp_home": gl_pp_home,
                        "goals_pk_home": gl_pk_home,
                        "goals_ev_away": gl_ev_away,
                        "goals_pp_away": gl_pp_away,
                        "goals_pk_away": gl_pk_away,
                        # blocks
                        "blocks_ev_home": bl_ev_home,
                        "blocks_pp_home": bl_pp_home,
                        "blocks_pk_home": bl_pk_home,
                        "blocks_ev_away": bl_ev_away,
                        "blocks_pp_away": bl_pp_away,
                        "blocks_pk_away": bl_pk_away,
                        # saves by strength
                        "saves_ev_home": max(0, sh_ev_away - gl_ev_away),
                        "saves_pp_home": max(0, sh_pp_away - gl_pp_away),
                        "saves_pk_home": max(0, sh_pk_away - gl_pk_away),
                        "saves_ev_away": max(0, sh_ev_home - gl_ev_home),
                        "saves_pp_away": max(0, sh_pp_home - gl_pp_home),
                        "saves_pk_away": max(0, sh_pk_home - gl_pk_home),
                    })
                except Exception:
                    pass
            except Exception as e:
                print({"simulate": "error", "gamePk": g.gamePk, "error": str(e)})
        games_df = pd.DataFrame(game_rows)
        boxes_df = pd.DataFrame(box_rows)
        events_df = pd.DataFrame(evt_rows)
        out_g = PROC_DIR / f"sim_games_pos_{d}.csv"
        out_b = PROC_DIR / f"sim_boxscores_pos_{d}.csv"
        out_e = PROC_DIR / f"sim_events_pos_{d}.csv"
        save_df(games_df, out_g)
        save_df(boxes_df, out_b)
        if not events_df.empty:
            save_df(events_df, out_e)
        print({"saved": {"games": str(out_g), "boxscores": str(out_b), "events": (str(out_e) if not events_df.empty else None)}, "counts": {"games": len(games_df), "box": len(boxes_df), "events": len(events_df)}})

    @app.command(name="game-calibrate-special-teams")
    def game_calibrate_special_teams_cmd(
        start: str = typer.Option(..., help="Start ET date YYYY-MM-DD"),
        end: str = typer.Option(..., help="End ET date YYYY-MM-DD"),
        target_pp_shot_frac: float = typer.Option(0.18, help="Target fraction of shots occurring on PP (league-wide heuristic)"),
        target_pp_goal_frac: float = typer.Option(0.24, help="Target fraction of goals occurring on PP (league-wide heuristic)"),
    ):
        import pandas as pd
        from pathlib import Path
        dates = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq="D")
        ev_sh = ev_gl = pp_sh = pp_gl = pk_sh = pk_gl = 0
        ev_bl = pp_bl = pk_bl = 0
        n_days = 0
        for dt in dates:
            d = dt.strftime("%Y-%m-%d")
            p = PROC_DIR / f"sim_events_pos_{d}.csv"
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
                # Sum EV/PP/PK shots+goals across both teams
                ev_sh += int(df[["shots_ev_home","shots_ev_away"]].sum().sum())
                pp_sh += int(df[["shots_pp_home","shots_pp_away"]].sum().sum())
                pk_sh += int(df[["shots_pk_home","shots_pk_away"]].sum().sum())
                ev_gl += int(df[["goals_ev_home","goals_ev_away"]].sum().sum())
                pp_gl += int(df[["goals_pp_home","goals_pp_away"]].sum().sum())
                pk_gl += int(df[["goals_pk_home","goals_pk_away"]].sum().sum())
                # Blocks by strength (sum home+away)
                try:
                    ev_bl += int(df[["blocks_ev_home","blocks_ev_away"]].sum().sum())
                    pp_bl += int(df[["blocks_pp_home","blocks_pp_away"]].sum().sum())
                    pk_bl += int(df[["blocks_pk_home","blocks_pk_away"]].sum().sum())
                except Exception:
                    pass
                n_days += 1
            except Exception:
                continue
        total_sh = ev_sh + pp_sh + pk_sh
        total_gl = ev_gl + pp_gl + pk_gl
        obs_pp_sh_frac = (pp_sh / total_sh) if total_sh > 0 else 0.0
        obs_pp_gl_frac = (pp_gl / total_gl) if total_gl > 0 else 0.0
        # Observed block rates per opponent shots (defensive exposure)
        obs_ev_blk_rate = (ev_bl / ev_sh) if ev_sh > 0 else 0.0
        # PK blocks: defending team blocks while short-handed; denominator = opponent PP shots
        obs_pk_blk_rate = (pk_bl / pp_sh) if pp_sh > 0 else 0.0
        # PP-def blocks: blocks while team is on PP (rare); denominator = opponent PK shots
        obs_pp_def_blk_rate = (pp_bl / pk_sh) if pk_sh > 0 else 0.0
        # Recommend multipliers to move observed toward targets (bounded)
        rec_pp_sh_mult = float(max(0.8, min(1.6, (target_pp_shot_frac / max(obs_pp_sh_frac, 1e-6)))))
        rec_pp_goal_mult = float(max(0.9, min(1.6, (target_pp_goal_frac / max(obs_pp_gl_frac, 1e-6)))))
        # PK shot/goal multipliers keep inverse relationship to PP
        rec_pk_sh_mult = float(max(0.6, min(1.2, 1.0 / rec_pp_sh_mult)))
        rec_pk_goal_mult = float(max(0.6, min(1.2, 1.0 / rec_pp_goal_mult)))
        out = {
            "window": {"start": start, "end": end, "days_with_events": n_days},
            "observed": {
                "shots": {"ev": ev_sh, "pp": pp_sh, "pk": pk_sh, "pp_frac": obs_pp_sh_frac},
                "goals": {"ev": ev_gl, "pp": pp_gl, "pk": pk_gl, "pp_frac": obs_pp_gl_frac},
                "blocks": {"ev": ev_bl, "pp": pp_bl, "pk": pk_bl, "ev_rate": obs_ev_blk_rate, "pk_rate": obs_pk_blk_rate, "pp_def_rate": obs_pp_def_blk_rate},
            },
            "targets": {"pp_shot_frac": target_pp_shot_frac, "pp_goal_frac": target_pp_goal_frac},
            "recommended": {
                "pp_shot_multiplier": rec_pp_sh_mult,
                "pk_shot_multiplier": rec_pk_sh_mult,
                "pp_goal_multiplier": rec_pp_goal_mult,
                "pk_goal_multiplier": rec_pk_goal_mult,
                # Blocks: recommend using observed rates as defaults for simulation
                "blocks_ev_rate": obs_ev_blk_rate,
                "blocks_pk_rate": obs_pk_blk_rate,
                "blocks_pp_def_rate": obs_pp_def_blk_rate,
            },
        }
        # Persist into sim_calibration.json and update model_calibration.json patch
        sim_cal_path = PROC_DIR / "sim_calibration.json"
        try:
            obj = {}
            if sim_cal_path.exists() and getattr(sim_cal_path.stat(), "st_size", 0) > 0:
                import json
                obj = json.loads(sim_cal_path.read_text(encoding="utf-8"))
            obj["special_teams"] = out
            sim_cal_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        except Exception:
            pass
        # Patch model_calibration.json (non-fatal if missing)
        mc_path = PROC_DIR / "model_calibration.json"
        try:
            import json
            mc = {}
            if mc_path.exists() and getattr(mc_path.stat(), "st_size", 0) > 0:
                mc = json.loads(mc_path.read_text(encoding="utf-8"))
            mc.setdefault("special_teams", {})
            mc["special_teams"].update(out["recommended"])  # store the multipliers and block rates
            mc_path.write_text(json.dumps(mc, indent=2), encoding="utf-8")
        except Exception:
            pass
        print({"calibrated": out["recommended"], "window": out["window"], "observed": out["observed"]})


@app.command(name="team-odds-collect")
def team_odds_collect(
    date: str = typer.Option(..., help="ET date YYYY-MM-DD"),
    markets: str = typer.Option("h2h,spreads,totals", help="Comma-separated team markets (OddsAPI keys)"),
):
    """Collect team odds for the slate and archive to data/odds/team/date=YYYY-MM-DD/oddsapi.{csv,parquet}."""
    try:
        from .data.team_odds import archive_team_odds_oddsapi
        res = archive_team_odds_oddsapi(date)
        print({
            "date": date,
            "source": "oddsapi",
            "input_rows": res.get("input_rows"),
            "output_rows": res.get("output_rows"),
            "csv_path": str(res.get("csv_path")),
        })
    except Exception as e:
        print(f"[team-odds-collect] failed: {e}")
        raise typer.Exit(code=1)


@app.command(name="games-archive")
def games_archive(
    date: str = typer.Option(..., help="ET date YYYY-MM-DD"),
):
    """Archive the day's scoreboard (games, status, scores) for reconciliation.

    Writes data/odds/games/date=YYYY-MM-DD/scoreboard.csv with upsert by gamePk where available.
    """
    base_dir = PROC_DIR.parent / "odds" / "games" / f"date={date}"
    csv_path = base_dir / "scoreboard.csv"
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        client = NHLWebClient()
        rows = client.scoreboard_day(date)
    except Exception as e:
        print(f"[games-archive] scoreboard fetch failed: {e}")
        raise typer.Exit(code=1)
    df = pd.DataFrame(rows or [])
    # Normalize minimal expected columns
    keep_cols = [
        "gamePk","state","start_time","home","away","home_goals","away_goals","venue","tv","gameState"
    ]
    cols = [c for c in keep_cols if c in df.columns]
    if cols:
        df = df[cols]
    # Upsert by gamePk when possible
    if csv_path.exists():
        try:
            old = pd.read_csv(csv_path)
        except Exception:
            old = pd.DataFrame()
        key = "gamePk" if "gamePk" in df.columns else None
        if key and key in old.columns and not old.empty:
            merged = pd.concat([old, df], ignore_index=True)
            merged = merged.sort_values(by=[key]).drop_duplicates(subset=[key], keep="last")
            df = merged
    df.to_csv(csv_path, index=False)
    print({"date": date, "rows": int(len(df.index)), "csv_path": str(csv_path)})


@app.command(name="game-accuracy-day")
def game_accuracy_day(
    date: str = typer.Option(..., help="ET date YYYY-MM-DD"),
    use_close: bool = typer.Option(True, help="Prefer closing odds/lines when available"),
):
    """Compute per-market accuracy for a single day from predictions_{date}.csv and write JSON.

    Outputs: data/processed/accuracy_{date}.json with fields per market and totals.
    Markets: moneyline, totals (exclude pushes), puckline (-1.5), first10 (if present).
    """
    import json, math
    import pandas as _pd
    from .utils.io import PROC_DIR
    pred_path = PROC_DIR / f"predictions_{date}.csv"
    if not pred_path.exists():
        print(json.dumps({"ok": False, "reason": "no_predictions", "path": str(pred_path)})); return
    try:
        df = _pd.read_csv(pred_path)
    except Exception as e:
        print(json.dumps({"ok": False, "reason": "read_failed", "error": str(e)})); return

    def _num(v):
        try:
            if v is None: return None
            f = float(v)
            return f if math.isfinite(f) else None
        except Exception:
            return None

    out = {}
    # Moneyline
    n=0; c=0
    for _, r in df.iterrows():
        fh=_num(r.get('final_home_goals')); fa=_num(r.get('final_away_goals'))
        ph=_num(r.get('p_home_ml')); pa=_num(r.get('p_away_ml'))
        if fh is None or fa is None or ph is None or pa is None: continue
        n += 1
        pick = 'home' if ph >= pa else 'away'
        won = (fh>fa) if pick=='home' else (fa>fh)
        if won: c += 1
    out['moneyline'] = {"n": n, "acc": (c/n) if n else None}

    # Totals (exclude pushes)
    n=0; c=0
    for _, r in df.iterrows():
        fh=_num(r.get('final_home_goals')); fa=_num(r.get('final_away_goals'))
        po=_num(r.get('p_over')); pu=_num(r.get('p_under'))
        tl=_num(r.get('close_total_line_used')) if use_close else _num(r.get('total_line_used'))
        if fh is None or fa is None or po is None or pu is None or tl is None: continue
        tot = fh + fa
        if abs(tot - tl) < 1e-9:  # push
            continue
        n += 1
        pick = 'over' if po >= pu else 'under'
        won = (tot>tl) if pick=='over' else (tot<tl)
        if won: c += 1
    out['totals'] = {"n": n, "acc": (c/n) if n else None}

    # Puckline (-1.5)
    n=0; c=0
    for _, r in df.iterrows():
        fh=_num(r.get('final_home_goals')); fa=_num(r.get('final_away_goals'))
        php=_num(r.get('p_home_pl_-1.5')); pap=_num(r.get('p_away_pl_+1.5'))
        if fh is None or fa is None or php is None or pap is None: continue
        n += 1
        pick = 'home' if php >= pap else 'away'
        diff = fh - fa
        won = (diff>1.5) if pick=='home' else (diff<1.5)
        if won: c += 1
    out['puckline'] = {"n": n, "acc": (c/n) if n else None}

    # First10 Yes/No if present
    if 'p_f10_yes' in df.columns and 'p_f10_no' in df.columns and 'result_first10' in df.columns:
        n=0; c=0
        for _, r in df.iterrows():
            py=_num(r.get('p_f10_yes')); pn=_num(r.get('p_f10_no'))
            rf=str(r.get('result_first10') or '').strip().lower()
            if py is None or pn is None or rf not in ('yes','no'): continue
            n += 1
            pick = 'yes' if py >= pn else 'no'
            if rf == pick: c += 1
        out['first10'] = {"n": n, "acc": (c/n) if n else None}

    # Write JSON
    out_path = PROC_DIR / f"accuracy_{date}.json"
    try:
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass
    print(json.dumps({"ok": True, "date": date, **out, "path": str(out_path)}))


if __name__ == "__main__":
    app()
