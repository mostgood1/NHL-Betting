import json
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional

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

from .models.poisson import PoissonGoals
from .models.trends import TrendAdjustments, team_keys, get_adjustment
from .utils.odds import american_to_decimal, decimal_to_implied_prob, remove_vig_two_way, ev_unit, kelly_stake
from .data.collect import collect_player_game_stats
from .models.props import SkaterShotsModel, GoalieSavesModel, SkaterGoalsModel, SkaterAssistsModel, SkaterPointsModel, SkaterBlocksModel
from .data.odds_api import OddsAPIClient, normalize_snapshot_to_rows
from .data.bovada import BovadaClient
from .data import player_props as props_data
from .data.rosters import build_all_team_roster_snapshots

app = typer.Typer(help="NHL Betting predictive engine CLI")


@app.command()
def fetch(
    season: Optional[int] = typer.Option(None, help="Season start year, e.g., 2023"),
    start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD"),
    source: str = typer.Option("web", help="Data source: 'web' (api-web.nhle.com), 'stats' (statsapi.web.nhl.com), or 'nhlpy' (nhl-api-py)"),
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

    # Simple Poisson baseline (load base_mu if available)
    base_mu = None
    cfg_path = MODEL_DIR / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            try:
                base_mu = float(json.load(f).get("base_mu", None))
            except Exception:
                base_mu = None
    pois = PoissonGoals(base_mu=base_mu or 3.05)
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
    elif odds_source == "bovada":
        try:
            bc = BovadaClient()
            df = bc.fetch_game_odds(date)
            if df is not None and not df.empty:
                # Normalize team keys
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                df["home_norm"] = df["home"].apply(norm_team)
                df["away_norm"] = df["away"].apply(norm_team)
                # Map to NHL abbreviations for robust matching (handles 'LA Kings' vs 'Los Angeles Kings')
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
            print("Failed to fetch odds from Bovada:", e)
            odds_df = None
    else:
        print("Unknown odds_source. Use 'csv', 'oddsapi', or 'bovada'.")
    for g in games:
        # Skip non-NHL matchups if any slipped through
        try:
            from .web.teams import get_team_assets
            if not get_team_assets(g.home).get("abbr") or not get_team_assets(g.away).get("abbr"):
                continue
        except Exception:
            pass
        p_home, p_away = elo.predict_moneyline_prob(g.home, g.away)
        # Apply ML subgroup adjustment (home side bias): team/div/conf of home team
        h_keys = team_keys(g.home)
        ml_delta = get_adjustment(trends.ml_home, h_keys)
        if ml_delta:
            p_home = min(max(p_home + ml_delta, 0.01), 0.99)
            p_away = 1.0 - p_home
        # Per-game total line: use odds total_line if available, else provided total_line
        per_game_total = total_line
        match_info = None  # store the matched odds row for later EV/edge calc
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
                match_info = m.iloc[0].to_dict()
                if "total_line" in match_info:
                    val = match_info.get("total_line")
                    try:
                        if pd.notna(val):
                            per_game_total = float(val)
                    except Exception:
                        per_game_total = total_line
        # Derive matchup-specific lambdas and probabilities
        lam_h, lam_a = pois.lambdas_from_total_split(per_game_total, p_home)
        # Apply goals adjustments separately for home and away based on each team subgroup
        gh_delta = get_adjustment(trends.goals, team_keys(g.home))
        ga_delta = get_adjustment(trends.goals, team_keys(g.away))
        lam_h = max(0.1, lam_h + gh_delta)
        lam_a = max(0.1, lam_a + ga_delta)
        p = pois.probs(total_line=per_game_total, lam_h=lam_h, lam_a=lam_a)

        # Optional: apply probability calibration if available
        try:
            from .utils.calibration import load_calibration
            cal_path = PROC_DIR / "model_calibration.json"
            ml_cal, tot_cal = load_calibration(cal_path)
            # moneyline: calibrate the stronger side, then derive the other by 1-p
            p_home_cal = float(ml_cal.apply(np.array([p.get("home_ml", 0.5)]))[0])
            p_away_cal = 1.0 - p_home_cal
            # totals: calibrate over; under=1-over
            p_over_cal = float(tot_cal.apply(np.array([p.get("over", 0.5)]))[0])
            p_under_cal = 1.0 - p_over_cal
        except Exception:
            p_home_cal = float(p.get("home_ml"))
            p_away_cal = float(p.get("away_ml"))
            p_over_cal = float(p.get("over"))
            p_under_cal = float(p.get("under"))
        # Derived projections (used by UI): per-team goals, total and spread
        proj_home_goals = float(lam_h)
        proj_away_goals = float(lam_a)
        model_total = float(lam_h + lam_a)
        model_spread = float(lam_h - lam_a)

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
            "p_home_ml": p_home_cal,
            "p_away_ml": p_away_cal,
            "p_over": p_over_cal,
            "p_under": p_under_cal,
            "p_home_pl_-1.5": float(p.get("home_puckline_-1.5")),
            "p_away_pl_+1.5": float(p.get("away_puckline_+1.5")),
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
                dec_home = american_to_decimal(_f(match_info.get("home_ml"))) if match_info.get("home_ml") is not None else None
                dec_away = american_to_decimal(_f(match_info.get("away_ml"))) if match_info.get("away_ml") is not None else None
                if dec_home is not None and dec_away is not None:
                    imp_h = decimal_to_implied_prob(dec_home)
                    imp_a = decimal_to_implied_prob(dec_away)
                    nv_h, nv_a = remove_vig_two_way(imp_h, imp_a)
                    row["ev_home_ml"] = round(ev_unit(row["p_home_ml"], dec_home), 4)
                    row["ev_away_ml"] = round(ev_unit(row["p_away_ml"], dec_away), 4)
                    row["edge_home_ml"] = round(row["p_home_ml"] - nv_h, 4)
                    row["edge_away_ml"] = round(row["p_away_ml"] - nv_a, 4)
                row["home_ml_odds"] = _f(match_info.get("home_ml")) if match_info.get("home_ml") is not None else None
                row["away_ml_odds"] = _f(match_info.get("away_ml")) if match_info.get("away_ml") is not None else None
                if match_info.get("home_ml_book") is not None:
                    row["home_ml_book"] = match_info.get("home_ml_book")
                if match_info.get("away_ml_book") is not None:
                    row["away_ml_book"] = match_info.get("away_ml_book")
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
    source: str = typer.Option("web", help="Data source: 'web' (api-web.nhle.com), 'stats' (statsapi.web.nhl.com), or 'nhlpy' (nhl-api-py)"),
    odds_source: str = typer.Option("bovada", help="Odds source: 'csv' (provide --odds-csv), 'oddsapi' (provide --snapshot), or 'bovada'"),
    snapshot: Optional[str] = typer.Option(None, help="When odds_source=oddsapi, ISO snapshot like 2024-03-01T12:00:00Z"),
    odds_regions: str = typer.Option("us", help="Odds API regions, e.g., us or us,us2"),
    odds_markets: str = typer.Option("h2h,totals,spreads", help="Odds API markets"),
    odds_bookmaker: Optional[str] = typer.Option(None, help="Preferred bookmaker key (e.g., pinnacle)"),
    odds_best: bool = typer.Option(False, help="Use best available odds across all bookmakers in snapshot"),
    bankroll: float = typer.Option(0.0, help="Bankroll amount for Kelly sizing (0 to disable)"),
    kelly_fraction_part: float = typer.Option(0.5, help="Fraction of Kelly to bet (e.g., 0.5 = half-Kelly)"),
):
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
    pois = PoissonGoals()
    p = pois.probs()
    print({"p_home_ml": p_home, "p_over": p["over"]})
    print("Smoke test complete.")


@app.command()
def collect_props(start: str = typer.Option(...), end: str = typer.Option(...), source: str = typer.Option("stats", help="Source for schedule/boxscores: web | stats | nhlpy")):
    df = collect_player_game_stats(start, end, source=source)
    print(f"Collected {len(df)} player-game rows into data/raw/player_game_stats.csv")


@app.command()
def predict_range(
    start: str = typer.Argument(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Argument(..., help="End date YYYY-MM-DD"),
    total_line: float = 6.0,
    odds_source: str = typer.Option("bovada", help="Odds source: csv|oddsapi|bovada"),
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
):
    """Build props recommendations_{date}.csv from canonical Parquet lines and simple Poisson projections.

    Reads data/props/player_props_lines/date=YYYY-MM-DD/*.parquet, computes p_over and EV for each OVER/UNDER pair
    using rolling-mean lambdas, and writes a denormalized recommendation list similar to NFL.
    """
    import os
    from glob import glob
    # Read canonical lines for date
    parts = []
    base = Path("data/props") / f"player_props_lines/date={date}"
    # Prefer Parquet files; fallback to CSVs if Parquet not present or unreadable
    cand_files = [base / "bovada.parquet", base / "oddsapi.parquet", base / "bovada.csv", base / "oddsapi.csv"]
    for f in cand_files:
        if f.exists():
            try:
                if f.suffix == ".parquet":
                    parts.append(pd.read_parquet(f, engine="pyarrow"))
                else:
                    parts.append(pd.read_csv(f))
            except Exception:
                pass
    if not parts:
        print("No props lines found for", date)
        raise typer.Exit(code=1)
    lines = pd.concat(parts, ignore_index=True)

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
                    player_team_map[_norm_name(nm)] = str(tm)
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
                            key = _norm_name(nm)
                            if key and key not in player_team_map:
                                ab = _abbr_team(tm)
                                if ab:
                                    player_team_map[key] = ab
                except Exception:
                    pass
    except Exception:
        player_team_map = player_team_map
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
                        key = (_norm_name(rr.get("player")).lower(), str(rr.get("market")).upper())
                        val = float(rr.get("proj_lambda")) if pd.notna(rr.get("proj_lambda")) else None
                        if key[0] and key[1] and val is not None:
                            lam_map[key] = val
                    except Exception:
                        pass
    except Exception:
        lam_map = lam_map
    # Instantiate models (probabilities only). We'll only compute lambdas from history for missing players lazily.
    shots = SkaterShotsModel(); saves = GoalieSavesModel(); goals = SkaterGoalsModel(); assists = SkaterAssistsModel(); points = SkaterPointsModel(); blocks = SkaterBlocksModel()
    hist = None  # defer loading
    def _ensure_hist():
        nonlocal hist
        if hist is None:
            stats_path = RAW_DIR / "player_game_stats.csv"
            if not stats_path.exists():
                try:
                    from datetime import datetime as _dt, timedelta as _td
                    end = date
                    start = (_dt.strptime(date, "%Y-%m-%d") - _td(days=365*2)).strftime("%Y-%m-%d")
                    # Prefer web (faster) with fallback
                    try:
                        collect_player_game_stats(start, end, source="web")
                    except Exception:
                        collect_player_game_stats(start, end, source="stats")
                except Exception:
                    pass
            try:
                hist = load_df(stats_path) if stats_path.exists() else pd.DataFrame()
            except Exception:
                try:
                    hist = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
                except Exception:
                    hist = pd.DataFrame()
        return hist
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
            # Lazy fallback: compute lambda from history just for this player if projections_all didn't have it
            H = _ensure_hist()
            try:
                if m == "SOG":
                    lam = shots.player_lambda(H, player)
                elif m == "SAVES":
                    lam = saves.player_lambda(H, player)
                elif m == "GOALS":
                    lam = goals.player_lambda(H, player)
                elif m == "ASSISTS":
                    lam = assists.player_lambda(H, player)
                elif m == "POINTS":
                    lam = points.player_lambda(H, player)
                elif m == "BLOCKS":
                    lam = blocks.player_lambda(H, player)
            except Exception:
                lam = None
            if lam is not None:
                lam_map[key] = lam  # cache for reuse within run
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
    # Combine rows: lines contain over_price and under_price per (market,player,line,book)
    # Vectorized EV computation using precomputed lambdas; fallback row-wise for misses
    import numpy as _np
    from scipy.stats import poisson as _poisson
    # Prepare normalized working frame (explicit core columns to avoid accidental column loss)
    core_cols = [c for c in [
        "market","player_name","player","team","line","over_price","under_price","book"
    ] if c in lines.columns]
    work = lines[core_cols].copy()
    # Uppercase market and optional filter
    if "market" in work.columns:
        work["market"] = work["market"].astype(str).str.upper()
        msel = (market or "").strip()
        if msel and msel.lower() not in ("all", ""): 
            work = work.loc[work["market"] == msel.upper()].copy()
    # Normalize player display and filter likely players
    work["player_display"] = work.apply(lambda r: _norm_player(r.get("player_name") or r.get("player")), axis=1)
    work = work.loc[work["player_display"].map(_looks_like_player)].copy()
    # Parse numeric line and keep valid (preserve all columns using loc)
    work["line_num"] = pd.to_numeric(work.get("line"), errors="coerce")
    work = work.loc[work["line_num"].notna()].copy()
    # Attach normalized name for join
    work["player_norm"] = work["player_display"].astype(str).map(_norm_name).str.lower()
    # Build lambda DataFrame from lam_map for merge
    lam_df = pd.DataFrame([{"player_norm": k[0], "market": k[1], "proj_lambda": v} for k, v in lam_map.items()]) if lam_map else pd.DataFrame(columns=["player_norm","market","proj_lambda"])
    merged = work.merge(lam_df, on=["player_norm", "market"], how="left")
    # Vectorized p_over for rows with proj_lambda available
    vec_mask = merged["proj_lambda"].notna()
    p_over_vec = pd.Series(_np.nan, index=merged.index)
    for mkt in ["SOG","SAVES","GOALS","ASSISTS","POINTS","BLOCKS"]:
        sel = vec_mask & (merged["market"] == mkt)
        if sel.any():
            lam_arr = merged.loc[sel, "proj_lambda"].astype(float).values
            line_arr = _np.floor(merged.loc[sel, "line_num"].astype(float).values + 1e-9).astype(int)
            p_over_vec.loc[sel] = _poisson.sf(line_arr, mu=lam_arr)
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
    dec_over = _american_to_decimal_series(merged.get("over_price"))
    dec_under = _american_to_decimal_series(merged.get("under_price"))
    p_over_s = pd.to_numeric(merged["p_over_vec"], errors="coerce")
    ev_over_s = p_over_s * (dec_over - 1.0) - (1.0 - p_over_s)
    p_under_s = (1.0 - p_over_s).clip(lower=0.0, upper=1.0)
    ev_under_s = p_under_s * (dec_under - 1.0) - (1.0 - p_under_s)
    # Choose side with better EV, handling NaNs
    over_better = (ev_under_s.isna()) | (~ev_over_s.isna() & (ev_over_s >= ev_under_s))
    chosen_side = _np.where(over_better, "Over", "Under")
    chosen_price = _np.where(over_better, merged.get("over_price"), merged.get("under_price"))
    chosen_ev = _np.where(over_better, ev_over_s, ev_under_s)
    # Build vectorized output rows (only where we had lambda and EV is finite)
    vec_out = merged[vec_mask].copy()
    vec_out["side"] = chosen_side
    vec_out["ev"] = pd.to_numeric(chosen_ev, errors="coerce")
    vec_out = vec_out[vec_out["ev"].notna() & (vec_out["ev"].astype(float) >= float(min_ev))]
    # Choose best team: prefer input team, else map
    vec_out["team_final"] = vec_out.get("team")
    try:
        missing_team = vec_out["team_final"].isna() | (vec_out["team_final"].astype(str).str.strip() == "")
        vec_out.loc[missing_team, "team_final"] = vec_out.loc[missing_team, "player_display"].map(lambda nm: player_team_map.get(_norm_name(nm)))
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
        lam, p_over = proj_and_prob(m, str(player), float(ln))
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
        if ev is None or float(ev) < float(min_ev):
            continue
        team_val = rr.get("team") or player_team_map.get(_norm_name(player))
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
    print(f"Wrote {out_path} with {len(out)} rows (normalized cols: {', '.join([c for c in ['proj_lambda','ev_over'] if c in out.columns])})")


@app.command()
def props_full(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD (ET)"),
    min_ev: float = typer.Option(0.0, help="Minimum EV threshold for recommendations"),
    top: int = typer.Option(200, help="Top N recommendations to keep"),
    sources: str = typer.Option("bovada,oddsapi", help="Comma list of sources to collect: bovada,oddsapi"),
    ensure_history_days: int = typer.Option(365, help="Days of player history to ensure before date"),
    market: str = typer.Option("", help="Optional market filter for recs: SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS"),
):
    """Run full props modeling for a date: collect lines, ensure stats, compute projections and recommendations.

    Outputs:
      - data/processed/props_projections_{date}.csv
      - data/processed/props_recommendations_{date}.csv
    """
    from datetime import datetime as _dt, timedelta as _td
    from .models.props import (
        SkaterShotsModel, GoalieSavesModel, SkaterGoalsModel,
        SkaterAssistsModel, SkaterPointsModel, SkaterBlocksModel,
    )
    from .data import player_props as props_data

    # 1) Collect canonical lines for date from requested sources
    srcs = [s.strip().lower() for s in str(sources).split(',') if s.strip()]
    base = Path("data/props") / f"player_props_lines/date={date}"
    base.mkdir(parents=True, exist_ok=True)
    wrote_any = False
    for sel in srcs:
        try:
            cfg = props_data.PropsCollectionConfig(output_root="data/props", book=("bovada" if sel=="bovada" else "oddsapi"), source=sel)
            res = props_data.collect_and_write(date, roster_df=None, cfg=cfg)
            cnt = int(res.get("combined_count") or 0)
            path = res.get("output_path")
            print(f"[collect] {sel}: rows={cnt} path={path}")
            wrote_any = wrote_any or (cnt > 0)
        except Exception as e:
            print(f"[collect] {sel} failed: {e}")
    # Validate we have at least one parquet
    parts = []
    for fname in ("bovada.parquet","oddsapi.parquet"):
        p = base / fname
        if p.exists():
            try:
                parts.append(pd.read_parquet(p, engine="pyarrow"))
            except Exception:
                pass
    if not parts:
        print("No props lines found; aborting.")
        raise typer.Exit(code=1)
    lines = pd.concat(parts, ignore_index=True)

    # Build a player->team mapping to ensure aligned team abbreviations in outputs
    from .web.teams import get_team_assets as _get_team_assets
    def _norm_name(x):
        try:
            s = str(x or "").strip()
            return " ".join(s.split())
        except Exception:
            return str(x)
    def _abbr_team(t):
        if not t:
            return None
        try:
            a = _get_team_assets(str(t)) or {}
            ab = a.get("abbr")
            return str(ab).upper() if ab else (str(t).strip().upper() if str(t).strip() else None)
        except Exception:
            return str(t).strip().upper() if isinstance(t, str) and t.strip() else None
    player_team_map = {}
    try:
        # 1) Roster enrichment: full_name -> team abbr, with name-variant keys (First Last, F Last, F. Last)
        try:
            from .data import player_props as _pp
            roster_df = _pp._build_roster_enrichment()
        except Exception:
            roster_df = None
        if roster_df is not None and not roster_df.empty and {"full_name","team"}.issubset(roster_df.columns):
            def _name_keys(full_name: str):
                try:
                    nm = str(full_name or "").strip()
                    parts = nm.split()
                    if len(parts) >= 2:
                        first = parts[0]; last = parts[-1]
                        keys = { _norm_name(nm).lower() }
                        fi = first[0]
                        keys.add(_norm_name(f"{fi} {last}").lower())
                        keys.add(_norm_name(f"{fi}. {last}").lower())
                        return keys
                    return { _norm_name(nm).lower() } if nm else set()
                except Exception:
                    return set()
            for _, row in roster_df.iterrows():
                full = row.get("full_name"); tm = row.get("team")
                ab = _abbr_team(tm)
                if not ab:
                    continue
                for k in _name_keys(full):
                    if k and k not in player_team_map:
                        player_team_map[k] = ab
        # 2) From lines parquet themselves (latest team seen)
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
                    player_team_map[_norm_name(nm).lower()] = str(tm)
        # Fallback: historical last known team per player
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
                        key = _norm_name(nm).lower()
                        if key and key not in player_team_map:
                            ab = _abbr_team(tm)
                            if ab:
                                player_team_map[key] = ab
            except Exception:
                pass
    except Exception:
        player_team_map = player_team_map

    # 2) Ensure player stats history exists (file present and non-empty)
    stats_path = RAW_DIR / "player_game_stats.csv"
    def _needs_history(p):
        try:
            if not p.exists() or p.stat().st_size == 0:
                return True
            try:
                tmp = pd.read_csv(p)
                return tmp.empty
            except Exception:
                return True
        except Exception:
            return True
    if _needs_history(stats_path):
        try:
            start = (_dt.strptime(date, "%Y-%m-%d") - _td(days=int(ensure_history_days))).strftime("%Y-%m-%d")
            # Try web first (faster), fallback to stats
            try:
                collect_player_game_stats(start, date, source="web")
            except Exception:
                collect_player_game_stats(start, date, source="stats")
        except Exception as e:
            print(f"[history] failed to ensure: {e}")
    try:
        hist = load_df(stats_path) if stats_path.exists() else pd.DataFrame()
    except Exception:
        hist = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    # Harmonize names like "T. Liljegren" to full names when possible using roster snapshots
    try:
        if hist is not None and not hist.empty and 'player' in hist.columns:
            import re, ast
            from .data import rosters as _rosters_mod  # type: ignore
            roster = _rosters_mod.build_all_team_roster_snapshots()
            last_to_full = {}
            if roster is not None and not roster.empty and 'full_name' in roster.columns:
                for nm in roster['full_name'].dropna().astype(str).unique().tolist():
                    parts = nm.strip().split(' ')
                    if len(parts) >= 2:
                        last = parts[-1].lower()
                        last_to_full.setdefault(last, set()).add(nm)
            def _extract_default(s: str):
                if isinstance(s, str) and s.strip().startswith('{'):
                    try:
                        d = ast.literal_eval(s)
                        if isinstance(d, dict):
                            v = d.get('default') or d.get('name') or ''
                            if isinstance(v, str):
                                return v
                    except Exception:
                        return s
                return s
            def _fix(n: str) -> str:
                n = _extract_default(n)
                m = re.match(r"^([A-Za-z])[\.]?\s+([A-Za-z\-']+)$", str(n).strip())
                if m:
                    ini = m.group(1).lower(); last = m.group(2).lower()
                    cands = list(last_to_full.get(last, []))
                    if len(cands) == 1:
                        return cands[0]
                    for c in cands:
                        first = c.split(' ')[0]
                        if first and first[0].lower() == ini:
                            return c
                return str(n)
            hist['player'] = hist['player'].astype(str).map(_fix)
    except Exception:
        pass

    # 3) Compute projections CSV (ev_over and p_over per line)
    shots = SkaterShotsModel(); saves = GoalieSavesModel(); goals = SkaterGoalsModel(); assists = SkaterAssistsModel(); points = SkaterPointsModel(); blocks = SkaterBlocksModel()
    def proj_prob(m, player, ln):
        m = (m or '').upper()
        if m == 'SOG':
            lam = shots.player_lambda(hist, player); return lam, shots.prob_over(lam, ln)
        if m == 'SAVES':
            lam = saves.player_lambda(hist, player); return lam, saves.prob_over(lam, ln)
        if m == 'GOALS':
            lam = goals.player_lambda(hist, player); return lam, goals.prob_over(lam, ln)
        if m == 'ASSISTS':
            lam = assists.player_lambda(hist, player); return lam, assists.prob_over(lam, ln)
        if m == 'POINTS':
            lam = points.player_lambda(hist, player); return lam, points.prob_over(lam, ln)
        if m == 'BLOCKS':
            lam = blocks.player_lambda(hist, player); return lam, blocks.prob_over(lam, ln)
        return None, None
    def _dec(a):
        try:
            a = float(a); return 1.0 + (a/100.0) if a > 0 else 1.0 + (100.0/abs(a))
        except Exception:
            return None
    proj_rows = []
    for _, r in lines.iterrows():
        player = r.get('player_name') or r.get('player')
        if not player:
            continue
        m = str(r.get('market') or '').upper()
        try:
            ln = float(r.get('line'))
        except Exception:
            ln = None
        lam, p_over = (None, None)
        if ln is not None:
            lam, p_over = proj_prob(m, str(player), ln)
        op = r.get('over_price'); up = r.get('under_price')
        ev_over = None
        if (p_over is not None) and (op is not None):
            dec = _dec(op)
            if dec is not None:
                ev_over = float(p_over) * (dec - 1.0) - (1.0 - float(p_over))
        team_val = r.get('team') or player_team_map.get(_norm_name(player).lower())
        proj_rows.append({
            'market': m,
            'player': player,
            'team': _abbr_team(team_val) if team_val else None,
            'line': ln,
            'over_price': op if pd.notna(op) else None,
            'under_price': up if pd.notna(up) else None,
            'proj_lambda': float(lam) if lam is not None else None,
            'p_over': float(p_over) if p_over is not None else None,
            'ev_over': float(ev_over) if ev_over is not None else None,
            'book': r.get('book'),
        })
    df_proj = pd.DataFrame(proj_rows)
    if not df_proj.empty:
        try:
            if df_proj['ev_over'].notna().any():
                df_proj = df_proj.sort_values(['ev_over','p_over'], ascending=[False, False])
            elif df_proj['p_over'].notna().any():
                df_proj = df_proj.sort_values('p_over', ascending=False)
        except Exception:
            pass
    out_proj = PROC_DIR / f"props_projections_{date}.csv"
    save_df(df_proj, out_proj)
    print(f"Wrote {out_proj} with {0 if df_proj is None or df_proj.empty else len(df_proj)} rows")

    # 4) Compute recommendations CSV using same logic as props_recommendations
    rec_rows = []
    for _, r in lines.iterrows():
        m = str(r.get('market') or '').upper()
        if market and m != market.upper():
            continue
        player = r.get('player_name') or r.get('player')
        if not player:
            continue
        try:
            ln = float(r.get('line'))
        except Exception:
            continue
        op = r.get('over_price'); up = r.get('under_price')
        if pd.isna(op) and pd.isna(up):
            continue
        lam, p_over = proj_prob(m, str(player), ln)
        if lam is None or p_over is None:
            continue
        ev_o = (p_over * (_dec(op)-1.0) - (1.0 - p_over)) if (op is not None and _dec(op) is not None) else None
        p_under = max(0.0, 1.0 - float(p_over))
        ev_u = (p_under * (_dec(up)-1.0) - (1.0 - p_under)) if (up is not None and _dec(up) is not None) else None
        side = None; ev = None; price = None
        if ev_o is not None or ev_u is not None:
            if (ev_u is None) or (ev_o is not None and ev_o >= ev_u):
                side = 'Over'; price = op; ev = ev_o
            else:
                side = 'Under'; price = up; ev = ev_u
        if ev is None or not (float(ev) >= float(min_ev)):
            continue
        team_val = r.get('team') or player_team_map.get(_norm_name(player).lower())
        rec_rows.append({
            'date': date,
            'player': player,
            'team': _abbr_team(team_val) if team_val else None,
            'market': m,
            'line': ln,
            'proj': float(lam),
            'p_over': float(p_over),
            'over_price': op if pd.notna(op) else None,
            'under_price': up if pd.notna(up) else None,
            'book': r.get('book'),
            'side': side,
            'ev': float(ev) if ev is not None else None,
        })
    df_rec = pd.DataFrame(rec_rows)
    if not df_rec.empty:
        df_rec = df_rec.sort_values('ev', ascending=False)
        if top and top > 0:
            df_rec = df_rec.head(top)
    out_rec = PROC_DIR / f"props_recommendations_{date}.csv"
    save_df(df_rec, out_rec)
    print(f"Wrote {out_rec} with {0 if df_rec is None or df_rec.empty else len(df_rec)} rows")


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
):
    """Compute model-only props projections (lambdas) for all rostered players on teams with a game.

    Writes data/processed/props_projections_all_{date}.csv with columns:
      [date, player, team, position, market, proj_lambda]
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
    if (str(_os.getenv("PROPS_SKIP_HISTORY", "")).strip().lower() not in ("1","true","yes")) and _needs_history(stats_path):
        try:
            start = (_dt.strptime(date, "%Y-%m-%d") - _td(days=int(ensure_history_days))).strftime("%Y-%m-%d")
            try:
                collect_player_game_stats(start, date, source="web")
            except Exception:
                collect_player_game_stats(start, date, source="stats")
        except Exception as e:
            print(f"[history] ensure failed: {e}")
    # We'll build any needed last-known positions lazily, filtered to slate roster only, using chunked reads
    hist = pd.DataFrame()
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
    # Try live roster via Stats API only if still empty; on failure, fallback to historical roster enrichment
    try:
        from .data.rosters import list_teams as _list_teams, fetch_current_roster as _fetch_current_roster
        teams = _list_teams()
        name_to_id = { str(t.get('name') or '').strip().lower(): int(t.get('id')) for t in teams }
        id_to_abbr = { int(t.get('id')): str(t.get('abbreviation') or '').upper() for t in teams }
        if roster_df.empty:
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
            roster_df = pd.DataFrame(rows_live)
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
    # Models
    shots = SkaterShotsModel(); saves = GoalieSavesModel(); goals = SkaterGoalsModel(); assists = SkaterAssistsModel(); points = SkaterPointsModel(); blocks = SkaterBlocksModel()
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
                    lam = saves.player_lambda(hist, player)
                    rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'SAVES', 'proj_lambda': float(lam) if lam is not None else None})
            else:
                lam = shots.player_lambda(hist, player); rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'SOG', 'proj_lambda': float(lam) if lam is not None else None})
                lam = goals.player_lambda(hist, player); rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'GOALS', 'proj_lambda': float(lam) if lam is not None else None})
                lam = assists.player_lambda(hist, player); rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'ASSISTS', 'proj_lambda': float(lam) if lam is not None else None})
                lam = points.player_lambda(hist, player); rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'POINTS', 'proj_lambda': float(lam) if lam is not None else None})
                lam = blocks.player_lambda(hist, player); rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'BLOCKS', 'proj_lambda': float(lam) if lam is not None else None})
        except Exception:
            continue
    out = pd.DataFrame(rows)
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


@app.command()
def odds_fetch_bovada(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD"),
    out_csv: str = typer.Option("", help="Optional output CSV path; defaults under data/raw/bovada_odds_YYYY-MM-DD.csv"),
):
    """Fetch pre-match Bovada odds for moneyline/totals/puckline and save to CSV."""
    bc = BovadaClient()
    df = bc.fetch_game_odds(date)
    if df is None or df.empty:
        print("No Bovada odds found for", date)
        return
    out_path = Path(out_csv) if out_csv else (RAW_DIR / f"bovada_odds_{date}.csv")
    save_df(df, out_path)
    print(df.head())
    print("Saved Bovada odds to", out_path)


@app.command()
def daily_update(days_ahead: int = typer.Option(2, help="How many days ahead to update (including today)")):
    """Refresh predictions with odds for today (+N days).

    Tries Bovada, then The Odds API (preferring DraftKings), then ensures a predictions CSV exists.
    """
    from datetime import datetime, timedelta, timezone
    base = datetime.now(timezone.utc)
    for i in range(0, max(1, days_ahead)):
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            predict_core(date=d, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True)
        except Exception:
            pass
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

    Strategy per date: Bovada -> The Odds API -> ensure predictions exist without odds.
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
        try:
            predict_core(date=d, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True, bankroll=bankroll, kelly_fraction_part=kelly_fraction_part)
        except Exception as e:
            print("Bovada step failed:", e)
        # If no odds present, try Odds API
        try:
            path = PROC_DIR / f"predictions_{d}.csv"
            df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
        if df.empty or not any(c in df.columns and df[c].notna().any() for c in ["home_ml_odds","away_ml_odds","over_odds","under_odds"]):
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

    Strategy per date: Bovada -> The Odds API -> ensure predictions exist without odds.
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
    """Fetch Bovada player props (SOG/GOALS/SAVES) and save to CSV."""
    bc = BovadaClient()
    df = bc.fetch_props_odds(date)
    if df is None or df.empty:
        print("No Bovada props found for", date)
        return
    if over_only:
        df = df[df["side"].str.upper() == "OVER"][['market','player','line','odds']]
    out_path = Path(out_csv) if out_csv else (RAW_DIR / f"bovada_props_{date}.csv")
    save_df(df, out_path)
    print(df.head())
    print("Saved Bovada props odds to", out_path)


@app.command()
def props_collect(
    date: str = typer.Option(..., help="Slate date YYYY-MM-DD"),
    output_root: str = typer.Option("data/props", help="Output root directory for Parquet files"),
    source: str = typer.Option("bovada", help="Source: bovada | oddsapi (requires ODDS_API_KEY)"),
):
    """Collect & normalize player props and write canonical Parquet under data/props/player_props_lines/date=YYYY-MM-DD.

    - bovada: scrape Bovada coupon JSON (SOG, GOALS, SAVES, ASSISTS, POINTS when available)
    - oddsapi: use The Odds API historical snapshot for player markets (requires ODDS_API_KEY)
    """
    src = source.lower().strip()
    cfg = props_data.PropsCollectionConfig(output_root=output_root, book=("bovada" if src=="bovada" else "oddsapi"), source=src)
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

    Strategy per day:
    - Try Bovada first. If zero combined rows, fallback to The Odds API (requires ODDS_API_KEY).
    """
    from datetime import datetime, timedelta
    def to_dt(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")
    cur = to_dt(start)
    end_dt = to_dt(end)
    cfg_b = props_data.PropsCollectionConfig(output_root=output_root, book="bovada", source="bovada")
    cfg_o = props_data.PropsCollectionConfig(output_root=output_root, book="oddsapi", source="oddsapi")
    total = 0
    days = 0
    while cur <= end_dt:
        d = cur.strftime("%Y-%m-%d")
        try:
            res = props_data.collect_and_write(d, roster_df=None, cfg=cfg_b)
            cnt = int(res.get("combined_count") or 0)
            if cnt == 0:
                # Fallback to Odds API
                try:
                    res2 = props_data.collect_and_write(d, roster_df=None, cfg=cfg_o)
                    cnt = int(res2.get("combined_count") or 0)
                except Exception as e2:
                    print(f"[backfill] oddsapi fallback failed for {d}: {e2}")
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
        for fname in ("bovada.parquet", "oddsapi.parquet"):
            p = base / fname
            if p.exists():
                try:
                    parts.append(pd.read_parquet(p))
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
            "staked": float(df["stake"].fillna(0).sum()),
            "pnl": float(df["payout"].fillna(0).sum()),
        }
        d["roi"] = (d["pnl"] / d["staked"]) if d["staked"] > 0 else None
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


    app()
