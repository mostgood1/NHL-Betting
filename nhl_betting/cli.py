import json
from datetime import datetime, timezone
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
from .models.poisson import PoissonGoals
from .models.trends import TrendAdjustments, team_keys, get_adjustment
from .utils.odds import american_to_decimal, decimal_to_implied_prob, remove_vig_two_way, ev_unit, kelly_stake
from .data.collect import collect_player_game_stats
from .models.props import SkaterShotsModel, GoalieSavesModel, SkaterGoalsModel
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
    for g in games:
        rows.append({
            "gamePk": g.gamePk,
            "date": g.gameDate,
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
    elo = Elo()
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
    cfg_path = MODEL_DIR / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"base_mu": base_mu}, f, indent=2)
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
    # Helper: normalize team names for robust matching
    import re, unicodedata
    def norm_team(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = s.lower()
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
    elo.ratings = ratings

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
            odds_df["date"] = pd.to_datetime(odds_df["date"]).dt.strftime("%Y-%m-%d")
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
        if odds_df is not None:
            key_date = pd.to_datetime(g.gameDate).strftime("%Y-%m-%d")
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
            # Try exact date+team match
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
            if not m.empty and "total_line" in m.columns:
                val = m.iloc[0].get("total_line")
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
        # Derivative model metrics: totals/ATS picks and edges
        pl_line_used = 1.5
        model_total = lam_h + lam_a
        model_spread = lam_h - lam_a  # home minus away
        # Edges in points (positive favors that side/total)
        edge_over_pts = model_total - per_game_total
        edge_under_pts = -edge_over_pts
        edge_home_pl_pts = model_spread - pl_line_used
        edge_away_pl_pts = pl_line_used - model_spread
        # Model picks by raw probability when odds are absent
        totals_pick = "Over" if p["over"] >= p["under"] else "Under"
        ats_pick = "home_-1.5" if p["home_puckline_-1.5"] >= p["away_puckline_+1.5"] else "away_+1.5"
        totals_edge_pts = float(edge_over_pts if totals_pick == "Over" else edge_under_pts)
        ats_edge_pts = float(edge_home_pl_pts if ats_pick == "home_-1.5" else edge_away_pl_pts)
        # Determine actuals if available (final games only in web schedule have goals)
        actual_total = None
        actual_home = None
        actual_away = None
        try:
            if g.home_goals is not None and g.away_goals is not None:
                actual_home = int(g.home_goals)
                actual_away = int(g.away_goals)
                actual_total = actual_home + actual_away
        except Exception:
            pass
        row = {
            "date": g.gameDate,
            "home": g.home,
            "away": g.away,
            # Transparency: applied subgroup adjustments
            "adj_home_ml_delta": round(float(ml_delta), 4) if ml_delta is not None else 0.0,
            "adj_home_goals_delta": round(float(gh_delta), 3) if gh_delta is not None else 0.0,
            "adj_away_goals_delta": round(float(ga_delta), 3) if ga_delta is not None else 0.0,
            "p_home_ml": round(p_home, 4),
            "p_away_ml": round(p_away, 4),
            "p_over": round(p["over"], 4),
            "p_under": round(p["under"], 4),
            "p_home_pl_-1.5": round(p["home_puckline_-1.5"], 4),
            "p_away_pl_+1.5": round(p["away_puckline_+1.5"], 4),
            "total_line_used": per_game_total,
            "proj_home_goals": round(lam_h, 2),
            "proj_away_goals": round(lam_a, 2),
            "model_total": round(model_total, 2),
            "model_spread": round(model_spread, 2),
            "pl_line_used": pl_line_used,
            "totals_pick": totals_pick,
            "totals_edge_pts": round(totals_edge_pts, 2),
            "ats_pick": ats_pick,
            "ats_edge_pts": round(ats_edge_pts, 2),
            "edge_over_pts": round(edge_over_pts, 2),
            "edge_under_pts": round(edge_under_pts, 2),
            "edge_home_pl_pts": round(edge_home_pl_pts, 2),
            "edge_away_pl_pts": round(edge_away_pl_pts, 2),
            "actual_home_goals": actual_home,
            "actual_away_goals": actual_away,
            "actual_total": actual_total,
            "venue": getattr(g, "venue", None),
            "game_state": getattr(g, "gameState", None),
        }
        # Winner predictions/accuracy if actuals present
        if actual_home is not None and actual_away is not None:
            row["winner_actual"] = g.home if actual_home > actual_away else (g.away if actual_away > actual_home else "Draw")
            row["winner_model"] = g.home if p_home >= p_away else g.away
            row["winner_correct"] = (row["winner_actual"] == row["winner_model"])
            # Model vs actual totals diff
            if actual_total is not None:
                row["total_diff"] = round(row["model_total"] - float(actual_total), 2)
                # Totals result/correctness
                if per_game_total is not None:
                    if float(actual_total) > float(per_game_total):
                        row["result_total"] = "Over"
                    elif float(actual_total) < float(per_game_total):
                        row["result_total"] = "Under"
                    else:
                        row["result_total"] = "Push"
                    row["totals_pick_correct"] = (row.get("result_total") != "Push" and row.get("result_total") == row.get("totals_pick"))
                # ATS result/correctness (half-line 1.5)
                diff = float(actual_home - actual_away)
                row["result_ats"] = "home_-1.5" if diff > pl_line_used else "away_+1.5"
                row["ats_pick_correct"] = (row.get("result_ats") == row.get("ats_pick"))
        # If odds provided, compute EVs
        if odds_df is not None:
            key_date = pd.to_datetime(g.gameDate).strftime("%Y-%m-%d")
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
            match = pd.DataFrame()
            if {"home_abbr","away_abbr"}.issubset(set(odds_df.columns)) and g_home_abbr and g_away_abbr:
                match = odds_df[(odds_df["date"] == key_date) & (odds_df["home_abbr"] == g_home_abbr) & (odds_df["away_abbr"] == g_away_abbr)]
            if match.empty:
                match = odds_df[(odds_df["date"] == key_date) & (odds_df["home_norm"] == g_home_n) & (odds_df["away_norm"] == g_away_n)]
            if match.empty:
                # Fallback: match by teams only (handles date boundary differences)
                if {"home_abbr","away_abbr"}.issubset(set(odds_df.columns)) and g_home_abbr and g_away_abbr:
                    match = odds_df[(odds_df["home_abbr"] == g_home_abbr) & (odds_df["away_abbr"] == g_away_abbr)]
                if match.empty:
                    match = odds_df[(odds_df["home_norm"] == g_home_n) & (odds_df["away_norm"] == g_away_n)]
            if match.empty:
                # Fallback: reversed sides
                if {"home_abbr","away_abbr"}.issubset(set(odds_df.columns)) and g_home_abbr and g_away_abbr:
                    match = odds_df[(odds_df["home_abbr"] == g_away_abbr) & (odds_df["away_abbr"] == g_home_abbr)]
                if match.empty:
                    match = odds_df[(odds_df["home_norm"] == g_away_n) & (odds_df["away_norm"] == g_home_n)]
            if not match.empty:
                m = match.iloc[0]
                # Moneyline EVs (two-way)
                def _to_float_odds(x):
                    if x is None or (isinstance(x, float) and pd.isna(x)):
                        return None
                    s = str(x).strip().upper()
                    if s in ("EVEN", "EV", "E"):
                        return 100.0
                    try:
                        return float(s)
                    except Exception:
                        return None
                dec_home = american_to_decimal(_to_float_odds(m["home_ml"])) if pd.notna(m.get("home_ml")) else None
                dec_away = american_to_decimal(_to_float_odds(m["away_ml"])) if pd.notna(m.get("away_ml")) else None
                if dec_home is not None and dec_away is not None:
                    imp_h = decimal_to_implied_prob(dec_home)
                    imp_a = decimal_to_implied_prob(dec_away)
                    nv_h, nv_a = remove_vig_two_way(imp_h, imp_a)
                    row["ev_home_ml"] = round(ev_unit(row["p_home_ml"], dec_home), 4)
                    row["ev_away_ml"] = round(ev_unit(row["p_away_ml"], dec_away), 4)
                    row["edge_home_ml"] = round(row["p_home_ml"] - nv_h, 4)
                    row["edge_away_ml"] = round(row["p_away_ml"] - nv_a, 4)
                # Save odds and bookmaker metadata if present
                try:
                    row["home_ml_odds"] = _to_float_odds(m.get("home_ml")) if pd.notna(m.get("home_ml")) else None
                except Exception:
                    row["home_ml_odds"] = None
                try:
                    row["away_ml_odds"] = _to_float_odds(m.get("away_ml")) if pd.notna(m.get("away_ml")) else None
                except Exception:
                    row["away_ml_odds"] = None
                if "home_ml_book" in m and pd.notna(m.get("home_ml_book")):
                    row["home_ml_book"] = m.get("home_ml_book")
                if "away_ml_book" in m and pd.notna(m.get("away_ml_book")):
                    row["away_ml_book"] = m.get("away_ml_book")
                if bankroll > 0:
                    row["stake_home_ml"] = round(kelly_stake(row["p_home_ml"], dec_home, bankroll, kelly_fraction_part), 2)
                    row["stake_away_ml"] = round(kelly_stake(row["p_away_ml"], dec_away, bankroll, kelly_fraction_part), 2)
                # Totals EVs (two-way at half-line)
                if all(k in m for k in ["over", "under"]):
                    dec_over = american_to_decimal(_to_float_odds(m["over"])) if pd.notna(m.get("over")) else None
                    dec_under = american_to_decimal(_to_float_odds(m["under"])) if pd.notna(m.get("under")) else None
                    if dec_over is not None and dec_under is not None:
                        imp_o = decimal_to_implied_prob(dec_over)
                        imp_u = decimal_to_implied_prob(dec_under)
                        nv_o, nv_u = remove_vig_two_way(imp_o, imp_u)
                        row["ev_over"] = round(ev_unit(row["p_over"], dec_over), 4)
                        row["ev_under"] = round(ev_unit(row["p_under"], dec_under), 4)
                        row["edge_over"] = round(row["p_over"] - nv_o, 4)
                        row["edge_under"] = round(row["p_under"] - nv_u, 4)
                    try:
                        row["over_odds"] = _to_float_odds(m.get("over")) if pd.notna(m.get("over")) else None
                    except Exception:
                        row["over_odds"] = None
                    try:
                        row["under_odds"] = _to_float_odds(m.get("under")) if pd.notna(m.get("under")) else None
                    except Exception:
                        row["under_odds"] = None
                    if "over_book" in m and pd.notna(m.get("over_book")):
                        row["over_book"] = m.get("over_book")
                    if "under_book" in m and pd.notna(m.get("under_book")):
                        row["under_book"] = m.get("under_book")
                    if bankroll > 0:
                        row["stake_over"] = round(kelly_stake(row["p_over"], dec_over, bankroll, kelly_fraction_part), 2)
                        row["stake_under"] = round(kelly_stake(row["p_under"], dec_under, bankroll, kelly_fraction_part), 2)
                # Puck line EVs (assume home -1.5 / away +1.5 two-way)
                if all(k in m for k in ["home_pl_-1.5", "away_pl_+1.5"]):
                    dec_hpl = american_to_decimal(_to_float_odds(m["home_pl_-1.5"])) if pd.notna(m.get("home_pl_-1.5")) else None
                    dec_apl = american_to_decimal(_to_float_odds(m["away_pl_+1.5"])) if pd.notna(m.get("away_pl_+1.5")) else None
                    if dec_hpl is not None and dec_apl is not None:
                        imp_hpl = decimal_to_implied_prob(dec_hpl)
                        imp_apl = decimal_to_implied_prob(dec_apl)
                        nv_hpl, nv_apl = remove_vig_two_way(imp_hpl, imp_apl)
                        row["ev_home_pl_-1.5"] = round(ev_unit(row["p_home_pl_-1.5"], dec_hpl), 4)
                        row["ev_away_pl_+1.5"] = round(ev_unit(row["p_away_pl_+1.5"], dec_apl), 4)
                        row["edge_home_pl_-1.5"] = round(row["p_home_pl_-1.5"] - nv_hpl, 4)
                        row["edge_away_pl_+1.5"] = round(row["p_away_pl_+1.5"] - nv_apl, 4)
                    try:
                        row["home_pl_-1.5_odds"] = _to_float_odds(m.get("home_pl_-1.5")) if pd.notna(m.get("home_pl_-1.5")) else None
                    except Exception:
                        row["home_pl_-1.5_odds"] = None
                    try:
                        row["away_pl_+1.5_odds"] = _to_float_odds(m.get("away_pl_+1.5")) if pd.notna(m.get("away_pl_+1.5")) else None
                    except Exception:
                        row["away_pl_+1.5_odds"] = None
                    if "home_pl_-1.5_book" in m and pd.notna(m.get("home_pl_-1.5_book")):
                        row["home_pl_-1.5_book"] = m.get("home_pl_-1.5_book")
                    if "away_pl_+1.5_book" in m and pd.notna(m.get("away_pl_+1.5_book")):
                        row["away_pl_+1.5_book"] = m.get("away_pl_+1.5_book")
                    if bankroll > 0:
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
    hist = load_df(stats_path)
    req = pd.read_csv(odds_csv)
    shots = SkaterShotsModel()
    saves = GoalieSavesModel()
    goals = SkaterGoalsModel()
    out_rows = []
    for _, r in req.iterrows():
        market = str(r["market"]).upper()
        player = str(r["player"]) 
        line = float(r["line"]) 
        odds = float(r["odds"]) 
        dec = american_to_decimal(odds)
        if market == "SOG":
            lam = shots.player_lambda(hist, player, r.get("team"))
            p_over = shots.prob_over(lam, line)
        elif market == "SAVES":
            lam = saves.player_lambda(hist, player)
            p_over = saves.prob_over(lam, line)
        elif market == "GOALS":
            lam = goals.player_lambda(hist, player)
            p_over = goals.prob_over(lam, line)
        else:
            continue
        out_rows.append({
            "market": market,
            "player": player,
            "line": line,
            "odds": odds,
            "proj_lambda": round(lam, 3),
            "p_over": round(p_over, 4),
            "ev_over": round(ev_unit(p_over, dec), 4)
        })
    out = pd.DataFrame(out_rows)
    out_path = PROC_DIR / "props_predictions.csv"
    save_df(out, out_path)
    print(out)
    print(f"Saved props predictions to {out_path}")


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
        try:
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

    Strategy per date: Bovada → The Odds API → ensure predictions exist without odds.
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
    dates = sorted({pd.to_datetime(getattr(g, 'gameDate')).strftime('%Y-%m-%d') for g in games})
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

    Strategy per date: Bovada → The Odds API → ensure predictions exist without odds.
    """
    build_range_core(start=start, end=end, source=source, bankroll=bankroll, kelly_fraction_part=kelly_fraction_part)

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

    Supports markets: SOG, GOALS, SAVES, ASSISTS, POINTS.
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
    # Ensure player stats exist over the range
    try:
        collect_player_game_stats(start, end, source="stats")
    except Exception:
        pass
    stats_path = RAW_DIR / "player_game_stats.csv"
    if not stats_path.exists():
        print("player_game_stats.csv missing; run collect_props first.")
        raise typer.Exit(code=1)
    stats = pd.read_csv(stats_path)
    # Filter stats to date window and build per-player per-date outcome columns
    stats["date_key"] = pd.to_datetime(stats["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    stats = stats[(stats["date_key"] >= start) & (stats["date_key"] <= end)]
    # Keep relevant columns
    keep = ["date_key","player","shots","goals","assists","saves"]
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
    print("[props_watch] no props found; finished attempts")


if __name__ == "__main__":
    app()
