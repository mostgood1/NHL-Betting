from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from nhl_betting.data.nhl_api_web import NHLWebClient
from nhl_betting.data.odds_api import OddsAPIClient, normalize_snapshot_to_rows


def _norm(s: object) -> str:
    return " ".join(str(s or "").strip().lower().split())


def _norm_team_name(s: object) -> str:
    # Normalize and strip accents-ish by removing non-ascii where possible.
    x = _norm(s)
    try:
        x = x.encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    return x


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return math.log(p / (1.0 - p))


def _american_to_implied_prob(price: Optional[float]) -> Optional[float]:
    try:
        if price is None:
            return None
        p = int(price)
        if p == 0:
            return None
        if p > 0:
            return 100.0 / (float(p) + 100.0)
        return float(-p) / (float(-p) + 100.0)
    except Exception:
        return None


def _profit_units(price: int, outcome: str) -> float:
    """1 unit stake; returns profit (win positive, loss -1, push 0)."""
    outcome = str(outcome).lower()
    if outcome == "push":
        return 0.0
    if outcome == "loss":
        return -1.0
    # win
    if price > 0:
        return float(price) / 100.0
    return 100.0 / float(abs(price))


def _poisson_cdf(k: int, mu: float) -> float:
    if mu <= 0:
        return 1.0 if k >= 0 else 0.0
    if k < 0:
        return 0.0
    pmf = math.exp(-float(mu))
    s = pmf
    for i in range(0, int(k)):
        pmf = pmf * float(mu) / float(i + 1)
        s += pmf
    return float(max(0.0, min(1.0, s)))


def _poisson_sf(k: int, mu: float) -> float:
    if k <= 0:
        return 1.0
    return float(max(0.0, 1.0 - _poisson_cdf(int(k) - 1, float(mu))))


def _total_probs(cur_total: int, line: float, mu_add: float) -> Tuple[float, float, float]:
    """Return (p_over, p_under, p_push) for totals given current total and Poisson(additional goals)."""
    is_int = abs(float(line) - round(float(line))) < 1e-9
    if is_int:
        line_i = int(round(float(line)))
        over_min_total = line_i + 1
        under_max_total = line_i - 1
        push_total = line_i
    else:
        over_min_total = int((float(line) // 1) + 1)
        under_max_total = int(float(line) // 1)
        push_total = None

    need_over = int(over_min_total) - int(cur_total)
    need_under = int(under_max_total) - int(cur_total)

    p_over = 1.0 if need_over <= 0 else _poisson_sf(int(need_over), float(mu_add))
    p_under = 0.0 if need_under < 0 else _poisson_cdf(int(need_under), float(mu_add))

    p_push = 0.0
    if push_total is not None:
        k_push = int(push_total) - int(cur_total)
        if k_push <= 0:
            # if already at/over push point, push probability is basically 0 in remaining goals model
            p_push = 0.0
        else:
            p_push = max(0.0, _poisson_cdf(k_push, float(mu_add)) - _poisson_cdf(k_push - 1, float(mu_add)))

    p_over = float(max(0.0, min(1.0, p_over)))
    p_under = float(max(0.0, min(1.0, p_under)))
    p_push = float(max(0.0, min(1.0, p_push)))
    return p_over, p_under, p_push


@dataclass
class Bet:
    game_pk: int
    matchup: str
    sample_min: float
    market: str
    side: str
    line: Optional[float]
    price: int
    p_model: float
    implied: float
    edge: float
    result: str
    profit: float


def _play_abs_sec(play: Dict[str, Any]) -> Optional[int]:
    try:
        pd = play.get("periodDescriptor") or {}
        per = int((pd or {}).get("number") or 0)
        tip = str(play.get("timeInPeriod") or "")
        if per < 1 or per > 3 or ":" not in tip:
            return None
        mm, ss = tip.split(":", 1)
        sec = 60 * int(mm) + int(ss)
        return (per - 1) * 1200 + sec
    except Exception:
        return None


def _state_at_abs_sec(pbp: Dict[str, Any], abs_sec: int) -> Dict[str, Any]:
    plays = pbp.get("plays") or []
    best = None
    best_sec = None
    for p in plays:
        if not isinstance(p, dict):
            continue
        s = _play_abs_sec(p)
        if s is None or s > abs_sec:
            continue
        if best_sec is None or s >= best_sec:
            best = p
            best_sec = s
    # Walk backward to find a play with scoreboard fields
    idx = None
    if best is not None:
        try:
            idx = plays.index(best)
        except Exception:
            idx = None
    home_score = away_score = None
    home_sog = away_sog = None
    if idx is not None:
        for j in range(idx, max(-1, idx - 60), -1):
            try:
                d = (plays[j] or {}).get("details") or {}
                if home_score is None and d.get("homeScore") is not None:
                    home_score = int(d.get("homeScore"))
                if away_score is None and d.get("awayScore") is not None:
                    away_score = int(d.get("awayScore"))
                if home_sog is None and d.get("homeSOG") is not None:
                    home_sog = int(d.get("homeSOG"))
                if away_sog is None and d.get("awaySOG") is not None:
                    away_sog = int(d.get("awaySOG"))
                if home_score is not None and away_score is not None and home_sog is not None and away_sog is not None:
                    break
            except Exception:
                continue

    # fallback: at least get final from pbp if missing
    try:
        if home_score is None:
            home_score = int((pbp.get("homeTeam") or {}).get("score"))
        if away_score is None:
            away_score = int((pbp.get("awayTeam") or {}).get("score"))
    except Exception:
        pass

    return {
        "home_score": home_score,
        "away_score": away_score,
        "home_sog": home_sog,
        "away_sog": away_sog,
    }


def _to_snapshot_iso(start_utc: str, abs_sec: int) -> str:
    # crude mapping: wall time is ~1.75x game clock + intermissions (18m each after p1,p2)
    st = datetime.fromisoformat(start_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    game_min = abs_sec / 60.0
    wall_min = 1.75 * game_min
    # intermissions: assume if abs_sec beyond 20 or 40
    if abs_sec >= 1200:
        wall_min += 18.0
    if abs_sec >= 2400:
        wall_min += 18.0
    # add small constant for stoppage/setup
    wall_min += 5.0
    snap = st + timedelta(minutes=wall_min)
    return snap.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _extract_prices(df_row: pd.Series, home_team: str, away_team: str) -> Dict[str, Any]:
    # df uses home/away names matching odds api; just pluck fields
    return {
        "home_ml": df_row.get("home_ml"),
        "away_ml": df_row.get("away_ml"),
        "over": df_row.get("over"),
        "under": df_row.get("under"),
        "total_line": df_row.get("total_line"),
    }


def main(date_ymd: str = "2026-02-26") -> None:
    web = NHLWebClient(timeout=15)
    oa = OddsAPIClient(rate_limit_per_sec=2.5)

    # Basic OddsAPI key visibility (do not print actual key)
    try:
        import os

        has_key = bool(os.environ.get("ODDS_API_KEY"))
        has_hist = bool(os.environ.get("ODDS_API_KEY_HISTORICAL"))
        print(f"OddsAPI env: ODDS_API_KEY={has_key} ODDS_API_KEY_HISTORICAL={has_hist}")
    except Exception:
        pass

    # NHL games for the date
    sb = web.scoreboard_day(date_ymd) or []
    print(f"date={date_ymd} games={len(sb)}")

    # Map NHL matchup -> OddsAPI event id via historical_list_events near each game's start.
    # list_events only returns upcoming games, so it will be empty for completed dates.
    ev_map: Dict[Tuple[str, str], str] = {}
    for g in sb:
        home = str(g.get("home") or "")
        away = str(g.get("away") or "")
        start_time_utc = str(g.get("gameDate") or "").strip()
        if not start_time_utc:
            continue
        st = datetime.fromisoformat(start_time_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
        snapshot = (st - timedelta(minutes=10)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        try:
            payload, _ = oa.historical_list_events("icehockey_nhl", snapshot_iso=snapshot)
            data = (payload or {}).get("data") or []
            # Match by team names
            want_ht = _norm_team_name(home)
            want_at = _norm_team_name(away)
            eid = None
            for ev in data:
                ht = _norm_team_name((ev or {}).get("home_team"))
                at = _norm_team_name((ev or {}).get("away_team"))
                if ht == want_ht and at == want_at:
                    eid = str((ev or {}).get("id") or "").strip()
                    break
            if eid:
                ev_map[(want_at, want_ht)] = eid
        except Exception as e:
            # keep going; diagnostics later
            continue

    print(f"OddsAPI mapped events: {len(ev_map)} / {len(sb)}")

    # Pregame model map from bundle
    from nhl_betting.web.app import _load_bundle_predictions_map, _norm_game_key

    pred_map = _load_bundle_predictions_map(date_ymd) or {}
    print(f"Bundle predictions map: {len(pred_map)} games")
    if pred_map:
        try:
            print("Bundle keys sample:", list(sorted(pred_map.keys()))[:5])
        except Exception:
            pass

    bets: List[Bet] = []

    matched_events = 0
    odds_snapshots_found = 0
    odds_snapshots_empty = 0
    odds_snapshot_errors = 0
    missing_model = 0

    sample_abs_secs = [1200, 2400, 3000]  # end 1st, end 2nd, 50:00 elapsed

    for g in sb:
        try:
            game_pk = int(g.get("gamePk"))
        except Exception:
            continue
        home = str(g.get("home") or "")
        away = str(g.get("away") or "")
        matchup = f"{away} @ {home}"

        # Pull pbp for state reconstruction
        pbp = web._get(f"/gamecenter/{game_pk}/play-by-play", None, 1)
        start_time_utc = pbp.get("startTimeUTC")
        if not start_time_utc:
            continue

        # Final result
        final_home = int((pbp.get("homeTeam") or {}).get("score") or 0)
        final_away = int((pbp.get("awayTeam") or {}).get("score") or 0)
        final_total = final_home + final_away

        # OddsAPI event id
        eid = ev_map.get((_norm_team_name(away), _norm_team_name(home)))
        if not eid:
            # can't score without odds
            continue
        matched_events += 1

        # Pred map key
        key = str(_norm_game_key(away, home) or "").strip().lower()
        pm = pred_map.get(key) or {}
        p0_home = pm.get("p_home_ml")
        model_total = pm.get("model_total")
        model_spread = pm.get("model_spread")
        if model_spread is None:
            try:
                phg = pm.get("proj_home_goals")
                pag = pm.get("proj_away_goals")
                if phg is not None and pag is not None:
                    model_spread = float(phg) - float(pag)
            except Exception:
                model_spread = None

        if p0_home is None or model_total is None:
            missing_model += 1
            continue

        for abs_sec in sample_abs_secs:
            em = abs_sec / 60.0
            rm = max(0.0, 60.0 - em)
            st = _state_at_abs_sec(pbp, abs_sec)
            hs = st.get("home_score")
            a_s = st.get("away_score")
            h_sog = st.get("home_sog")
            a_sog = st.get("away_sog")
            if hs is None or a_s is None:
                continue
            gd = int(hs) - int(a_s)
            sd = (int(h_sog) - int(a_sog)) if (h_sog is not None and a_sog is not None) else None

            # Fetch historical in-play odds snapshot near this time (best-effort)
            snap_guess = _to_snapshot_iso(str(start_time_utc), abs_sec)
            odds_df = None
            last_err = None
            for delta_min in [0, -6, 6, -12, 12, -18, 18]:
                snap = datetime.fromisoformat(snap_guess.replace("Z", "+00:00")).astimezone(timezone.utc) + timedelta(minutes=delta_min)
                snap_iso = snap.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                try:
                    ev_odds, _ = oa.historical_event_odds(
                        "icehockey_nhl",
                        eid,
                        markets="h2h,totals,spreads",
                        snapshot_iso=snap_iso,
                        regions="us",
                    )
                    tmp = normalize_snapshot_to_rows([ev_odds], bookmaker=None, best_of_all=True)
                    if tmp is not None and not tmp.empty:
                        odds_df = tmp
                        break
                except Exception as e:
                    last_err = e
                    odds_snapshot_errors += 1
                    continue

            if odds_df is None or odds_df.empty:
                if last_err is not None:
                    odds_snapshots_empty += 1
                    # Print a small sample of failures for first couple games/times
                    if odds_snapshots_empty <= 5:
                        print(
                            f"No odds snapshot for {matchup} at em={em:.1f} (guess={snap_guess}); last_err={type(last_err).__name__}: {last_err}"
                        )
                else:
                    odds_snapshots_empty += 1
                continue

            odds_snapshots_found += 1

            row = odds_df.iloc[0]
            prices = _extract_prices(row, home, away)

            # --- ML signal heuristic (mirrors app.py but without pbp_ctx extras) ---
            z = _logit(float(p0_home))
            w = 0.5 + 1.5 * (em / 60.0)
            z += 1.15 * float(gd) * float(w)
            if sd is not None:
                z += 0.03 * float(max(-20, min(20, sd)))
            p_home_live = float(max(0.01, min(0.99, _sigmoid(z))))
            p_away_live = 1.0 - p_home_live

            for side in ["HOME", "AWAY"]:
                if side == "HOME":
                    price = prices.get("home_ml")
                    p_model = p_home_live
                else:
                    price = prices.get("away_ml")
                    p_model = p_away_live
                try:
                    price_i = int(price)
                except Exception:
                    continue
                implied = _american_to_implied_prob(price_i)
                if implied is None:
                    continue
                edge = float(p_model) - float(implied)
                action = "WATCH"
                if edge >= 0.03:
                    if em >= 8.0 and edge >= 0.045:
                        action = "BET"
                    if em >= 4.0 and edge >= 0.06:
                        action = "BET"

                if action == "BET":
                    # result
                    if side == "HOME":
                        result = "win" if final_home > final_away else ("push" if final_home == final_away else "loss")
                    else:
                        result = "win" if final_away > final_home else ("push" if final_home == final_away else "loss")
                    bets.append(
                        Bet(
                            game_pk=game_pk,
                            matchup=matchup,
                            sample_min=float(em),
                            market="ML",
                            side=side,
                            line=None,
                            price=int(price_i),
                            p_model=float(p_model),
                            implied=float(implied),
                            edge=float(edge),
                            result=result,
                            profit=float(_profit_units(int(price_i), result)),
                        )
                    )

            # --- TOTAL signal heuristic (simplified) ---
            try:
                line = float(prices.get("total_line"))
                over_price = prices.get("over")
                under_price = prices.get("under")
                if over_price is None or under_price is None:
                    continue
                over_i = int(over_price)
                under_i = int(under_price)

                cur_total = int(hs) + int(a_s)
                mu_add = float(model_total) * (float(rm) / 60.0)
                mu_add = max(0.0, mu_add)
                p_over, p_under, p_push = _total_probs(cur_total, line, mu_add)

                implied_over = _american_to_implied_prob(over_i)
                implied_under = _american_to_implied_prob(under_i)
                if implied_over is None or implied_under is None:
                    continue
                edge_over = p_over - implied_over
                edge_under = p_under - implied_under

                if edge_over >= edge_under:
                    side = "OVER"
                    price_i = over_i
                    p_model = p_over
                    implied = implied_over
                    edge = edge_over
                else:
                    side = "UNDER"
                    price_i = under_i
                    p_model = p_under
                    implied = implied_under
                    edge = edge_under

                abs_edge = abs(float(edge))
                action = "WATCH"
                if em >= 10.0 and abs_edge >= 0.04:
                    action = "BET"
                if em >= 6.0 and abs_edge >= 0.06:
                    action = "BET"

                # If already reached line, don't count
                if cur_total >= line:
                    action = "WATCH"

                if action == "BET":
                    # settle
                    is_int = abs(float(line) - round(float(line))) < 1e-9
                    if is_int and int(final_total) == int(round(line)):
                        result = "push"
                    else:
                        if side == "OVER":
                            result = "win" if final_total > line else "loss"
                        else:
                            result = "win" if final_total < line else "loss"
                    bets.append(
                        Bet(
                            game_pk=game_pk,
                            matchup=matchup,
                            sample_min=float(em),
                            market="TOTAL",
                            side=side,
                            line=float(line),
                            price=int(price_i),
                            p_model=float(p_model),
                            implied=float(implied),
                            edge=float(edge),
                            result=result,
                            profit=float(_profit_units(int(price_i), result)),
                        )
                    )
            except Exception:
                pass

    if not bets:
        print(
            "No BET signals generated. Diagnostics: "
            f"matched_events={matched_events} missing_model={missing_model} odds_found={odds_snapshots_found} odds_empty={odds_snapshots_empty} odds_errors={odds_snapshot_errors}"
        )
        return

    df = pd.DataFrame([b.__dict__ for b in bets])
    print("\nBETS summary")
    print(df.groupby(["market"]).agg(
        n=("profit","size"),
        win_rate=("result", lambda s: float((s=="win").mean())),
        avg_edge=("edge","mean"),
        roi_units=("profit","sum"),
        avg_profit=("profit","mean"),
    ).reset_index())

    print("\nTop losses (by -profit)")
    try:
        print(df.sort_values("profit").head(10)[["market","side","line","price","sample_min","matchup","edge","result","profit"]])
    except Exception:
        pass

    print("\nTop wins (by profit)")
    try:
        print(df.sort_values("profit", ascending=False).head(10)[["market","side","line","price","sample_min","matchup","edge","result","profit"]])
    except Exception:
        pass


if __name__ == "__main__":
    main("2026-02-26")
