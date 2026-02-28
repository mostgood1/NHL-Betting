from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.io import PROC_DIR


def _clamp01(x: float | None) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return None
        return max(0.0, min(1.0, xf))
    except Exception:
        return None


def _num(x: object) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            xf = float(x)
            return xf if math.isfinite(xf) else None
        s = str(x).strip().replace(",", "")
        if s == "":
            return None
        xf = float(s)
        return xf if math.isfinite(xf) else None
    except Exception:
        return None


def _support01(signed_value: float | None, scale: float) -> float | None:
    """Map a signed support value into [0,1] with tanh smoothing.

    signed_value > 0 => supports the bet; signed_value < 0 => opposes.
    """
    if signed_value is None:
        return None
    try:
        v = float(signed_value)
        if not math.isfinite(v):
            return None
        sc = float(scale) if scale and math.isfinite(scale) and scale > 0 else 1.0
        return 0.5 + 0.5 * math.tanh(v / sc)
    except Exception:
        return None


def _fmt_signed(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return "—"
        s = f"{xf:+.{digits}f}"
        # avoid +0.00
        if s.startswith("+0"):
            return s.replace("+0", "+0", 1)
        return s
    except Exception:
        return "—"


def _safe_parse_slate_dt(pred: pd.DataFrame) -> pd.Series:
    """Return a datetime64 series representing the slate date (ET) as best-effort."""
    if pred is None or pred.empty:
        return pd.Series([], dtype="datetime64[ns]")
    if "date_et" in pred.columns:
        return pd.to_datetime(pred["date_et"], errors="coerce")
    # Fallback: first 10 chars of ISO-ish timestamp
    s = pred.get("date")
    if s is None:
        return pd.to_datetime(pd.Series([pd.NA] * len(pred)), errors="coerce")
    return pd.to_datetime(s.astype(str).str.slice(0, 10), errors="coerce")


@lru_cache(maxsize=4)
def _load_team_games(proc_dir: Path = PROC_DIR) -> pd.DataFrame:
    p = proc_dir / "team_games.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        tg = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    if tg is None or tg.empty:
        return pd.DataFrame()
    try:
        tg = tg.copy()
        tg["date_dt"] = pd.to_datetime(tg["date"], errors="coerce")
        tg = tg.dropna(subset=["date_dt", "team"])
        tg["team"] = tg["team"].astype(str)
        tg = tg.sort_values(["team", "date_dt"])
    except Exception:
        return pd.DataFrame()
    return tg


def _attach_team_form(pred: pd.DataFrame, proc_dir: Path = PROC_DIR) -> pd.DataFrame:
    """Attach last-known roll10/rest/b2b features for home/away teams."""
    if pred is None or pred.empty:
        return pred

    tg = _load_team_games(proc_dir)
    if tg is None or tg.empty:
        return pred

    out = pred.copy()
    try:
        out["slate_dt"] = _safe_parse_slate_dt(out)
        out = out.dropna(subset=["slate_dt", "home", "away"])
        out["home"] = out["home"].astype(str)
        out["away"] = out["away"].astype(str)

        # Build a (slate_dt, team) table to merge-asof by team.
        teams = pd.concat(
            [
                out[["slate_dt", "home"]].rename(columns={"home": "team"}),
                out[["slate_dt", "away"]].rename(columns={"away": "team"}),
            ],
            ignore_index=True,
        ).drop_duplicates()
        teams = teams.sort_values(["team", "slate_dt"])

        # merge_asof requires both sides sorted.
        merged = pd.merge_asof(
            teams,
            tg,
            left_on="slate_dt",
            right_on="date_dt",
            by="team",
            direction="backward",
        )

        # Keep only the columns we actually use.
        keep_cols = {
            "team",
            "slate_dt",
            "goals_for_roll10",
            "goals_against_roll10",
            "rest_days",
            "b2b",
        }
        merged = merged[[c for c in merged.columns if c in keep_cols]].copy()

        # Attach for home
        out = out.merge(
            merged.rename(
                columns={
                    "team": "home",
                    "goals_for_roll10": "home_gf10",
                    "goals_against_roll10": "home_ga10",
                    "rest_days": "home_rest_days",
                    "b2b": "home_b2b",
                }
            ),
            on=["slate_dt", "home"],
            how="left",
        )
        # Attach for away
        out = out.merge(
            merged.rename(
                columns={
                    "team": "away",
                    "goals_for_roll10": "away_gf10",
                    "goals_against_roll10": "away_ga10",
                    "rest_days": "away_rest_days",
                    "b2b": "away_b2b",
                }
            ),
            on=["slate_dt", "away"],
            how="left",
        )

    except Exception:
        return pred

    return out


@dataclass(frozen=True)
class _EdgeSpec:
    market_group: str
    bet: str
    prob_col: str
    odds_col: str
    book_col: str | None = None
    needs_totals_line: bool = False


_EDGE_SPECS: dict[str, _EdgeSpec] = {
    "ev_home_ml": _EdgeSpec("moneyline", "home_ml", "p_home_ml", "home_ml_odds", "home_ml_book"),
    "ev_away_ml": _EdgeSpec("moneyline", "away_ml", "p_away_ml", "away_ml_odds", "away_ml_book"),
    "ev_over": _EdgeSpec("totals", "over", "p_over", "over_odds", "over_book", needs_totals_line=True),
    "ev_under": _EdgeSpec("totals", "under", "p_under", "under_odds", "under_book", needs_totals_line=True),
    "ev_home_pl_-1.5": _EdgeSpec("puckline", "home_pl_-1.5", "p_home_pl_-1.5", "home_pl_-1.5_odds", "home_pl_-1.5_book"),
    "ev_away_pl_+1.5": _EdgeSpec("puckline", "away_pl_+1.5", "p_away_pl_+1.5", "away_pl_+1.5_odds", "away_pl_+1.5_book"),
    "ev_f10_yes": _EdgeSpec("first10", "f10_yes", "p_f10_yes", "f10_yes_odds", None),
    "ev_f10_no": _EdgeSpec("first10", "f10_no", "p_f10_no", "f10_no_odds", None),
    "ev_p1_over": _EdgeSpec("periods", "p1_over", "p1_over_prob", "p1_over_odds", None),
    "ev_p1_under": _EdgeSpec("periods", "p1_under", "p1_under_prob", "p1_under_odds", None),
    "ev_p2_over": _EdgeSpec("periods", "p2_over", "p2_over_prob", "p2_over_odds", None),
    "ev_p2_under": _EdgeSpec("periods", "p2_under", "p2_under_prob", "p2_under_odds", None),
    "ev_p3_over": _EdgeSpec("periods", "p3_over", "p3_over_prob", "p3_over_odds", None),
    "ev_p3_under": _EdgeSpec("periods", "p3_under", "p3_under_prob", "p3_under_odds", None),
}


def attach_game_edge_signals(
    date: str,
    edges: pd.DataFrame,
    predictions: Optional[pd.DataFrame] = None,
    proc_dir: Path = PROC_DIR,
) -> pd.DataFrame:
    """Attach non-EV edge signals to long-form game edges.

    Adds:
    - market_group, bet
    - prob, price, book
    - totals_line (totals only, if available)
    - edge_score in [0,1]
    - edge_reasons (human-readable)

    This is designed to be backward compatible: existing columns are untouched.
    """
    if edges is None or edges.empty:
        return edges

    # Load predictions if not provided
    pred = predictions
    if pred is None:
        p_sim = proc_dir / f"predictions_sim_{date}.csv"
        p_legacy = proc_dir / f"predictions_{date}.csv"
        p = p_sim if p_sim.exists() else p_legacy
        if not p.exists():
            return edges
        try:
            pred = pd.read_csv(p)
        except Exception:
            return edges

    if pred is None or pred.empty:
        return edges

    pred = pred.copy()
    try:
        # Normalize total line column
        if "total_line_used" not in pred.columns and "totals_line_used" in pred.columns:
            pred["total_line_used"] = pred["totals_line_used"]
    except Exception:
        pass

    # Attach team form context (recency, rest)
    pred = _attach_team_form(pred, proc_dir=proc_dir)

    # Join predictions into edges on slate date + teams
    e = edges.copy()
    try:
        e["date_key"] = e["date"].astype(str).str.slice(0, 10)
    except Exception:
        e["date_key"] = str(date)
    try:
        pred["date_key"] = pred.get("date_et")
        if pred["date_key"].isna().all():
            pred["date_key"] = pred.get("date")
        pred["date_key"] = pred["date_key"].astype(str).str.slice(0, 10)
    except Exception:
        pred["date_key"] = str(date)

    # Keep only columns needed to compute edge signals
    pred_keep = {
        "date_key",
        "home",
        "away",
        "total_line_used",
        "model_total",
        "model_spread",
        "period1_home_proj",
        "period1_away_proj",
        "home_gf10",
        "home_ga10",
        "away_gf10",
        "away_ga10",
        "home_rest_days",
        "away_rest_days",
        "home_b2b",
        "away_b2b",
    }
    for spec in _EDGE_SPECS.values():
        pred_keep.add(spec.prob_col)
        pred_keep.add(spec.odds_col)
        if spec.book_col:
            pred_keep.add(spec.book_col)

    pred2 = pred[[c for c in pred.columns if c in pred_keep]].copy()
    # Join on matchup keys only.
    # Many artifacts have mixed UTC/ET dates in the `date` string, so joining on a derived
    # date_key can fail even when home/away match perfectly.
    try:
        pred_join = pred2.drop(columns=["date_key"], errors="ignore")
        pred_join = pred_join.drop_duplicates(subset=["home", "away"], keep="last")
    except Exception:
        pred_join = pred2.drop(columns=["date_key"], errors="ignore")
    merged = e.merge(pred_join, on=["home", "away"], how="left")

    # Derive per-game features used across markets
    def _gd(gf: object, ga: object) -> float | None:
        gff = _num(gf)
        gaa = _num(ga)
        if gff is None or gaa is None:
            return None
        return gff - gaa

    merged["home_gd10"] = merged.apply(lambda r: _gd(r.get("home_gf10"), r.get("home_ga10")), axis=1)
    merged["away_gd10"] = merged.apply(lambda r: _gd(r.get("away_gf10"), r.get("away_ga10")), axis=1)
    merged["form_delta"] = merged.apply(
        lambda r: (_num(r.get("home_gd10")) - _num(r.get("away_gd10")))
        if (_num(r.get("home_gd10")) is not None and _num(r.get("away_gd10")) is not None)
        else None,
        axis=1,
    )
    merged["rest_delta"] = merged.apply(
        lambda r: (_num(r.get("home_rest_days")) - _num(r.get("away_rest_days")))
        if (_num(r.get("home_rest_days")) is not None and _num(r.get("away_rest_days")) is not None)
        else None,
        axis=1,
    )
    merged["b2b_delta"] = merged.apply(
        lambda r: (_num(r.get("away_b2b")) - _num(r.get("home_b2b")))
        if (_num(r.get("away_b2b")) is not None and _num(r.get("home_b2b")) is not None)
        else None,
        axis=1,
    )
    merged["total_bias"] = merged.apply(
        lambda r: (_num(r.get("model_total")) - _num(r.get("total_line_used")))
        if (_num(r.get("model_total")) is not None and _num(r.get("total_line_used")) is not None)
        else None,
        axis=1,
    )

    def _row_edge_score_and_reasons(r: pd.Series) -> tuple[float | None, str]:
        m = str(r.get("market") or "")
        spec = _EDGE_SPECS.get(m)
        if spec is None:
            return None, ""

        prob = _num(r.get(spec.prob_col))
        if prob is None:
            return None, ""

        model_conf = _clamp01(abs(prob - 0.5) * 2.0)

        spread = _num(r.get("model_spread"))
        total_bias = _num(r.get("total_bias"))
        form_delta = _num(r.get("form_delta"))
        rest_delta = _num(r.get("rest_delta"))
        b2b_delta = _num(r.get("b2b_delta"))

        reasons: list[str] = []

        # By default, score mostly on model confidence, then layer matchup context.
        edge_score = None

        if spec.market_group == "moneyline":
            is_home = spec.bet.startswith("home_")
            direction = 1.0 if is_home else -1.0
            spread_support = _support01(direction * spread if spread is not None else None, scale=0.65)
            form_support = _support01(direction * form_delta if form_delta is not None else None, scale=1.25)
            rest_adv = None
            if rest_delta is not None or b2b_delta is not None:
                rest_adv = (rest_delta or 0.0) + 1.5 * (b2b_delta or 0.0)
            rest_support = _support01(direction * rest_adv if rest_adv is not None else None, scale=2.0)

            parts = [
                (0.50, model_conf),
                (0.25, spread_support),
                (0.15, form_support),
                (0.10, rest_support),
            ]
            s = 0.0
            w = 0.0
            for ww, vv in parts:
                if vv is None:
                    continue
                s += ww * float(vv)
                w += ww
            edge_score = (s / w) if w > 0 else None

            reasons.append(f"p={prob:.3f}")
            if spread is not None:
                reasons.append(f"spread {_fmt_signed(spread, 2)}")
            if form_delta is not None:
                reasons.append(f"L10 GDΔ {_fmt_signed(form_delta, 2)}")
            if rest_adv is not None and (rest_delta is not None or b2b_delta is not None):
                # expose raw components
                rd = _fmt_signed(rest_delta, 0) if rest_delta is not None else "—"
                bd = _fmt_signed(b2b_delta, 0) if b2b_delta is not None else "—"
                reasons.append(f"restΔ {rd}, b2bΔ {bd}")

        elif spec.market_group == "totals":
            is_over = spec.bet == "over"
            direction = 1.0 if is_over else -1.0
            line_support = _support01(direction * total_bias if total_bias is not None else None, scale=0.75)

            # Pace hint from P1 projections (quickly reacts to roster/goalie updates).
            p1h = _num(r.get("period1_home_proj"))
            p1a = _num(r.get("period1_away_proj"))
            p1_sum = (p1h + p1a) if (p1h is not None and p1a is not None) else None
            pace_support = _support01(direction * ((p1_sum or 0.0) - 2.0) if p1_sum is not None else None, scale=0.45)

            parts = [
                (0.45, model_conf),
                (0.40, line_support),
                (0.15, pace_support),
            ]
            s = 0.0
            w = 0.0
            for ww, vv in parts:
                if vv is None:
                    continue
                s += ww * float(vv)
                w += ww
            edge_score = (s / w) if w > 0 else None

            tl = _num(r.get("total_line_used"))
            mt = _num(r.get("model_total"))
            if mt is not None and tl is not None:
                reasons.append(f"total {mt:.2f} vs line {tl:.1f} ({_fmt_signed(mt - tl, 2)})")
            reasons.append(f"p={prob:.3f}")
            if p1_sum is not None:
                reasons.append(f"P1 pace {p1_sum:.2f}")

        elif spec.market_group == "puckline":
            is_home = spec.bet.startswith("home_")
            direction = 1.0 if is_home else -1.0
            cover_margin = None
            if spread is not None:
                cover_margin = (spread - 1.5) if is_home else ((-spread) - 1.5)
            cover_support = _support01(cover_margin, scale=1.0)
            form_support = _support01(direction * form_delta if form_delta is not None else None, scale=1.5)

            parts = [
                (0.55, model_conf),
                (0.30, cover_support),
                (0.15, form_support),
            ]
            s = 0.0
            w = 0.0
            for ww, vv in parts:
                if vv is None:
                    continue
                s += ww * float(vv)
                w += ww
            edge_score = (s / w) if w > 0 else None

            reasons.append(f"p={prob:.3f}")
            if spread is not None:
                reasons.append(f"spread {_fmt_signed(spread, 2)}")
            if cover_margin is not None:
                reasons.append(f"cover margin {_fmt_signed(cover_margin, 2)}")
            if form_delta is not None:
                reasons.append(f"L10 GDΔ {_fmt_signed(form_delta, 2)}")

        elif spec.market_group in ("first10", "periods"):
            # For these markets we generally have less stable opponent context, so we
            # rank by confidence first and add a small pace/spread hint.
            p1h = _num(r.get("period1_home_proj"))
            p1a = _num(r.get("period1_away_proj"))
            p1_sum = (p1h + p1a) if (p1h is not None and p1a is not None) else None
            pace_support = _support01((p1_sum - 2.0) if p1_sum is not None else None, scale=0.55)

            parts = [
                (0.80, model_conf),
                (0.20, pace_support),
            ]
            s = 0.0
            w = 0.0
            for ww, vv in parts:
                if vv is None:
                    continue
                s += ww * float(vv)
                w += ww
            edge_score = (s / w) if w > 0 else None

            reasons.append(f"p={prob:.3f}")
            if p1_sum is not None:
                reasons.append(f"P1 pace {p1_sum:.2f}")

        return _clamp01(edge_score) if edge_score is not None else None, "; ".join([x for x in reasons if x])

    merged["market_group"] = merged["market"].map(lambda m: (_EDGE_SPECS.get(str(m)) or _EdgeSpec("", "", "", "")).market_group)
    merged["bet"] = merged["market"].map(lambda m: (_EDGE_SPECS.get(str(m)) or _EdgeSpec("", "", "", "")).bet)
    def _row_spec(r: pd.Series) -> _EdgeSpec | None:
        try:
            return _EDGE_SPECS.get(str(r.get("market") or ""))
        except Exception:
            return None

    def _row_prob(r: pd.Series) -> float | None:
        spec = _row_spec(r)
        return _num(r.get(spec.prob_col)) if spec is not None else None

    def _row_price(r: pd.Series) -> float | None:
        spec = _row_spec(r)
        return _num(r.get(spec.odds_col)) if spec is not None else None

    def _row_book(r: pd.Series) -> str | None:
        spec = _row_spec(r)
        if spec is None or not spec.book_col:
            return None
        try:
            v = r.get(spec.book_col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            s = str(v).strip()
            return s or None
        except Exception:
            return None

    merged["prob"] = merged.apply(_row_prob, axis=1)
    merged["price"] = merged.apply(_row_price, axis=1)
    merged["book"] = merged.apply(_row_book, axis=1)

    merged["totals_line"] = None
    try:
        totals_mask = merged["market_group"] == "totals"
        if bool(totals_mask.any()) and "total_line_used" in merged.columns:
            merged.loc[totals_mask, "totals_line"] = pd.to_numeric(merged.loc[totals_mask, "total_line_used"], errors="coerce")
    except Exception:
        pass

    # Human readable side label (kept simple for now)
    def _side_label(r: pd.Series) -> str:
        mg = str(r.get("market_group") or "")
        bet = str(r.get("bet") or "")
        if mg == "moneyline":
            return str(r.get("home")) if bet == "home_ml" else str(r.get("away"))
        if mg == "totals":
            ln = _num(r.get("totals_line"))
            core = "Over" if bet == "over" else "Under"
            return f"{core} {ln:.1f}" if ln is not None else core
        if mg == "puckline":
            if bet == "home_pl_-1.5":
                return f"{r.get('home')} -1.5"
            if bet == "away_pl_+1.5":
                return f"{r.get('away')} +1.5"
        if mg == "first10":
            return "Yes" if bet == "f10_yes" else "No"
        if mg == "periods":
            return bet.upper()
        return ""

    merged["side"] = merged.apply(_side_label, axis=1)

    # Compute scores
    scores = merged.apply(_row_edge_score_and_reasons, axis=1, result_type="expand")
    merged["edge_score"] = scores[0]
    merged["edge_reasons"] = scores[1]

    return merged
