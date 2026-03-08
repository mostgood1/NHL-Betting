from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.io import DATA_DIR, PROC_DIR


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


def _american_to_decimal(x: object) -> float | None:
    price = _num(x)
    if price is None or price == 0:
        return None
    try:
        return 1.0 + (price / 100.0) if price > 0 else 1.0 + (100.0 / abs(price))
    except Exception:
        return None


def _combine_support(parts: list[tuple[float, float | None]]) -> float | None:
    score = 0.0
    weight = 0.0
    for part_weight, value in parts:
        if value is None:
            continue
        try:
            vv = float(value)
        except Exception:
            continue
        if not math.isfinite(vv):
            continue
        score += float(part_weight) * vv
        weight += float(part_weight)
    if weight <= 0:
        return None
    return max(0.0, min(1.0, score / weight))


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


def _safe_read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _norm_game_key(away: object, home: object) -> str:
    away_s = str(away or "").strip().lower()
    home_s = str(home or "").strip().lower()
    if not away_s or not home_s:
        return ""
    return f"{away_s}@{home_s}"


def _games_map(obj: object) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not isinstance(obj, dict):
        return out
    games = obj.get("games") or []
    if not isinstance(games, list):
        return out
    for game in games:
        if not isinstance(game, dict):
            continue
        key = str(game.get("key") or _norm_game_key(game.get("away"), game.get("home")) or "").strip().lower()
        if not key:
            continue
        out[key] = game
    return out


@lru_cache(maxsize=16)
def _load_team_snapshot_games(date_ymd: str, data_root: str) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    snap_dir = Path(data_root) / "odds_snapshots" / "team_odds" / f"date={date_ymd}"
    open_obj = _safe_read_json(snap_dir / "open.json")
    prev_obj = _safe_read_json(snap_dir / "prev.json")
    cur_obj = _safe_read_json(snap_dir / "current.json")
    return _games_map(open_obj), _games_map(prev_obj), _games_map(cur_obj)


def _fmt_american(x: object) -> str:
    price = _num(x)
    if price is None:
        return "-"
    try:
        return f"{int(round(price)):+d}"
    except Exception:
        return "-"


def _driver_from_support(value: float | None, positive_tag: str, negative_tag: str, cutoff: float = 0.58) -> str | None:
    if value is None:
        return None
    try:
        vv = float(value)
    except Exception:
        return None
    if not math.isfinite(vv):
        return None
    lo = 1.0 - float(cutoff)
    if vv >= float(cutoff):
        return positive_tag
    if vv <= lo:
        return negative_tag
    return None


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


def _recommendation_signal_market(market: object, bet: object) -> str | None:
    key = (str(market or "").strip().lower(), str(bet or "").strip().lower())
    mapping = {
        ("moneyline", "home_ml"): "ev_home_ml",
        ("moneyline", "away_ml"): "ev_away_ml",
        ("totals", "over"): "ev_over",
        ("totals", "under"): "ev_under",
        ("puckline", "home_pl_-1.5"): "ev_home_pl_-1.5",
        ("puckline", "away_pl_+1.5"): "ev_away_pl_+1.5",
        ("first10", "f10_yes"): "ev_f10_yes",
        ("first10", "f10_no"): "ev_f10_no",
        ("periods", "p1_over"): "ev_p1_over",
        ("periods", "p1_under"): "ev_p1_under",
        ("periods", "p2_over"): "ev_p2_over",
        ("periods", "p2_under"): "ev_p2_under",
        ("periods", "p3_over"): "ev_p3_over",
        ("periods", "p3_under"): "ev_p3_under",
    }
    return mapping.get(key)


def _movement_support_for_edge(
    r: pd.Series,
    open_games: dict[str, dict],
    prev_games: dict[str, dict],
    cur_games: dict[str, dict],
) -> tuple[float | None, str, str]:
    spec = _EDGE_SPECS.get(str(r.get("market") or ""))
    if spec is None or spec.market_group not in {"moneyline", "totals", "puckline"}:
        return None, "", ""

    key = _norm_game_key(r.get("away"), r.get("home"))
    if not key:
        return None, "", ""

    open_game = open_games.get(key) or {}
    prev_game = prev_games.get(key) or {}
    cur_game = cur_games.get(key) or {}
    if not cur_game:
        return None, "", ""

    tags: list[str] = []
    reasons: list[str] = []
    support_parts: list[tuple[float, float | None]] = []

    def _price_block(game: dict) -> dict:
        block_name = "ml" if spec.market_group == "moneyline" else ("total" if spec.market_group == "totals" else "puckline")
        block = game.get(block_name) or {}
        return block if isinstance(block, dict) else {}

    def _price_support(open_price: object, prev_price: object, cur_price: object) -> float | None:
        cur_dec = _american_to_decimal(cur_price)
        if cur_dec is None:
            return None
        parts: list[tuple[float, float | None]] = []

        open_dec = _american_to_decimal(open_price)
        if open_dec is not None:
            delta_open = cur_dec - open_dec
            parts.append((0.65, _support01(delta_open, scale=0.12)))
            if abs(delta_open) >= 0.01:
                tags.append("PRICE+" if delta_open > 0 else "PRICE-")
                reasons.append(f"price {_fmt_american(open_price)}→{_fmt_american(cur_price)}")

        prev_dec = _american_to_decimal(prev_price)
        if prev_dec is not None:
            delta_prev = cur_dec - prev_dec
            parts.append((0.35, _support01(delta_prev, scale=0.08)))
            if abs(delta_prev) >= 0.01:
                reasons.append(f"tick {_fmt_american(prev_price)}→{_fmt_american(cur_price)}")

        return _combine_support(parts)

    if spec.market_group == "moneyline":
        side_key = "home" if spec.bet == "home_ml" else "away"
        open_price = _price_block(open_game).get(side_key)
        prev_price = _price_block(prev_game).get(side_key)
        cur_price = _price_block(cur_game).get(side_key)
        support_parts.append((1.0, _price_support(open_price, prev_price, cur_price)))

    elif spec.market_group == "puckline":
        side_key = "home_-1.5" if spec.bet == "home_pl_-1.5" else "away_+1.5"
        open_price = _price_block(open_game).get(side_key)
        prev_price = _price_block(prev_game).get(side_key)
        cur_price = _price_block(cur_game).get(side_key)
        support_parts.append((1.0, _price_support(open_price, prev_price, cur_price)))

    elif spec.market_group == "totals":
        block_open = _price_block(open_game)
        block_prev = _price_block(prev_game)
        block_cur = _price_block(cur_game)
        direction = -1.0 if spec.bet == "over" else 1.0
        open_line = _num(block_open.get("line"))
        prev_line = _num(block_prev.get("line"))
        cur_line = _num(block_cur.get("line"))

        line_parts: list[tuple[float, float | None]] = []
        if open_line is not None and cur_line is not None:
            favorable_open = direction * (cur_line - open_line)
            line_parts.append((0.65, _support01(favorable_open, scale=0.5)))
            if abs(cur_line - open_line) >= 0.01:
                tags.append("MOVE+" if favorable_open > 0 else "MOVE-")
                reasons.append(f"line {open_line:.1f}→{cur_line:.1f}")
        if prev_line is not None and cur_line is not None:
            favorable_prev = direction * (cur_line - prev_line)
            line_parts.append((0.35, _support01(favorable_prev, scale=0.25)))
            if abs(cur_line - prev_line) >= 0.01:
                reasons.append(f"tick {prev_line:.1f}→{cur_line:.1f}")

        side_key = "over" if spec.bet == "over" else "under"
        open_price = block_open.get(side_key)
        prev_price = block_prev.get(side_key)
        cur_price = block_cur.get(side_key)
        support_parts.append((0.60, _combine_support(line_parts)))
        support_parts.append((0.40, _price_support(open_price, prev_price, cur_price)))

    movement_support = _combine_support(support_parts)

    if tags:
        deduped: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            if tag in seen:
                continue
            deduped.append(tag)
            seen.add(tag)
        tags = deduped

    if reasons:
        deduped_reasons: list[str] = []
        seen_reason: set[str] = set()
        for reason in reasons:
            if reason in seen_reason:
                continue
            deduped_reasons.append(reason)
            seen_reason.add(reason)
        reasons = deduped_reasons

    return movement_support, " · ".join(tags), "; ".join(reasons)


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

    try:
        open_games, prev_games, cur_games = _load_team_snapshot_games(str(date), str(DATA_DIR.resolve()))
    except Exception:
        open_games, prev_games, cur_games = {}, {}, {}

    def _row_edge_score_and_reasons(r: pd.Series) -> tuple[float | None, str, str]:
        m = str(r.get("market") or "")
        spec = _EDGE_SPECS.get(m)
        if spec is None:
            return None, "", ""

        prob = _num(r.get(spec.prob_col))
        if prob is None:
            return None, "", ""

        model_conf = _clamp01(abs(prob - 0.5) * 2.0)

        spread = _num(r.get("model_spread"))
        total_bias = _num(r.get("total_bias"))
        form_delta = _num(r.get("form_delta"))
        rest_delta = _num(r.get("rest_delta"))
        b2b_delta = _num(r.get("b2b_delta"))

        reasons: list[str] = []
        drivers: list[str] = []

        movement_support, movement_tags, movement_reasons = _movement_support_for_edge(r, open_games, prev_games, cur_games)

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
                (0.42, model_conf),
                (0.22, spread_support),
                (0.13, form_support),
                (0.08, rest_support),
                (0.15, movement_support),
            ]
            s = 0.0
            w = 0.0
            for ww, vv in parts:
                if vv is None:
                    continue
                s += ww * float(vv)
                w += ww
            edge_score = (s / w) if w > 0 else None

            drivers.append("MODEL HOME" if is_home else "MODEL AWAY")
            for tag in (
                _driver_from_support(spread_support, "SPREAD+", "SPREAD-"),
                _driver_from_support(form_support, "FORM+", "FORM-"),
                _driver_from_support(rest_support, "REST+", "REST-"),
            ):
                if tag:
                    drivers.append(tag)

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
                (0.38, model_conf),
                (0.32, line_support),
                (0.15, pace_support),
                (0.15, movement_support),
            ]
            s = 0.0
            w = 0.0
            for ww, vv in parts:
                if vv is None:
                    continue
                s += ww * float(vv)
                w += ww
            edge_score = (s / w) if w > 0 else None

            drivers.append("MODEL OVR" if is_over else "MODEL UND")
            for tag in (
                _driver_from_support(line_support, "LINE+", "LINE-"),
                _driver_from_support(pace_support, "PACE+", "PACE-"),
            ):
                if tag:
                    drivers.append(tag)

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
                (0.45, model_conf),
                (0.23, cover_support),
                (0.17, form_support),
                (0.15, movement_support),
            ]
            s = 0.0
            w = 0.0
            for ww, vv in parts:
                if vv is None:
                    continue
                s += ww * float(vv)
                w += ww
            edge_score = (s / w) if w > 0 else None

            drivers.append("MODEL HOME" if is_home else "MODEL AWAY")
            for tag in (
                _driver_from_support(cover_support, "COVER+", "COVER-"),
                _driver_from_support(form_support, "FORM+", "FORM-"),
            ):
                if tag:
                    drivers.append(tag)

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

            if spec.market_group == "first10":
                drivers.append("MODEL YES" if spec.bet == "f10_yes" else "MODEL NO")
            elif spec.bet.endswith("_over"):
                drivers.append("MODEL OVR")
            elif spec.bet.endswith("_under"):
                drivers.append("MODEL UND")
            reasons.append(f"p={prob:.3f}")
            if p1_sum is not None:
                reasons.append(f"P1 pace {p1_sum:.2f}")
            pace_tag = _driver_from_support(pace_support, "PACE+", "PACE-")
            if pace_tag:
                drivers.append(pace_tag)

        if movement_tags:
            drivers.extend([tag for tag in movement_tags.split(" · ") if tag])
        if movement_reasons:
            reasons.append(movement_reasons)

        deduped_drivers: list[str] = []
        seen_drivers: set[str] = set()
        for driver in drivers:
            driver_s = str(driver or "").strip()
            if not driver_s or driver_s in seen_drivers:
                continue
            deduped_drivers.append(driver_s)
            seen_drivers.add(driver_s)

        return (
            _clamp01(edge_score) if edge_score is not None else None,
            "; ".join([x for x in reasons if x]),
            " · ".join(deduped_drivers[:7]),
        )

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
    merged["edge_drivers"] = scores[2]

    return merged


def attach_game_recommendation_signals(
    date: str,
    recommendations: pd.DataFrame,
    *,
    edges: Optional[pd.DataFrame] = None,
    predictions: Optional[pd.DataFrame] = None,
    proc_dir: Path = PROC_DIR,
) -> pd.DataFrame:
    if recommendations is None:
        return recommendations

    out = recommendations.copy()
    for col in ("edge_score", "edge_reasons", "edge_drivers"):
        if col in out.columns:
            out = out.drop(columns=[col], errors="ignore")

    if out.empty:
        out["edge_score"] = pd.Series(dtype="float64")
        out["edge_reasons"] = pd.Series(dtype="object")
        out["edge_drivers"] = pd.Series(dtype="object")
        return out

    edge_df = edges.copy() if edges is not None else None
    pred = predictions

    if edge_df is None or edge_df.empty:
        if pred is None:
            p_sim = proc_dir / f"predictions_sim_{date}.csv"
            p_legacy = proc_dir / f"predictions_{date}.csv"
            p = p_sim if p_sim.exists() else p_legacy
            if p.exists():
                try:
                    pred = pd.read_csv(p)
                except Exception:
                    pred = None
        if pred is None or pred.empty:
            out["edge_score"] = pd.NA
            out["edge_reasons"] = ""
            out["edge_drivers"] = ""
            return out
        ev_cols = [c for c in pred.columns if str(c).startswith("ev_")]
        if ev_cols:
            edge_df = pred.melt(id_vars=["date", "home", "away"], value_vars=ev_cols, var_name="market", value_name="ev").dropna()
        else:
            synth = out[[c for c in ["date", "home", "away", "market", "bet", "ev"] if c in out.columns]].copy()
            if synth.empty:
                out["edge_score"] = pd.NA
                out["edge_reasons"] = ""
                out["edge_drivers"] = ""
                return out
            synth["market"] = synth.apply(
                lambda r: _recommendation_signal_market(r.get("market"), r.get("bet")),
                axis=1,
            )
            synth = synth.dropna(subset=["market"])
            edge_df = synth[[c for c in ["date", "home", "away", "market", "ev"] if c in synth.columns]].copy()

    if edge_df is None or edge_df.empty:
        out["edge_score"] = pd.NA
        out["edge_reasons"] = ""
        out["edge_drivers"] = ""
        return out

    if ("edge_score" not in edge_df.columns) or (("edge_reasons" not in edge_df.columns) and ("edge_drivers" not in edge_df.columns)):
        edge_df = attach_game_edge_signals(date, edge_df, predictions=pred, proc_dir=proc_dir)

    if "edge_drivers" not in edge_df.columns:
        edge_df = edge_df.copy()
        edge_df["edge_drivers"] = edge_df.get("edge_reasons", "")
    if "edge_reasons" not in edge_df.columns:
        edge_df = edge_df.copy()
        edge_df["edge_reasons"] = edge_df.get("edge_drivers", "")

    signal_cols = [c for c in ["home", "away", "market", "edge_score", "edge_reasons", "edge_drivers"] if c in edge_df.columns]
    signal_df = edge_df[signal_cols].copy()
    signal_df = signal_df.rename(columns={"market": "_signal_market"})
    signal_df = signal_df.drop_duplicates(subset=["home", "away", "_signal_market"], keep="first")

    out["_signal_market"] = out.apply(
        lambda r: _recommendation_signal_market(r.get("market"), r.get("bet")),
        axis=1,
    )
    out = out.merge(signal_df, on=["home", "away", "_signal_market"], how="left")
    out = out.drop(columns=["_signal_market"], errors="ignore")

    if "edge_score" not in out.columns:
        out["edge_score"] = pd.NA
    if "edge_reasons" not in out.columns:
        out["edge_reasons"] = ""
    if "edge_drivers" not in out.columns:
        out["edge_drivers"] = out.get("edge_reasons", "")
    return out
