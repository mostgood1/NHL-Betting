"""Generate a tuning report from a settled Live Lens bet ledger.

Input can be:
- JSONL produced by check_live_lens_betting_performance.py (preferred; preserves nested fields)
- CSV produced by check_live_lens_betting_performance.py

Example:
  python scripts/live_lens_tuning_report.py \
    --ledger data/processed/live_lens/perf/live_lens_bets_all.jsonl \
        --season-type playoff \
    --min-bets 10 \
    --out data/processed/live_lens/perf/live_lens_tuning_report.md
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


_LIVE_LENS_MAX_ELAPSED_SECONDS = 65 * 60


def _format_elapsed_mmss(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def _elapsed_seconds(elapsed_min: Any) -> Optional[float]:
    em = _safe_float(elapsed_min)
    if em is None:
        return None
    sec = max(0.0, float(em) * 60.0)
    if sec >= float(_LIVE_LENS_MAX_ELAPSED_SECONDS):
        sec = float(_LIVE_LENS_MAX_ELAPSED_SECONDS) - 1e-9
    return sec


def _elapsed_bucket(elapsed_min: Any, *, bin_seconds: int = 60) -> str:
    sec = _elapsed_seconds(elapsed_min)
    if sec is None:
        return "(missing)"
    bin_seconds = max(1, int(bin_seconds))
    start = int(math.floor(sec / float(bin_seconds))) * bin_seconds
    start = max(0, min(start, _LIVE_LENS_MAX_ELAPSED_SECONDS - bin_seconds))
    end = min(_LIVE_LENS_MAX_ELAPSED_SECONDS, start + bin_seconds)
    return f"{_format_elapsed_mmss(start)}-{_format_elapsed_mmss(end)}"


def _elapsed_bucket_start_seconds(bucket: Any) -> Optional[int]:
    try:
        s = str(bucket or "").strip()
    except Exception:
        return None
    if not s or s in {"(none)", "(missing)"}:
        return None
    if ":" in s and "-" in s:
        try:
            start, _ = s.split("-", 1)
            mm, ss = start.split(":", 1)
            return max(0, min(_LIVE_LENS_MAX_ELAPSED_SECONDS - 1, (int(mm) * 60) + int(ss)))
        except Exception:
            return None
    if s.startswith(">="):
        try:
            return max(0, int(float(s[2:].strip()) * 60.0))
        except Exception:
            return None
    if "-" in s:
        try:
            start, _ = s.split("-", 1)
            return max(0, int(float(start.strip()) * 60.0))
        except Exception:
            return None
    return None


def _prepare_elapsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    try:
        legacy_bucket = out["elapsed_bucket"] if "elapsed_bucket" in out.columns else None
        legacy_bucket_1m = out["elapsed_bucket_1m"] if "elapsed_bucket_1m" in out.columns else None
        if "elapsed_min" in out.columns:
            out["elapsed_min"] = pd.to_numeric(out["elapsed_min"], errors="coerce")
            out["elapsed_bucket"] = out["elapsed_min"].apply(_elapsed_bucket)
            if legacy_bucket_1m is not None:
                out["elapsed_bucket"] = out["elapsed_bucket"].where(out["elapsed_min"].notna(), legacy_bucket_1m)
            elif legacy_bucket is not None:
                out["elapsed_bucket"] = out["elapsed_bucket"].where(out["elapsed_min"].notna(), legacy_bucket)
            out["elapsed_bucket_1m"] = out["elapsed_bucket"]
            out["elapsed_bucket_15s"] = out["elapsed_min"].apply(lambda x: _elapsed_bucket(x, bin_seconds=15))
        elif "elapsed_bucket_1m" in out.columns:
            out["elapsed_bucket"] = out["elapsed_bucket_1m"]
    except Exception:
        return out
    return out


def _read_ledger(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def _parse_maybe_dict(x: Any) -> Optional[Dict[str, Any]]:
    if isinstance(x, dict):
        return x
    if x is None:
        return None
    s = str(x).strip()
    if not s or s == "nan":
        return None
    # Try JSON first, then literal_eval
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if (not path.exists()) or path.stat().st_size <= 0:
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _guess_priors_path(ledger_path: Path) -> Path:
    try:
        if str(ledger_path.parent.name).strip().lower() == "perf":
            return ledger_path.parent.parent / "live_lens_driver_tag_priors.json"
    except Exception:
        pass
    return ledger_path.parent / "live_lens_driver_tag_priors.json"


def _season_type_from_game_pk(game_pk: Any) -> Optional[str]:
    try:
        s = str(int(game_pk))
    except Exception:
        s = str(game_pk or "").strip()
    if len(s) >= 6:
        type_code = s[4:6]
        if type_code == "02":
            return "regular"
        if type_code == "03":
            return "playoff"
    return None


def _filter_ledger(
    df: pd.DataFrame,
    *,
    season_type: str = "all",
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if start_date and "date" in out.columns:
        out = out[out["date"].astype(str).str[:10] >= str(start_date)].copy()
    if end_date and "date" in out.columns:
        out = out[out["date"].astype(str).str[:10] <= str(end_date)].copy()
    season_key = str(season_type or "all").strip().lower()
    if season_key not in {"all", "regular", "playoff"}:
        raise ValueError(f"unsupported season_type: {season_type}")
    if season_key == "all":
        return out
    if "gamePk" not in out.columns:
        return out.iloc[0:0].copy()
    game_pk_season = out["gamePk"].apply(_season_type_from_game_pk)
    return out[game_pk_season == season_key].copy()


def _driver_tag_priors_table(obj: Dict[str, Any]) -> pd.DataFrame:
    markets = obj.get("markets") if isinstance(obj, dict) else None
    if not isinstance(markets, dict):
        return pd.DataFrame()
    rows = []
    for scope, grp in markets.items():
        if not isinstance(grp, dict):
            continue
        for tag, spec in grp.items():
            if not isinstance(spec, dict):
                continue
            rows.append(
                {
                    "scope": str(scope),
                    "driver_tag": str(tag),
                    "edge_delta": _safe_float(spec.get("edge_delta")),
                    "reliability": _safe_float(spec.get("reliability")),
                    "bets": int(spec.get("bets") or 0),
                    "roi": _safe_float(spec.get("roi")),
                    "baseline_roi": _safe_float(spec.get("baseline_roi")),
                    "roi_gap": _safe_float(spec.get("roi_gap")),
                    "win_rate": _safe_float(spec.get("win_rate")),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["abs_edge_delta"] = out["edge_delta"].abs()
    out = out.sort_values(["abs_edge_delta", "bets", "scope", "driver_tag"], ascending=[False, False, True, True]).reset_index(drop=True)
    return out


def _driver_tag_prior_sections(priors_path: Path) -> list[str]:
    obj = _read_json(priors_path)
    if not isinstance(obj, dict):
        return []

    priors = _driver_tag_priors_table(obj)
    if priors.empty:
        return []

    defaults = obj.get("defaults") if isinstance(obj, dict) else {}
    try:
        cap = _safe_float((defaults or {}).get("max_total_edge_adjustment"))
    except Exception:
        cap = None

    lines: list[str] = []
    lines.append("## Learned driver-tag priors\n")
    lines.append(f"source: `{priors_path.as_posix()}`\n")
    if cap is not None:
        lines.append(f"max edge-adjustment cap: {float(cap):+.3f}\n")

    loosen = priors[priors["edge_delta"].notna() & (priors["edge_delta"] < 0)].copy()
    if not loosen.empty:
        loosen = loosen.sort_values(["edge_delta", "bets"], ascending=[True, False]).reset_index(drop=True)
        lines.append("### Tags that loosen gates\n")
        lines.append(_to_md_table(loosen.drop(columns=["abs_edge_delta"], errors="ignore"), max_rows=20))
        lines.append("")

    tighten = priors[priors["edge_delta"].notna() & (priors["edge_delta"] > 0)].copy()
    if not tighten.empty:
        tighten = tighten.sort_values(["edge_delta", "bets"], ascending=[False, False]).reset_index(drop=True)
        lines.append("### Tags that tighten gates\n")
        lines.append(_to_md_table(tighten.drop(columns=["abs_edge_delta"], errors="ignore"), max_rows=20))
        lines.append("")

    return lines


def _roi_table(df: pd.DataFrame, group_col: str, min_bets: int) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    d = df[df["profit_units"].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    def _wins(x: pd.Series) -> int:
        return int((x == "WIN").sum())

    def _losses(x: pd.Series) -> int:
        return int((x == "LOSE").sum())

    g = (
        d.groupby(group_col, dropna=False)
        .agg(
            bets=("profit_units", "size"),
            units=("profit_units", "sum"),
            avg_edge=("edge", "mean"),
            wins=("result", _wins),
            losses=("result", _losses),
        )
        .reset_index()
    )
    g["roi"] = g["units"] / g["bets"].replace(0, float("nan"))
    g["win_rate"] = g.apply(lambda r: (r["wins"] / (r["wins"] + r["losses"])) if (r["wins"] + r["losses"]) > 0 else float("nan"), axis=1)
    g = g[g["bets"] >= int(min_bets)].copy()
    if group_col in {"elapsed_bucket", "elapsed_bucket_1m", "elapsed_bucket_15s"}:
        g["_elapsed_sort"] = g[group_col].apply(_elapsed_bucket_start_seconds)
        g["_elapsed_sort"] = g["_elapsed_sort"].fillna(10**9)
        g = g.sort_values(["_elapsed_sort", group_col], ascending=[True, True]).drop(columns=["_elapsed_sort"]).reset_index(drop=True)
    else:
        g = g.sort_values(["bets", "roi"], ascending=[False, False]).reset_index(drop=True)
    return g


def _roi_table_multi(df: pd.DataFrame, group_cols: list[str], min_bets: int) -> pd.DataFrame:
    if df.empty or any(col not in df.columns for col in group_cols):
        return pd.DataFrame()
    d = df[df["profit_units"].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    def _wins(x: pd.Series) -> int:
        return int((x == "WIN").sum())

    def _losses(x: pd.Series) -> int:
        return int((x == "LOSE").sum())

    g = (
        d.groupby(group_cols, dropna=False)
        .agg(
            bets=("profit_units", "size"),
            units=("profit_units", "sum"),
            avg_edge=("edge", "mean"),
            wins=("result", _wins),
            losses=("result", _losses),
        )
        .reset_index()
    )
    g["roi"] = g["units"] / g["bets"].replace(0, float("nan"))
    g["win_rate"] = g.apply(lambda r: (r["wins"] / (r["wins"] + r["losses"])) if (r["wins"] + r["losses"]) > 0 else float("nan"), axis=1)
    g = g[g["bets"] >= int(min_bets)].copy()
    return g.sort_values(["roi", "bets"], ascending=[True, False]).reset_index(drop=True)


def _playoff_gate_hits(rec: pd.Series) -> list[str]:
    hits: list[str] = []
    market = str(rec.get("market") or "").strip().upper()
    side = str(rec.get("side") or "").strip().upper()
    elapsed_min = _safe_float(rec.get("elapsed_min"))
    score_state_age_sec = _safe_float(rec.get("meta_score_state_age_sec"))
    try:
        score_home = int(rec.get("score_home")) if rec.get("score_home") is not None else None
        score_away = int(rec.get("score_away")) if rec.get("score_away") is not None else None
        score_diff = (int(score_home) - int(score_away)) if score_home is not None and score_away is not None else None
    except Exception:
        score_diff = None

    if market == "TOTAL" and side == "OVER":
        if elapsed_min is not None and 5.0 <= float(elapsed_min) < 20.0:
            hits.append("gate:playoff_total_over_block_5_20")
        if score_diff == 0 and score_state_age_sec is not None and float(score_state_age_sec) >= 300.0:
            hits.append("gate:playoff_total_over_tied_stale_5m")
        return hits

    if market == "PERIOD_TOTAL" and side == "OVER":
        try:
            period = int(rec.get("sig_period") if pd.notna(rec.get("sig_period")) else rec.get("period"))
        except Exception:
            period = None
        period_elapsed = None
        if elapsed_min is not None and period is not None:
            period_elapsed = float(elapsed_min) - (20.0 * max(0, int(period) - 1))
        if period == 1 and period_elapsed is not None and 5.0 <= float(period_elapsed) < 15.0:
            hits.append("gate:playoff_p1_over_block_5_15")
        if score_state_age_sec is not None and 120.0 <= float(score_state_age_sec) < 600.0:
            hits.append("gate:playoff_period_total_over_stale_score_state_2_10m")
        if score_diff == 0:
            hits.append("gate:playoff_period_total_over_tied")
        return hits

    if market == "ML":
        edge = _safe_float(rec.get("edge"))
        if side == "HOME" and score_diff == 0 and edge is not None and float(edge) < 0.08:
            hits.append("gate:playoff_home_ml_tied_edge>=0.08")
        if side == "AWAY" and score_diff is not None and score_diff < 0 and elapsed_min is not None and 35.0 <= float(elapsed_min) < 50.0:
            hits.append("gate:playoff_away_ml_leading_block_35_50")
        return hits

    if market == "TOTAL" and side == "UNDER":
        tags = _normalize_driver_tags(rec.get("driver_tags"))
        has_recent_goal = any(tag in {"goal_home", "goal_away"} for tag in tags)
        if score_diff is not None and score_diff < 0 and elapsed_min is not None and 35.0 <= float(elapsed_min) < 60.0 and not has_recent_goal:
            hits.append("gate:playoff_under_away_leading_stale_block_35_60")
    return hits


def _allowed_playoff_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["playoff_gate_hits"] = out.apply(_playoff_gate_hits, axis=1)
    out["is_allowed_playoff_row"] = out["playoff_gate_hits"].apply(lambda x: len(x) == 0)
    return out[out["is_allowed_playoff_row"]].copy()


def _top_n_edge_table(df: pd.DataFrame, *, top_ns: list[int], group_col: str = "market") -> pd.DataFrame:
    if df.empty or group_col not in df.columns or "edge" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group_value, group_df in df.groupby(group_col, dropna=False):
        sub = group_df[group_df["edge"].notna() & group_df["profit_units"].notna()].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["edge", "profit_units"], ascending=[False, False]).reset_index(drop=True)
        for top_n in top_ns:
            head = sub.head(min(int(top_n), len(sub))).copy()
            if head.empty:
                continue
            rows.append(
                {
                    group_col: group_value,
                    "top_n": int(top_n),
                    "bets": int(len(head)),
                    "units": float(head["profit_units"].sum()),
                    "roi": float(head["profit_units"].mean()),
                    "avg_edge": float(head["edge"].mean()),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values([group_col, "top_n"]).reset_index(drop=True)


def _explode_driver_tags(df: pd.DataFrame) -> pd.DataFrame:
    if "driver_tags" not in df.columns:
        return df.assign(driver_tag="(none)")
    out = df.copy()
    # Normalize to list
    def _norm_tags(x: Any):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return []
        if isinstance(x, list):
            return [str(t) for t in x if str(t).strip()]
        if isinstance(x, str):
            # might be a stringified list in CSV
            s = x.strip()
            if s.startswith("["):
                try:
                    v = ast.literal_eval(s)
                    if isinstance(v, list):
                        return [str(t) for t in v if str(t).strip()]
                except Exception:
                    pass
            return [s] if s else []
        return []

    out["driver_tag"] = out["driver_tags"].apply(_norm_tags)
    out = out.explode("driver_tag")
    out["driver_tag"] = out["driver_tag"].fillna("(none)")
    out.loc[out["driver_tag"].astype(str).str.strip() == "", "driver_tag"] = "(none)"
    return out


def _flatten_driver_meta(df: pd.DataFrame) -> pd.DataFrame:
    if "driver_meta" not in df.columns:
        return df
    meta = df["driver_meta"].apply(_parse_maybe_dict)
    keys = set()
    for m in meta.dropna().tolist():
        keys.update(list(m.keys()))
    keys = {k for k in keys if isinstance(k, str) and k}
    out = df.copy()
    for k in sorted(keys):
        out[f"meta_{k}"] = meta.apply(lambda d: (d.get(k) if isinstance(d, dict) else None))
    return out


def _normalize_driver_tags(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(t) for t in x if str(t).strip()]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("["):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(t) for t in v if str(t).strip()]
            except Exception:
                pass
        return [s] if s else []
    return []


def _merge_driver_tags(primary: Any, fallback: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in _normalize_driver_tags(primary) + _normalize_driver_tags(fallback):
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _prepare_flow_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def _pick(tags: list[str], mapping: list[tuple[str, str]], default: str) -> str:
        seen = set(tags)
        for tag, label in mapping:
            if tag in seen:
                return label
        return default

    def _age_bucket(seconds: Any) -> str:
        sec = _safe_float(seconds)
        if sec is None:
            return "(missing)"
        if sec <= 120:
            return "<=2m"
        if sec <= 300:
            return "2-5m"
        if sec <= 600:
            return "5-10m"
        return ">10m"

    def _pp_age_bucket(seconds: Any, pp_team: Any) -> str:
        team = str(pp_team or "").strip().lower()
        if team not in {"home", "away"}:
            return "none"
        sec = _safe_float(seconds)
        if sec is None:
            return "(missing)"
        if sec <= 30:
            return "<=30s"
        if sec <= 60:
            return "30-60s"
        if sec <= 90:
            return "60-90s"
        return "90-120s"

    out = df.copy()
    if "meta_trigger_tags" in out.columns:
        out["_driver_tags_norm"] = out.apply(lambda r: _merge_driver_tags(r.get("driver_tags"), r.get("meta_trigger_tags")), axis=1)
    elif "driver_tags" in out.columns:
        out["_driver_tags_norm"] = out["driver_tags"].apply(_normalize_driver_tags)
    else:
        out["_driver_tags_norm"] = [[] for _ in range(len(out))]

    out["flow_score_state"] = out["_driver_tags_norm"].apply(
        lambda tags: _pick(tags, [("score:tied", "tied"), ("score:home_leading", "home_leading"), ("score:away_leading", "away_leading")], "unknown")
    )
    out["flow_pressure_state"] = out["_driver_tags_norm"].apply(
        lambda tags: _pick(tags, [("pressure:home", "home"), ("pressure:away", "away"), ("pressure:even", "even")], "unknown")
    )
    out["flow_recent_goal_state"] = out["_driver_tags_norm"].apply(
        lambda tags: "recent_goal" if ("goal_home" in set(tags) or "goal_away" in set(tags)) else "stale"
    )
    out["flow_recent_goal_team"] = out["_driver_tags_norm"].apply(
        lambda tags: "both" if ({"goal_home", "goal_away"} <= set(tags)) else ("home" if "goal_home" in set(tags) else ("away" if "goal_away" in set(tags) else "none"))
    )
    out["flow_manpower_state"] = out["_driver_tags_norm"].apply(
        lambda tags: _pick(tags, [("manpower:5v3", "5v3"), ("manpower:pp_home", "pp_home"), ("manpower:pp_away", "pp_away")], "even")
    )
    out["flow_empty_net_state"] = out["_driver_tags_norm"].apply(
        lambda tags: "both" if ({"empty_net:home", "empty_net:away"} <= set(tags)) else ("home" if "empty_net:home" in set(tags) else ("away" if "empty_net:away" in set(tags) else "none"))
    )
    out["flow_late_state"] = out["_driver_tags_norm"].apply(
        lambda tags: _pick(tags, [("late:empty_net", "one_goal_empty_net"), ("late:one_goal", "one_goal"), ("late:multi_goal", "multi_goal"), ("late:tied", "tied_late")], "normal")
    )
    out["flow_goalie_state"] = out["_driver_tags_norm"].apply(
        lambda tags: _pick(tags, [("goalie:strong", "strong"), ("goalie:weak", "weak")], "neutral")
    )
    out["flow_pace_state"] = out["_driver_tags_norm"].apply(
        lambda tags: _pick(tags, [("pace:down", "down"), ("pace:up", "up")], "neutral")
    )
    out["flow_goal_age_bucket"] = out["meta_time_since_last_goal_sec"].apply(_age_bucket) if "meta_time_since_last_goal_sec" in out.columns else "(missing)"
    out["flow_score_state_age_bucket"] = out["meta_score_state_age_sec"].apply(_age_bucket) if "meta_score_state_age_sec" in out.columns else "(missing)"
    if "meta_market_blend_weight" in out.columns:
        def _blend_bucket(x: Any) -> str:
            val = _safe_float(x)
            if val is None:
                return "(missing)"
            if val <= 0.20:
                return "<=0.20"
            if val <= 0.30:
                return "0.20-0.30"
            return ">0.30"
        out["flow_market_blend_bucket"] = out["meta_market_blend_weight"].apply(_blend_bucket)
    else:
        out["flow_market_blend_bucket"] = "(missing)"
    if "meta_pp_state_age_sec" in out.columns or "meta_pp_team" in out.columns:
        out["flow_pp_state_age_bucket"] = out.apply(
            lambda r: _pp_age_bucket(r.get("meta_pp_state_age_sec"), r.get("meta_pp_team")),
            axis=1,
        )
    else:
        out["flow_pp_state_age_bucket"] = "none"
    out["flow_shape_compact"] = out.apply(
        lambda r: " | ".join(
            [
                str(r.get("market") or "?"),
                str(r.get("flow_score_state") or "unknown"),
                str(r.get("flow_recent_goal_state") or "stale"),
                str(r.get("flow_late_state") or "normal"),
                str(r.get("flow_manpower_state") or "even"),
            ]
        ),
        axis=1,
    )
    return out


def _flow_section(df: pd.DataFrame, *, min_bets: int) -> list[str]:
    if df.empty:
        return []
    lines: list[str] = []
    lines.append("## Flow-First Playoff Evaluation\n")
    lines.append("Game-shape summaries below use live state and trigger context as the primary lens; historical priors remain an appendix only.\n")

    sections = [
        ("flow_score_state", "### ROI by score state", 10),
        ("flow_pressure_state", "### ROI by pressure state", 10),
        ("flow_recent_goal_state", "### ROI by recent goal freshness", 10),
        ("flow_goal_age_bucket", "### ROI by time since last goal", 10),
        ("flow_score_state_age_bucket", "### ROI by score-state age", 10),
        ("flow_pp_state_age_bucket", "### ROI by power-play state age", 10),
        ("flow_manpower_state", "### ROI by manpower state", 10),
        ("flow_empty_net_state", "### ROI by empty-net state", 10),
        ("flow_late_state", "### ROI by late-state mode", 10),
        ("flow_shape_compact", "### ROI by compact flow shape", 20),
    ]
    for col, header, max_rows in sections:
        if col not in df.columns:
            continue
        threshold = int(min_bets)
        if col == "flow_shape_compact":
            threshold = max(4, min(int(min_bets), 8))
        t = _roi_table(df, col, min_bets=threshold)
        lines.append(f"{header}\n")
        lines.append(_to_md_table(t, max_rows=max_rows))
        lines.append("")

    if {"market", "side", "flow_pressure_state"} <= set(df.columns):
        period_total_pressure = _roi_table_multi(
            df[df["market"] == "PERIOD_TOTAL"].copy(),
            ["side", "flow_pressure_state"],
            min_bets=max(2, min(int(min_bets), 8)),
        )
        if not period_total_pressure.empty:
            lines.append("### ROI by PERIOD_TOTAL side and pressure\n")
            lines.append(_to_md_table(period_total_pressure, max_rows=20))
            lines.append("")

        total_pressure = _roi_table_multi(
            df[df["market"] == "TOTAL"].copy(),
            ["side", "flow_pressure_state"],
            min_bets=max(2, min(int(min_bets), 8)),
        )
        if not total_pressure.empty:
            lines.append("### ROI by TOTAL side and pressure\n")
            lines.append(_to_md_table(total_pressure, max_rows=20))
            lines.append("")

    if {"market", "side", "flow_late_state"} <= set(df.columns):
        period_total_late = _roi_table_multi(
            df[df["market"] == "PERIOD_TOTAL"].copy(),
            ["side", "flow_late_state"],
            min_bets=max(2, min(int(min_bets), 8)),
        )
        if not period_total_late.empty:
            lines.append("### ROI by PERIOD_TOTAL side and late state\n")
            lines.append(_to_md_table(period_total_late, max_rows=20))
            lines.append("")

        total_late = _roi_table_multi(
            df[df["market"] == "TOTAL"].copy(),
            ["side", "flow_late_state"],
            min_bets=max(2, min(int(min_bets), 8)),
        )
        if not total_late.empty:
            lines.append("### ROI by TOTAL side and late state\n")
            lines.append(_to_md_table(total_late, max_rows=20))
            lines.append("")

    if {"market", "side", "flow_market_blend_bucket"} <= set(df.columns):
        total_blend = _roi_table_multi(
            df[df["market"] == "TOTAL"].copy(),
            ["side", "flow_market_blend_bucket"],
            min_bets=max(2, min(int(min_bets), 8)),
        )
        total_blend = total_blend[total_blend["flow_market_blend_bucket"] != "(missing)"].copy()
        if not total_blend.empty:
            lines.append("### ROI by TOTAL side and market-blend bucket\n")
            lines.append(_to_md_table(total_blend, max_rows=20))
            lines.append("")

        shape = _roi_table(df, "flow_shape_compact", min_bets=max(4, min(int(min_bets), 8)))

    if "flow_shape_compact" in df.columns:
        shape = _roi_table(df, "flow_shape_compact", min_bets=max(4, min(int(min_bets), 8)))
        if not shape.empty:
            worst = shape.sort_values(["roi", "bets"], ascending=[True, False]).reset_index(drop=True)
            best = shape.sort_values(["roi", "bets"], ascending=[False, False]).reset_index(drop=True)
            lines.append("### Worst compact flow shapes\n")
            lines.append(_to_md_table(worst, max_rows=10))
            lines.append("")
            lines.append("### Best compact flow shapes\n")
            lines.append(_to_md_table(best, max_rows=10))
            lines.append("")

    allowed = _allowed_playoff_rows(df)
    if not allowed.empty:
        lines.append("## Allowed-Row Ranking Quality\n")
        lines.append("These tables remove rows already blocked by the shipped playoff gates so ranking quality is measured only on bets the current logic would still allow.\n")

        allowed_market = _roi_table(allowed, "market", min_bets=max(2, min(int(min_bets), 8)))
        if not allowed_market.empty:
            lines.append("### Allowed playoff rows by market\n")
            lines.append(_to_md_table(allowed_market, max_rows=20))
            lines.append("")

        if {"market", "edge_bucket"} <= set(allowed.columns):
            allowed_edge = _roi_table_multi(
                allowed[allowed["market"].isin(["ML", "TOTAL", "PERIOD_TOTAL"])].copy(),
                ["market", "edge_bucket"],
                min_bets=max(2, min(int(min_bets), 8)),
            )
            if not allowed_edge.empty:
                lines.append("### Allowed playoff rows by market and edge bucket\n")
                lines.append(_to_md_table(allowed_edge, max_rows=30))
                lines.append("")

        top_n = _top_n_edge_table(
            allowed[allowed["market"].isin(["ML", "TOTAL", "PERIOD_TOTAL"])].copy(),
            top_ns=[5, 10, 20, 30, 50],
        )
        if not top_n.empty:
            lines.append("### Allowed playoff rows top-N by edge\n")
            lines.append(_to_md_table(top_n, max_rows=30))
            lines.append("")

    return lines


def _to_md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "(no rows)"
    show = df.head(max_rows).copy()
    # Compact formatting
    for c in ["roi", "units", "avg_edge", "win_rate", "reliability", "baseline_roi", "roi_gap", "edge_delta"]:
        if c in show.columns:
            show[c] = show[c].apply(lambda x: (f"{float(x):+.3f}" if c in {"units", "avg_edge", "roi_gap", "edge_delta"} else f"{float(x):+.3%}") if _safe_float(x) is not None else "")

    cols = list(show.columns)
    # Build markdown manually to avoid optional tabulate dependency.
    def _cell(v: Any) -> str:
        if v is None:
            return ""
        s = str(v)
        s = s.replace("\n", " ").replace("\r", " ")
        s = s.replace("|", "\\|")
        return s

    header = "| " + " | ".join(_cell(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(_cell(show.iloc[i][c]) for c in cols) + " |"
        for i in range(len(show))
    ]
    return "\n".join([header, sep] + rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True, help="Path to settled ledger CSV/JSONL")
    ap.add_argument("--min-bets", type=int, default=10)
    ap.add_argument("--priors-json", default="", help="Optional learned driver-tag priors JSON; defaults near the ledger")
    ap.add_argument("--season-type", default="all", choices=["all", "regular", "playoff"], help="Optional season-type filter derived from gamePk")
    ap.add_argument("--start-date", default="", help="Optional inclusive start date YYYY-MM-DD")
    ap.add_argument("--end-date", default="", help="Optional inclusive end date YYYY-MM-DD")
    ap.add_argument("--out", default="", help="Optional markdown output path")
    args = ap.parse_args()

    ledger_path = Path(args.ledger)
    df = _read_ledger(ledger_path)
    if df.empty:
        raise SystemExit("Ledger is empty")

    df = _filter_ledger(
        df,
        season_type=str(args.season_type or "all"),
        start_date=str(args.start_date or "").strip(),
        end_date=str(args.end_date or "").strip(),
    )
    if df.empty:
        raise SystemExit("Ledger is empty after filters")

    df = _prepare_elapsed_columns(df)

    # Flatten and normalize
    df = _flatten_driver_meta(df)
    df = _prepare_flow_columns(df)
    df_tags = _explode_driver_tags(df)

    # Core summaries
    lines = []
    lines.append(f"# Live Lens Tuning Report\n")
    lines.append(f"source: `{ledger_path.as_posix()}`\n")
    filters = []
    if str(args.season_type or "all").strip().lower() != "all":
        filters.append(f"season_type={str(args.season_type).lower()}")
    if str(args.start_date or "").strip():
        filters.append(f"start_date={str(args.start_date).strip()}")
    if str(args.end_date or "").strip():
        filters.append(f"end_date={str(args.end_date).strip()}")
    if filters:
        lines.append(f"filters: `{', '.join(filters)}`\n")
    lines.append(f"rows: `{int(len(df))}`\n")

    # Overall / by market
    by_market = _roi_table(df, "market", min_bets=max(1, int(args.min_bets)))
    lines.append("## ROI by market\n")
    lines.append(_to_md_table(by_market, max_rows=25))
    lines.append("")

    # Driver tags
    by_tag = _roi_table(df_tags, "driver_tag", min_bets=int(args.min_bets))
    lines.append("## ROI by driver_tag\n")
    lines.append(_to_md_table(by_tag, max_rows=30))
    lines.append("")

    if str(args.season_type or "all").strip().lower() == "playoff":
        lines.extend(_flow_section(df, min_bets=int(args.min_bets)))

    # Common buckets from the ledger script
    for col, header, max_rows in (
        ("edge_bucket", "## ROI by edge bucket", 20),
        ("elapsed_bucket", "## ROI by elapsed minute bucket", 70),
    ):
        if col in df.columns:
            t = _roi_table(df, col, min_bets=int(args.min_bets))
            lines.append(f"{header}\n")
            lines.append(_to_md_table(t, max_rows=max_rows))
            lines.append("")

    # Odds staleness if available
    if "meta_odds_age_sec" in df.columns or "odds_staleness_bucket" in df.columns:
        if "meta_odds_age_sec" in df.columns:
            d2 = df.copy()
            d2["odds_age_min_bucket"] = d2["meta_odds_age_sec"].apply(lambda x: "(missing)" if _safe_float(x) is None else ("<=1m" if float(x) <= 60 else ("1-3m" if float(x) <= 180 else ("3-6m" if float(x) <= 360 else ">6m"))))
            by_stale = _roi_table(d2, "odds_age_min_bucket", min_bets=int(args.min_bets))
            lines.append("## ROI by odds age\n")
            lines.append(_to_md_table(by_stale, max_rows=10))
            lines.append("")
        elif "odds_staleness_bucket" in df.columns:
            by_stale = _roi_table(df, "odds_staleness_bucket", min_bets=int(args.min_bets))
            lines.append("## ROI by odds staleness bucket\n")
            lines.append(_to_md_table(by_stale, max_rows=10))
            lines.append("")

    # Goal diff buckets (game markets)
    if "meta_gd" in df.columns:
        d3 = df.copy()
        d3["gd_bucket"] = d3["meta_gd"].apply(lambda x: "(missing)" if _safe_float(x) is None else ("home+" if float(x) >= 1 else ("away+" if float(x) <= -1 else "tie")))
        by_gd = _roi_table(d3, "gd_bucket", min_bets=int(args.min_bets))
        lines.append("## ROI by goal-diff bucket\n")
        lines.append(_to_md_table(by_gd, max_rows=10))
        lines.append("")

    # ML probability source (logit vs poisson)
    if "meta_prob_source" in df.columns:
        by_src = _roi_table(df, "meta_prob_source", min_bets=int(args.min_bets))
        lines.append("## ROI by ML prob_source\n")
        lines.append(_to_md_table(by_src, max_rows=10))
        lines.append("")

    # Pace multiplier buckets (totals/props)
    if "meta_pace_mult" in df.columns:
        d4 = df.copy()
        def _pace_bucket(x: Any) -> str:
            v = _safe_float(x)
            if v is None:
                return "(missing)"
            if v <= 0.92:
                return "<=0.92"
            if v <= 1.00:
                return "0.92-1.00"
            if v <= 1.08:
                return "1.00-1.08"
            return ">1.08"
        d4["pace_bucket"] = d4["meta_pace_mult"].apply(_pace_bucket)
        by_pace = _roi_table(d4, "pace_bucket", min_bets=int(args.min_bets))
        lines.append("## ROI by pace_mult bucket\n")
        lines.append(_to_md_table(by_pace, max_rows=10))
        lines.append("")

    # Totals remaining-goals mean buckets
    if "meta_mu_remaining" in df.columns:
        d5 = df.copy()
        def _mu_bucket(x: Any) -> str:
            v = _safe_float(x)
            if v is None:
                return "(missing)"
            if v <= 0.8:
                return "<=0.8"
            if v <= 1.4:
                return "0.8-1.4"
            if v <= 2.2:
                return "1.4-2.2"
            return ">2.2"
        d5["mu_remaining_bucket"] = d5["meta_mu_remaining"].apply(_mu_bucket)
        by_mu = _roi_table(d5, "mu_remaining_bucket", min_bets=int(args.min_bets))
        lines.append("## ROI by mu_remaining bucket\n")
        lines.append(_to_md_table(by_mu, max_rows=10))
        lines.append("")

    priors_path = Path(args.priors_json) if str(args.priors_json or "").strip() else _guess_priors_path(ledger_path)
    lines.extend(_driver_tag_prior_sections(priors_path))

    md = "\n".join(lines).strip() + "\n"
    print(md)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"wrote={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
