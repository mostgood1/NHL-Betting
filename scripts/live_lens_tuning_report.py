"""Generate a tuning report from a settled Live Lens bet ledger.

Input can be:
- JSONL produced by check_live_lens_betting_performance.py (preferred; preserves nested fields)
- CSV produced by check_live_lens_betting_performance.py

Example:
  python scripts/live_lens_tuning_report.py \
    --ledger data/processed/live_lens/perf/live_lens_bets_all.jsonl \
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
    ap.add_argument("--out", default="", help="Optional markdown output path")
    args = ap.parse_args()

    ledger_path = Path(args.ledger)
    df = _read_ledger(ledger_path)
    if df.empty:
        raise SystemExit("Ledger is empty")

    df = _prepare_elapsed_columns(df)

    # Flatten and normalize
    df = _flatten_driver_meta(df)
    df_tags = _explode_driver_tags(df)

    # Core summaries
    lines = []
    lines.append(f"# Live Lens Tuning Report\n")
    lines.append(f"source: `{ledger_path.as_posix()}`\n")

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
