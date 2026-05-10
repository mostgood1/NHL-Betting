"""Summarize the realized impact of playoff-specific Live Lens recommendation gates.

Reads the settled Live Lens bet ledger and computes the playoff-only delta for the
currently implemented recommendation gates:
- gate:playoff_total_over_block_5_20
- gate:playoff_total_over_tied_stale_5m
- gate:playoff_p1_over_block_5_15
- gate:playoff_period_total_over_stale_score_state_2_10m
- gate:playoff_period_total_over_tied
- gate:playoff_home_ml_tied_edge>=0.08
- gate:playoff_away_ml_leading_block_35_50
- gate:playoff_under_away_leading_stale_block_35_60

Example:
  python scripts/live_lens_playoff_gate_delta_report.py \
    --ledger data/processed/live_lens/perf/live_lens_bets_all.jsonl \
    --out data/processed/live_lens/perf/live_lens_playoff_gate_delta_report.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _read_ledger(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rows.append(json.loads(s))
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def _season_type_from_game_pk(game_pk: Any) -> str | None:
    try:
        s = str(int(game_pk))
    except Exception:
        s = str(game_pk or "").strip()
    if len(s) >= 6:
        code = s[4:6]
        if code == "02":
            return "regular"
        if code == "03":
            return "playoff"
    return None


def _period_number(rec: pd.Series) -> int | None:
    period = rec.get("sig_period")
    if period is None:
        period = rec.get("period")
    return _safe_int(period)


def _playoff_gate_tag(rec: pd.Series) -> str | None:
    if _season_type_from_game_pk(rec.get("gamePk")) != "playoff":
        return None
    market = str(rec.get("market") or "").strip().upper()
    side = str(rec.get("side") or "").strip().upper()
    score_home = _safe_int(rec.get("score_home"))
    score_away = _safe_int(rec.get("score_away"))
    score_diff = None
    if score_home is not None and score_away is not None:
        score_diff = int(score_home) - int(score_away)
    edge = _safe_float(rec.get("edge"))

    if market == "ML":
        elapsed = _safe_float(rec.get("elapsed_min"))
        if side == "HOME" and score_diff == 0:
            min_required_edge = 0.08
            if edge is not None and float(edge) < float(min_required_edge):
                return f"gate:playoff_home_ml_tied_edge>={float(min_required_edge):.02f}"
            return None
        if side == "AWAY" and score_diff is not None and score_diff < 0:
            if elapsed is not None and 35.0 <= float(elapsed) < 50.0:
                return "gate:playoff_away_ml_leading_block_35_50"
        return None
    if market == "TOTAL" and side == "UNDER":
        elapsed = _safe_float(rec.get("elapsed_min"))
        tags = rec.get("driver_tags") if isinstance(rec.get("driver_tags"), list) else []
        has_recent_goal = any(str(t or "").strip() in {"goal_home", "goal_away"} for t in tags)
        if score_diff is not None and score_diff < 0 and elapsed is not None and 35.0 <= float(elapsed) < 60.0 and not has_recent_goal:
            return "gate:playoff_under_away_leading_stale_block_35_60"

    if side != "OVER":
        return None

    elapsed = _safe_float(rec.get("elapsed_min"))
    if market == "TOTAL":
        meta = rec.get("driver_meta") if isinstance(rec.get("driver_meta"), dict) else {}
        score_state_age_sec = _safe_float((meta or {}).get("score_state_age_sec"))
        if elapsed is not None and 5.0 <= float(elapsed) < 20.0:
            return "gate:playoff_total_over_block_5_20"
        if score_diff == 0 and score_state_age_sec is not None and float(score_state_age_sec) >= 300.0:
            return "gate:playoff_total_over_tied_stale_5m"
        return None

    if market == "PERIOD_TOTAL":
        meta = rec.get("driver_meta") if isinstance(rec.get("driver_meta"), dict) else {}
        score_state_age_sec = _safe_float((meta or {}).get("score_state_age_sec"))
        period = _period_number(rec)
        period_elapsed = None
        if elapsed is not None and period is not None:
            period_elapsed = float(elapsed) - (20.0 * float(max(0, int(period) - 1)))
        if period == 1 and period_elapsed is not None and 5.0 <= float(period_elapsed) < 15.0:
            return "gate:playoff_p1_over_block_5_15"
        if score_state_age_sec is not None and 120.0 <= float(score_state_age_sec) < 600.0:
            return "gate:playoff_period_total_over_stale_score_state_2_10m"
        if score_home is not None and score_away is not None and int(score_home) == int(score_away):
            return "gate:playoff_period_total_over_tied"
    return None


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"bets": 0, "units": 0.0, "roi": None, "wins": 0, "losses": 0, "win_rate": None}
    wins = int((df["result"].astype(str).str.upper() == "WIN").sum()) if "result" in df.columns else 0
    losses = int((df["result"].astype(str).str.upper() == "LOSE").sum()) if "result" in df.columns else 0
    units = float(df["profit_units"].sum())
    bets = int(len(df))
    roi = (units / float(bets)) if bets > 0 else None
    win_rate = (wins / float(wins + losses)) if (wins + losses) > 0 else None
    return {
        "bets": bets,
        "units": units,
        "roi": roi,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
    }


def _format_num(x: Any, *, pct: bool = False, signed: bool = False) -> str:
    v = _safe_float(x)
    if v is None:
        return ""
    if pct:
        return f"{float(v):+.3%}" if signed else f"{float(v):.3%}"
    return f"{float(v):+.3f}" if signed else f"{float(v):.3f}"


def _to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    show = df.copy()
    for col in ["removed_units", "kept_units", "delta_units"]:
        if col in show.columns:
            show[col] = show[col].apply(lambda x: _format_num(x, signed=True))
    for col in ["removed_roi", "kept_roi", "base_roi", "delta_roi", "removed_win_rate", "kept_win_rate"]:
        if col in show.columns:
            show[col] = show[col].apply(lambda x: _format_num(x, pct=True, signed=True))
    cols = list(show.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in show.iterrows():
        vals = []
        for c in cols:
            s = "" if row[c] is None else str(row[c])
            vals.append(s.replace("|", "\\|"))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def build_gate_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    playoff = df[df["season_type"] == "playoff"].copy()
    if playoff.empty:
        return pd.DataFrame()
    base = _summary(playoff)
    rows: list[dict[str, Any]] = []
    gate_order = [
        "gate:playoff_total_over_block_5_20",
        "gate:playoff_total_over_tied_stale_5m",
        "gate:playoff_p1_over_block_5_15",
        "gate:playoff_period_total_over_stale_score_state_2_10m",
        "gate:playoff_period_total_over_tied",
        "gate:playoff_home_ml_tied_edge>=0.08",
        "gate:playoff_away_ml_leading_block_35_50",
        "gate:playoff_under_away_leading_stale_block_35_60",
    ]
    for gate in gate_order:
        removed = playoff[playoff["gate_tag"] == gate].copy()
        kept = playoff[playoff["gate_tag"] != gate].copy()
        removed_s = _summary(removed)
        kept_s = _summary(kept)
        rows.append(
            {
                "gate": gate,
                "removed_bets": int(removed_s["bets"]),
                "removed_units": removed_s["units"],
                "removed_roi": removed_s["roi"],
                "removed_win_rate": removed_s["win_rate"],
                "kept_bets": int(kept_s["bets"]),
                "kept_units": kept_s["units"],
                "kept_roi": kept_s["roi"],
                "kept_win_rate": kept_s["win_rate"],
                "base_roi": base["roi"],
                "delta_units": float((kept_s["units"] or 0.0) - float(base["units"] or 0.0)),
                "delta_roi": (None if kept_s["roi"] is None or base["roi"] is None else float(kept_s["roi"]) - float(base["roi"])),
            }
        )
    removed_any = playoff[playoff["gate_tag"].notna()].copy()
    kept_any = playoff[playoff["gate_tag"].isna()].copy()
    removed_s = _summary(removed_any)
    kept_s = _summary(kept_any)
    rows.append(
        {
            "gate": "all_playoff_gates_combined",
            "removed_bets": int(removed_s["bets"]),
            "removed_units": removed_s["units"],
            "removed_roi": removed_s["roi"],
            "removed_win_rate": removed_s["win_rate"],
            "kept_bets": int(kept_s["bets"]),
            "kept_units": kept_s["units"],
            "kept_roi": kept_s["roi"],
            "kept_win_rate": kept_s["win_rate"],
            "base_roi": base["roi"],
            "delta_units": float((kept_s["units"] or 0.0) - float(base["units"] or 0.0)),
            "delta_roi": (None if kept_s["roi"] is None or base["roi"] is None else float(kept_s["roi"]) - float(base["roi"])),
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True, help="Path to settled ledger CSV/JSONL")
    ap.add_argument("--out", default="", help="Optional markdown output path")
    args = ap.parse_args()

    ledger_path = Path(args.ledger)
    df = _read_ledger(ledger_path)
    if df.empty:
        raise SystemExit("Ledger is empty")
    if "profit_units" not in df.columns:
        raise SystemExit("Ledger missing profit_units")

    out = df.copy()
    out["season_type"] = out["gamePk"].apply(_season_type_from_game_pk) if "gamePk" in out.columns else None
    out["gate_tag"] = out.apply(_playoff_gate_tag, axis=1)
    table = build_gate_delta_table(out)
    if table.empty:
        raise SystemExit("No playoff rows found")

    playoff = out[out["season_type"] == "playoff"].copy()
    base = _summary(playoff)
    lines = [
        "# Live Lens Playoff Gate Delta Report\n",
        f"source: `{ledger_path.as_posix()}`\n",
        f"playoff_rows: `{int(base['bets'])}`\n",
        f"playoff_units: `{_format_num(base['units'], signed=True)}`\n",
        f"playoff_roi: `{_format_num(base['roi'], pct=True, signed=True)}`\n",
        "## Gate Delta Table\n",
        _to_md_table(table),
        "",
    ]
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