"""Fit learned driver-tag edge priors for Live Lens game and period markets.

Reads the settled Live Lens ledger and estimates small, conservative edge-threshold
adjustments for historically strong or weak learnable driver tags.

Output artifact schema:
    {
      "defaults": {...},
      "markets": {
        "TOTAL": {
          "pace:up": {
            "edge_delta": -0.004,
            "reliability": 0.61,
            "bets": 42,
            ...
          },
          ...
        },
        "ML": {...},
        "__all__": {...}
      }
    }

Negative `edge_delta` loosens the required edge. Positive `edge_delta` tightens it.
This is safe to run in the daily update pipeline.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# Ensure repo root is on sys.path so `import nhl_betting` works
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nhl_betting.utils.io import PROC_DIR


SUPPORTED_MARKETS = {
    "TOTAL",
    "ML",
    "PUCKLINE",
    "REG_3WAY",
    "PERIOD_TOTAL",
    "PERIOD_ML",
    "PERIOD_SPREAD",
    "PERIOD_3WAY",
}
TOTAL_MARKETS = {"TOTAL", "PERIOD_TOTAL"}
SIDE_MARKETS = {"ML", "PUCKLINE", "REG_3WAY", "PERIOD_ML", "PERIOD_SPREAD", "PERIOD_3WAY"}
TOTAL_SIDES = {"OVER", "UNDER"}
TEAM_SIDES = {"HOME", "AWAY", "DRAW"}


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if (not path.exists()) or path.stat().st_size <= 0:
        return []
    suf = path.suffix.lower()
    if suf in {".jsonl", ".ndjson"}:
        return list(_iter_jsonl(path))
    rows: list[dict[str, Any]] = []
    try:
        import csv

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if isinstance(row, dict):
                    rows.append(dict(row))
    except Exception:
        return []
    return rows


def _parse_tag_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t) for t in x if str(t or "").strip()]
    s = str(x).strip()
    if not s or s == "nan":
        return []
    if s.startswith("["):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(t) for t in obj if str(t or "").strip()]
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [str(t) for t in obj if str(t or "").strip()]
        except Exception:
            pass
    return [s]


def _is_learnable_driver_tag(tag: object) -> bool:
    try:
        s = str(tag or "").strip().lower()
        if not s:
            return False
        if s.startswith(("market:", "edge:", "gate:", "prob_source:", "guard:", "odds:", "book:")):
            return False
        if s in {"total_already_reached", "(none)"}:
            return False
        if s in {"goals_ahead", "goals_behind", "goals_on_track", "pressure_high", "pressure_low"}:
            return True
        return s.startswith(("pace:", "goalie:", "manpower:", "empty_net:", "pressure:", "late:", "score:"))
    except Exception:
        return False


def _normalize_learnable_tags(tags: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in _parse_tag_list(tags):
        try:
            s = str(raw or "").strip()
        except Exception:
            continue
        if not s or not _is_learnable_driver_tag(s):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_market(x: Any) -> Optional[str]:
    try:
        s = str(x or "").strip().upper()
    except Exception:
        return None
    if not s or s not in SUPPORTED_MARKETS:
        return None
    return s


def _normalize_side(market: str, x: Any) -> Optional[str]:
    try:
        s = str(x or "").strip().upper()
    except Exception:
        return None
    if not s:
        return None
    if market in TOTAL_MARKETS:
        return s if s in TOTAL_SIDES else None
    if market in SIDE_MARKETS:
        return s if s in TEAM_SIDES else None
    return None


def _scopes_for_market(market: str, side: Optional[str] = None) -> list[str]:
    scopes = [str(market)]
    side_s = _normalize_side(market, side)
    if side_s:
        scopes.append(f"{market}:{side_s}")
    if market in TOTAL_MARKETS and "TOTAL" not in scopes:
        scopes.append("TOTAL")
        if side_s:
            scopes.append(f"TOTAL:{side_s}")
    elif market in SIDE_MARKETS and "ML" not in scopes:
        scopes.append("ML")
        if side_s:
            scopes.append(f"ML:{side_s}")
    scopes.append("__all__")
    # stable dedupe
    out: list[str] = []
    seen: set[str] = set()
    for s in scopes:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _extract_date(rec: dict[str, Any]) -> Optional[str]:
    for key in ("date", "signal_date"):
        try:
            s = str(rec.get(key) or "").strip()
        except Exception:
            continue
        if len(s) >= 10:
            return s[:10]
    return None


def _in_date_range(rec_date: Optional[str], start: str, end: str) -> bool:
    if not rec_date:
        return True
    return str(start) <= str(rec_date) <= str(end)


def _new_stats() -> dict[str, Any]:
    return {"bets": 0, "units": 0.0, "wins": 0, "losses": 0, "pushes": 0}


def _update_stats(stats: dict[str, Any], rec: dict[str, Any]) -> None:
    stats["bets"] = int(stats.get("bets", 0) or 0) + 1
    stats["units"] = float(stats.get("units", 0.0) or 0.0) + float(_safe_float(rec.get("profit_units")) or 0.0)
    result = str(rec.get("result") or "").strip().upper()
    if result == "WIN":
        stats["wins"] = int(stats.get("wins", 0) or 0) + 1
    elif result == "LOSE":
        stats["losses"] = int(stats.get("losses", 0) or 0) + 1
    elif result == "PUSH":
        stats["pushes"] = int(stats.get("pushes", 0) or 0) + 1


def _to_public_stats(stats: dict[str, Any]) -> dict[str, Any]:
    bets = int(stats.get("bets", 0) or 0)
    wins = int(stats.get("wins", 0) or 0)
    losses = int(stats.get("losses", 0) or 0)
    units = float(stats.get("units", 0.0) or 0.0)
    roi = (units / float(bets)) if bets > 0 else 0.0
    win_rate = (wins / float(wins + losses)) if (wins + losses) > 0 else None
    return {
        "bets": bets,
        "units": float(units),
        "roi": float(roi),
        "wins": wins,
        "losses": losses,
        "pushes": int(stats.get("pushes", 0) or 0),
        "win_rate": (float(win_rate) if win_rate is not None else None),
    }


def build_driver_tag_priors(
    rows: Iterable[dict[str, Any]],
    *,
    start: str,
    end: str,
    min_bets: int,
    min_market_bets: int,
    shrink_bets: float,
    min_roi_gap: float,
    roi_to_edge: float,
    max_edge_adjustment: float,
) -> dict[str, Any]:
    baseline_stats: dict[str, dict[str, Any]] = defaultdict(_new_stats)
    tag_stats: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(_new_stats))

    total_rows = 0
    kept_rows = 0
    for rec in rows:
        total_rows += 1
        market = _normalize_market(rec.get("market"))
        if market is None:
            continue
        profit_units = _safe_float(rec.get("profit_units"))
        if profit_units is None:
            continue
        rec_date = _extract_date(rec)
        if not _in_date_range(rec_date, start, end):
            continue
        side = _normalize_side(market, rec.get("side"))
        scopes = _scopes_for_market(market, side)
        for scope in scopes:
            _update_stats(baseline_stats[scope], rec)
        tags = _normalize_learnable_tags(rec.get("driver_tags"))
        if not tags:
            continue
        kept_rows += 1
        for scope in scopes:
            for tag in tags:
                _update_stats(tag_stats[scope][tag], rec)

    markets: dict[str, dict[str, Any]] = {}
    for scope, base in sorted(baseline_stats.items()):
        base_public = _to_public_stats(base)
        base_bets = int(base_public.get("bets", 0) or 0)
        base_roi = float(base_public.get("roi", 0.0) or 0.0)
        if base_bets < int(min_market_bets):
            continue
        scope_tags: dict[str, Any] = {}
        for tag, stats in sorted(tag_stats.get(scope, {}).items()):
            pub = _to_public_stats(stats)
            bets = int(pub.get("bets", 0) or 0)
            if bets < int(min_bets):
                continue
            roi = float(pub.get("roi", 0.0) or 0.0)
            roi_gap = float(roi - base_roi)
            reliability = float(max(0.0, min(1.0, float(bets) / max(1.0, float(bets) + float(shrink_bets)))))
            shrunk_roi_gap = float(roi_gap * reliability)
            edge_delta = float(-shrunk_roi_gap * float(roi_to_edge))
            edge_delta = float(max(-float(max_edge_adjustment), min(float(max_edge_adjustment), edge_delta)))
            if abs(shrunk_roi_gap) < float(min_roi_gap) or abs(edge_delta) < 1e-4:
                continue
            scope_tags[tag] = {
                "edge_delta": float(edge_delta),
                "reliability": float(reliability),
                "bets": bets,
                "units": float(pub.get("units", 0.0) or 0.0),
                "roi": float(roi),
                "baseline_roi": float(base_roi),
                "roi_gap": float(roi_gap),
                "shrunk_roi_gap": float(shrunk_roi_gap),
                "wins": int(pub.get("wins", 0) or 0),
                "losses": int(pub.get("losses", 0) or 0),
                "pushes": int(pub.get("pushes", 0) or 0),
                "win_rate": pub.get("win_rate"),
            }
        if scope_tags:
            markets[scope] = dict(
                sorted(
                    scope_tags.items(),
                    key=lambda kv: (abs(float((kv[1] or {}).get("edge_delta", 0.0) or 0.0)), int((kv[1] or {}).get("bets", 0) or 0), str(kv[0])),
                    reverse=True,
                )
            )

    return {
        "defaults": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "start": str(start),
            "end": str(end),
            "rows_seen": int(total_rows),
            "rows_used": int(kept_rows),
            "min_bets": int(min_bets),
            "min_market_bets": int(min_market_bets),
            "shrink_bets": float(shrink_bets),
            "min_roi_gap": float(min_roi_gap),
            "roi_to_edge": float(roi_to_edge),
            "max_total_edge_adjustment": float(max_edge_adjustment),
        },
        "markets": markets,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=str(PROC_DIR / "live_lens" / "perf" / "live_lens_bets_all.jsonl"), help="Settled Live Lens ledger JSONL/CSV")
    ap.add_argument("--start", default="", help="Start date YYYY-MM-DD (default: today-60)")
    ap.add_argument("--end", default="", help="End date YYYY-MM-DD (default: today)")
    ap.add_argument("--out-json", default=str(PROC_DIR / "live_lens" / "live_lens_driver_tag_priors.json"), help="Output priors JSON")
    ap.add_argument("--min-bets", type=int, default=8)
    ap.add_argument("--min-market-bets", type=int, default=15)
    ap.add_argument("--shrink-bets", type=float, default=40.0)
    ap.add_argument("--min-roi-gap", type=float, default=0.015)
    ap.add_argument("--roi-to-edge", type=float, default=0.12)
    ap.add_argument("--max-edge-adjustment", type=float, default=0.015)
    args = ap.parse_args()

    end = str(args.end or "").strip()
    start = str(args.start or "").strip()
    if not end:
        end = datetime.now(timezone.utc).date().isoformat()
    if not start:
        start = (datetime.now(timezone.utc).date() - timedelta(days=60)).isoformat()

    ledger_path = Path(args.ledger)
    rows = _read_ledger(ledger_path)

    obj = build_driver_tag_priors(
        rows,
        start=start,
        end=end,
        min_bets=max(1, int(args.min_bets)),
        min_market_bets=max(1, int(args.min_market_bets)),
        shrink_bets=max(1.0, float(args.shrink_bets)),
        min_roi_gap=max(0.0, float(args.min_roi_gap)),
        roi_to_edge=max(0.0, float(args.roi_to_edge)),
        max_edge_adjustment=max(0.001, float(args.max_edge_adjustment)),
    )
    obj["defaults"]["source_ledger"] = str(ledger_path)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

    n_markets = len((obj.get("markets") or {})) if isinstance(obj, dict) else 0
    n_tags = 0
    try:
        for grp in (obj.get("markets") or {}).values():
            if isinstance(grp, dict):
                n_tags += len(grp)
    except Exception:
        n_tags = 0
    print(f"[fit_live_lens_driver_tag_priors] wrote {out_path} scopes={n_markets} tags={n_tags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
