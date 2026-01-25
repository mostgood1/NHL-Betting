import argparse
import json
import csv
from pathlib import Path


def load_json(path: str):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def pick_metrics(obj):
    if not obj:
        return None
    ov = obj.get("overall") or {}
    by_market = obj.get("by_market") or {}
    def _acc(m):
        try:
            return float((by_market.get(m) or {}).get("accuracy"))
        except Exception:
            return None
    return {
        "picks": ov.get("picks"),
        "decided": ov.get("decided"),
        "accuracy": ov.get("accuracy"),
        "brier": ov.get("brier"),
        "acc_sog": _acc("SOG"),
        "acc_goals": _acc("GOALS"),
        "acc_assists": _acc("ASSISTS"),
        "acc_points": _acc("POINTS"),
    }


def main():
    ap = argparse.ArgumentParser(description="Merge projections and sim-backed backtest summaries into a CSV dashboard.")
    ap.add_argument("--proj", type=str, default="", help="Path to projections backtest summary JSON")
    ap.add_argument("--sim", type=str, default="", help="Path to sim-backed backtest summary JSON")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path")
    args = ap.parse_args()

    proj = load_json(args.proj)
    sim = load_json(args.sim)

    rows = []
    if proj:
        m = pick_metrics(proj)
        if m:
            m["source"] = "projections"
            rows.append(m)
    if sim:
        m = pick_metrics(sim)
        if m:
            m["source"] = "sim"
            rows.append(m)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source","picks","decided","accuracy","brier",
        "acc_sog","acc_goals","acc_assists","acc_points",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
