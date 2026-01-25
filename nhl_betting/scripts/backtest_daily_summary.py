import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
def load_summary(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)
def extract_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    if not summary:
        return {}
    overall = summary.get('overall', {})
    per_market = summary.get('per_market', {})
    metrics = {
        'picks': overall.get('picks'),
        'decided': overall.get('decided'),
        'accuracy': overall.get('accuracy'),
        'brier': overall.get('brier'),
    }
    for mkt, vals in per_market.items():
        acc_key = f"acc_{mkt.lower()}"
        br_key = f"brier_{mkt.lower()}"
        metrics[acc_key] = vals.get('accuracy')
        metrics[br_key] = vals.get('brier')
    return metrics

def add_deltas(curr: Dict[str, Any], prev: Dict[str, Any]) -> Dict[str, Any]:
    if not curr:
        return {}
    deltas: Dict[str, Any] = {}
    keys = set(curr.keys())
    for key in keys:
        # Only compute deltas for numeric fields
        curr_val = curr.get(key)
        prev_val = prev.get(key) if prev else None
        if isinstance(curr_val, (int, float)) and isinstance(prev_val, (int, float)):
            deltas[f"{key}_delta"] = round(curr_val - prev_val, 6)
    return deltas
def main(argv: List[str]) -> int:
    # Args: proj_json sim_json out_csv [proj_prev_json] [sim_prev_json]
    if len(argv) < 4:
        print("Usage: backtest_daily_summary.py <proj_json> <sim_json> <out_csv> [proj_prev_json] [sim_prev_json]")
        return 2
    proj_json = Path(argv[1]) if argv[1] != 'None' else None
    sim_json = Path(argv[2]) if argv[2] != 'None' else None
    out_csv = Path(argv[3])
    proj_prev_json = Path(argv[4]) if len(argv) > 4 and argv[4] != 'None' else None
    sim_prev_json = Path(argv[5]) if len(argv) > 5 and argv[5] != 'None' else None
    proj_summary = load_summary(proj_json) if proj_json else {}
    sim_summary = load_summary(sim_json) if sim_json else {}
    proj_prev_summary = load_summary(proj_prev_json) if proj_prev_json else {}
    sim_prev_summary = load_summary(sim_prev_json) if sim_prev_json else {}
    proj_metrics = extract_metrics(proj_summary)
    sim_metrics = extract_metrics(sim_summary)
    proj_deltas = add_deltas(proj_metrics, extract_metrics(proj_prev_summary)) if proj_metrics else {}
    sim_deltas = add_deltas(sim_metrics, extract_metrics(sim_prev_summary)) if sim_metrics else {}
    headers: List[str] = [
        'source','picks','decided','accuracy','brier',
        'acc_sog','brier_sog','acc_goals','brier_goals','acc_assists','brier_assists',
        'acc_points','brier_points','acc_saves','brier_saves','acc_blocks','brier_blocks',
        'picks_delta','decided_delta','accuracy_delta','brier_delta',
        'acc_sog_delta','brier_sog_delta','acc_goals_delta','brier_goals_delta','acc_assists_delta','brier_assists_delta',
        'acc_points_delta','brier_points_delta','acc_saves_delta','brier_saves_delta','acc_blocks_delta','brier_blocks_delta'
    ]
    rows: List[List[Any]] = []
    if proj_metrics:
        rows.append([
            'projections',
            proj_metrics.get('picks'), proj_metrics.get('decided'), proj_metrics.get('accuracy'), proj_metrics.get('brier'),
            proj_metrics.get('acc_sog'), proj_metrics.get('brier_sog'), proj_metrics.get('acc_goals'), proj_metrics.get('brier_goals'),
            proj_metrics.get('acc_assists'), proj_metrics.get('brier_assists'), proj_metrics.get('acc_points'), proj_metrics.get('brier_points'),
            proj_metrics.get('acc_saves'), proj_metrics.get('brier_saves'), proj_metrics.get('acc_blocks'), proj_metrics.get('brier_blocks'),
            proj_deltas.get('picks_delta'), proj_deltas.get('decided_delta'), proj_deltas.get('accuracy_delta'), proj_deltas.get('brier_delta'),
            proj_deltas.get('acc_sog_delta'), proj_deltas.get('brier_sog_delta'), proj_deltas.get('acc_goals_delta'), proj_deltas.get('brier_goals_delta'),
            proj_deltas.get('acc_assists_delta'), proj_deltas.get('brier_assists_delta'), proj_deltas.get('acc_points_delta'), proj_deltas.get('brier_points_delta'),
            proj_deltas.get('acc_saves_delta'), proj_deltas.get('brier_saves_delta'), proj_deltas.get('acc_blocks_delta'), proj_deltas.get('brier_blocks_delta'),
        ])
    if sim_metrics:
        rows.append([
            'simulations',
            sim_metrics.get('picks'), sim_metrics.get('decided'), sim_metrics.get('accuracy'), sim_metrics.get('brier'),
            sim_metrics.get('acc_sog'), sim_metrics.get('brier_sog'), sim_metrics.get('acc_goals'), sim_metrics.get('brier_goals'),
            sim_metrics.get('acc_assists'), sim_metrics.get('brier_assists'), sim_metrics.get('acc_points'), sim_metrics.get('brier_points'),
            sim_metrics.get('acc_saves'), sim_metrics.get('brier_saves'), sim_metrics.get('acc_blocks'), sim_metrics.get('brier_blocks'),
            sim_deltas.get('picks_delta'), sim_deltas.get('decided_delta'), sim_deltas.get('accuracy_delta'), sim_deltas.get('brier_delta'),
            sim_deltas.get('acc_sog_delta'), sim_deltas.get('brier_sog_delta'), sim_deltas.get('acc_goals_delta'), sim_deltas.get('brier_goals_delta'),
            sim_deltas.get('acc_assists_delta'), sim_deltas.get('brier_assists_delta'), sim_deltas.get('acc_points_delta'), sim_deltas.get('brier_points_delta'),
            sim_deltas.get('acc_saves_delta'), sim_deltas.get('brier_saves_delta'), sim_deltas.get('acc_blocks_delta'), sim_deltas.get('brier_blocks_delta'),
        ])
    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8') as f:
        f.write(','.join(headers) + '\n')
        for row in rows:
            f.write(','.join('' if v is None else str(v) for v in row) + '\n')
if __name__ == '__main__':
    sys.exit(main(sys.argv))
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
