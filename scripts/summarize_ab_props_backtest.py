import argparse
import json
from pathlib import Path


def _maybe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _load_rows_metrics(path: Path, *, min_ev: float | None = None, max_ev: float | None = None) -> dict:
    """Compute betting metrics from a backtest rows CSV.

    Expected profit/ROI are computed from `ev` as:
      expected_profit = sum(ev * stake)
      expected_roi = expected_profit / total_stake
    where `ev` is assumed to be per-$1-staked expected value.
    """
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "pandas is required to compute betting metrics from rows CSVs. "
            "Install it or re-run without rows metrics."
        ) from e

    df = pd.read_csv(path)

    if min_ev is not None:
        df = df[df["ev"] >= min_ev]
    if max_ev is not None:
        df = df[df["ev"] <= max_ev]

    if df.empty:
        return {
            "rows_path": str(path),
            "n": 0,
            "stake": 0.0,
            "payout": 0.0,
            "profit": 0.0,
            "roi": None,
            "expected_profit": 0.0,
            "expected_roi": None,
            "mean_ev": None,
            "decided": 0,
            "wins": 0,
            "losses": 0,
            "accuracy": None,
            "brier": None,
            "avg_prob": None,
        }

    # required columns: ev, stake, payout
    for col in ["ev", "stake", "payout"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in rows CSV: {path}")

    stake = float(df["stake"].sum())
    payout = float(df["payout"].sum())
    profit = payout  # payout already includes -stake on losses

    expected_profit = float((df["ev"] * df["stake"]).sum())
    expected_roi = _safe_div(expected_profit, stake)

    roi = _safe_div(profit, stake)

    mean_ev = _maybe_float(df["ev"].mean())

    decided = None
    wins = None
    losses = None
    accuracy = None
    brier = None
    avg_prob = None

    if {"p_over", "result"}.issubset(set(df.columns)):
        # For props boxscore backtests, the rows CSV stores `p_over` as the chosen-side
        # probability (chosen_prob). So Brier uses p_sel = p_over directly.
        res = df["result"].astype(str).str.lower()
        decided_mask = res.isin(["win", "loss"])
        decided = int(decided_mask.sum())
        wins = int((res == "win").sum())
        losses = int((res == "loss").sum())
        accuracy = _safe_div(wins, decided)
        y = (res == "win").astype(float)
        p_sel = df["p_over"].astype(float).clip(0.0, 1.0)
        brier = _maybe_float(((y - p_sel) ** 2).mean())
        avg_prob = _maybe_float(p_sel.mean())

    return {
        "rows_path": str(path),
        "n": int(len(df)),
        "stake": stake,
        "payout": payout,
        "profit": profit,
        "roi": roi,
        "expected_profit": expected_profit,
        "expected_roi": expected_roi,
        "mean_ev": mean_ev,
        "decided": decided,
        "wins": wins,
        "losses": losses,
        "accuracy": accuracy,
        "brier": brier,
        "avg_prob": avg_prob,
    }


def _load_rows_metrics_by_market(path: Path, *, min_ev: float | None = None, max_ev: float | None = None) -> dict:
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "pandas is required to compute betting metrics from rows CSVs. "
            "Install it or re-run without rows metrics."
        ) from e

    df = pd.read_csv(path)
    if min_ev is not None:
        df = df[df["ev"] >= min_ev]
    if max_ev is not None:
        df = df[df["ev"] <= max_ev]
    if df.empty or "market" not in df.columns:
        return {}

    out = {}
    for market, g in df.groupby("market", dropna=False):
        stake = float(g["stake"].sum())
        payout = float(g["payout"].sum())
        profit = payout
        expected_profit = float((g["ev"] * g["stake"]).sum())
        out[str(market)] = {
            "n": int(len(g)),
            "stake": stake,
            "payout": payout,
            "profit": profit,
            "roi": _safe_div(profit, stake),
            "expected_profit": expected_profit,
            "expected_roi": _safe_div(expected_profit, stake),
            "mean_ev": _maybe_float(g["ev"].mean()),
        }
    return out


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x: float | None, nd: int = 6) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize A/B props backtest deltas from two summary JSONs produced by "
            "`props-backtest-from-boxscores`."
        )
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--a-prefix",
        default="sched",
        help="Prefix for variant A (default: sched)",
    )
    parser.add_argument(
        "--b-prefix",
        default="legacy",
        help="Prefix for variant B (default: legacy)",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(Path("data") / "processed"),
        help="Directory containing backtest summary JSONs",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write delta JSON (default: data/processed/ab_props_backtest_delta_<start>_to_<end>.json)",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=None,
        help="If set, compute an additional BETTING_SUBSET section using only rows with ev >= min_ev",
    )
    parser.add_argument(
        "--max-ev",
        type=float,
        default=None,
        help="If set, compute an additional BETTING_SUBSET section using only rows with ev <= max_ev",
    )

    args = parser.parse_args()

    if args.out is not None and str(args.out).endswith("_roi.json"):
        raise SystemExit(
            "Refusing to write redundant ROI-suffixed delta JSON. "
            "ROI is already included in the canonical delta output; "
            "use an output path without the '_roi.json' suffix."
        )

    processed_dir = Path(args.processed_dir)
    a_path = processed_dir / f"{args.a_prefix}_ab_props_backtest_boxscores_summary_{args.start}_to_{args.end}.json"
    b_path = processed_dir / f"{args.b_prefix}_ab_props_backtest_boxscores_summary_{args.start}_to_{args.end}.json"

    a_rows_path = processed_dir / f"{args.a_prefix}_ab_props_backtest_boxscores_rows_{args.start}_to_{args.end}.csv"
    b_rows_path = processed_dir / f"{args.b_prefix}_ab_props_backtest_boxscores_rows_{args.start}_to_{args.end}.csv"

    if not a_path.exists():
        raise SystemExit(f"Missing A summary: {a_path}")
    if not b_path.exists():
        raise SystemExit(f"Missing B summary: {b_path}")

    a = _load_json(a_path)
    b = _load_json(b_path)

    a_overall = a.get("overall", {})
    b_overall = b.get("overall", {})

    delta = {
        "window": {"start": args.start, "end": args.end},
        "a": {"prefix": args.a_prefix, "path": str(a_path)},
        "b": {"prefix": args.b_prefix, "path": str(b_path)},
        "overall": {},
        "by_market": {},
    }

    # Betting metrics from rows (profit/ROI/expected ROI). Optional.
    if a_rows_path.exists() and b_rows_path.exists():
        a_rows = _load_rows_metrics(a_rows_path)
        b_rows = _load_rows_metrics(b_rows_path)
        delta["a"]["rows_path"] = a_rows["rows_path"]
        delta["b"]["rows_path"] = b_rows["rows_path"]
        delta["overall"]["stake"] = {"a": a_rows["stake"], "b": b_rows["stake"], "delta": a_rows["stake"] - b_rows["stake"]}
        delta["overall"]["profit"] = {"a": a_rows["profit"], "b": b_rows["profit"], "delta": a_rows["profit"] - b_rows["profit"]}
        delta["overall"]["roi"] = {"a": a_rows["roi"], "b": b_rows["roi"], "delta": (a_rows["roi"] - b_rows["roi"]) if (a_rows["roi"] is not None and b_rows["roi"] is not None) else None}
        delta["overall"]["expected_profit"] = {"a": a_rows["expected_profit"], "b": b_rows["expected_profit"], "delta": a_rows["expected_profit"] - b_rows["expected_profit"]}
        delta["overall"]["expected_roi"] = {"a": a_rows["expected_roi"], "b": b_rows["expected_roi"], "delta": (a_rows["expected_roi"] - b_rows["expected_roi"]) if (a_rows["expected_roi"] is not None and b_rows["expected_roi"] is not None) else None}
        delta["overall"]["mean_ev"] = {"a": a_rows["mean_ev"], "b": b_rows["mean_ev"], "delta": (a_rows["mean_ev"] - b_rows["mean_ev"]) if (a_rows["mean_ev"] is not None and b_rows["mean_ev"] is not None) else None}

        a_rows_by_mkt = _load_rows_metrics_by_market(a_rows_path)
        b_rows_by_mkt = _load_rows_metrics_by_market(b_rows_path)
        for m in sorted(set(a_rows_by_mkt.keys()) | set(b_rows_by_mkt.keys())):
            am = a_rows_by_mkt.get(m, {})
            bm = b_rows_by_mkt.get(m, {})
            delta["by_market"].setdefault(m, {})
            for key in ["stake", "profit", "roi", "expected_profit", "expected_roi", "mean_ev"]:
                av = am.get(key)
                bv = bm.get(key)
                if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                    delta["by_market"][m][key] = {"a": av, "b": bv, "delta": av - bv}
                else:
                    delta["by_market"][m][key] = {"a": av, "b": bv, "delta": None}

        if args.min_ev is not None or args.max_ev is not None:
            a_bets = _load_rows_metrics(a_rows_path, min_ev=args.min_ev, max_ev=args.max_ev)
            b_bets = _load_rows_metrics(b_rows_path, min_ev=args.min_ev, max_ev=args.max_ev)
            delta["betting_subset"] = {
                "filter": {"min_ev": args.min_ev, "max_ev": args.max_ev},
                "overall": {
                    "n": {"a": a_bets["n"], "b": b_bets["n"], "delta": a_bets["n"] - b_bets["n"]},
                    "profit": {"a": a_bets["profit"], "b": b_bets["profit"], "delta": a_bets["profit"] - b_bets["profit"]},
                    "roi": {"a": a_bets["roi"], "b": b_bets["roi"], "delta": (a_bets["roi"] - b_bets["roi"]) if (a_bets["roi"] is not None and b_bets["roi"] is not None) else None},
                    "expected_profit": {"a": a_bets["expected_profit"], "b": b_bets["expected_profit"], "delta": a_bets["expected_profit"] - b_bets["expected_profit"]},
                    "expected_roi": {"a": a_bets["expected_roi"], "b": b_bets["expected_roi"], "delta": (a_bets["expected_roi"] - b_bets["expected_roi"]) if (a_bets["expected_roi"] is not None and b_bets["expected_roi"] is not None) else None},
                    "accuracy": {"a": a_bets["accuracy"], "b": b_bets["accuracy"], "delta": (a_bets["accuracy"] - b_bets["accuracy"]) if (a_bets["accuracy"] is not None and b_bets["accuracy"] is not None) else None},
                    "brier": {"a": a_bets["brier"], "b": b_bets["brier"], "delta": (a_bets["brier"] - b_bets["brier"]) if (a_bets["brier"] is not None and b_bets["brier"] is not None) else None},
                },
            }

    for key in ["picks", "decided", "wins", "losses", "pushes", "accuracy", "brier", "avg_prob"]:
        av = a_overall.get(key)
        bv = b_overall.get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            delta["overall"][key] = {"a": av, "b": bv, "delta": av - bv}
        else:
            delta["overall"][key] = {"a": av, "b": bv, "delta": None}

    a_mkts = a.get("by_market", {}) or {}
    b_mkts = b.get("by_market", {}) or {}
    all_mkts = sorted(set(a_mkts.keys()) | set(b_mkts.keys()))

    for m in all_mkts:
        am = a_mkts.get(m, {})
        bm = b_mkts.get(m, {})
        out = delta["by_market"].get(m, {})
        for key in ["picks", "decided", "wins", "losses", "pushes", "accuracy", "brier", "avg_prob"]:
            av = am.get(key)
            bv = bm.get(key)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                out[key] = {"a": av, "b": bv, "delta": av - bv}
            else:
                out[key] = {"a": av, "b": bv, "delta": None}
        delta["by_market"][m] = out

    # Print a compact, human-friendly diff
    print("A/B props backtest delta")
    print(f" window: {args.start}..{args.end}")
    print(f" A: {args.a_prefix}")
    print(f" B: {args.b_prefix}")
    print("\nOVERALL")
    for key in ["picks", "decided", "wins", "accuracy", "brier", "avg_prob", "expected_roi", "roi", "expected_profit", "profit", "mean_ev"]:
        if key not in delta["overall"]:
            continue
        row = delta["overall"][key]
        if key in {"picks", "decided", "wins"}:
            print(f" {key:>13}: a={row['a']} b={row['b']} delta={row['delta']}")
        elif key in {"profit", "expected_profit"}:
            print(
                f" {key:>13}: a={_fmt(row['a'], 2)} b={_fmt(row['b'], 2)} delta={_fmt(row['delta'], 2)}"
            )
        else:
            print(
                f" {key:>13}: a={_fmt(row['a'])} b={_fmt(row['b'])} delta={_fmt(row['delta'])}"
            )

    print("\nBY_MARKET (accuracy/brier/roi)")
    for m in all_mkts:
        acc = delta["by_market"][m]["accuracy"]
        br = delta["by_market"][m]["brier"]
        roi = delta["by_market"][m].get("roi")
        print(
            f" {m:>8}: acc a={_fmt(acc['a'])} b={_fmt(acc['b'])} d={_fmt(acc['delta'])}"
            f" | brier a={_fmt(br['a'])} b={_fmt(br['b'])} d={_fmt(br['delta'])}"
            + (f" | roi a={_fmt(roi['a'])} b={_fmt(roi['b'])} d={_fmt(roi['delta'])}" if isinstance(roi, dict) else "")
        )

    if "betting_subset" in delta:
        bs = delta["betting_subset"]
        f = bs["filter"]
        o = bs["overall"]
        print(f"\nBETTING_SUBSET (ev >= {f.get('min_ev')} ev <= {f.get('max_ev')})")
        for key in ["n", "accuracy", "brier", "expected_roi", "roi", "expected_profit", "profit"]:
            row = o.get(key)
            if row is None:
                continue
            if key == "n":
                print(f" {key:>13}: a={row['a']} b={row['b']} delta={row['delta']}")
            elif key in {"profit", "expected_profit"}:
                print(
                    f" {key:>13}: a={_fmt(row['a'], 2)} b={_fmt(row['b'], 2)} delta={_fmt(row['delta'], 2)}"
                )
            else:
                print(
                    f" {key:>13}: a={_fmt(row['a'])} b={_fmt(row['b'])} delta={_fmt(row['delta'])}"
                )

    out_path = Path(args.out) if args.out else (processed_dir / f"ab_props_backtest_delta_{args.start}_to_{args.end}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(delta, f, indent=2, sort_keys=True)

    print(f"\nWrote delta JSON: {out_path}")


if __name__ == "__main__":
    main()
