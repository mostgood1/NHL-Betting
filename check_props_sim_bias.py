from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _finite_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x[np.isfinite(x)]


def _summ(x: pd.Series) -> dict:
    x = _finite_series(x)
    if x.empty:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize sim-vs-history bias from sanity_sim_vs_averages_{date}.csv")
    ap.add_argument("--date", help="YYYY-MM-DD (loads data/processed/sanity_sim_vs_averages_{date}.csv)")
    ap.add_argument("--path", help="Path to a sanity_sim_vs_averages CSV")
    ap.add_argument(
        "--lined-only",
        action="store_true",
        help="Restrict to players with current props lines (is_current==True) for --date.",
    )
    ap.add_argument(
        "--lines-source",
        default="oddsapi",
        help="Canonical lines source to use when --lined-only is set (default: oddsapi).",
    )
    args = ap.parse_args()

    if not args.path:
        if not args.date:
            ap.error("Provide either --date or --path")
        args.path = str(Path("data/processed") / f"sanity_sim_vs_averages_{args.date}.csv")

    p = Path(args.path)
    if not p.exists() or p.stat().st_size == 0:
        raise SystemExit(f"Missing report: {p}")

    df = pd.read_csv(p)
    print(f"report={p} rows={len(df)}")

    if args.lined_only:
        if not args.date:
            raise SystemExit("--lined-only requires --date (to locate canonical lines partition)")
        base = Path("data/props/player_props_lines") / f"date={args.date}"
        lp = base / f"{args.lines_source}.parquet"
        lc = base / f"{args.lines_source}.csv"
        lines = None
        if lp.exists():
            try:
                lines = pd.read_parquet(lp)
            except Exception:
                lines = None
        if lines is None and lc.exists():
            try:
                lines = pd.read_csv(lc)
            except Exception:
                lines = None
        if lines is None or lines.empty:
            raise SystemExit(f"Missing canonical lines for {args.date}: {lp} / {lc}")
        if "is_current" in lines.columns:
            cur = pd.to_numeric(lines["is_current"], errors="coerce")
            lines = lines[cur == 1].copy()
        if lines.empty:
            raise SystemExit(f"No current lines for {args.date} in {args.lines_source}")
        if "player_id" not in lines.columns or "player_id" not in df.columns:
            raise SystemExit("player_id column missing in lines or sanity report; cannot apply --lined-only")
        ids = pd.to_numeric(lines["player_id"], errors="coerce").dropna()
        ids = ids.astype(int).unique().tolist()
        before = len(df)
        pid = pd.to_numeric(df["player_id"], errors="coerce").dropna().astype(int)
        df = df.loc[pid.index[pid.isin(ids)]].copy()
        print(f"lined_only=1 source={args.lines_source} current_players={len(ids)} rows={len(df)}/{before}")

    if "is_dressed" in df.columns:
        dressed = pd.to_numeric(df["is_dressed"], errors="coerce").fillna(0).astype(int)
        n_dressed = int((dressed == 1).sum())
        n_undressed = int((dressed == 0).sum())
        print(f"is_dressed: dressed={n_dressed} undressed={n_undressed}")

    metrics = ["shots", "assists", "points", "goals", "blocks", "sim_toi_min"]

    subsets: list[tuple[str, pd.DataFrame]] = [("ALL", df)]
    if "is_dressed" in df.columns:
        dressed = pd.to_numeric(df["is_dressed"], errors="coerce").fillna(0).astype(int)
        subsets.append(("DRESSED", df[dressed == 1].copy()))

    for label, d in subsets:
        print(f"\n==== {label} ====")
        for base in ("season", "rolling", "actual"):
            print(f"\n== delta_sim_vs_{base} ==")
            for m in metrics:
                col = f"delta_sim_vs_{base}_{m}"
                if col not in d.columns:
                    continue
                s = _summ(d[col])
                if not s.get("n"):
                    continue
                print(
                    f"{m:12s} n={s['n']:4d} mean={s['mean']:+.3f} med={s['median']:+.3f} p10={s['p10']:+.3f} p90={s['p90']:+.3f}"
                )

        # Ratios vs rolling averages are a nice sanity signal
        print("\n== ratio sim/rolling ==")
        pairs = [
            ("shots", "roll_shots"),
            ("assists", "roll_assists"),
            ("points", "roll_points"),
            ("sim_toi_min", "roll_toi_min"),
        ]
        for sim_col, roll_col in pairs:
            if sim_col not in d.columns or roll_col not in d.columns:
                continue
            a = _finite_series(d[sim_col])
            b = _finite_series(d[roll_col])
            r = (a / b).replace([np.inf, -np.inf], np.nan)
            r = r[np.isfinite(r)]
            r = r[(r > 0) & (r < 5)]
            s = _summ(r)
            if not s.get("n"):
                continue
            print(
                f"{sim_col:12s} n={s['n']:4d} mean={s['mean']:.3f} med={s['median']:.3f} p10={s['p10']:.3f} p90={s['p90']:.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
