import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    n: int
    stake: float
    profit: float
    roi: float | None
    expected_profit: float
    expected_roi: float | None
    mean_ev: float | None


def compute_metrics(rows_path: Path, *, min_ev: float | None = None) -> Metrics:
    df = pd.read_csv(rows_path)
    if min_ev is not None:
        df = df[pd.to_numeric(df.get("ev"), errors="coerce") >= float(min_ev)]

    if df.empty:
        return Metrics(
            n=0,
            stake=0.0,
            profit=0.0,
            roi=None,
            expected_profit=0.0,
            expected_roi=None,
            mean_ev=None,
        )

    stake = float(pd.to_numeric(df["stake"], errors="coerce").fillna(0.0).sum())
    profit = float(pd.to_numeric(df["payout"], errors="coerce").fillna(0.0).sum())

    ev = pd.to_numeric(df.get("ev"), errors="coerce")
    expected_profit = float((ev.fillna(0.0) * pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)).sum())

    roi = (profit / stake) if stake else None
    expected_roi = (expected_profit / stake) if stake else None
    mean_ev = float(ev.mean()) if ev.notna().any() else None

    return Metrics(
        n=int(len(df)),
        stake=stake,
        profit=profit,
        roi=roi,
        expected_profit=expected_profit,
        expected_roi=expected_roi,
        mean_ev=mean_ev,
    )


def fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "None"
    return f"{100.0 * float(x):.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare odds-clamp backtest rows vs baseline")
    ap.add_argument("--base", required=True, help="Baseline rows CSV path")
    ap.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant label=path, e.g. c300=data/processed/foo.csv (repeatable)",
    )
    ap.add_argument("--min-ev", type=float, default=None, help="Optional EV floor for subset metrics")

    args = ap.parse_args()

    base_path = Path(args.base)
    base = compute_metrics(base_path, min_ev=args.min_ev)

    variants: list[tuple[str, Path]] = []
    for v in args.variant:
        if "=" not in v:
            raise SystemExit(f"Bad --variant '{v}'. Use label=path")
        label, path = v.split("=", 1)
        variants.append((label.strip(), Path(path.strip())))

    print(f"BASE: {base_path}")
    print(
        "  n={n} stake={stake:.0f} profit={profit:.0f} roi={roi} exp_roi={exp_roi} mean_ev={mean_ev}".format(
            n=base.n,
            stake=base.stake,
            profit=base.profit,
            roi=fmt_pct(base.roi),
            exp_roi=fmt_pct(base.expected_roi),
            mean_ev=(f"{base.mean_ev:.4f}" if base.mean_ev is not None else "None"),
        )
    )

    for label, path in variants:
        m = compute_metrics(path, min_ev=args.min_ev)
        d_profit = m.profit - base.profit
        d_n = m.n - base.n
        d_roi = (m.roi - base.roi) if (m.roi is not None and base.roi is not None) else None
        print(f"\n{label}: {path}")
        print(
            "  n={n} ({d_n:+d}) profit={profit:.0f} (\u0394{d_profit:+.0f}) roi={roi} (\u0394{d_roi}) exp_roi={exp_roi} mean_ev={mean_ev}".format(
                n=m.n,
                d_n=d_n,
                profit=m.profit,
                d_profit=d_profit,
                roi=fmt_pct(m.roi),
                d_roi=(fmt_pct(d_roi) if d_roi is not None else "None"),
                exp_roi=fmt_pct(m.expected_roi),
                mean_ev=(f"{m.mean_ev:.4f}" if m.mean_ev is not None else "None"),
            )
        )


if __name__ == "__main__":
    main()
