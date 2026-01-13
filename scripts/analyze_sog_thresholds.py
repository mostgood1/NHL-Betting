import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv", help="Path to backtest rows CSV")
    ap.add_argument("--thresholds", default="0.66,0.68,0.7,0.72,0.75")
    args = ap.parse_args()
    p = Path(args.rows_csv)
    df = pd.read_csv(p)
    psel = np.where(df["side"].astype(str) == "Over", df["p_over"].astype(float), 1.0 - df["p_over"].astype(float))
    df["p_sel"] = psel
    D = df[(df["market"] == "SOG") & (df["result"].isin(["win", "loss"]))]
    thrs = [float(x) for x in str(args.thresholds).split(",") if x]
    out = []
    for thr in thrs:
        sub = D[D["p_sel"] >= thr]
        acc = (sub["result"] == "win").mean() if len(sub) > 0 else None
        out.append({"thr": thr, "picks": int(len(sub)), "acc": (None if acc is None else float(acc))})
    import json
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
