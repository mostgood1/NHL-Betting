import argparse
import glob
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", help="Start date YYYY-MM-DD")
    ap.add_argument("end", help="End date YYYY-MM-DD")
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    counts = {}
    cur = start
    while cur <= end:
        d = cur.strftime("%Y-%m-%d")
        p = Path(f"data/processed/props_simulations_{d}.csv")
        if p.exists():
            try:
                df = pd.read_csv(p)
                for mkt, cnt in df["market"].value_counts().to_dict().items():
                    counts[mkt] = counts.get(mkt, 0) + int(cnt)
            except Exception:
                pass
        cur += timedelta(days=1)
    import json
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
