import sys
from pathlib import Path
import pandas as pd
import json

PROC = Path('data/processed')
HIST = PROC / 'props_reconciliations_history.csv'


def load_day(date: str) -> pd.DataFrame:
    p = PROC / f'props_reconciliations_{date}.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        df['date'] = date
        return df
    except Exception:
        return pd.DataFrame()


def append_history(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    if HIST.exists():
        try:
            cur = pd.read_csv(HIST)
        except Exception:
            cur = pd.DataFrame()
        out = pd.concat([cur, df], ignore_index=True)
        out.to_csv(HIST, index=False)
    else:
        df.to_csv(HIST, index=False)


def write_day_summary(date: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    summ = {
        'date': date,
        'totals': df['result'].value_counts().to_dict(),
        'by_market': df.groupby(['market','result']).size().unstack(fill_value=0).to_dict(),
    }
    out = PROC / f'props_reconciliations_summary_{date}.json'
    out.write_text(json.dumps(summ))
    print(f'[aggregate] wrote {out}')


def main(dates: list[str]) -> None:
    all_df = []
    for d in dates:
        df = load_day(d)
        if df is None or df.empty:
            print(f'[aggregate] missing day {d}')
            continue
        append_history(df)
        write_day_summary(d, df)
        all_df.append(df)
    if all_df:
        agg = pd.concat(all_df, ignore_index=True)
        tot = agg['result'].value_counts().to_dict()
        print('[aggregate] combined totals:', tot)
        print('[aggregate] by-market head:\n', agg.groupby(['market','result']).size().unstack(fill_value=0).head(20).to_string())

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: props_reconciliations_aggregate.py YYYY-MM-DD [YYYY-MM-DD ...]')
        sys.exit(2)
    main(sys.argv[1:])
