import sys
import os
import duckdb

path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/nhl_pbp/pbp_2025.parquet'
con = duckdb.connect()
print('exists', os.path.exists(path))
try:
    cnt = con.execute(f"SELECT COUNT(1) FROM read_parquet('{path}')").fetchall()[0][0]
    print('rows', cnt)
    df = con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 5").fetchdf()
    print('cols', df.columns.tolist())
    print(df.to_string(index=False))
except Exception as e:
    print('duckdb error', e)
