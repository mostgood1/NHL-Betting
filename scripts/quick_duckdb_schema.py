import sys
import duckdb

path = sys.argv[1]
con = duckdb.connect()
print(con.execute(f"SELECT * FROM parquet_schema('{path}')").fetchdf().to_string(index=False))
