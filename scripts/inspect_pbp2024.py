import duckdb


def main():
    path = "data/raw/nhl_pbp/pbp_2024.parquet"
    con = duckdb.connect()
    print("[info] counting rows...")
    print(con.execute(f"select count(*) from read_parquet('{path}')").fetchall())
    print("[info] date range...")
    print(con.execute(f"select min(game_date), max(game_date) from read_parquet('{path}')").fetchall())
    print("[info] seasons present...")
    print(con.execute(f"select distinct season from read_parquet('{path}') order by 1").fetchall())


if __name__ == "__main__":
    main()
