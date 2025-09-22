from pathlib import Path
import pandas as pd

from nhl_betting.data.nhl_api import NHLClient
from nhl_betting.utils.io import RAW_DIR, save_df
from nhl_betting.cli import featurize, train


def main(start: str = "2022-09-01", end: str = "2024-08-01"):
    client = NHLClient()
    games = client.schedule(start, end)
    rows = []
    for g in games:
        rows.append({
            "gamePk": g.gamePk,
            "date": g.gameDate,
            "season": g.season,
            "type": g.gameType,
            "home": g.home,
            "away": g.away,
            "home_goals": g.home_goals,
            "away_goals": g.away_goals,
        })
    df = pd.DataFrame(rows)
    out = RAW_DIR / "games.csv"
    save_df(df, out)
    print(f"Saved {len(df)} games to {out}")

    # Feature engineering and training
    featurize()
    train()


if __name__ == "__main__":
    main()
