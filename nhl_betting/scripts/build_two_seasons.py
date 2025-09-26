from __future__ import annotations

import pandas as pd

from ..data.nhl_api import NHLClient
from ..utils.io import RAW_DIR, save_df
from ..cli import featurize, train


def run(start: str = "2022-09-01", end: str = "2024-08-01") -> None:
    client = NHLClient()
    games = client.schedule_range(start, end, step_days=60)
    rows = []
    # ET calendar date helper
    from datetime import datetime, timezone
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        ZoneInfo = None  # type: ignore
    def _to_et(iso_utc: str):
        if not iso_utc:
            return None
        try:
            s = str(iso_utc).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if ZoneInfo:
                return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            try:
                dt = datetime.fromisoformat(str(iso_utc)[:19]).replace(tzinfo=timezone.utc)
                if ZoneInfo:
                    return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
                return dt.strftime("%Y-%m-%d")
            except Exception:
                return None
    for g in games:
        rows.append({
            "gamePk": g.gamePk,
            "date": g.gameDate,
            "date_et": _to_et(g.gameDate),
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

    featurize()
    train()
