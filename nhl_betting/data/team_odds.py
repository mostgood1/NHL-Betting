import os
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import pandas as pd

try:
    from nhl_betting.data.odds_api import OddsAPIClient  # type: ignore
except Exception:  # pragma: no cover
    OddsAPIClient = None  # type: ignore


def _date_range_utc(day: datetime) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    # widen window a bit to catch late listings
    return (start - timedelta(hours=8), end + timedelta(hours=8))


def _flatten_team_odds(event: dict, bookmaker: dict, market: dict) -> List[dict]:
    rows: List[dict] = []
    event_id = event.get("id") or event.get("id")
    commence_time = event.get("commence_time")
    home = (event.get("home_team") or "").strip()
    away = (event.get("away_team") or "").strip()

    bm_key = bookmaker.get("key")
    bm_title = bookmaker.get("title")
    bm_last = bookmaker.get("last_update")

    mkt_key = market.get("key")
    mkt_outcomes = market.get("outcomes") or []

    for out in mkt_outcomes:
        rows.append(
            {
                "event_id": event_id,
                "commence_time": commence_time,
                "home": home,
                "away": away,
                "bookmaker_key": bm_key,
                "bookmaker": bm_title,
                "book_last_update": bm_last,
                "market": mkt_key,
                "outcome_name": out.get("name"),
                "outcome_price": out.get("price"),
                "outcome_point": out.get("point"),
            }
        )
    return rows


def collect_oddsapi_team_odds(date: str, markets: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if OddsAPIClient is None:
        raise RuntimeError("OddsAPIClient not available; ensure Odds API support is installed.")

    day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_utc, end_utc = _date_range_utc(day)

    market_list: List[str] = (
        list(markets)
        if markets is not None
        else os.getenv("TEAM_ODDS_MARKETS", "h2h,spreads,totals").split(",")
    )

    client = OddsAPIClient()
    events, _ = client.list_events(
        sport="icehockey_nhl",
        commence_from_iso=start_utc.isoformat().replace("+00:00", "Z"),
        commence_to_iso=end_utc.isoformat().replace("+00:00", "Z"),
    )

    all_rows: List[dict] = []
    for ev in events or []:
        odds, _ = client.event_odds(
            sport="icehockey_nhl",
            event_id=str(ev.get("id")),
            markets=",".join(market_list),
        )
        for bm in odds.get("bookmakers", []):
            for mk in bm.get("markets", []):
                all_rows.extend(_flatten_team_odds(ev, bm, mk))

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["date"] = date
    return df


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _merge_upsert(existing: pd.DataFrame, incoming: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if existing is None or existing.empty:
        return incoming
    if incoming is None or incoming.empty:
        return existing
    incoming = incoming.drop_duplicates(subset=keys)
    existing = existing.drop_duplicates(subset=keys)
    # prefer latest prices by book_last_update if present
    def _latest(df: pd.DataFrame) -> pd.DataFrame:
        if "book_last_update" in df.columns:
            return df.sort_values("book_last_update").drop_duplicates(subset=keys, keep="last")
        return df.drop_duplicates(subset=keys, keep="last")

    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = _latest(combined)
    return combined


def write_team_odds(df: pd.DataFrame, date: str, source: str = "oddsapi") -> dict:
    base_dir = os.path.join("data", "odds", "team", f"date={date}")
    csv_path = os.path.join(base_dir, f"{source}.csv")
    pq_path = os.path.join(base_dir, f"{source}.parquet")
    _ensure_dir(csv_path)

    existing = None
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
        except Exception:
            existing = None

    keys = [
        "event_id",
        "bookmaker_key",
        "market",
        "outcome_name",
        "outcome_point",
    ]
    merged = _merge_upsert(existing if isinstance(existing, pd.DataFrame) else pd.DataFrame(), df, keys)

    # Always write CSV sidecar
    merged.to_csv(csv_path, index=False)
    try:
        merged.to_parquet(pq_path, index=False)
    except Exception:
        pass

    return {
        "input_rows": 0 if df is None else int(len(df.index)),
        "output_rows": 0 if merged is None else int(len(merged.index)),
        "csv_path": csv_path,
        "parquet_path": pq_path,
    }


def archive_team_odds_oddsapi(date: str) -> dict:
    markets_env = os.getenv("TEAM_ODDS_MARKETS")
    markets = markets_env.split(",") if markets_env else None
    df = collect_oddsapi_team_odds(date, markets)
    return write_team_odds(df, date, source="oddsapi")
