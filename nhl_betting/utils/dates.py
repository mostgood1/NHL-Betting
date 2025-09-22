from __future__ import annotations

from datetime import datetime, timedelta, timezone

DATE_FMT = "%Y-%m-%d"


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, DATE_FMT).replace(tzinfo=timezone.utc)


def today_utc() -> datetime:
    return datetime.now(timezone.utc)


def ymd(dt: datetime) -> str:
    return dt.strftime(DATE_FMT)


def days_ago(n: int) -> datetime:
    return today_utc() - timedelta(days=n)
