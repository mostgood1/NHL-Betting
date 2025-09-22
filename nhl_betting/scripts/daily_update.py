from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nhl_betting.cli import predict_core


def today_ymd() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def run(days_ahead: int = 1) -> None:
    base = datetime.now(timezone.utc)
    for i in range(0, max(1, days_ahead)):
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        # Try Bovada first, then The Odds API, fallback to schedule-only
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            predict_core(date=d, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True)
        except Exception:
            pass
        try:
            # If odds absent, attempt The Odds API preferring DraftKings
            predict_core(date=d, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings")
        except Exception:
            pass
        try:
            # Ensure predictions exist even without odds
            predict_core(date=d, source="web", odds_source="csv")
        except Exception:
            pass


if __name__ == "__main__":
    run(days_ahead=2)
