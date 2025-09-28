"""
Recompute EVs, edges, and recommendations for a given slate date.

Usage (PowerShell):
  .\.venv\Scripts\Activate.ps1; python scripts\recompute_for_date.py 2025-09-28

This invokes the FastAPI app helper to:
  - ensure predictions_<date>.csv has EV fields populated (recompute if missing)
  - write edges_<date>.csv and recommendations_<date>.csv into data/processed
  - best-effort push CSVs to GitHub if GITHUB_* env vars are configured
"""

import asyncio
import sys
from datetime import datetime


def _today_et() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d")


async def _main(date: str) -> None:
    # Import inside to avoid heavy imports if just asking for help
    import nhl_betting.web.app as app
    await app._recompute_edges_and_recommendations(date)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else _today_et()
    # Basic validation
    try:
        datetime.fromisoformat(d)
    except Exception:
        print(f"Invalid date format: {d}. Expected YYYY-MM-DD.")
        sys.exit(2)
    asyncio.run(_main(d))
    print(f"Recompute complete for {d}")
