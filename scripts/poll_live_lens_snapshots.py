"""Poll /v1/live-lens for a date to generate JSONL snapshots on disk.

This is useful for collecting tagged live-lens snapshots consistently
for later ROI + driver analysis.

It calls the FastAPI app route via TestClient (no server needed).

Example (poll today every 60s for 2 hours):
  python scripts/poll_live_lens_snapshots.py --date 2026-03-01 --seconds 7200 --interval-sec 60 --inplay --best

Notes:
- Snapshot writing is controlled by LIVE_LENS_DIR/NHL_LIVE_LENS_DIR and throttled
  by LIVE_LENS_SNAPSHOT_MIN_SECONDS inside the endpoint.
- Set LIVE_LENS_SNAPSHOT_ALWAYS=1 to also log no-signal snapshots.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


# Ensure repo root is on sys.path for package imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--regions", default="us")
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--inplay", action="store_true")
    ap.add_argument("--include-non-live", action="store_true")
    ap.add_argument("--interval-sec", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=3600.0)
    ap.add_argument("--timeout-sec", type=float, default=3.0)
    args = ap.parse_args()

    os.environ["LIVE_LENS_ODDS_TIMEOUT_SEC"] = str(args.timeout_sec)
    os.environ["LIVE_LENS_SCHEDULE_TIMEOUT_SEC"] = str(args.timeout_sec)
    os.environ["LIVE_LENS_PBP_TIMEOUT_SEC"] = str(args.timeout_sec)

    from fastapi.testclient import TestClient
    from nhl_betting.web.app import app

    client = TestClient(app)

    url = (
        f"/v1/live-lens/{args.date}"
        f"?include_non_live={1 if args.include_non_live else 0}"
        f"&include_pbp=0"
        f"&inplay={1 if args.inplay else 0}"
        f"&regions={args.regions}"
        f"&best={1 if args.best else 0}"
    )

    t_end = time.time() + float(max(1.0, args.seconds))
    n = 0
    ok = 0
    while time.time() < t_end:
        n += 1
        try:
            r = client.get(url)
            if r.status_code == 200 and (r.json() or {}).get("ok") is True:
                ok += 1
            print(f"{n} status={r.status_code} ok={ok}")
        except Exception as e:
            print(f"{n} error={type(e).__name__}: {e}")
        time.sleep(float(max(1.0, args.interval_sec)))

    print(f"done requests={n} ok={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
