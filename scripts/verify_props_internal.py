"""Internal verification of props recommendations enrichment.

Uses FastAPI TestClient (no network socket) to hit /props/recommendations with debug=2
and asserts that we have a non-zero number of player cards and at least some remote
headshot photos. This bypasses PowerShell quoting issues and isolates whether the
route logic itself is populating photo URLs.

Exit codes:
 0 success
 1 generic failure / exception
 2 zero cards
 3 zero remote photos
"""
from __future__ import annotations

import sys
import datetime as _dt
import pathlib

# Ensure repository root is on sys.path so 'nhl_betting' can be imported even if
# PYTHONPATH is not set. This makes the script robust when launched directly
# (e.g., `python scripts/verify_props_internal.py`).
_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent  # repo root (contains nhl_betting/)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from fastapi.testclient import TestClient
except Exception as e:  # pragma: no cover
    print("[fail] fastapi.testclient import error:", e, file=sys.stderr)
    sys.exit(1)

try:
    from nhl_betting.web.app import app  # type: ignore
except Exception as e:  # pragma: no cover
    print("[fail] could not import app:", e, file=sys.stderr)
    sys.exit(1)


def main(date: str | None = None) -> int:
    d = date or _dt.date.today().strftime('%Y-%m-%d')
    client = TestClient(app)
    url = f"/props/recommendations?date={d}&debug=2"
    try:
        r = client.get(url)
    except Exception as e:
        print(f"[fail] request error: {e}")
        return 1
    if r.status_code != 200:
        print(f"[fail] HTTP {r.status_code}")
        return 1
    try:
        data = r.json()
    except Exception as e:
        print("[fail] not JSON:", e)
        return 1
    cards = data.get('cards', 0)
    photos_remote = data.get('photos_remote', 0)
    silhouettes = data.get('silhouettes', 0)
    logos = data.get('logos', 0)
    print(f"date={d} cards={cards} photos_remote={photos_remote} silhouettes={silhouettes} logos={logos}")
    if cards == 0:
        print('[fail] zero cards')
        return 2
    if photos_remote == 0:
        print('[fail] zero remote photos')
        return 3
    missing_sample = data.get('sample_players_no_photo') or []
    if missing_sample:
        print('[info] sample missing photos:', ', '.join(missing_sample))
    print('[ok] enrichment metrics look healthy')
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
