import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from nhl_betting.web.app import app

if __name__ == '__main__':
    import json, sys
    date = '2025-10-14'
    client = TestClient(app)
    r = client.get(f'/props/recommendations?date={date}&debug=3')
    print('status', r.status_code)
    try:
        js = r.json()
    except Exception as e:
        print('Failed to parse JSON:', e)
        print(r.text[:500])
        sys.exit(1)
    # Pretty print key diagnostics
    print(json.dumps(js, indent=2)[:8000])
