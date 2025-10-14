from nhl_betting.web.app import app
from fastapi.testclient import TestClient

# Pick a known valid NHL player id from our diagnostics sample
KNOWN_PIDS = [8471811, 8480426, 8482055]

client = TestClient(app)

for pid in KNOWN_PIDS:
    r = client.get(f"/img/headshot/{pid}.jpg")
    print(pid, r.status_code, len(r.content) if r.status_code == 200 else 0)
