import json, os, time, urllib.request
from datetime import date

BASE = os.environ.get('PROPS_BASE_URL','http://127.0.0.1:8020')
D = date.today().strftime('%Y-%m-%d')
url = f"{BASE}/props/recommendations?date={D}&debug=2"
print('[verify] GET', url)
try:
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read().decode())
except Exception as e:
    print('[verify] FAILED request:', e)
    raise SystemExit(1)
print('[verify] metrics:', data)
photos = data.get('photos_remote')
cards = data.get('cards')
if photos is None:
    print('[verify] missing photos_remote metric')
    raise SystemExit(2)
if photos == 0 and cards > 0:
    print('[verify] FAIL: 0 remote photos; sample no-photo players:', data.get('sample_players_no_photo'))
    raise SystemExit(3)
print('[verify] PASS: remote photos present (', photos, '/', cards, ')')
