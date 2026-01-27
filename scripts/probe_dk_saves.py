import json
from urllib.request import Request, urlopen

ids = [421, 42648, 88808, 88807, 88806, 85507]
headers = { 'User-Agent': 'Mozilla/5.0' }

def fetch(url):
    req = Request(url, headers=headers)
    with urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode('utf-8'))

for id_ in ids:
    url = f"https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{id_}?format=json"
    try:
        js = fetch(url)
    except Exception as e:
        print(f"id={id_} failed: {e}")
        continue
    eg = js.get('eventGroup') or {}
    name = eg.get('name')
    ocs = eg.get('offerCategories') or []
    has_saves = False
    for oc in ocs:
        for sc in (oc.get('offerSubcategoryDescriptors') or []):
            for offers in (sc.get('offers') or []):
                for m in (offers or []):
                    label = (m.get('label') or '').lower()
                    if 'save' in label:
                        has_saves = True
                        break
    print(f"id={id_} name={name!r} saves={has_saves}")
