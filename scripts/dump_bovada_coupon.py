import json
from urllib.request import Request, urlopen

hosts = ["https://www.bovada.lv","https://www.bovada.com","https://www.bodog.eu","https://www.bodog.com"]
paths = [
    "/services/sports/event/coupon/events/A/description/ice-hockey/nhl",
    "/services/sports/event/coupon/events/A/description/hockey/nhl",
]
filters = ["def","players","player","props","playerprops"]
headers = { 'User-Agent': 'Mozilla/5.0' }

def fetch(url):
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))

for h in hosts:
    for p in paths:
        for f in filters:
            url = f"{h}{p}?marketFilterId={f}&preMatchOnly=true&includeParticipants=true&lang=en"
            try:
                js = fetch(url)
            except Exception as e:
                print('fail', url, e)
                continue
            events = []
            if isinstance(js, list):
                for grp in js:
                    if isinstance(grp, dict) and isinstance(grp.get('events'), list):
                        events.extend(grp['events'])
            elif isinstance(js, dict) and isinstance(js.get('events'), list):
                events = js['events']
            print('ok', url, 'events=', len(events))
            saves = 0
            blocks = 0
            for ev in events[:5]:
                dgs = ev.get('displayGroups') or []
                for dg in dgs:
                    mkts = dg.get('markets') or []
                    for m in mkts:
                        desc = (m.get('description') or m.get('displayKey') or m.get('key') or '').lower()
                        if 'save' in desc:
                            saves += 1
                        if 'block' in desc:
                            blocks += 1
            print('  first5: saves mkts=', saves, 'blocks mkts=', blocks)
