import json
from urllib.request import Request, urlopen

hosts = ["https://www.bovada.lv","https://www.bovada.com","https://www.bodog.eu","https://www.bodog.com"]
paths = [
    "/services/sports/event/coupon/mostpopular/ice-hockey/nhl",
    "/services/sports/event/coupon/mostpopular/hockey/nhl",
    "/services/sports/event/coupon/featured-events/ice-hockey/nhl",
    "/services/sports/event/coupon/featured-events/hockey/nhl",
]
headers = { 'User-Agent': 'Mozilla/5.0' }

def fetch(url):
    req = Request(url, headers=headers)
    with urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode('utf-8'))

for h in hosts:
    for p in paths:
        url = h + p
        try:
            js = fetch(url)
        except Exception as e:
            print('fail', url, e)
            continue
        def walk_has_saves(o):
            if isinstance(o, dict):
                for k,v in o.items():
                    if k in ('description','displayKey','key','name','label') and isinstance(v,str):
                        if 'save' in v.lower():
                            return True
                    if walk_has_saves(v):
                        return True
            elif isinstance(o, list):
                for x in o:
                    if walk_has_saves(x):
                        return True
            return False
        has = walk_has_saves(js)
        print('ok', url, 'saves=', has)
