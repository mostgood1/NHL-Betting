import json
from urllib.parse import quote
from urllib.request import Request, urlopen

slug = 'boston-bruins-new-york-rangers-202601261910'
url = f'https://www.bovada.lv/services/sports/event/coupon/events/A/description/hockey/nhl/{quote(slug)}?preMatchOnly=true&includeParticipants=true&lang=en'
headers = { 'User-Agent': 'Mozilla/5.0' }
req = Request(url, headers=headers)
try:
    with urlopen(req, timeout=25) as r:
        js = json.loads(r.read().decode('utf-8'))
    if isinstance(js, list):
        print('list len=', len(js))
        print('first keys:', list(js[0].keys()) if js else [])
    else:
        print('dict keys:', list(js.keys()))
except Exception as e:
    print('fail', e)
