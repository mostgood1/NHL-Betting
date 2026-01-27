import re, sys, json
from urllib.request import Request, urlopen

url = sys.argv[1] if len(sys.argv) > 1 else 'https://www.bovada.lv/sports/hockey/nhl/boston-bruins-new-york-rangers-202601261910'
headers = { 'User-Agent': 'Mozilla/5.0' }
req = Request(url, headers=headers)
with urlopen(req, timeout=30) as r:
    html = r.read().decode('utf-8', errors='ignore')

# Try to find an event id
m = re.findall(r'eventId\":\"?(\d+)\"?', html)
ids = list(dict.fromkeys(m))
print('eventIds:', ids[:10])
# Look for embedded JSON payloads
scripts = re.findall(r'<script type="application/json"[^>]*>(.*?)</script>', html, re.S|re.I)
print('json_scripts:', len(scripts))
for i, s in enumerate(scripts[:3]):
    if 'saves' in s.lower() or 'blocked' in s.lower():
        print('script', i, 'has saves/blocked markers, length', len(s))
        print(s[:1000])
        break
