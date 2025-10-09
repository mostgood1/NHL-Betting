from fastapi.testclient import TestClient
from nhl_betting.web.app import app
import os
client = TestClient(app)
print('PROPS_PAGE_SIZE', os.getenv('PROPS_PAGE_SIZE'))
# Page 1
r1 = client.get('/props', params={'date':'today'})
print('props page1 status', r1.status_code, 'len', len(r1.text), 'X-Cache', r1.headers.get('X-Cache'))
# Page 2
r2 = client.get('/props', params={'date':'today','page':2})
print('props page2 status', r2.status_code, 'len', len(r2.text))
print('page2 different len?', len(r2.text)!=len(r1.text))
# JSON page 1
api1 = client.get('/api/props/all.json', params={'date':'today','page':1})
print('json status', api1.status_code)
js = api1.json()
print('meta total_rows', js['total_rows'], 'filtered', js['filtered_rows'], 'page_size', js['page_size'], 'total_pages', js['total_pages'], 'rows_returned', len(js['data']))
etag = api1.headers.get('ETag')
print('etag present', bool(etag))
# Conditional
api2 = client.get('/api/props/all.json', params={'date':'today','page':1}, headers={'If-None-Match': etag})
print('conditional status', api2.status_code)
# Cache hit
r1b = client.get('/props', params={'date':'today'})
print('props page1 second fetch X-Cache', r1b.headers.get('X-Cache'))
