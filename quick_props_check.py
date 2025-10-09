from fastapi.testclient import TestClient
from nhl_betting.web.app import app

# Choose a date that already has processed prediction files to avoid extra fetching work
TEST_DATE = '2025-10-08'

client = TestClient(app)

r = client.get('/props', params={'date': TEST_DATE})
print('HTML /props status:', r.status_code, 'len:', len(r.text), 'cache:', r.headers.get('X-Cache'))

r2 = client.get('/props', params={'date': TEST_DATE})
print('HTML /props repeat status:', r2.status_code, 'len:', len(r2.text), 'cache:', r2.headers.get('X-Cache'))

page2 = client.get('/props', params={'date': TEST_DATE, 'page': 2})
print('HTML /props page=2 status:', page2.status_code, 'len:', len(page2.text))

j = client.get('/api/props/all.json', params={'date': TEST_DATE, 'page': 1})
print('JSON status:', j.status_code)
print('JSON meta: total_rows', j.json().get('total_rows'), 'page_size', j.json().get('page_size'), 'data_len', len(j.json().get('data', [])))
E = j.headers.get('ETag')
print('ETag present:', bool(E))
if E:
    j304 = client.get('/api/props/all.json', params={'date': TEST_DATE, 'page': 1}, headers={'If-None-Match': E})
    print('Conditional GET status (expect 304 if unchanged):', j304.status_code)
