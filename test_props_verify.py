from fastapi.testclient import TestClient
from nhl_betting.web.app import app

client = TestClient(app)

print('--- HTML /props first call ---')
r1 = client.get('/props')
print('status:', r1.status_code, 'length:', len(r1.text))
print('X-Cache:', r1.headers.get('X-Cache'))

print('\n--- HTML /props second call (should HIT cache if caching eligible) ---')
r2 = client.get('/props')
print('status:', r2.status_code, 'length:', len(r2.text))
print('X-Cache:', r2.headers.get('X-Cache'))

print('\n--- HTML /props/all?page=2 ---')
r_page2 = client.get('/props/all?page=2')
print('status:', r_page2.status_code, 'length:', len(r_page2.text))

print('\n--- JSON /api/props/all.json page 1 ---')
j1 = client.get('/api/props/all.json?page=1')
print('status:', j1.status_code)
print('headers etag:', j1.headers.get('ETag'))
body1 = j1.json()
print('json keys:', list(body1.keys()))
print('page meta:', body1.get('page'), body1.get('page_size'), body1.get('total_pages'))

etag = j1.headers.get('ETag')
if etag:
    print('\n--- Conditional GET with If-None-Match ---')
    j2 = client.get('/api/props/all.json?page=1', headers={'If-None-Match': etag})
    print('conditional status:', j2.status_code, '(304 expected if no data changed)')
else:
    print('No ETag returned; cannot test conditional request.')
