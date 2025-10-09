from fastapi.testclient import TestClient
from nhl_betting.web.app import app
import os

client = TestClient(app)

def test_projections_history_empty_ok():
    os.environ['FAST_PROPS_TEST'] = '1'
    r = client.get('/api/props/projections/history.json', params={'date':'today','days':7})
    assert r.status_code == 200
    js = r.json()
    assert 'data' in js
    # lookback_days appears only when rows exist
    if js.get('total_rows',0) > 0:
        assert 'lookback_days' in js
