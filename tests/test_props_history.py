from fastapi.testclient import TestClient
from nhl_betting.web.app import app
import os

client = TestClient(app)

def test_recommendations_history_empty_ok():
    os.environ['FAST_PROPS_TEST'] = '1'
    r = client.get('/api/props/recommendations/history.json', params={'date':'today','days':7})
    assert r.status_code == 200
    js = r.json()
    assert 'data' in js
    # If history file absent, minimal payload without lookback_days is acceptable
    if js.get('total_rows',0) > 0:
        assert 'lookback_days' in js
