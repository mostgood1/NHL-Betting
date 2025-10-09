import os
import json
from fastapi.testclient import TestClient
from nhl_betting.web.app import app

client = TestClient(app)

def test_props_all_basic():
    # Fast synthetic mode for speed
    os.environ['FAST_PROPS_TEST'] = '1'
    r = client.get('/api/props/all.json', params={'date':'today'})
    assert r.status_code in (200,304)
    data = r.json()
    assert 'total_rows' in data
    assert 'page' in data and data['page'] == 1


def test_recommendations_json_empty_ok():
    # Likely empty (no file) but should still return structured payload
    r = client.get('/api/props/recommendations.json', params={'date':'1900-01-01'})
    assert r.status_code in (200,304)
    data = r.json()
    assert 'total_rows' in data
    assert isinstance(data['data'], list)


def test_etag_behavior():
    os.environ['FAST_PROPS_TEST'] = '1'
    first = client.get('/api/props/all.json', params={'date':'today'})
    assert first.status_code == 200
    etag = first.headers.get('etag')
    if etag:
        second = client.get('/api/props/all.json', params={'date':'today'}, headers={'If-None-Match': etag})
        assert second.status_code == 304
