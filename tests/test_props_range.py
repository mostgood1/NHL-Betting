import os
from fastapi.testclient import TestClient
from nhl_betting.web.app import app

client = TestClient(app)

def test_props_range_unauthorized():
    os.environ['FAST_PROPS_TEST'] = '1'
    # Ensure token is set but not supplied
    os.environ['REFRESH_CRON_TOKEN'] = 'RANGE_TOKEN'
    r = client.post('/api/cron/props-range?back=0&ahead=0&mode=projections')
    assert r.status_code == 401


def test_props_range_projections_mode():
    os.environ['FAST_PROPS_TEST'] = '1'
    os.environ['REFRESH_CRON_TOKEN'] = 'RANGE_TOKEN'
    headers = {'Authorization': 'Bearer RANGE_TOKEN'}
    r = client.post('/api/cron/props-range?back=0&ahead=0&mode=projections', headers=headers)
    assert r.status_code == 200
    js = r.json()
    assert js.get('ok') is True
    assert js.get('mode') == 'projections'
    assert 'dates' in js and len(js['dates']) >= 1
    # results should have entries for each date
    for d in js['dates']:
        assert d in js['results']
        assert 'projections' in js['results'][d]
