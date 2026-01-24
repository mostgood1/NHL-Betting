from fastapi.testclient import TestClient
from nhl_betting.web.app import app

client = TestClient(app)


def test_props_html_has_eff_lambda_header():
    r = client.get('/props/all', params={'date':'today'})
    assert r.status_code == 200
    text = r.text
    # Ensure header for effective lambda is present
    assert 'Proj λ eff' in text


def test_api_projections_include_eff_lambda_key():
    r = client.get('/api/props/projections', params={'date':'today'})
    assert r.status_code == 200
    js = r.json()
    rows = js.get('data') or []
    # If there are rows, at least one should include proj_lambda_eff key
    if rows:
        assert any('proj_lambda_eff' in row and row.get('proj_lambda_eff') is not None for row in rows), 'Expected proj_lambda_eff present in at least one row'
