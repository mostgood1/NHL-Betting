import os
from fastapi.testclient import TestClient
from nhl_betting.web.app import app

client = TestClient(app)

def test_props_all_html_renders():
    # Ensure we exercise the real template, not the fast synthetic fallback
    if 'FAST_PROPS_TEST' in os.environ:
        del os.environ['FAST_PROPS_TEST']
    r = client.get('/props/all', params={'date': 'today'})
    assert r.status_code == 200
    text = r.text.lower()
    assert 'props' in text
    assert 'source-toggle' in text, f"expected source-toggle in template, got snippet: {text[:160]}"
