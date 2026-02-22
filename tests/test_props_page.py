from fastapi.testclient import TestClient
from nhl_betting.web.app import app

client = TestClient(app)

def test_cards_root_html_renders():
    r = client.get('/')
    assert r.status_code == 200
    text = r.text.lower()
    assert 'nhl betting' in text
    assert 'toggle theme' in text
