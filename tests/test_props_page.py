from fastapi.testclient import TestClient
from nhl_betting.web.app import app

client = TestClient(app)

def test_cards_root_html_renders():
    r = client.get('/')
    assert r.status_code == 200
    text = r.text
    assert 'NHL Game Cards' in text
    assert 'cards-control-card' in text
    assert 'cards-scoreboard' in text
    assert 'Player props pulse' in text
    assert "label: 'Projected'" in text
    assert "label: 'Actual'" in text
    assert "label: 'Edge'" in text
    assert "label: 'Sim mean'" not in text
    assert 'cards-tabs-rail' in text
    assert 'Away goalie' in text
    assert 'cards-overview-main-grid' in text
    assert 'cards-hr-targets-card' in text
    assert 'cards-strip-starters' in text
    assert 'cards-head-team' in text
    assert 'cards-game-time-label' in text
    assert 'prop-overview-lens' in text
    assert 'actual-totals' in text
    assert 'sim-totals' in text
    assert 'actual-box' in text
    assert 'sim-box' in text
    assert 'data-role="game-lens"' in text
    assert 'data-role="overview-bars"' in text
    assert 'Opening period' in text
    assert 'Middle period' in text
    assert 'Closing period' in text
    assert 'Extra-time path' in text
    assert 'Open game view' in text


def test_game_route_redirects_to_cards_view():
    r = client.get('/game/12345?date=2026-02-02', follow_redirects=False)
    assert r.status_code == 307
    assert r.headers['location'] == '/?date=2026-02-02&gamePk=12345'
