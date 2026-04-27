from fastapi.testclient import TestClient
import json

import pandas as pd

import nhl_betting.web.app as web_app_module
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
    assert 'Tracking' in text
    assert 'Current snap' in text
    assert 'Open / prev' in text
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


def test_props_cards_prefers_tracked_movement_rows(monkeypatch, tmp_path):
    df = pd.DataFrame(
        [
            {
                'player': 'Untracked Star',
                'team': 'PHI',
                'opp': 'PIT',
                'market': 'SOG',
                'side': 'Over',
                'book': 'pinnacle',
                'line': 2.5,
                'price': -105,
                'ev': 0.40,
                'prob': 0.67,
                'edge_reasons': 'MODEL OVR',
            },
            {
                'player': 'Tracked Mover',
                'team': 'PIT',
                'opp': 'PHI',
                'market': 'SOG',
                'side': 'Under',
                'book': 'draftkings',
                'line': 2.5,
                'price': -110,
                'ev': 0.22,
                'prob': 0.61,
                'edge_reasons': 'MODEL UND',
            },
            {
                'player': 'Tracked Flat',
                'team': 'VGK',
                'opp': 'UTA',
                'market': 'POINTS',
                'side': 'Over',
                'book': 'draftkings',
                'line': 0.5,
                'price': 100,
                'ev': 0.18,
                'prob': 0.57,
                'edge_reasons': 'MODEL OVR',
            },
        ]
    )

    proc_dir = tmp_path / 'processed'
    proc_dir.mkdir()
    df.to_csv(proc_dir / 'props_recommendations_2026-04-27.csv', index=False)

    snap_dir = tmp_path / 'snapshots'
    snap_dir.mkdir()
    open_obj = {
        'asof_utc': '2026-04-27T20:00:00+00:00',
        'rows': [
            {'team': 'PIT', 'player_name': 'Tracked Mover', 'market': 'SOG', 'book': 'draftkings', 'line': 2.5, 'over_price': -110, 'under_price': -105},
            {'team': 'VGK', 'player_name': 'Tracked Flat', 'market': 'POINTS', 'book': 'draftkings', 'line': 0.5, 'over_price': 100, 'under_price': -120},
        ],
    }
    prev_obj = {
        'asof_utc': '2026-04-27T21:00:00+00:00',
        'rows': [
            {'team': 'PIT', 'player_name': 'Tracked Mover', 'market': 'SOG', 'book': 'draftkings', 'line': 2.5, 'over_price': -118, 'under_price': -102},
            {'team': 'VGK', 'player_name': 'Tracked Flat', 'market': 'POINTS', 'book': 'draftkings', 'line': 0.5, 'over_price': 100, 'under_price': -120},
        ],
    }
    cur_obj = {
        'asof_utc': '2026-04-27T22:00:00+00:00',
        'rows': [
            {'team': 'PIT', 'player_name': 'Tracked Mover', 'market': 'SOG', 'book': 'draftkings', 'line': 3.5, 'over_price': -140, 'under_price': 115},
            {'team': 'VGK', 'player_name': 'Tracked Flat', 'market': 'POINTS', 'book': 'draftkings', 'line': 0.5, 'over_price': 100, 'under_price': -120},
        ],
    }
    for name, obj in (('open.json', open_obj), ('prev.json', prev_obj), ('current.json', cur_obj)):
        (snap_dir / name).write_text(json.dumps(obj), encoding='utf-8')

    monkeypatch.setattr(web_app_module, 'PROC_DIR', proc_dir)
    monkeypatch.setattr(web_app_module, '_props_odds_snapshots_dir', lambda _date: snap_dir)
    monkeypatch.setattr(web_app_module, '_seed_repo_props_artifacts_to_active_dirs', lambda _dates=None: {})
    monkeypatch.setattr(web_app_module, '_maybe_self_heal_props_recommendations', lambda *_args, **_kwargs: {'status': 'fresh'})
    monkeypatch.setattr(web_app_module, '_load_bundle_props_recommendations_df', lambda _date: pd.DataFrame())
    monkeypatch.setattr(web_app_module, '_read_props_lines_latest', lambda *_args, **_kwargs: pd.DataFrame())

    r = client.get('/v1/props-cards/2026-04-27?top=2')
    assert r.status_code == 200
    payload = r.json()
    assert payload['tracked_cards'] == 2
    assert payload['moving_cards'] == 1
    assert payload['cards'][0]['player'] == 'Tracked Mover'
    assert payload['cards'][0]['has_current_snapshot'] is True
    assert payload['cards'][0]['has_movement'] is True
    assert payload['cards'][1]['player'] == 'Tracked Flat'
    assert payload['cards'][1]['has_snapshot'] is True
