import os
from fastapi.testclient import TestClient
from nhl_betting.web.app import app

client = TestClient(app)

def test_recommendations_etag_and_ev_consistency():
    # Enable fast synthetic generation if available
    os.environ['FAST_PROPS_TEST'] = '1'
    r1 = client.get('/api/props/recommendations.json', params={'date':'today'})
    assert r1.status_code in (200, 304)
    etag = r1.headers.get('etag')
    data = r1.json()
    assert 'data' in data
    # Conditional request if we have an ETag and at least one row
    if etag and data.get('total_rows'):
        r2 = client.get('/api/props/recommendations.json', params={'date':'today'}, headers={'If-None-Match': etag})
        assert r2.status_code in (200, 304)
        if r2.status_code == 304:
            # Nothing changed; acceptable
            pass
    # EV consistency: ev_over (or ev) ≈ p_over * (dec_odds - 1) - (1-p_over) where dec_odds from over_price
    rows = data.get('data') or []
    checked = 0
    for row in rows:
        # Only assert for Over side where we have over_price and ev_over
        if str(row.get('side')).lower() != 'over':
            continue
        p_over = row.get('p_over')
        ev_val = row.get('ev_over') if row.get('ev_over') is not None else row.get('ev')
        over_price = row.get('over_price') or row.get('price')
        if p_over is None or ev_val is None or over_price is None:
            continue
        try:
            p = float(p_over)
            price = float(over_price)
        except Exception:
            continue
        # american to decimal conversion
        dec = (price / 100.0 + 1.0) if price > 0 else (100.0 / abs(price) + 1.0)
        theo = p * (dec - 1.0) - (1.0 - p)
        assert not (theo != theo), 'NaN theoretical EV'
        assert abs(theo - float(ev_val)) < 0.15, f"EV mismatch: theo={theo} stored={ev_val} row={row}"
        checked += 1
        if checked >= 10:
            break
