import pandas as pd
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import nhl_betting.web.app as web_app


def test_live_lens_emits_period_total_and_ml(monkeypatch: pytest.MonkeyPatch):
    date = "2099-01-01"
    game_key = "mtl @ bos"

    async def fake_v1_live(d: str):
        assert d == date
        return JSONResponse(
            {
                "ok": True,
                "date": date,
                "games": [
                    {
                        "date": date,
                        "gamePk": 1,
                        "home": "BOS",
                        "away": "MTL",
                        "key": game_key,
                        "gameState": "LIVE",
                        "period": 2,
                        "clock": "10:00",
                        "score": {"home": 1, "away": 1},
                        "lens": {
                            "totals": {"home": {"sog": 10}, "away": {"sog": 12}},
                            "periods": [
                                {"period": 1, "home": {"goals": 1, "sog": 5}, "away": {"goals": 1, "sog": 6}},
                                {"period": 2, "home": {"goals": 0, "sog": 5}, "away": {"goals": 0, "sog": 6}},
                            ],
                            "players": {"home": [], "away": []},
                            "goalies": {
                                "home": [{"name": "Test Goalie", "saves": 11, "shots_against": 12, "sv_pct": 0.917}],
                                "away": [{"name": "Away Goalie", "saves": 9, "shots_against": 10, "sv_pct": 0.9}],
                            },
                        },
                    }
                ],
            }
        )

    def fake_v1_odds_payload(*_args, **_kwargs):
        # Provide in-play period lines (P2) + core markets.
        return {
            "ok": True,
            "asof_utc": "2099-01-01T00:00:00Z",
            "date": date,
            "games": [
                {
                    "date": date,
                    "home": "BOS",
                    "away": "MTL",
                    "key": game_key,
                    "ml": {"home": -115, "away": 105},
                    "total": {"line": 5.5, "over": -110, "under": -110},
                    "puckline": {"home_-1.5": -110, "away_+1.5": -110},
                    "period_totals": {
                        "p2": {"line": 0.5, "over": -110, "under": -110},
                    },
                    "period_lines": {
                        "p2": {
                            "ml": {"home": 120, "away": -140},
                            "total": {"line": 0.5, "over": -110, "under": -110},
                        },
                    },
                    # Back-compat keys tolerated by older codepaths.
                    "h2h": {"home": -115, "away": 105},
                    "totals": {"line": 5.5, "over": -110, "under": -110},
                    "spreads": {"home": -110, "away": -110, "line": 1.5},
                }
            ],
        }

    def fake_v1_props_odds_payload(*_args, **_kwargs):
        return {
            "ok": True,
            "date": date,
            "asof_utc": "2099-01-01T00:00:00Z",
            "games": [{"date": date, "home": "BOS", "away": "MTL", "key": game_key, "props": {}}],
        }

    def fake_read_all_players_projections(d: str):
        assert d == date
        return pd.DataFrame(columns=["date", "player", "team", "position", "market", "proj_lambda"])

    # Critical for period signals: pregame model_total/model_spread must exist.
    def fake_bundle_predictions_map(d: str):
        assert d == date
        return {
            game_key: {
                "model_total": 9.0,
                "model_spread": 2.0,
                "total_line_used": 5.5,
            }
        }

    monkeypatch.setattr(web_app, "v1_live", fake_v1_live, raising=True)
    monkeypatch.setattr(web_app, "_v1_odds_payload", fake_v1_odds_payload, raising=True)
    monkeypatch.setattr(web_app, "_v1_props_odds_payload", fake_v1_props_odds_payload, raising=True)
    monkeypatch.setattr(web_app, "_read_all_players_projections", fake_read_all_players_projections, raising=True)
    monkeypatch.setattr(web_app, "_load_bundle_predictions_map", fake_bundle_predictions_map, raising=True)

    client = TestClient(web_app.app)
    resp = client.get(f"/v1/live-lens/{date}?regions=us&best=1&include_non_live=1&inplay=1")
    assert resp.status_code == 200
    js = resp.json()
    assert js.get("ok") is True
    assert js.get("games")

    sigs = (js["games"][0].get("signals") or [])

    period_total = [s for s in sigs if s.get("market") == "PERIOD_TOTAL"]
    period_ml = [s for s in sigs if s.get("market") == "PERIOD_ML"]

    assert period_total, "Expected PERIOD_TOTAL signal"
    assert period_ml, "Expected PERIOD_ML signal"

    # Settlement pipeline reads sig['period'].
    assert period_total[0].get("period") == 2
    assert period_ml[0].get("period") == 2


def test_settle_row_period_total_and_ml():
    from check_live_lens_betting_performance import _settle_row

    final_scores = {1: (3, 2)}
    final_period_scores = {
        (1, 2): (1, 0),  # BOS wins P2 1-0
        (1, 1): (1, 2),
    }

    # PERIOD_TOTAL OVER 0.5 should win (1 goal).
    row_total = pd.Series(
        {
            "gamePk": 1,
            "market": "PERIOD_TOTAL",
            "side": "OVER",
            "line": 0.5,
            "sig_period": 2,
            "price_american": -110,
        }
    )
    out_total = _settle_row(row_total, final_scores=final_scores, final_period_scores=final_period_scores)
    assert out_total["result"] == "WIN"
    assert out_total["profit_units"] is not None

    # PERIOD_ML HOME should win (1-0).
    row_ml = pd.Series(
        {
            "gamePk": 1,
            "market": "PERIOD_ML",
            "side": "HOME",
            "line": None,
            "sig_period": 2,
            "price_american": 120,
        }
    )
    out_ml = _settle_row(row_ml, final_scores=final_scores, final_period_scores=final_period_scores)
    assert out_ml["result"] == "WIN"
    assert out_ml["profit_units"] is not None

    # Tie period -> PUSH for 2-way PERIOD_ML.
    final_period_scores_tie = {(1, 2): (0, 0)}
    out_ml_push = _settle_row(row_ml, final_scores=final_scores, final_period_scores=final_period_scores_tie)
    assert out_ml_push["result"] == "PUSH"
    assert out_ml_push["profit_units"] == 0.0
