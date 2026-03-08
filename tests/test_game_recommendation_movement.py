from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nhl_betting.core import game_edge_signals as signals_mod
from nhl_betting.core import recs_shared
from nhl_betting.web import app as app_mod


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _set_temp_data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    proc_dir = data_dir / "processed"

    monkeypatch.setenv("ODDS_SNAPSHOT_SCHED_MINUTES", "0")
    monkeypatch.setattr(app_mod, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(app_mod, "RAW_DIR", data_dir / "raw", raising=True)
    monkeypatch.setattr(app_mod, "PROC_DIR", proc_dir, raising=True)
    monkeypatch.setattr(app_mod, "MODEL_DIR", data_dir / "models", raising=True)
    monkeypatch.setattr(app_mod, "_gh_upsert_file_if_configured", lambda *_a, **_k: None, raising=True)

    monkeypatch.setattr(signals_mod, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(signals_mod, "PROC_DIR", proc_dir, raising=True)
    monkeypatch.setattr(recs_shared, "PROC_DIR", proc_dir, raising=True)

    signals_mod._load_team_snapshot_games.cache_clear()
    signals_mod._load_team_games.cache_clear()
    return data_dir, proc_dir


def _seed_team_snapshot_set(data_dir: Path) -> None:
    snap_dir = data_dir / "odds_snapshots" / "team_odds" / "date=2026-03-08"
    open_obj = {
        "asof_utc": "2026-03-08T14:00:00+00:00",
        "games": [
            {
                "key": "dallas stars@colorado avalanche",
                "away": "Dallas Stars",
                "home": "Colorado Avalanche",
                "ml": {"away": 110, "away_book": "draftkings", "home": -130, "home_book": "draftkings"},
                "total": {"line": 6.5, "over": -110, "over_book": "draftkings", "under": -110, "under_book": "draftkings"},
                "puckline": {"away_+1.5": -180, "away_+1.5_book": "draftkings", "home_-1.5": 150, "home_-1.5_book": "draftkings"},
            },
            {
                "key": "florida panthers@toronto maple leafs",
                "away": "Florida Panthers",
                "home": "Toronto Maple Leafs",
                "ml": {"away": 100, "away_book": "draftkings", "home": -110, "home_book": "draftkings"},
                "total": {"line": 6.5, "over": -110, "over_book": "draftkings", "under": -110, "under_book": "draftkings"},
                "puckline": {"away_+1.5": -175, "away_+1.5_book": "draftkings", "home_-1.5": 145, "home_-1.5_book": "draftkings"},
            },
        ],
    }
    prev_obj = {
        "asof_utc": "2026-03-08T15:00:00+00:00",
        "games": [
            {
                "key": "dallas stars@colorado avalanche",
                "away": "Dallas Stars",
                "home": "Colorado Avalanche",
                "ml": {"away": 105, "away_book": "draftkings", "home": -125, "home_book": "draftkings"},
                "total": {"line": 6.5, "over": -112, "over_book": "draftkings", "under": -108, "under_book": "draftkings"},
                "puckline": {"away_+1.5": -182, "away_+1.5_book": "draftkings", "home_-1.5": 148, "home_-1.5_book": "draftkings"},
            },
            {
                "key": "florida panthers@toronto maple leafs",
                "away": "Florida Panthers",
                "home": "Toronto Maple Leafs",
                "ml": {"away": 102, "away_book": "draftkings", "home": -115, "home_book": "draftkings"},
                "total": {"line": 6.5, "over": -108, "over_book": "draftkings", "under": -112, "under_book": "draftkings"},
                "puckline": {"away_+1.5": -176, "away_+1.5_book": "draftkings", "home_-1.5": 146, "home_-1.5_book": "draftkings"},
            },
        ],
    }
    current_obj = {
        "asof_utc": "2026-03-08T16:00:00+00:00",
        "games": [
            {
                "key": "dallas stars@colorado avalanche",
                "away": "Dallas Stars",
                "home": "Colorado Avalanche",
                "ml": {"away": 100, "away_book": "draftkings", "home": -120, "home_book": "draftkings"},
                "total": {"line": 6.0, "over": -110, "over_book": "draftkings", "under": -110, "under_book": "draftkings"},
                "puckline": {"away_+1.5": -178, "away_+1.5_book": "draftkings", "home_-1.5": 152, "home_-1.5_book": "draftkings"},
            },
            {
                "key": "florida panthers@toronto maple leafs",
                "away": "Florida Panthers",
                "home": "Toronto Maple Leafs",
                "ml": {"away": 100, "away_book": "draftkings", "home": -120, "home_book": "draftkings"},
                "total": {"line": 6.0, "over": -110, "over_book": "draftkings", "under": -110, "under_book": "draftkings"},
                "puckline": {"away_+1.5": -178, "away_+1.5_book": "draftkings", "home_-1.5": 152, "home_-1.5_book": "draftkings"},
            },
        ],
    }
    _write_json(snap_dir / "open.json", open_obj)
    _write_json(snap_dir / "prev.json", prev_obj)
    _write_json(snap_dir / "current.json", current_obj)
    signals_mod._load_team_snapshot_games.cache_clear()


def _seed_predictions(proc_dir: Path) -> None:
    _write_csv(
        proc_dir / "predictions_2026-03-08.csv",
        "date,home,away,p_home_ml,p_away_ml,home_ml_odds,away_ml_odds,home_ml_book,away_ml_book,model_spread\n"
        "2026-03-08,Colorado Avalanche,Dallas Stars,0.60,0.40,-120,100,draftkings,draftkings,0.70\n"
        "2026-03-08,Toronto Maple Leafs,Florida Panthers,0.60,0.40,-120,100,draftkings,draftkings,0.70\n",
    )


def test_attach_game_edge_signals_adds_totals_move_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, proc_dir = _set_temp_data_dirs(tmp_path, monkeypatch)
    _seed_team_snapshot_set(data_dir)

    predictions = pd.DataFrame(
        [
            {
                "date": "2026-03-08",
                "home": "Colorado Avalanche",
                "away": "Dallas Stars",
                "p_over": 0.56,
                "over_odds": -110,
                "under_odds": -110,
                "total_line_used": 6.0,
                "model_total": 6.7,
                "period1_home_proj": 1.1,
                "period1_away_proj": 0.9,
            }
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "date": "2026-03-08",
                "home": "Colorado Avalanche",
                "away": "Dallas Stars",
                "market": "ev_over",
                "ev": 0.08,
            }
        ]
    )

    out = signals_mod.attach_game_edge_signals("2026-03-08", edges, predictions=predictions, proc_dir=proc_dir)

    assert float(out.loc[0, "edge_score"]) > 0
    assert "MOVE+" in str(out.loc[0, "edge_drivers"])


def test_api_recommendations_uses_movement_signals_for_sorting_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, proc_dir = _set_temp_data_dirs(tmp_path, monkeypatch)
    _seed_team_snapshot_set(data_dir)
    _seed_predictions(proc_dir)

    client = TestClient(app_mod.app)
    response = client.get("/api/recommendations", params={"date": "2026-03-08", "markets": "moneyline", "top": 2})
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 2
    assert rows[0]["home"] == "Colorado Avalanche"
    assert "PRICE+" in str(rows[0].get("edge_drivers") or "")
    assert "PRICE-" in str(rows[1].get("edge_drivers") or "")
    assert float(rows[0]["edge_score"]) > float(rows[1]["edge_score"])

    saved = pd.read_csv(proc_dir / "recommendations_2026-03-08.csv")
    assert {"edge_score", "edge_drivers", "edge_reasons"}.issubset(saved.columns)
    assert saved.iloc[0]["home"] == "Colorado Avalanche"


def test_shared_recompute_uses_movement_signals_and_persists_edge_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, proc_dir = _set_temp_data_dirs(tmp_path, monkeypatch)
    _seed_team_snapshot_set(data_dir)
    _seed_predictions(proc_dir)

    rows = recs_shared.recompute_edges_and_recommendations("2026-03-08", 0.0)

    assert len(rows) == 2
    assert rows[0]["home"] == "Colorado Avalanche"
    assert "PRICE+" in str(rows[0].get("edge_drivers") or "")
    assert "PRICE-" in str(rows[1].get("edge_drivers") or "")

    saved = pd.read_csv(proc_dir / "recommendations_2026-03-08.csv")
    assert {"edge_score", "edge_drivers", "edge_reasons"}.issubset(saved.columns)
    assert saved.iloc[0]["home"] == "Colorado Avalanche"