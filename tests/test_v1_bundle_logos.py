from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from nhl_betting.web import app as app_mod


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _set_render_like_paths(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "disk" / "data"
    proc_dir = data_dir / "processed"

    monkeypatch.setattr(app_mod, "ROOT_DIR", repo_root)
    monkeypatch.setattr(app_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(app_mod, "RAW_DIR", data_dir / "raw")
    monkeypatch.setattr(app_mod, "PROC_DIR", proc_dir)
    monkeypatch.setattr(app_mod, "MODEL_DIR", data_dir / "models")
    monkeypatch.delenv("NHL_PROPS_DIR", raising=False)
    monkeypatch.delenv("PROPS_DIR", raising=False)
    return repo_root, data_dir, proc_dir


def test_v1_bundle_enriches_team_logos(tmp_path: Path, monkeypatch):
    # Point the API at a temp processed directory.
    monkeypatch.setattr(app_mod, "PROC_DIR", tmp_path)

    # Minimal predictions artifact that build_daily_bundle will pick up.
    _write_csv(
        tmp_path / "predictions_2026-02-02.csv",
        "date,home,away,venue,game_state,p_home_ml,p_away_ml\n"
        "2026-02-02,Boston Bruins,New York Rangers,TD Garden,PRE,0.55,0.45\n",
    )

    client = TestClient(app_mod.app)
    r = client.get("/v1/bundle/2026-02-02")
    assert r.status_code == 200
    payload = r.json()
    assert payload.get("ok") is True

    rows = (
        (payload.get("data") or {})
        .get("games", {})
        .get("predictions", {})
        .get("rows", [])
    )
    assert isinstance(rows, list)
    assert len(rows) >= 1

    row0 = rows[0]
    assert row0.get("home") == "Boston Bruins"
    assert row0.get("away") == "New York Rangers"

    assert row0.get("home_abbr") in {"BOS", "bos"}
    assert row0.get("away_abbr") in {"NYR", "nyr"}

    assert isinstance(row0.get("home_logo"), str)
    assert isinstance(row0.get("away_logo"), str)
    assert "assets.nhle.com/logos/nhl/svg" in row0["home_logo"]
    assert "assets.nhle.com/logos/nhl/svg" in row0["away_logo"]


def test_v1_bundle_seeds_repo_bundle_to_active_proc_dir(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    proc_dir = tmp_path / "disk" / "processed"

    monkeypatch.setattr(app_mod, "ROOT_DIR", repo_root)
    monkeypatch.setattr(app_mod, "PROC_DIR", proc_dir)

    bundle_obj = {
        "data": {
            "games": {
                "predictions": {
                    "count": 1,
                    "rows": [
                        {
                            "home": "Boston Bruins",
                            "away": "New York Rangers",
                            "venue": "TD Garden",
                            "game_state": "PRE",
                            "gamePk": 2026020201,
                            "scheduled_start_utc": "2026-02-02T00:00:00Z",
                            "p_home_ml": 0.55,
                            "p_away_ml": 0.45,
                        }
                    ],
                }
            }
        }
    }
    _write_json(repo_root / "data" / "processed" / "bundles" / "date=2026-02-02" / "bundle.json", bundle_obj)

    from nhl_betting.publish import daily_bundles as bundle_mod

    def _fail_build(*_args, **_kwargs):
        raise AssertionError("build_daily_bundle should not run when repo bundle exists")

    monkeypatch.setattr(bundle_mod, "build_daily_bundle", _fail_build, raising=True)

    client = TestClient(app_mod.app)
    r = client.get("/v1/bundle/2026-02-02")
    assert r.status_code == 200

    payload = r.json()
    assert payload.get("ok") is True
    assert payload.get("note") is None
    assert (proc_dir / "bundles" / "date=2026-02-02" / "bundle.json").exists()

    rows = (
        (payload.get("data") or {})
        .get("games", {})
        .get("predictions", {})
        .get("rows", [])
    )
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0].get("home") == "Boston Bruins"
    assert rows[0].get("away") == "New York Rangers"


def test_v1_dates_seeds_repo_manifest_to_active_proc_dir(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    proc_dir = tmp_path / "disk" / "processed"

    monkeypatch.setattr(app_mod, "ROOT_DIR", repo_root)
    monkeypatch.setattr(app_mod, "PROC_DIR", proc_dir)

    manifest_obj = {
        "generated_at_utc": "2026-03-06T00:00:00+00:00",
        "latest": "2026-03-06",
        "dates": ["2026-03-05", "2026-03-06"],
        "bundles": {
            "2026-03-05": {"exists": True, "path": "data/processed/bundles/date=2026-03-05/bundle.json"},
            "2026-03-06": {"exists": True, "path": "data/processed/bundles/date=2026-03-06/bundle.json"},
        },
    }
    _write_json(repo_root / "data" / "processed" / "bundles" / "manifest.json", manifest_obj)

    from nhl_betting.publish import daily_bundles as bundle_mod

    def _fail_manifest(*_args, **_kwargs):
        raise AssertionError("build_manifest should not run when repo manifest exists")

    monkeypatch.setattr(bundle_mod, "build_manifest", _fail_manifest, raising=True)

    client = TestClient(app_mod.app)
    r = client.get("/v1/dates")
    assert r.status_code == 200

    payload = r.json()
    assert payload.get("ok") is True
    assert payload.get("latest") == "2026-03-06"
    assert payload.get("dates") == ["2026-03-05", "2026-03-06"]
    assert (proc_dir / "bundles" / "manifest.json").exists()


def test_v1_props_cards_seed_repo_recommendations_to_active_proc_dir(tmp_path: Path, monkeypatch):
    repo_root, data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)

    _write_csv(
        repo_root / "data" / "processed" / "props_recommendations_2026-03-06.csv",
        "player,team,opp,market,side,book,line,price,ev\n"
        "Nathan MacKinnon,COL,DAL,SOG,Over,draftkings,3.5,-110,0.08\n",
    )
    monkeypatch.setattr(app_mod, "_github_raw_read_csv", lambda *_a, **_k: None, raising=True)

    client = TestClient(app_mod.app)
    r = client.get("/v1/props-cards/2026-03-06?top=12")
    assert r.status_code == 200

    payload = r.json()
    assert payload.get("ok") is True
    assert payload.get("note") is None
    assert len(payload.get("cards") or []) == 1
    assert payload["cards"][0].get("player") == "Nathan MacKinnon"
    assert (proc_dir / "props_recommendations_2026-03-06.csv").exists()


def test_v1_props_cards_include_team_logo_and_headshot(tmp_path: Path, monkeypatch):
    repo_root, _data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)

    _write_csv(
        repo_root / "data" / "processed" / "props_recommendations_2026-03-06.csv",
        "player,team,opp,market,side,book,line,price,ev\n"
        "Cutter Gauthier,ANA,MTL,SOG,Under,betmgm,3.5,100,0.654\n",
    )
    _write_csv(
        proc_dir / "roster_master.csv",
        "player_id,full_name,team_abbr,image_url\n"
        "8483445,Cutter Gauthier,ANA,https://assets.nhle.com/mugs/nhl/20252026/ANA/8483445.png\n",
    )
    monkeypatch.setattr(app_mod, "_github_raw_read_csv", lambda *_a, **_k: None, raising=True)

    client = TestClient(app_mod.app)
    r = client.get("/v1/props-cards/2026-03-06?top=12")
    assert r.status_code == 200

    payload = r.json()
    assert payload.get("ok") is True
    cards = payload.get("cards") or []
    assert len(cards) == 1
    assert cards[0].get("player") == "Cutter Gauthier"
    assert cards[0].get("headshot_url") == "https://assets.nhle.com/mugs/nhl/20252026/ANA/8483445.png"
    assert isinstance(cards[0].get("team_logo"), str)
    assert "assets.nhle.com/logos/nhl/svg/ANA" in cards[0]["team_logo"]


def test_v1_props_cards_fallback_to_bundle_recommendations(tmp_path: Path, monkeypatch):
    repo_root, _data_dir, _proc_dir = _set_render_like_paths(tmp_path, monkeypatch)

    bundle_obj = {
        "data": {
            "props": {
                "recommendations": {
                    "rows": [
                        {
                            "player": "Connor McDavid",
                            "team": "EDM",
                            "opp": "CGY",
                            "market": "POINTS",
                            "side": "Over",
                            "book": "draftkings",
                            "line": 1.5,
                            "price": 105,
                            "ev": 0.06,
                        }
                    ]
                }
            }
        }
    }
    _write_json(repo_root / "data" / "processed" / "bundles" / "date=2026-03-06" / "bundle.json", bundle_obj)
    monkeypatch.setattr(app_mod, "_github_raw_read_csv", lambda *_a, **_k: None, raising=True)

    client = TestClient(app_mod.app)
    r = client.get("/v1/props-cards/2026-03-06?top=12")
    assert r.status_code == 200

    payload = r.json()
    assert payload.get("ok") is True
    assert payload.get("note") is None
    assert len(payload.get("cards") or []) == 1
    assert payload["cards"][0].get("player") == "Connor McDavid"


def test_api_player_props_seeds_repo_line_files_to_active_props_dir(tmp_path: Path, monkeypatch):
    repo_root, data_dir, _proc_dir = _set_render_like_paths(tmp_path, monkeypatch)

    _write_csv(
        repo_root / "data" / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv",
        "date,player_name,player_id,team,market,line,over_price,under_price,book,is_current\n"
        "2026-03-06,Nathan MacKinnon,8477492,COL,SOG,3.5,-110,-110,draftkings,True\n",
    )

    client = TestClient(app_mod.app)
    r = client.get("/api/player-props?date=2026-03-06")
    assert r.status_code == 200

    payload = r.json()
    assert payload.get("date") == "2026-03-06"
    data = payload.get("data") or []
    assert len(data) == 1
    assert data[0].get("player") == "Nathan MacKinnon"
    assert (data_dir / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv").exists()
