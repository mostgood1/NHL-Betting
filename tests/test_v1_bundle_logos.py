from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from nhl_betting.web import app as app_mod


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


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
