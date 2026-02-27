from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nhl_betting.web import app as app_mod


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


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
