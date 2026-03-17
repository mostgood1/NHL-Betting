from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd

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
    assert payload.get("note") is None
    assert (tmp_path / "bundles" / "date=2026-02-02" / "bundle.json").exists()
    assert (tmp_path / "bundles" / "manifest.json").exists()

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


def test_v1_bundle_persists_repo_fallback_bundle_to_active_proc_dir(tmp_path: Path, monkeypatch):
    repo_root, _data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)

    _write_csv(
        repo_root / "data" / "processed" / "predictions_2026-02-02.csv",
        "date,home,away,venue,game_state,p_home_ml,p_away_ml\n"
        "2026-02-02,Boston Bruins,New York Rangers,TD Garden,PRE,0.55,0.45\n",
    )

    client = TestClient(app_mod.app)
    r = client.get("/v1/bundle/2026-02-02")
    assert r.status_code == 200

    payload = r.json()
    assert payload.get("ok") is True
    assert payload.get("note") is None

    bundle_file = proc_dir / "bundles" / "date=2026-02-02" / "bundle.json"
    manifest_file = proc_dir / "bundles" / "manifest.json"
    assert bundle_file.exists()
    assert manifest_file.exists()

    bundle_obj = json.loads(bundle_file.read_text(encoding="utf-8"))
    rows = (
        (bundle_obj.get("data") or {})
        .get("games", {})
        .get("predictions", {})
        .get("rows", [])
    )
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0].get("home") == "Boston Bruins"
    assert rows[0].get("away") == "New York Rangers"


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


def test_seed_repo_props_artifacts_restores_empty_recommendations_csv(tmp_path: Path, monkeypatch):
    repo_root, _data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)

    _write_csv(
        repo_root / "data" / "processed" / "props_recommendations_2026-03-06.csv",
        "player,team,opp,market,side,book,line,price,ev\n"
        "Nathan MacKinnon,COL,DAL,SOG,Over,draftkings,3.5,-110,0.08\n",
    )
    _write_csv(proc_dir / "props_recommendations_2026-03-06.csv", "player,team,opp,market,side,book,line,price,ev\n")

    stats = app_mod._seed_repo_props_artifacts_to_active_dirs(["2026-03-06"])

    assert int(stats.get("copied") or 0) >= 1
    text = (proc_dir / "props_recommendations_2026-03-06.csv").read_text(encoding="utf-8")
    assert "Nathan MacKinnon" in text


def test_v1_props_cards_refreshes_stale_recommendations_from_lines(tmp_path: Path, monkeypatch):
    _repo_root, data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)

    rec_path = proc_dir / "props_recommendations_2026-03-06.csv"
    _write_csv(
        rec_path,
        "player,team,opp,market,side,book,line,price,ev\n"
        "Old Player,COL,DAL,SOG,Over,draftkings,3.5,-110,0.08\n",
    )

    lines_path = data_dir / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv"
    _write_csv(
        lines_path,
        "date,player_name,player_id,team,market,line,over_price,under_price,book,is_current\n"
        "2026-03-06,New Player,1,COL,SOG,3.5,-110,-110,draftkings,True\n",
    )

    os.utime(rec_path, (1000, 1000))
    os.utime(lines_path, (2000, 2000))

    def _fake_refresh(date: str, min_ev: float = 0.0, top: int = 200):
        _write_csv(
            rec_path,
            "player,team,opp,market,side,book,line,price,ev\n"
            "New Player,COL,DAL,SOG,Over,draftkings,3.5,-105,0.12\n",
        )
        return {"ok": True, "date": date, "rows": 1}

    monkeypatch.setattr(app_mod, "_refresh_props_recommendations", _fake_refresh, raising=True)
    monkeypatch.setattr(app_mod, "_github_raw_read_csv", lambda *_a, **_k: None, raising=True)

    client = TestClient(app_mod.app)
    r = client.get("/v1/props-cards/2026-03-06?top=12")
    assert r.status_code == 200

    payload = r.json()
    assert payload.get("ok") is True
    assert len(payload.get("cards") or []) == 1
    assert payload["cards"][0].get("player") == "New Player"


def test_v1_props_cards_self_heals_missing_local_recommendations(tmp_path: Path, monkeypatch):
    _repo_root, _data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)

    rec_path = proc_dir / "props_recommendations_2026-03-06.csv"
    calls: list[tuple[str, float, int]] = []

    def _fake_refresh(date: str, min_ev: float = 0.0, top: int = 200):
        calls.append((date, min_ev, top))
        _write_csv(
            rec_path,
            "player,team,opp,market,side,book,line,price,ev\n"
            "Recovered Player,COL,DAL,SOG,Over,draftkings,3.5,-105,0.12\n",
        )
        return {"ok": True, "date": date, "rows": 1}

    monkeypatch.setattr(app_mod, "_refresh_props_recommendations", _fake_refresh, raising=True)
    monkeypatch.setattr(app_mod, "_github_raw_read_csv", lambda *_a, **_k: None, raising=True)
    monkeypatch.setattr(app_mod, "_read_only", lambda *_a, **_k: False, raising=True)

    client = TestClient(app_mod.app)
    r = client.get("/v1/props-cards/2026-03-06?top=12")
    assert r.status_code == 200

    payload = r.json()
    assert payload.get("ok") is True
    assert len(payload.get("cards") or []) == 1
    assert payload["cards"][0].get("player") == "Recovered Player"
    assert calls == [("2026-03-06", 0.0, 200)]


def test_refresh_props_recommendations_collects_missing_lines_on_demand(tmp_path: Path, monkeypatch):
    _repo_root, data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)
    from nhl_betting.core import props_edge_signals as edge_mod

    _write_csv(
        proc_dir / "props_projections_all_2026-03-06.csv",
        "player,team,market,proj_lambda\n"
        "Collected Player,COL,SOG,4.2\n",
    )

    calls: list[tuple[str, bool, tuple[str, ...]]] = []

    def _fake_collect(date: str, push_to_github: bool = True, books: tuple[str, ...] = ("oddsapi", "bovada")):
        calls.append((date, push_to_github, tuple(books)))
        _write_csv(
            data_dir / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv",
            "date,player_name,team,market,line,over_price,under_price,book\n"
            "2026-03-06,Collected Player,COL,SOG,1.5,100,-200,draftkings\n",
        )
        return {"date": date, "written": ["oddsapi.csv"], "errors": []}

    monkeypatch.setattr(app_mod, "_collect_props_lines_for_date", _fake_collect, raising=True)
    monkeypatch.setattr(app_mod, "_github_raw_read_csv", lambda *_a, **_k: pd.DataFrame(), raising=True)
    monkeypatch.setattr(app_mod, "_github_raw_read_parquet", lambda *_a, **_k: pd.DataFrame(), raising=True)
    monkeypatch.setattr(app_mod, "_gh_upsert_file_if_better_or_same", lambda *_a, **_k: {"ok": True}, raising=True)
    monkeypatch.setattr(edge_mod, "attach_prop_edge_signals", lambda **kwargs: kwargs["props"], raising=True)

    res = app_mod._refresh_props_recommendations("2026-03-06", min_ev=0.0, top=50)

    assert res.get("ok") is True
    assert calls == [("2026-03-06", True, ("oddsapi", "bovada"))]
    text = (proc_dir / "props_recommendations_2026-03-06.csv").read_text(encoding="utf-8")
    assert "Collected Player" in text


def test_refresh_props_recommendations_reads_csv_lines_fallback(tmp_path: Path, monkeypatch):
    _repo_root, data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)

    _write_csv(
        data_dir / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv",
        "date,player_name,team,market,line,over_price,under_price,book\n"
        "2026-03-06,CSV Player,COL,SOG,1.5,100,-200,draftkings\n",
    )
    _write_csv(
        proc_dir / "props_projections_all_2026-03-06.csv",
        "player,market,proj_lambda\n"
        "CSV Player,SOG,4.2\n",
    )

    monkeypatch.setattr(app_mod, "_gh_upsert_file_if_better_or_same", lambda *_a, **_k: {"ok": True}, raising=True)

    res = app_mod._refresh_props_recommendations("2026-03-06", min_ev=0.0, top=50)

    assert res.get("ok") is True
    assert int(res.get("rows") or 0) >= 1

    out_path = proc_dir / "props_recommendations_2026-03-06.csv"
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "CSV Player" in text


def test_refresh_props_recommendations_falls_back_to_existing_recs_when_proj_all_empty(tmp_path: Path, monkeypatch):
    _repo_root, data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)
    from nhl_betting.core import props_edge_signals as edge_mod

    rec_path = proc_dir / "props_recommendations_2026-03-06.csv"
    _write_csv(
        data_dir / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv",
        "date,player_name,team,market,line,over_price,under_price,book\n"
        "2026-03-06,Fallback Player,COL,SOG,1.5,100,-200,draftkings\n",
    )
    _write_csv(
        proc_dir / "props_projections_all_2026-03-06.csv",
        "player,team,market,proj_lambda\n",
    )
    _write_csv(
        rec_path,
        "player,team,market,proj_lambda,proj\n"
        "Fallback Player,COL,SOG,4.2,4.2\n",
    )

    os.utime(rec_path, (1000, 1000))
    os.utime(data_dir / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv", (2000, 2000))

    monkeypatch.setattr(app_mod, "_gh_upsert_file_if_better_or_same", lambda *_a, **_k: {"ok": True}, raising=True)
    monkeypatch.setattr(edge_mod, "attach_prop_edge_signals", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not run")), raising=True)

    res = app_mod._refresh_props_recommendations("2026-03-06", min_ev=0.0, top=50)

    assert res.get("ok") is True
    assert res.get("projection_source") == "recommendations_local"

    out_path = proc_dir / "props_recommendations_2026-03-06.csv"
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "Fallback Player" in text


def test_refresh_props_recommendations_uses_csv_when_parquet_is_empty(tmp_path: Path, monkeypatch):
    _repo_root, data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)
    from nhl_betting.core import props_edge_signals as edge_mod
    import pandas as pd

    lines_dir = data_dir / "props" / "player_props_lines" / "date=2026-03-06"
    lines_dir.mkdir(parents=True, exist_ok=True)
    (lines_dir / "oddsapi.parquet").write_bytes(b"stub")
    _write_csv(
        lines_dir / "oddsapi.csv",
        "date,player_name,team,market,line,over_price,under_price,book\n"
        "2026-03-06,Parquet Fallback Player,COL,SOG,1.5,100,-200,draftkings\n",
    )
    _write_csv(
        proc_dir / "props_projections_all_2026-03-06.csv",
        "player,team,market,proj_lambda\n",
    )
    _write_csv(
        proc_dir / "props_recommendations_2026-03-06.csv",
        "player,team,market,proj_lambda,proj\n"
        "Parquet Fallback Player,COL,SOG,4.2,4.2\n",
    )
    os.utime(proc_dir / "props_recommendations_2026-03-06.csv", (1000, 1000))
    os.utime(lines_dir / "oddsapi.parquet", (2000, 2000))
    os.utime(lines_dir / "oddsapi.csv", (2000, 2000))

    orig_read_parquet = app_mod.pd.read_parquet

    def _fake_read_parquet(path, *args, **kwargs):
        if str(path).endswith("oddsapi.parquet"):
            return pd.DataFrame(columns=["date", "player_name", "team", "market", "line", "over_price", "under_price", "book"])
        return orig_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(app_mod, "_gh_upsert_file_if_better_or_same", lambda *_a, **_k: {"ok": True}, raising=True)
    monkeypatch.setattr(edge_mod, "attach_prop_edge_signals", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not run")), raising=True)
    monkeypatch.setattr(app_mod.pd, "read_parquet", _fake_read_parquet, raising=True)

    res = app_mod._refresh_props_recommendations("2026-03-06", min_ev=0.0, top=50)

    assert res.get("ok") is True
    assert res.get("projection_source") == "recommendations_local"
    text = (proc_dir / "props_recommendations_2026-03-06.csv").read_text(encoding="utf-8")
    assert "Parquet Fallback Player" in text


def test_refresh_props_recommendations_keeps_csv_matches_when_parquet_rows_do_not_match(tmp_path: Path, monkeypatch):
    _repo_root, data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)
    from nhl_betting.core import props_edge_signals as edge_mod
    import pandas as pd

    lines_dir = data_dir / "props" / "player_props_lines" / "date=2026-03-06"
    lines_dir.mkdir(parents=True, exist_ok=True)
    (lines_dir / "oddsapi.parquet").write_bytes(b"stub")
    _write_csv(
        lines_dir / "oddsapi.csv",
        "date,player_name,team,market,line,over_price,under_price,book\n"
        "2026-03-06,CSV Rescue Player,COL,SOG,1.5,100,-200,draftkings\n",
    )
    _write_csv(
        proc_dir / "props_projections_all_2026-03-06.csv",
        "player,team,market,proj_lambda\n",
    )
    _write_csv(
        proc_dir / "props_recommendations_2026-03-06.csv",
        "player,team,market,proj_lambda,proj\n"
        "CSV Rescue Player,COL,SOG,4.2,4.2\n",
    )
    os.utime(proc_dir / "props_recommendations_2026-03-06.csv", (1000, 1000))
    os.utime(lines_dir / "oddsapi.parquet", (2000, 2000))
    os.utime(lines_dir / "oddsapi.csv", (2000, 2000))

    orig_read_parquet = app_mod.pd.read_parquet

    def _fake_read_parquet(path, *args, **kwargs):
        if str(path).endswith("oddsapi.parquet"):
            return pd.DataFrame([
                {
                    "date": "2026-03-06",
                    "player_name": "Unmatched Player",
                    "team": "COL",
                    "market": "SOG",
                    "line": 1.5,
                    "over_price": 100,
                    "under_price": -200,
                    "book": "draftkings",
                }
            ])
        return orig_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(app_mod, "_gh_upsert_file_if_better_or_same", lambda *_a, **_k: {"ok": True}, raising=True)
    monkeypatch.setattr(edge_mod, "attach_prop_edge_signals", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not run")), raising=True)
    monkeypatch.setattr(app_mod.pd, "read_parquet", _fake_read_parquet, raising=True)

    res = app_mod._refresh_props_recommendations("2026-03-06", min_ev=0.0, top=50)

    assert res.get("ok") is True
    text = (proc_dir / "props_recommendations_2026-03-06.csv").read_text(encoding="utf-8")
    assert "CSV Rescue Player" in text


def test_refresh_props_recommendations_keeps_existing_cache_when_refresh_would_be_empty(tmp_path: Path, monkeypatch):
    _repo_root, data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)
    from nhl_betting.core import props_edge_signals as edge_mod

    _write_csv(
        data_dir / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv",
        "date,player_name,team,market,line,over_price,under_price,book\n"
        "2026-03-06,Sticky Player,COL,HITS,1.5,100,-200,draftkings\n",
    )
    _write_csv(
        proc_dir / "props_projections_all_2026-03-06.csv",
        "player,team,market,proj_lambda\n",
    )
    rec_path = proc_dir / "props_recommendations_2026-03-06.csv"
    _write_csv(
        rec_path,
        "player,team,market,proj_lambda,proj\n"
        "Sticky Player,COL,HITS,1.0,1.0\n",
    )
    os.utime(rec_path, (1000, 1000))
    os.utime(data_dir / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv", (2000, 2000))

    monkeypatch.setattr(app_mod, "_gh_upsert_file_if_better_or_same", lambda *_a, **_k: {"ok": True}, raising=True)
    monkeypatch.setattr(edge_mod, "attach_prop_edge_signals", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not run")), raising=True)

    res = app_mod._refresh_props_recommendations("2026-03-06", min_ev=0.0, top=50)

    assert res.get("ok") is True
    assert res.get("reason") == "empty-refresh-output-kept-existing"
    text = rec_path.read_text(encoding="utf-8")
    assert "Sticky Player" in text


def test_refresh_props_recommendations_applies_cli_under_side_gates(tmp_path: Path, monkeypatch):
    _repo_root, data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)
    from nhl_betting.core import props_edge_signals as edge_mod

    _write_csv(
        data_dir / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv",
        "date,player_name,team,market,line,over_price,under_price,book\n"
        "2026-03-06,Over Favorite,COL,POINTS,0.5,-210,165,draftkings\n"
        "2026-03-06,Points Under Keep,COL,POINTS,1.5,110,-120,draftkings\n"
        "2026-03-06,Points Under Drop,COL,POINTS,1.5,110,-130,draftkings\n"
        "2026-03-06,Goals Under Keep,COL,GOALS,0.5,110,-160,draftkings\n"
        "2026-03-06,Goals Under Drop,COL,GOALS,0.5,110,-175,draftkings\n"
        "2026-03-06,Under Low Ev,COL,POINTS,1.5,110,-120,draftkings\n",
    )
    _write_csv(
        proc_dir / "props_projections_all_2026-03-06.csv",
        "player,team,market,proj_lambda\n"
        "Over Favorite,COL,POINTS,1.2\n"
        "Points Under Keep,COL,POINTS,1.0\n"
        "Points Under Drop,COL,POINTS,1.0\n"
        "Goals Under Keep,COL,GOALS,0.4\n"
        "Goals Under Drop,COL,GOALS,0.4\n"
        "Under Low Ev,COL,POINTS,1.0\n",
    )

    side_map = {
        "Over Favorite": "Over",
        "Points Under Keep": "Under",
        "Points Under Drop": "Under",
        "Goals Under Keep": "Under",
        "Goals Under Drop": "Under",
        "Under Low Ev": "Under",
    }
    prob_map = {
        "Over Favorite": 0.70,
        "Points Under Keep": 0.70,
        "Points Under Drop": 0.72,
        "Goals Under Keep": 0.82,
        "Goals Under Drop": 0.82,
        "Under Low Ev": 0.60,
    }
    edge_map = {
        "Over Favorite": 0.90,
        "Points Under Keep": 0.80,
        "Points Under Drop": 0.70,
        "Goals Under Keep": 0.60,
        "Goals Under Drop": 0.50,
        "Under Low Ev": 0.40,
    }

    def _fake_attach_prop_edge_signals(*, date, props):
        out = props.copy()
        out["side_suggested"] = out["player"].map(side_map)
        out["chosen_prob"] = pd.to_numeric(out["player"].map(prob_map), errors="coerce")
        out["edge_score"] = pd.to_numeric(out["player"].map(edge_map), errors="coerce")
        out["edge_reasons"] = "test"
        out["edge_drivers"] = "test"
        return out

    monkeypatch.setattr(app_mod, "_gh_upsert_file_if_better_or_same", lambda *_a, **_k: {"ok": True}, raising=True)
    monkeypatch.setattr(app_mod, "_github_raw_read_csv", lambda *_a, **_k: pd.DataFrame(), raising=True)
    monkeypatch.setattr(app_mod, "_github_raw_read_parquet", lambda *_a, **_k: pd.DataFrame(), raising=True)
    monkeypatch.setattr(edge_mod, "attach_prop_edge_signals", _fake_attach_prop_edge_signals, raising=True)

    res = app_mod._refresh_props_recommendations("2026-03-06", min_ev=0.0, top=50)

    assert res.get("ok") is True
    out = pd.read_csv(proc_dir / "props_recommendations_2026-03-06.csv")

    assert set(out["player"].tolist()) == {
        "Over Favorite",
        "Points Under Keep",
        "Goals Under Keep",
    }


def test_api_props_recommendations_self_heal_route_applies_shared_under_gates(tmp_path: Path, monkeypatch):
    _repo_root, data_dir, proc_dir = _set_render_like_paths(tmp_path, monkeypatch)
    from nhl_betting.core import props_edge_signals as edge_mod

    _write_csv(
        data_dir / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv",
        "date,player_name,team,market,line,over_price,under_price,book\n"
        "2026-03-06,Over Favorite,COL,POINTS,0.5,-210,165,draftkings\n"
        "2026-03-06,Points Under Keep,COL,POINTS,1.5,110,-120,draftkings\n"
        "2026-03-06,Points Under Drop,COL,POINTS,1.5,110,-130,draftkings\n"
        "2026-03-06,Goals Under Keep,COL,GOALS,0.5,110,-160,draftkings\n"
        "2026-03-06,Goals Under Drop,COL,GOALS,0.5,110,-175,draftkings\n"
        "2026-03-06,Under Low Ev,COL,POINTS,1.5,110,-120,draftkings\n",
    )
    _write_csv(
        proc_dir / "props_projections_all_2026-03-06.csv",
        "player,team,market,proj_lambda\n"
        "Over Favorite,COL,POINTS,1.2\n"
        "Points Under Keep,COL,POINTS,1.0\n"
        "Points Under Drop,COL,POINTS,1.0\n"
        "Goals Under Keep,COL,GOALS,0.4\n"
        "Goals Under Drop,COL,GOALS,0.4\n"
        "Under Low Ev,COL,POINTS,1.0\n",
    )

    side_map = {
        "Over Favorite": "Over",
        "Points Under Keep": "Under",
        "Points Under Drop": "Under",
        "Goals Under Keep": "Under",
        "Goals Under Drop": "Under",
        "Under Low Ev": "Under",
    }
    prob_map = {
        "Over Favorite": 0.70,
        "Points Under Keep": 0.70,
        "Points Under Drop": 0.72,
        "Goals Under Keep": 0.82,
        "Goals Under Drop": 0.82,
        "Under Low Ev": 0.60,
    }
    edge_map = {
        "Over Favorite": 0.90,
        "Points Under Keep": 0.80,
        "Points Under Drop": 0.70,
        "Goals Under Keep": 0.60,
        "Goals Under Drop": 0.50,
        "Under Low Ev": 0.40,
    }

    def _fake_attach_prop_edge_signals(*, date, props):
        out = props.copy()
        out["side_suggested"] = out["player"].map(side_map)
        out["chosen_prob"] = pd.to_numeric(out["player"].map(prob_map), errors="coerce")
        out["edge_score"] = pd.to_numeric(out["player"].map(edge_map), errors="coerce")
        out["edge_reasons"] = "test"
        out["edge_drivers"] = "test"
        return out

    monkeypatch.setattr(app_mod, "_read_only", lambda *_a, **_k: False, raising=True)
    monkeypatch.setattr(app_mod, "_seed_repo_props_artifacts_to_active_dirs", lambda *_a, **_k: {"ok": True}, raising=True)
    monkeypatch.setattr(app_mod, "_gh_upsert_file_if_better_or_same", lambda *_a, **_k: {"ok": True}, raising=True)
    monkeypatch.setattr(app_mod, "_github_raw_read_csv", lambda *_a, **_k: pd.DataFrame(), raising=True)
    monkeypatch.setattr(app_mod, "_github_raw_read_parquet", lambda *_a, **_k: pd.DataFrame(), raising=True)
    monkeypatch.setattr(edge_mod, "attach_prop_edge_signals", _fake_attach_prop_edge_signals, raising=True)

    client = TestClient(app_mod.app)
    r = client.get("/api/props/recommendations", params={"date": "2026-03-06", "top": 50})

    assert r.status_code == 200
    payload = r.json()
    assert payload.get("date") == "2026-03-06"
    assert {row.get("player") for row in (payload.get("data") or [])} == {
        "Over Favorite",
        "Points Under Keep",
        "Goals Under Keep",
    }

    out = pd.read_csv(proc_dir / "props_recommendations_2026-03-06.csv")
    assert set(out["player"].tolist()) == {
        "Over Favorite",
        "Points Under Keep",
        "Goals Under Keep",
    }


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


def test_v1_props_cards_recovers_headshot_from_lines_when_snapshot_join_misses(tmp_path: Path, monkeypatch):
    repo_root, data_dir, _proc_dir = _set_render_like_paths(tmp_path, monkeypatch)

    _write_csv(
        repo_root / "data" / "processed" / "props_recommendations_2026-03-06.csv",
        "player,team,opp,market,side,book,line,price,ev\n"
        "Cutter Gauthier,ANA,MTL,POINTS,Under,betonlineag,0.5,-105,0.11\n",
    )
    _write_csv(
        data_dir / "props" / "player_props_lines" / "date=2026-03-06" / "oddsapi.csv",
        "date,player_id,player_name,team,market,line,over_price,under_price,book,first_seen_at,last_seen_at,is_current,_merge_key\n"
        "2026-03-06,8483445,Cutter Gauthier,,POINTS,0.5,-125,105,draftkings,2026-03-06T14:32:42Z,2026-03-06T14:33:06Z,True,id::8483445\n",
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
    assert cards[0].get("player_id") == "8483445"
    assert cards[0].get("headshot_url") == "https://assets.nhle.com/mugs/nhl/20252026/ANA/8483445.png"


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
