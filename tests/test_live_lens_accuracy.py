import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import nhl_betting.web.app as web_app


def test_live_lens_accuracy_data_reads_perf_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    perf_dir = tmp_path / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)

    # Minimal settled-ledger records (as produced by check_live_lens_betting_performance.py --out *.jsonl)
    recs = [
        {
            "date": "2099-01-01",
            "gamePk": 1,
            "market": "TOTAL",
            "elapsed_bucket": "20-30",
            "edge_bucket": ">=0.06",
            "driver_tags": ["market:TOTAL", "pace:down"],
            "result": "WIN",
            "profit_units": 0.9,
        },
        {
            "date": "2099-01-01",
            "gamePk": 2,
            "market": "TOTAL",
            "elapsed_bucket": "20-30",
            "edge_bucket": ">=0.06",
            "driver_tags": ["market:TOTAL"],
            "result": "LOSE",
            "profit_units": -1.0,
        },
        {
            "date": "2099-01-01",
            "gamePk": 3,
            "market": "PERIOD_TOTAL",
            "elapsed_bucket": "10-20",
            "edge_bucket": "0.03-0.04",
            "driver_tags": [],
            "result": "PUSH",
            "profit_units": 0.0,
        },
    ]

    p = perf_dir / "live_lens_bets_2099-01-01_test.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    monkeypatch.setenv("LIVE_LENS_PERF_DIR", str(perf_dir))

    # Avoid cross-test cache contamination.
    try:
        web_app._CACHE.clear()
    except Exception:
        pass

    client = TestClient(web_app.app)
    r = client.get("/api/live-lens-accuracy/data?date=2099-01-01")
    assert r.status_code == 200

    obj = r.json()
    assert obj.get("ok") is True

    s = obj.get("summary")
    assert s is not None
    assert s["bets"] == 3
    assert s["wins"] == 1
    assert s["losses"] == 1
    assert s["pushes"] == 1
    assert abs(float(s["units"]) - (-0.1)) < 1e-9

    by_market = obj.get("by_market") or []
    mk = {str(x.get("key")): x for x in by_market}
    assert "TOTAL" in mk
    assert mk["TOTAL"]["bets"] == 2

    by_date = obj.get("by_date") or []
    dd = {str(x.get("key")): x for x in by_date}
    assert "2099-01-01" in dd
    assert dd["2099-01-01"]["bets"] == 3

    by_edge = obj.get("by_edge_bucket") or []
    eb = {str(x.get("key")): x for x in by_edge}
    assert ">=0.06" in eb
    assert eb[">=0.06"]["bets"] == 2

    by_tag = obj.get("by_driver_tag") or []
    tg = {str(x.get("key")): x for x in by_tag}
    assert "market:TOTAL" in tg
    assert tg["market:TOTAL"]["bets"] == 2

    by_tt = obj.get("by_tag_type") or []
    tt = {str(x.get("key")): x for x in by_tt}
    assert "market" in tt
    assert tt["market"]["bets"] == 2
    assert "pace" in tt
    assert tt["pace"]["bets"] == 1
    assert "(none)" in tt
    assert tt["(none)"]["bets"] == 1


def test_live_lens_accuracy_page_alias_renders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    perf_dir = tmp_path / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)

    p = perf_dir / "live_lens_bets_2099-01-01_test.jsonl"
    p.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("LIVE_LENS_PERF_DIR", str(perf_dir))

    client = TestClient(web_app.app)
    r = client.get("/live_lens_accuracy?date=2099-01-01")
    assert r.status_code == 200
    assert "Live Lens Accuracy" in r.text


def test_live_lens_accuracy_seeds_repo_and_falls_back_to_latest_settled_date(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    active_proc = tmp_path / "active" / "processed"
    active_perf = active_proc / "live_lens" / "perf"
    active_perf.mkdir(parents=True, exist_ok=True)

    repo_proc = tmp_path / "repo" / "processed"
    repo_perf = repo_proc / "live_lens" / "perf"
    repo_perf.mkdir(parents=True, exist_ok=True)

    recs = [
        {
            "date": "2099-01-01",
            "market": "TOTAL",
            "result": "WIN",
            "profit_units": 0.9,
        },
        {
            "date": "2099-01-03",
            "market": "ML",
            "result": "LOSE",
            "profit_units": -1.0,
        },
    ]
    with (repo_perf / "live_lens_bets_all.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    monkeypatch.setattr(web_app, "PROC_DIR", active_proc)
    monkeypatch.setattr(web_app, "_repo_proc_dir", lambda: repo_proc)
    monkeypatch.setattr(web_app, "_seed_repo_bundle_artifacts_to_proc_dir", lambda *args, **kwargs: {"checked": 0, "copied": 0})
    monkeypatch.setattr(web_app, "_seed_repo_props_artifacts_to_active_dirs", lambda *args, **kwargs: {"checked": 0, "copied": 0})
    monkeypatch.setenv("LIVE_LENS_PERF_DIR", str(active_perf))

    try:
        web_app._CACHE.clear()
    except Exception:
        pass

    with TestClient(web_app.app) as client:
        r = client.get("/api/live-lens-accuracy/data?date=2099-01-10")

    assert r.status_code == 200
    obj = r.json()
    assert obj.get("ok") is True
    assert obj.get("requested_date") == "2099-01-10"
    assert obj.get("start") == "2099-01-03"
    assert obj.get("end") == "2099-01-03"
    assert obj.get("latest_available_date") == "2099-01-03"
    assert obj.get("fallback_applied") is True
    assert obj.get("rows") == 1
    assert (active_perf / "live_lens_bets_all.jsonl").exists()


def test_pregame_accuracy_data_reads_logs_and_normalizes_game_dates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    active_proc = tmp_path / "active" / "processed"
    active_proc.mkdir(parents=True, exist_ok=True)
    repo_proc = tmp_path / "repo" / "processed"
    repo_proc.mkdir(parents=True, exist_ok=True)

    (repo_proc / "reconciliations_log.csv").write_text(
        "date,home,away,market,bet,ev,price,result,stake,payout\n"
        "2099-01-01T23:30:00Z,Home A,Away A,moneyline,home_ml,0.11,-110,win,100,90.9091\n"
        "2099-01-02T00:30:00Z,Home B,Away B,totals,under,0.07,-105,loss,100,-100\n",
        encoding="utf-8",
    )
    (repo_proc / "props_reconciliations_log.csv").write_text(
        "date,market,player,line,side,odds,ev,actual,result,stake,payout\n"
        "2099-01-01,SOG,Player One,2.5,Over,-110,0.14,3,win,100,90.9091\n"
        "2099-01-01,GOALS,Player Two,0.5,Under,120,0.05,1,loss,100,-100\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(web_app, "PROC_DIR", active_proc)
    monkeypatch.setattr(web_app, "_repo_proc_dir", lambda: repo_proc)
    monkeypatch.setattr(web_app, "_seed_repo_bundle_artifacts_to_proc_dir", lambda *args, **kwargs: {"checked": 0, "copied": 0})
    monkeypatch.setattr(web_app, "_seed_repo_props_artifacts_to_active_dirs", lambda *args, **kwargs: {"checked": 0, "copied": 0})

    try:
        web_app._CACHE.clear()
    except Exception:
        pass

    with TestClient(web_app.app) as client:
        r = client.get("/api/pregame-accuracy/data?date=2099-01-01")

    assert r.status_code == 200
    obj = r.json()
    assert obj.get("ok") is True
    assert obj.get("start") == "2099-01-01"
    assert obj.get("end") == "2099-01-01"
    assert obj.get("fallback_applied") is False

    games = obj.get("games") or {}
    props = obj.get("props") or {}
    combined = obj.get("combined") or {}

    assert games.get("rows") == 2
    assert props.get("rows") == 2
    assert (combined.get("summary") or {}).get("bets") == 4

    by_game_date = {str(row.get("key")): row for row in (games.get("by_date") or [])}
    assert by_game_date["2099-01-01"]["bets"] == 2

    by_game_market = {str(row.get("key")): row for row in (games.get("by_market") or [])}
    assert by_game_market["moneyline"]["bets"] == 1
    assert by_game_market["totals"]["bets"] == 1

    by_props_market = {str(row.get("key")): row for row in (props.get("by_market") or [])}
    assert by_props_market["SOG"]["bets"] == 1
    assert by_props_market["GOALS"]["bets"] == 1
