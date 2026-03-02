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

    by_edge = obj.get("by_edge_bucket") or []
    eb = {str(x.get("key")): x for x in by_edge}
    assert ">=0.06" in eb
    assert eb[">=0.06"]["bets"] == 2

    by_tag = obj.get("by_driver_tag") or []
    tg = {str(x.get("key")): x for x in by_tag}
    assert "market:TOTAL" in tg
    assert tg["market:TOTAL"]["bets"] == 2
