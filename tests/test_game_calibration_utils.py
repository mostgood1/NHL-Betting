import json
from pathlib import Path

from nhl_betting.utils.calibration import BinaryCalibration, load_calibration, load_game_calibration


def test_binary_calibration_two_way_preserves_push_mass() -> None:
    cal = BinaryCalibration(t=2.0, b=0.0)

    p_over, p_under = cal.apply_two_way(0.56, push_mass=0.08)

    assert abs((p_over + p_under) - 0.92) < 1e-9
    assert 0.0 < p_over < 0.56
    assert 0.36 < p_under < 0.92


def test_load_game_calibration_reads_flat_keys(tmp_path: Path) -> None:
    path = tmp_path / "model_calibration.json"
    path.write_text(
        """
{
  "dc_rho": -0.03,
  "market_anchor_w": 0.21,
  "market_anchor_w_totals": 0.34,
  "ml_temp": 1.15,
  "ml_bias": -0.05,
  "totals_temp": 1.25
}
""".strip(),
        encoding="utf-8",
    )

    cfg = load_game_calibration(path)
    ml_cal, tot_cal = load_calibration(path)

    assert cfg["dc_rho"] == -0.03
    assert cfg["market_anchor_w_ml"] == 0.21
    assert cfg["market_anchor_w_totals"] == 0.34
    assert cfg["moneyline"].t == 1.15
    assert cfg["moneyline"].b == -0.05
    assert cfg["totals"].t == 1.25
    assert ml_cal.t == 1.15
    assert ml_cal.b == -0.05
    assert tot_cal.t == 1.25
    assert tot_cal.b == 0.0


def test_load_game_calibration_prefers_nested_blocks(tmp_path: Path) -> None:
    path = tmp_path / "model_calibration.json"
    path.write_text(
        """
{
  "ml_temp": 1.05,
  "moneyline": {"t": 1.3, "b": 0.07},
  "totals": {"t": 1.2, "b": -0.01}
}
""".strip(),
        encoding="utf-8",
    )

    cfg = load_game_calibration(path)

    assert cfg["moneyline"].t == 1.3
    assert cfg["moneyline"].b == 0.07
    assert cfg["totals"].t == 1.2
    assert cfg["totals"].b == -0.01


def test_game_auto_calibrate_defaults_start_from_end(tmp_path: Path, monkeypatch) -> None:
    from nhl_betting import cli as cli_module

    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()
    proc_dir.mkdir()

    monkeypatch.setattr("nhl_betting.utils.io.RAW_DIR", raw_dir)
    monkeypatch.setattr("nhl_betting.utils.io.PROC_DIR", proc_dir)

    out_path = proc_dir / "model_calibration.json"
    cli_module.game_auto_calibrate(start=None, end="2026-03-16", out_json=str(out_path))

    cfg = json.loads(out_path.read_text(encoding="utf-8"))

    assert cfg["range"] == {"start": "2025-09-01", "end": "2026-03-16"}
    assert cfg["ml_fit_range"]["start"] == "2025-09-01"
    assert cfg["ml_fit_range"]["end"] == "2026-03-16"
    assert cfg["ml_fit_range"]["mode"] == "fallback_all"


def test_game_auto_calibrate_preserves_existing_keys(tmp_path: Path, monkeypatch) -> None:
    from nhl_betting import cli as cli_module

    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()
    proc_dir.mkdir()

    monkeypatch.setattr("nhl_betting.utils.io.RAW_DIR", raw_dir)
    monkeypatch.setattr("nhl_betting.utils.io.PROC_DIR", proc_dir)

    out_path = proc_dir / "model_calibration.json"
    out_path.write_text(
        json.dumps(
            {
                "min_ev_ml": 0.095,
                "min_ev_totals": 0.075,
                "min_ev_pl": 0.095,
                "custom_marker": "keep",
            }
        ),
        encoding="utf-8",
    )

    cli_module.game_auto_calibrate(start=None, end="2026-03-16", out_json=str(out_path))

    cfg = json.loads(out_path.read_text(encoding="utf-8"))

    assert cfg["min_ev_ml"] == 0.095
    assert cfg["min_ev_totals"] == 0.075
    assert cfg["min_ev_pl"] == 0.095
    assert cfg["custom_marker"] == "keep"
