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
