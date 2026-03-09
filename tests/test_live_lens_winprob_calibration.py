import pytest

import nhl_betting.web.app as web_app
from nhl_betting.utils.live_lens_time import (
    live_lens_calibration_segment_candidates,
    pick_live_lens_calibration_spec,
)


def test_live_lens_calibration_segment_candidates_use_ot_15s_buckets():
    keys = live_lens_calibration_segment_candidates(
        "poisson",
        period=4,
        clock="04:31",
        remaining_min=0.0,
    )

    assert keys[:4] == [
        "src=poisson|phase=OT|t15=60:15-60:30",
        "src=poisson|phase=OT|t1=60:00-61:00",
        "src=poisson|phase=OT",
        "src=poisson|rm=0-5",
    ]
    assert keys[-1] == "src=poisson"


def test_pick_live_lens_calibration_spec_prefers_finer_time_segment():
    obj = {
        "default": {"kind": "temp_shift", "t": 1.0, "b": 0.0},
        "segments": {
            "src=poisson|phase=REG|t1=20:00-21:00": {"kind": "temp_shift", "t": 0.9, "b": 0.0},
            "src=poisson|rm=20-40": {"kind": "temp_shift", "t": 0.8, "b": 0.0},
        },
    }

    spec, key = pick_live_lens_calibration_spec(
        obj,
        "poisson",
        elapsed_min=20.25,
        remaining_min=39.75,
    )

    assert key == "src=poisson|phase=REG|t1=20:00-21:00"
    assert pytest.approx(float(spec.get("t") or 0.0), abs=1e-9) == 0.9


def test_web_pick_live_lens_calibration_spec_falls_back_to_legacy_rm_key():
    obj = {
        "default": {"kind": "temp_shift", "t": 1.0, "b": 0.0},
        "segments": {
            "src=poisson|rm=20-40": {"kind": "temp_shift", "t": 0.7, "b": 0.1},
        },
    }

    spec = web_app._pick_live_lens_winprob_calibration_spec(
        obj,
        39.75,
        "poisson",
        elapsed_min=20.25,
        period=2,
        clock="19:45",
    )

    assert pytest.approx(float(spec.get("t") or 0.0), abs=1e-9) == 0.7
    assert pytest.approx(float(spec.get("b") or 0.0), abs=1e-9) == 0.1


def test_web_apply_live_lens_winprob_calibration_prefers_15_second_key(monkeypatch: pytest.MonkeyPatch):
    obj = {
        "default": {"kind": "temp_shift", "t": 1.0, "b": 0.0},
        "segments": {
            "src=poisson|phase=REG|t15=20:15-20:30": {"kind": "temp_shift", "t": 0.5, "b": 0.0},
            "src=poisson|phase=REG|t1=20:00-21:00": {"kind": "temp_shift", "t": 2.0, "b": 0.0},
        },
    }

    monkeypatch.setattr(web_app, "_load_live_lens_winprob_calibration", lambda: obj)

    calibrated = web_app._apply_live_lens_winprob_calibration(
        0.60,
        39.75,
        "poisson",
        elapsed_min=20.25,
        period=2,
        clock="19:45",
    )

    assert calibrated > 0.60