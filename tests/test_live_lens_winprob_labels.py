import pytest

from scripts.fit_live_lens_winprob_calibration import _extract_home_win_label as fit_extract_home_win_label
from scripts.report_live_lens_winprob_monitor import _extract_home_win_label as monitor_extract_home_win_label


@pytest.mark.parametrize("extractor", [fit_extract_home_win_label, monitor_extract_home_win_label])
def test_extract_home_win_label_uses_explicit_label(extractor):
    rec = {"home_win": 1, "final": True, "home_goals_final": 1, "away_goals_final": 4}

    assert extractor(rec) == 1


@pytest.mark.parametrize("extractor", [fit_extract_home_win_label, monitor_extract_home_win_label])
def test_extract_home_win_label_falls_back_to_final_scores(extractor):
    assert extractor({"home_goals_final": 4, "away_goals_final": 2}) == 1
    assert extractor({"home_goals_final": 1, "away_goals_final": 3}) == 0


@pytest.mark.parametrize("extractor", [fit_extract_home_win_label, monitor_extract_home_win_label])
def test_extract_home_win_label_respects_non_final_rows(extractor):
    rec = {"final": False, "home_win": 1, "home_goals_final": 4, "away_goals_final": 2}

    assert extractor(rec) is None